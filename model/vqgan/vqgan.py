from types import MappingProxyType
from typing import Optional

import lightning as L
import torch
import torch.nn.functional as F

from .discriminator import VQLPIPSWithDiscriminator
from .quantize import VectorQuantizer
from .vqgan_blocks import Encoder, Decoder

dd_config = MappingProxyType(dict(
    double_z=False,
    z_channels=256,
    resolution=512,
    in_channels=3,
    out_ch=3,
    ch=128,
    ch_mult=[1, 1, 2, 2, 4],  # num_down = len(ch_mult)-1
    num_res_blocks=2,
    attn_resolutions=[],
))
loss_config = MappingProxyType(dict(
    disc_conditional=False,
    disc_in_channels=3,
    disc_start=10000,
    disc_weight=0.8,
    codebook_weight=1.0,
))


class VQModel(L.LightningModule):
    # https://github.com/CompVis/taming-transformers/blob/master/taming/models/vqgan.py
    def __init__(self, ddconfig=dd_config,  lossconfig: Optional[dict] = None, n_embed: int=8192, embed_dim: int=256, colorize_nlabels=None, monitor=None, remap=None, sane_index_shape=False,):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25, remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.learning_rate = 2e-4
        if colorize_nlabels is not None: self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        self.monitor = monitor
        self.loss = VQLPIPSWithDiscriminator(**lossconfig) if lossconfig is not None else None

        self.automatic_optimization = False

    def encode(self, x): return self.quantize(self.quant_conv(self.encoder(x)))

    def decode(self, quant):  return self.decoder(self.post_quant_conv(quant))

    def decode_code(self, code_b): return self.decode(self.quantize.embed_code(code_b))

    def forward(self, x):
        quant, diff, _ = self.encode(x)
        return self.decode(quant), diff

    @staticmethod
    def get_input(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3: x = x.unsqueeze(0)
        return x.contiguous()

    def training_step(self, x, **kwargs):
        x = self.get_input(x)
        xrec, qloss = self(x)

        opt_ae, opt_disc = self.optimizers()

        ######################
        # Optimize Generator #
        ######################
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step, last_layer=self.get_last_layer(), split="train")
        opt_ae.zero_grad()
        self.manual_backward(aeloss)
        opt_ae.step()

        self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        ##########################
        # Optimize Discriminator #
        ##########################
        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step, last_layer=self.get_last_layer(), split="train")

        opt_disc.zero_grad()
        self.manual_backward(discloss)
        opt_disc.step()

        self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)

    def validation_step(self, x, **kwargs):
        x = self.get_input(x)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step, last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step, last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+list(self.decoder.parameters())+list(self.quantize.parameters())+list(self.quant_conv.parameters())+list(self.post_quant_conv.parameters()), lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch).to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x
