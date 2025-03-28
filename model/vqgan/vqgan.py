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
    def __init__(self, ddconfig=dd_config,  lossconfig: Optional[dict] = None, n_embed: int=8192, embed_dim: int=256, grad_accum_steps: int=1):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.learning_rate = 5e-4
        self.loss = VQLPIPSWithDiscriminator(**lossconfig) if lossconfig is not None else None

        self.automatic_optimization = False
        self.grad_accum_steps = grad_accum_steps

    def encode(self, x): return self.quantize(self.quant_conv(self.encoder(x)))

    def decode(self, quant): return self.decoder(self.post_quant_conv(quant))

    def decode_code(self, code_b): return self.decode(self.quantize.embed_code(code_b))

    def forward(self, x):
        quant, diff, _ = self.encode(x)
        return self.decode(quant), diff

    @staticmethod
    def get_input(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3: x = x.unsqueeze(0)
        return x.contiguous()

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch)
        xrec, qloss = self(x)

        opt_ae, opt_disc = self.optimizers()

        ######################
        # Optimize Generator #
        ######################
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step, last_layer=self.get_last_layer(), split="train")
        self.manual_backward(aeloss / self.grad_accum_steps)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        if (batch_idx + 1) % self.grad_accum_steps == 0:
            opt_ae.step()
            opt_ae.zero_grad()

        self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        ##########################
        # Optimize Discriminator #
        ##########################
        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step, last_layer=self.get_last_layer(), split="train")
        self.manual_backward(discloss / self.grad_accum_steps)

        if (batch_idx + 1) % self.grad_accum_steps == 0:
            opt_disc.step()
            opt_disc.zero_grad()

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
        opt_ae = torch.optim.AdamW(list(self.encoder.parameters())+list(self.decoder.parameters())+list(self.quantize.parameters())+list(self.quant_conv.parameters())+list(self.post_quant_conv.parameters()), lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.AdamW(self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self): return self.decoder.conv_out.weight
