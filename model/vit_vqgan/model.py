
import lightning as L
import torch
from typing import Optional, Tuple
from torch import Tensor

from model.vit_vqgan.layers import Config
from model.vit_vqgan.loss import VQLPIPSWithDiscriminator
from model.vit_vqgan.quantizer import FSQ
from model.vit_vqgan.layers import ViTEncoder, ViTDecoder, Config


class ViTVQGAN(L.LightningModule):
    def __init__(self, config: Config, learning_rate: float = 1e-4, grad_accum_steps: int = 1):
        super().__init__()
        self.config = config

        self.encoder = ViTEncoder(config)
        self.quantize = FSQ(
            config['levels'], dim=config['hidden_size'], num_codebooks=config['num_codebooks'])
        self.decoder = ViTDecoder(config)
        self.loss = VQLPIPSWithDiscriminator(
            disc_start=10000, perceptual_weight=0, codebook_weight=0, disc_weight=0.1, disc_in_channels=config['num_channels'],
            disc_conditional=False,)
        self.learning_rate = learning_rate
        self.grad_accum_steps = grad_accum_steps
        self.automatic_optimization = False

    def encode(self, pixels: Tensor, interpolate_pos_encoding: Optional[bool] = False) -> Tuple[Tensor, Tensor]:
        x = self.encoder(pixels, interpolate_pos_encoding)
        x, indices = self.quantize(x)
        return x, indices

    def decode(self, indices: Tensor) -> Tensor:
        hidden = self.quantize.indices_to_codes(indices)
        return self.decoder(hidden)

    def forward(self, pixels: Tensor, interpolate_pos_encoding: Optional[bool] = False) -> Tuple[Tensor, Tensor]:
        x = self.encoder(pixels, interpolate_pos_encoding)
        x, indices = self.quantize(x)
        recon = self.decoder(x)
        return recon, indices

    @staticmethod
    def get_input(x: Tensor) -> Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(0)
        return x.contiguous()

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch)
        xrec, _ = self(x)

        qloss = torch.tensor(0.0, device=x.device)

        opt_ae, opt_disc = self.optimizers()

        ######################
        # Optimize Generator #
        ######################
        aeloss, log_dict_ae = self.loss(
            qloss, x, xrec, 0, self.global_step, last_layer=self.get_last_layer(), split="train")
        self.manual_backward(aeloss / self.grad_accum_steps)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        if (batch_idx + 1) % self.grad_accum_steps == 0:
            opt_ae.step()
            opt_ae.zero_grad()

        self.log("train/aeloss", aeloss, prog_bar=True,
                 logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False,
                      logger=True, on_step=True, on_epoch=True)

        ##########################
        # Optimize Discriminator #
        ##########################
        discloss, log_dict_disc = self.loss(
            qloss, x, xrec, 1, self.global_step, last_layer=self.get_last_layer(), split="train")
        self.manual_backward(discloss / self.grad_accum_steps)

        if (batch_idx + 1) % self.grad_accum_steps == 0:
            opt_disc.step()
            opt_disc.zero_grad()

        self.log("train/discloss", discloss, prog_bar=True,
                 logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False,
                      logger=True, on_step=True, on_epoch=True)

    def validation_step(self, x, **kwargs):
        x = self.get_input(x)
        xrec, _ = self(x)
        qloss = torch.tensor(0.0, device=x.device)
        aeloss, log_dict_ae = self.loss(
            qloss, x, xrec, 0, self.global_step, last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(
            qloss, x, xrec, 1, self.global_step, last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss, prog_bar=True, logger=True,
                 on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss, prog_bar=True, logger=True,
                 on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.AdamW(list(self.encoder.parameters(
        ))+list(self.decoder.parameters())+list(self.quantize.parameters()), lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.AdamW(
            self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self): return self.decoder.get_last_layer()
