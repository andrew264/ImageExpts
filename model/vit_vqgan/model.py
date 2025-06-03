import lightning as L
import torch
from typing import Tuple
from torch import Tensor, nn

from model.vit_vqgan.config import ViTVQGANConfig
from model.vit_vqgan.layers import ViTEncoder, ViTDecoder
from model.vit_vqgan.quantizer import FSQ
from model.vit_vqgan.loss import VQLPIPSWithDiscriminator


class ViTVQGAN(L.LightningModule):
    def __init__(self, config: ViTVQGANConfig, learning_rate: float = 1e-4, grad_accum_steps: int = 1):
        super().__init__()
        self.config = config

        self.encoder = ViTEncoder(
            encoder_config=config.encoder_config,
            patch_size=config.patch_size,
            num_channels=config.num_channels
        )
        
        self.quantize = FSQ(
            levels=config.fsq_config.levels,
            dim=config.encoder_config.hidden_size, 
            output_dim=config.decoder_config.hidden_size,
            num_codebooks=config.fsq_config.num_codebooks
        )
        
        self.decoder = ViTDecoder(
            decoder_config=config.decoder_config,
            patch_size=config.patch_size,
            num_channels=config.num_channels
        )
        
        self.loss = VQLPIPSWithDiscriminator(
            disc_start=config.disc_start,
            disc_weight=config.disc_weight,
            disc_num_layers=config.disc_num_layers,
            disc_in_channels=config.num_channels,
            disc_ndf=config.disc_ndf,
            disc_loss_type=config.disc_loss_type
        )
        
        self.learning_rate = learning_rate
        self.grad_accum_steps = grad_accum_steps
        self.automatic_optimization = False

    def encode(self, pixels: Tensor) -> Tuple[Tensor, Tensor, Tuple[int, int]]:
        encoded_features, grid_hw = self.encoder(pixels)
        quantized_features, indices = self.quantize(encoded_features)
        return quantized_features, indices, grid_hw

    def decode(self, features_or_indices: Tensor, grid_hw: Tuple[int, int], is_indices: bool = True) -> Tensor:
        if is_indices:
            hidden = self.quantize.indices_to_codes(features_or_indices)
        else:
            hidden = features_or_indices
        return self.decoder(hidden, grid_hw)

    def forward(self, pixels: Tensor) -> Tuple[Tensor, Tensor]:
        encoded_features, grid_hw = self.encoder(pixels)
        quantized_features, indices = self.quantize(encoded_features)
        recon = self.decoder(quantized_features, grid_hw)
        return recon, indices

    @staticmethod
    def get_input(batch_item: Tensor) -> Tensor:
        if batch_item.ndim == 3:
            batch_item = batch_item.unsqueeze(0)
        return batch_item.contiguous()

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch)
        xrec, _ = self(x)

        opt_ae, opt_disc = self.optimizers()

        # Generator (autoencoder) optimization
        aeloss, log_dict_ae = self.loss(
            inputs=x, 
            reconstructions=xrec, 
            optimizer_idx=0, 
            global_step=self.global_step, 
            last_layer_params=[self.get_last_layer()], 
            split="train"
        )
        
        self.manual_backward(aeloss / self.grad_accum_steps)
        
        if (batch_idx + 1) % self.grad_accum_steps == 0:
            self.clip_gradients(opt_ae, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            opt_ae.step()
            opt_ae.zero_grad()

        self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        # Discriminator optimization
        discloss, log_dict_disc = self.loss(
            inputs=x, 
            reconstructions=xrec, 
            optimizer_idx=1, 
            global_step=self.global_step,
            last_layer_params=[self.get_last_layer()],
            split="train"
        )
        
        self.manual_backward(discloss / self.grad_accum_steps)

        if (batch_idx + 1) % self.grad_accum_steps == 0:
            self.clip_gradients(opt_disc, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            opt_disc.step()
            opt_disc.zero_grad()

        self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch)
        xrec, _ = self(x)
        
        aeloss, log_dict_ae = self.loss(
            x, xrec, 0, self.global_step, last_layer_params=[self.get_last_layer()], split="val")

        discloss, log_dict_disc = self.loss(
            x, xrec, 1, self.global_step, last_layer_params=[self.get_last_layer()], split="val")
        
        self.log("val/nll_loss", log_dict_ae["val/nll_loss"], prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/discloss", discloss, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        
        filtered_log_dict_ae = {k: v for k, v in log_dict_ae.items() if k not in ["val/nll_loss", "val/aeloss"]}
        self.log_dict(filtered_log_dict_ae, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        
        filtered_log_dict_disc = {k: v for k, v in log_dict_disc.items() if k not in ["val/discloss"]}
        self.log_dict(filtered_log_dict_disc, logger=True, on_step=False, on_epoch=True, sync_dist=True)


    def configure_optimizers(self):
        lr = self.learning_rate
        
        ae_params = list(self.encoder.parameters()) + \
                    list(self.quantize.parameters()) + \
                    list(self.decoder.parameters())
        
        opt_ae = torch.optim.AdamW(ae_params, lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.AdamW(self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
        
        return [opt_ae, opt_disc], []

    def get_last_layer(self) -> nn.Parameter:
        return self.decoder.get_last_layer()