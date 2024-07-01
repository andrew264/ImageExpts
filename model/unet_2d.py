from dataclasses import dataclass
from typing import Optional, Union

import lightning as L
import torch
from diffusers import DDPMScheduler, get_cosine_schedule_with_warmup
from torch import nn

from .config import UNet2DConfig
from .embedding import Timesteps, TimestepEmbedding
from .unet_2d_blocks import get_down_block, UNetMidBlock2D, get_up_block


@dataclass
class UNet2DOutput:
    sample: torch.Tensor


class UNet2DModel(L.LightningModule):
    def __init__(self, config: Optional[UNet2DConfig] = None, ):
        super(UNet2DModel, self).__init__()

        self._noise_scheduler = None
        self.config = config or UNet2DConfig()

        sample_size = self.config.sample_size
        in_channels = self.config.in_channels
        out_channels = self.config.out_channels
        down_block_types = self.config.down_block_types
        up_block_types = self.config.up_block_types
        block_out_channels = self.config.block_out_channels
        layers_per_block = self.config.layers_per_block
        norm_num_groups = self.config.norm_num_groups
        act_fn = self.config.act_fn
        attention_head_dim = self.config.attention_head_dim

        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4

        if len(down_block_types) != len(up_block_types):
            raise ValueError("Down and Up block types must have the same length")

        if len(block_out_channels) != len(down_block_types):
            raise ValueError("Block out channels must have the same length as block types")

        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))

        self.time_proj = Timesteps(block_out_channels[0], True)
        timestep_input_dim = block_out_channels[0]
        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim, act_fn=act_fn)

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        self.loss_fn = nn.MSELoss()

        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else output_channel,
            )
            self.down_blocks.append(down_block)

        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            resnet_act_fn=act_fn,
            attention_head_dim=attention_head_dim if attention_head_dim is not None else block_out_channels[-1],
            resnet_groups=norm_num_groups,
            attn_groups=None,
            add_attention=True,
        )

        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=not is_final_block,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else output_channel,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        num_groups_out = norm_num_groups if norm_num_groups is not None else min(block_out_channels[0] // 4, 32)
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=num_groups_out, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(
            self,
            sample: torch.Tensor,
            timesteps: Union[torch.Tensor, float, int], ) -> UNet2DOutput:

        if not isinstance(timesteps, torch.Tensor):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif isinstance(timesteps, torch.Tensor) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)

        t_emb = self.time_proj(timesteps).to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        skip_sample = sample
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample, temb=emb, skip_sample=skip_sample
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(sample, emb)

        # 5. up
        skip_sample = None
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample = upsample_block(sample, res_samples, emb, skip_sample)
            else:
                sample = upsample_block(sample, res_samples, emb)

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if skip_sample is not None:
            sample += skip_sample

        return UNet2DOutput(sample=sample)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
        }

    @property
    def noise_scheduler(self):
        if self._noise_scheduler is None:
            self._noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
        return self._noise_scheduler

    def training_step(self, batch, batch_idx):
        clean_images = batch["images"]
        noise = torch.randn(clean_images.shape, device=clean_images.device)
        bs = clean_images.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config['num_train_timesteps'], (bs,), device=clean_images.device,
            dtype=torch.int64
        )
        noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)

        # Forward pass
        noise_pred = self.forward(noisy_images, timesteps, ).sample
        loss = self.loss_fn(noise_pred, clean_images)
        self.log("train_loss", loss, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        clean_images = batch["images"]
        noise = torch.randn(clean_images.shape, device=clean_images.device)
        bs = clean_images.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config['num_train_timesteps'], (bs,), device=clean_images.device,
            dtype=torch.int64
        )
        noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)

        # Forward pass
        noise_pred = self.forward(noisy_images, timesteps, ).sample
        loss = self.loss_fn(noise_pred, clean_images)
        self.log("val_loss", loss, prog_bar=True)
        return {"loss": loss}
