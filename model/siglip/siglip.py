from typing import TypedDict, Optional, Tuple

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import lightning as L
from transformers import get_cosine_schedule_with_warmup

from model.siglip.blocks import SiglipTextConfig, SiglipVisionConfig, TextTransformer, VisionTransformer, VisionEmbeddings, Attention, MLP, MultiheadAttentionPoolingHead
from model.siglip.initialization import default_flax_embed_init, lecun_normal_

class SiglipConfig(TypedDict):
    text_config: SiglipTextConfig
    vision_config: SiglipVisionConfig
    
class SiglipModel(L.LightningModule):
    def __init__(self, config: SiglipConfig):
        super().__init__()
        self.config = config
        text_config = config['text_config']
        vision_config = config['vision_config']

        self.text_model = TextTransformer(text_config)
        self.vision_model = VisionTransformer(vision_config)

        self.logit_scale = nn.Parameter(torch.randn(1))
        self.logit_bias = nn.Parameter(torch.randn(1))

    def _init_weights(self, module):
        if isinstance(module, VisionEmbeddings):
            width = self.config['vision_config']['hidden_size']
            nn.init.normal_(module.position_embedding.weight, std=1 / np.sqrt(width))
        elif isinstance(module, nn.Embedding):
            default_flax_embed_init(module.weight)
        elif isinstance(module, Attention):
            nn.init.xavier_uniform_(module.q_proj.weight)
            nn.init.xavier_uniform_(module.k_proj.weight)
            nn.init.xavier_uniform_(module.v_proj.weight)
            nn.init.xavier_uniform_(module.out_proj.weight)
            nn.init.zeros_(module.q_proj.bias)
            nn.init.zeros_(module.k_proj.bias)
            nn.init.zeros_(module.v_proj.bias)
            nn.init.zeros_(module.out_proj.bias)
        elif isinstance(module, MLP):
            nn.init.xavier_uniform_(module.fc1.weight)
            nn.init.xavier_uniform_(module.fc2.weight)
            nn.init.normal_(module.fc1.bias, std=1e-6)
            nn.init.normal_(module.fc2.bias, std=1e-6)
        elif isinstance(module, MultiheadAttentionPoolingHead):
            nn.init.xavier_uniform_(module.probe.data)
            nn.init.xavier_uniform_(module.attention.in_proj_weight.data)
            nn.init.zeros_(module.attention.in_proj_bias.data)
        elif isinstance(module, SiglipModel):
            logit_scale_init = torch.log(torch.tensor(1.0))
            module.logit_scale.data.fill_(logit_scale_init)
            module.logit_bias.data.zero_()
        elif isinstance(module, (nn.Linear, nn.Conv2d)):
            lecun_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_text_features(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None, position_ids: Optional[Tensor] = None,) -> Tensor:
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,)
        pooled_output = text_outputs[1]
        return pooled_output

    def get_image_features(self, pixel_values: Tensor, interpolate_pos_encoding: bool = False) -> Tensor:
        vision_outputs = self.vision_model(pixel_values=pixel_values, interpolate_pos_encoding=interpolate_pos_encoding,)
        pooled_output = vision_outputs[1]
        return pooled_output
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        warmup_steps = max(100, int(self.trainer.estimated_stepping_batches * .1))
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=self.trainer.estimated_stepping_batches,)
        lr_scheduler_config = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1,}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config,}
    
    def loss_fn(self, logits: Tensor) -> Tensor:
        eye = torch.eye(logits.size(0), device=logits.device)
        m1_diag1 = -torch.ones_like(logits) + 2 * eye
        loglik = F.logsigmoid(m1_diag1 * logits)
        nll = -torch.sum(loglik, dim=-1)
        return nll.mean()

    def forward(self, input_ids: Tensor, pixel_values: Tensor, attention_mask: Optional[Tensor] = None, position_ids: Optional[Tensor] = None, interpolate_pos_encoding: bool = False,) -> Tuple:
        vision_outputs = self.vision_model(pixel_values=pixel_values, interpolate_pos_encoding=interpolate_pos_encoding,)
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,)

        image_embeds = vision_outputs[1]
        text_embeds = text_outputs[1]

        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        logits_per_text = (torch.matmul(text_embeds, image_embeds.t().to(text_embeds.device)) * self.logit_scale.exp() + self.logit_bias)
        logits_per_image = logits_per_text.t()

        return (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
    
    def training_step(self, batch, batch_idx):
        input_ids, pixel_values, attention_mask = batch['input_ids'], batch['pixel_values'], batch['attention_mask']
        output = self(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
        loss = self.loss_fn(output[1])
        self.log("loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        input_ids, pixel_values, attention_mask = batch['input_ids'], batch['pixel_values'], batch['attention_mask']
        output = self(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
        loss = self.loss_fn(output[1])
        return {"loss": loss}