from functools import partial
from typing import Tuple, Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from model.vit_vqgan.config import BaseTransformerConfig, DecoderTransformerConfig, EncoderTransformerConfig


class VisionEmbeddings(nn.Module):
    def __init__(self, patch_size: int, num_channels: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_embedding = nn.Conv2d(
            in_channels=num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

    def forward(self, pixel_values: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        target_dtype = self.patch_embedding.weight.dtype
        
        _, _, H, W = pixel_values.shape
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"Input image dimensions ({H}x{W}) are not divisible by patch size ({self.patch_size})."
            )
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size
        
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        return embeddings, (grid_h, grid_w)


class RotaryEmbedding(nn.Module):
    def __init__(self, config: EncoderTransformerConfig):
        super().__init__()
        self.dim = config.hidden_size // config.num_attention_heads
        self.base = config.rope_base
        self.inv_freq: Tensor
        self._init_rope()

    def _init_rope(self):
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, position_ids: Tensor, dtype: torch.dtype = torch.bfloat16) -> Tuple[Tensor, Tensor]:
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = position_ids.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)

def rotate_half(x: Tensor) -> Tensor:
    return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, unsqueeze_dim: int = 1) -> Tuple[Tensor, Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Attention(nn.Module):
    def __init__(self, config: BaseTransformerConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:" f" {self.num_heads}).")
        
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden: Tensor, freqs: Optional[Tuple[Tensor, Tensor]] = None, attention_mask: Optional[Tensor] = None,) -> Tensor:
        batch_size, q_len, _ = hidden.size()

        query_states = self.q_proj(hidden)
        key_states = self.k_proj(hidden)
        value_states = self.v_proj(hidden)

        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        if freqs is not None:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, *freqs)

        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=attention_mask,
                                                     dropout_p=self.dropout if self.training else 0.0, is_causal=False,)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, q_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output
    
class MLP(nn.Module):
    def __init__(self, config: BaseTransformerConfig):
        super().__init__()
        self.activation_fn = partial(F.gelu, approximate="tanh")
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden: Tensor) -> Tensor:
        hidden = self.fc1(hidden)
        hidden = self.activation_fn(hidden)
        hidden = self.fc2(hidden)
        return hidden
    
class Block(nn.Module):
    def __init__(self, config: BaseTransformerConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = Attention(config=config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = MLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden: Tensor, freqs: Optional[Tuple[Tensor, Tensor]], attention_mask: Optional[Tensor]) -> Tensor:
        residual = hidden
        hidden = self.layer_norm1(hidden)
        hidden = self.self_attn(hidden=hidden, freqs=freqs, attention_mask=attention_mask)
        hidden = residual + hidden

        residual = hidden
        hidden = self.layer_norm2(hidden)
        hidden = self.mlp(hidden)
        hidden = residual + hidden
        return hidden
    
class Transformer(nn.Module):
    def __init__(self, config: BaseTransformerConfig):
        super().__init__()
        self.layers = nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers)])

    def forward(self, inputs_embeds: Tensor, freqs: Optional[Tuple[Tensor, Tensor]] = None, attention_mask: Optional[Tensor] = None) -> Tensor:
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = checkpoint(encoder_layer, hidden_states, freqs, attention_mask, use_reentrant=False)
        return hidden_states

class ViTEncoder(nn.Module):
    def __init__(self, encoder_config: EncoderTransformerConfig, patch_size: int, num_channels: int):
        super().__init__()
        self.embeddings = VisionEmbeddings(patch_size, num_channels, encoder_config.hidden_size)
        self.encoder = Transformer(encoder_config)
        self.post_layernorm = nn.LayerNorm(encoder_config.hidden_size, eps=encoder_config.layer_norm_eps)
        self.rotary_emb = RotaryEmbedding(encoder_config)

    def forward(self, pixel_values: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        hidden_states, grid_hw = self.embeddings(pixel_values)
        
        num_patches = hidden_states.shape[1]
        position_ids = torch.arange(num_patches, device=hidden_states.device).unsqueeze(0)
        freqs = self.rotary_emb(position_ids, dtype=hidden_states.dtype)
        
        encoder_outputs = self.encoder(inputs_embeds=hidden_states, freqs=freqs)
        last_hidden_state = self.post_layernorm(encoder_outputs)
        return last_hidden_state, grid_hw

class ViTDecoder(nn.Module):
    def __init__(self, decoder_config: DecoderTransformerConfig, patch_size: int, num_channels: int):
        super().__init__()
        self.decoder_hidden_size = decoder_config.hidden_size
            
        self.decoder = Transformer(decoder_config)
        self.post_layernorm = nn.LayerNorm(self.decoder_hidden_size, eps=decoder_config.layer_norm_eps)
        
        self.final_conv = nn.ConvTranspose2d(
            self.decoder_hidden_size, 
            num_channels, 
            kernel_size=patch_size, 
            stride=patch_size
        )
    
    def forward(self, hidden_states: Tensor, grid_hw: Tuple[int, int]) -> Tensor:
        grid_h, grid_w = grid_hw
        
        decoder_outputs = self.decoder(hidden_states, freqs=None)
        x = self.post_layernorm(decoder_outputs)
        
        x = rearrange(x, 'b (h w) c -> b c h w', h=grid_h, w=grid_w)
        
        return F.tanh(self.final_conv(x))

    def get_last_layer(self) -> nn.Parameter:
        return self.final_conv.weight