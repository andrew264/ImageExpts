from functools import partial
from typing import Tuple, TypedDict, Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch.utils.checkpoint import checkpoint


class Config(TypedDict):
    hidden_size: int
    image_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    patch_size: int
    num_channels: int
    attention_dropout: float
    layer_norm_eps: float
    levels: list[int]
    num_codebooks: int
    rope_base: int

class VisionEmbeddings(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embed_dim = config['hidden_size']
        self.image_size = config['image_size']
        self.patch_size = config['patch_size']

        self.patch_embedding = nn.Conv2d(
            in_channels=config['num_channels'],
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches

    def forward(self, pixel_values: Tensor) -> Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        return embeddings


class RotaryEmbedding(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.dim = cfg['hidden_size'] // cfg['num_attention_heads']
        self.base = cfg['rope_base']
        self.inv_freq: Tensor
        self.rope_init()

    def rope_init(self):
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
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embed_dim = config['hidden_size']
        self.num_heads = config['num_attention_heads']
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:" f" {self.num_heads}).")
        
        self.scale = self.head_dim**-0.5
        self.dropout = config['attention_dropout']

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
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.activation_fn = partial(F.gelu, approximate="tanh")
        self.fc1 = nn.Linear(config['hidden_size'], config['intermediate_size'])
        self.fc2 = nn.Linear(config['intermediate_size'], config['hidden_size'])

    def forward(self, hidden: Tensor) -> Tensor:
        hidden = self.fc1(hidden)
        hidden = self.activation_fn(hidden)
        hidden = self.fc2(hidden)
        return hidden
    
class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.embed_dim = config['hidden_size']
        self.self_attn = Attention(config=config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config['layer_norm_eps'])
        self.mlp = MLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config['layer_norm_eps'])

    def forward(self, hidden: Tensor, freqs: Optional[Tuple[Tensor, Tensor]], attention_mask: Tensor,) -> Tensor:
        residual = hidden

        hidden = self.layer_norm1(hidden)
        hidden = self.self_attn(hidden=hidden, freqs=freqs, attention_mask=attention_mask,)
        hidden = residual + hidden

        residual = hidden
        hidden = self.layer_norm2(hidden)
        hidden = self.mlp(hidden)
        hidden = residual + hidden

        return hidden
    
class Transformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([Block(config) for _ in range(config['num_hidden_layers'])])

    def forward(self, inputs_embeds, freqs: Optional[Tuple[Tensor, Tensor]] = None, attention_mask: Optional[Tensor] = None) -> Tensor:
        hidden = inputs_embeds
        for encoder_layer in self.layers:
            hidden = checkpoint(encoder_layer, hidden, freqs, attention_mask, use_reentrant=False)
        return hidden

    
class ViTEncoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        embed_dim = config['hidden_size']

        self.embeddings = VisionEmbeddings(config)
        self.encoder = Transformer(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config['layer_norm_eps'])
        self.rotary_emb = RotaryEmbedding(config)

    def forward(self, pixel_values) -> Tensor:
        hidden = self.embeddings(pixel_values)
        input_pos = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
        freqs = self.rotary_emb(input_pos)
        last_hidden_state = self.encoder(inputs_embeds=hidden, freqs=freqs)
        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state

class ViTDecoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        embed_dim = config['hidden_size']
        self.decoder = Transformer(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config['layer_norm_eps'])
        self.to_pixel = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=config['image_size'] // config['patch_size']),
            nn.ConvTranspose2d(embed_dim, config['num_channels'], kernel_size=config['patch_size'], stride=config['patch_size'])
        )
    
    def forward(self, hidden: Tensor) -> Tensor:
        x = self.post_layernorm(self.decoder(hidden))
        return F.tanh(self.to_pixel(x))
    

    def get_last_layer(self) -> nn.Parameter:
        return self.to_pixel[-1].weight
