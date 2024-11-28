from functools import partial
from typing import TypedDict, Optional, Union, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask


class SiglipVisionConfig(TypedDict):
    hidden_size: int
    image_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    patch_size: int
    num_channels: int
    attention_dropout: float
    layer_norm_eps: float

class SiglipTextConfig(TypedDict):
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    vocab_size: int
    max_position_embeddings: int
    attention_dropout: float
    layer_norm_eps: float




class VisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
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
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def interpolate_pos_encoding(self, embeddings: Tensor, height: int, width: int) -> Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution images.
        """

        num_patches = embeddings.shape[1]
        num_positions = self.position_embedding.weight.shape[0]

        if num_patches == num_positions and height == width:
            return self.position_embedding(self.position_ids)

        patch_pos_embed = self.position_embedding.weight.unsqueeze(0)

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_height, new_width), mode="bicubic", align_corners=False,)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def forward(self, pixel_values: Tensor, interpolate_pos_encoding=False) -> Tensor:
        _, _, height, width = pixel_values.shape
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class TextEmbeddings(nn.Module):
    def __init__(self, config: SiglipTextConfig):
        super().__init__()
        embed_dim = config['hidden_size']

        self.token_embedding = nn.Embedding(config['vocab_size'], embed_dim)
        self.position_embedding = nn.Embedding(config['max_position_embeddings'], embed_dim)
        self.register_buffer("position_ids", torch.arange(config['max_position_embeddings']).expand((1, -1)), persistent=False)

    def forward(self, input_ids: Tensor, position_ids: Optional[Tensor] = None) -> Tensor:
        seq_length = input_ids.shape[-1]
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings
    
class Attention(nn.Module):
    def __init__(self, config: Union[SiglipTextConfig, SiglipVisionConfig]):
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
        self.is_causal = False

    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None,) -> Tensor:
        batch_size, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        is_causal = True if self.is_causal and q_len > 1 else False

        attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=attention_mask,
                                                     dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal,)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, q_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output
    
class MLP(nn.Module):
    def __init__(self, config: Union[SiglipTextConfig, SiglipVisionConfig]):
        super().__init__()
        self.config = config
        self.activation_fn = partial(F.gelu, approximate="tanh")
        self.fc1 = nn.Linear(config['hidden_size'], config['intermediate_size'])
        self.fc2 = nn.Linear(config['intermediate_size'], config['hidden_size'])

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states
    
class EncoderLayer(nn.Module):
    def __init__(self, config: Union[SiglipTextConfig, SiglipVisionConfig]):
        super().__init__()
        self.embed_dim = config['hidden_size']
        self.self_attn = Attention(config=config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config['layer_norm_eps'])
        self.mlp = MLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config['layer_norm_eps'])

    def forward(self, hidden_states: Tensor, attention_mask: Tensor,) -> Tensor:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask,)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
    
class Encoder(nn.Module):
    def __init__(self, config: Union[SiglipTextConfig, SiglipVisionConfig]):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config['num_hidden_layers'])])

    def forward(self, inputs_embeds, attention_mask: Optional[Tensor] = None) -> Tensor:
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states, attention_mask)
        return hidden_states

class TextTransformer(nn.Module):
    def __init__(self, config: SiglipTextConfig):
        super().__init__()
        self.config = config
        embed_dim = config['hidden_size']

        self.embeddings = TextEmbeddings(config)
        self.encoder = Encoder(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config['layer_norm_eps'])
        self.head = nn.Linear(embed_dim, embed_dim)

    def forward(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None, position_ids: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        # note: SigLIP's text model does not use a causal mask, unlike the original CLIP model.
        if attention_mask is not None:
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        last_hidden_state = self.encoder(inputs_embeds=hidden_states, attention_mask=attention_mask)
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        pooled_output = last_hidden_state[:, -1, :]
        pooled_output = self.head(pooled_output)

        return last_hidden_state, pooled_output
    
class VisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config['hidden_size']

        self.embeddings = VisionEmbeddings(config)
        self.encoder = Encoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config['layer_norm_eps'])
        self.head = MultiheadAttentionPoolingHead(config)

    def forward(self, pixel_values, interpolate_pos_encoding: Optional[bool] = False) -> Tuple[Tensor, Tensor]:
        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        last_hidden_state = self.encoder(inputs_embeds=hidden_states,)
        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooler_output = self.head(last_hidden_state)
        return last_hidden_state, pooler_output
    
class MultiheadAttentionPoolingHead(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.probe = nn.Parameter(torch.randn(1, 1, config['hidden_size']))
        self.attention = torch.nn.MultiheadAttention(config['hidden_size'], config['num_attention_heads'], batch_first=True)
        self.layernorm = nn.LayerNorm(config['hidden_size'], eps=config['layer_norm_eps'])
        self.mlp = MLP(config)

    def forward(self, hidden_state):
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        hidden_state = self.attention(probe, hidden_state, hidden_state)[0]

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state[:, 0]
