from dataclasses import dataclass, field
from typing import List

@dataclass
class BaseTransformerConfig:
    """Common parameters for Encoder/Decoder Transformer blocks."""
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    attention_dropout: float = 0.0
    layer_norm_eps: float = 1e-6

@dataclass
class EncoderTransformerConfig(BaseTransformerConfig):
    """Configuration for the ViT Encoder's Transformer blocks."""
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    rope_base: int = 10000

@dataclass
class DecoderTransformerConfig(BaseTransformerConfig):
    """Configuration for the ViT Decoder's Transformer blocks (No RoPE)."""
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_attention_heads: int = 12
    num_hidden_layers: int = 12


@dataclass
class FSQInitConfig:
    """Parameters to initialize the FSQ module."""
    levels: List[int] = field(default_factory=lambda: [8, 8, 4, 4])
    num_codebooks: int = 8


@dataclass
class ViTVQGANConfig:
    """Main configuration for the ViT-VQGAN model."""
    num_channels: int = 4
    patch_size: int = 16

    encoder_config: EncoderTransformerConfig = field(default_factory=EncoderTransformerConfig)
    decoder_config: DecoderTransformerConfig = field(default_factory=DecoderTransformerConfig)
    fsq_config: FSQInitConfig = field(default_factory=FSQInitConfig)

    # Loss function parameters
    disc_start: int = 10000
    disc_weight: float = 0.1
    disc_num_layers: int = 3
    disc_ndf: int = 64
    disc_loss_type: str = "vanilla"