from typing import Optional, Union, Tuple


class UNet2DConfig:
    sample_size: Optional[Union[int, Tuple[int, int]]] = None,
    in_channels: int = 3
    out_channels: int = 3
    down_block_types: Tuple[str, ...] = (
        "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D")
    up_block_types: Tuple[str, ...] = ("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D")
    block_out_channels: Tuple[int, ...] = (224, 448, 672, 896)
    layers_per_block: int = 2
    norm_num_groups: int = 32
    act_fn = "silu"
    attention_head_dim: Optional[int] = 8
