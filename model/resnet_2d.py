from typing import Optional

import torch
from torch import nn
from transformers.activations import get_activation


class ResnetBlock2D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            temb_channels: int = 512,
            groups: int = 32,
            non_linearity: str = "swish",
            use_in_shortcut: Optional[bool] = None,
            conv_shortcut_bias: bool = True,
            conv_2d_out_channels: Optional[int] = None,
    ):
        super(ResnetBlock2D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        eps = 1e-6

        self.norm_1 = nn.GroupNorm(groups, in_channels, eps=eps, affine=True)

        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.time_emb_proj = nn.Linear(temb_channels, out_channels)

        self.norm_2 = nn.GroupNorm(groups, out_channels, eps=eps, affine=True)

        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.non_linearity = get_activation(non_linearity)

        conv_2d_out_channels = conv_2d_out_channels or out_channels
        self.use_in_shortcut = self.in_channels != conv_2d_out_channels if use_in_shortcut is None else use_in_shortcut
        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv2d(
                in_channels,
                conv_2d_out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=conv_shortcut_bias,
            )

    def forward(self, input_tensor: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        hidden_states = input_tensor

        hidden_states = self.norm_1(hidden_states)
        hidden_states = self.non_linearity(hidden_states)
        hidden_states = self.conv_1(hidden_states)

        temb = self.time_emb_proj(self.non_linearity(temb))[:, :, None, None]
        hidden_states = hidden_states + temb
        hidden_states = self.norm_2(hidden_states)
        hidden_states = self.non_linearity(hidden_states)

        hidden_states = self.conv_2(hidden_states)

        if self.use_in_shortcut:
            input_tensor = self.conv_shortcut(input_tensor)

        return input_tensor + hidden_states
