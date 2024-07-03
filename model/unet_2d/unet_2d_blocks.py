from typing import Optional, Tuple

import torch
from torch import nn

from ..attention import Attention
from ..resnet_2d import ResnetBlock2D


def get_down_block(
        down_block_type: str,
        num_layers: int,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        add_downsample: bool,
        resnet_act_fn: str,
        resnet_groups: Optional[int] = None,
        attention_head_dim: Optional[int] = None,
        downsample_type: Optional[str] = None,
) -> nn.Module:
    if down_block_type == "DownBlock2D":
        return DownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
        )
    elif down_block_type == "AttnDownBlock2D":
        if add_downsample is False:
            downsample_type = None
        else:
            downsample_type = downsample_type or "conv"
        return AttnDownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            attention_head_dim=attention_head_dim,
            downsample_type=downsample_type,
        )

    else:
        raise ValueError(f"Unknown down block type {down_block_type}")


def get_up_block(
        up_block_type: str,
        num_layers: int,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        add_upsample: bool,
        resnet_act_fn: str,
        resnet_groups: Optional[int] = None,
        attention_head_dim: Optional[int] = None,
        upsample_type: Optional[str] = None,
) -> nn.Module:
    if up_block_type == "UpBlock2D":
        return UpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
        )
    elif up_block_type == "AttnUpBlock2D":
        if add_upsample is False:
            upsample_type = None
        else:
            upsample_type = upsample_type or "conv"
        return AttnUpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            attention_head_dim=attention_head_dim,
            upsample_type=upsample_type,
        )
    else:
        raise ValueError(f"Unknown up block type {up_block_type}")


class Downsample2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(Downsample2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(Upsample2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, hidden_states: torch.Tensor, output_size: Optional[int] = None, ) -> torch.Tensor:
        assert hidden_states.shape[1] == self.in_channels
        if output_size is None:
            hidden_states = nn.functional.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        else:
            hidden_states = nn.functional.interpolate(hidden_states, size=output_size, mode="nearest")

        return self.conv(hidden_states)


class DownBlock2D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            temb_channels: int,
            num_layers: int = 1,
            resnet_act_fn: str = "swish",
            resnet_groups: int = 32,
            add_downsample: bool = True,
    ):
        super().__init__()

        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels,
                    out_channels,
                    temb_channels,
                    resnet_groups,
                    resnet_act_fn,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList([
                Downsample2D(out_channels, out_channels, )
            ])
        else:
            self.downsamplers = None

    def forward(
            self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        output_states = ()

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsample in self.downsamplers:
                hidden_states = downsample(hidden_states)
            output_states += (hidden_states,)

        return hidden_states, output_states


class UpBlock2D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            prev_output_channel: int,
            out_channels: int,
            temb_channels: int,
            num_layers: int = 1,
            resnet_act_fn: str = "swish",
            resnet_groups: int = 32,
            add_upsample: bool = True,
    ):
        super().__init__()

        resnets = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    groups=resnet_groups,
                    non_linearity=resnet_act_fn,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([
                Upsample2D(out_channels, out_channels)
            ])
        else:
            self.upsamplers = None

    def forward(
            self,
            hidden_states: torch.Tensor,
            res_hidden_states_tuple: Tuple[torch.Tensor, ...],
            temb: Optional[torch.Tensor] = None,
            upsample_size: Optional[int] = None,
    ) -> torch.Tensor:
        for resnet in self.resnets:
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            for upsample in self.upsamplers:
                hidden_states = upsample(hidden_states, upsample_size)

        return hidden_states


class AttnDownBlock2D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            temb_channels: int,
            num_layers: int = 1,
            resnet_act_fn: str = "swish",
            resnet_groups: int = 32,
            attention_head_dim: int = 1,
            downsample_type: str = "conv",
    ):
        super().__init__()

        resnets = []
        attentions = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    groups=resnet_groups,
                    non_linearity=resnet_act_fn,
                )
            )
            attentions.append(
                Attention(
                    out_channels,
                    heads=out_channels // attention_head_dim,
                    dim_heads=attention_head_dim,
                    bias=True,
                    norm_num_groups=resnet_groups,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)

        if downsample_type == "conv":
            self.downsamplers = nn.ModuleList([
                Downsample2D(out_channels, out_channels)
            ])
        else:
            self.downsamplers = None

    def forward(
            self,
            hidden_states: torch.Tensor,
            temb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        output_states = ()

        for resnet, attention in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attention(hidden_states)
            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsample in self.downsamplers:
                hidden_states = downsample(hidden_states)
            output_states += (hidden_states,)

        return hidden_states, output_states


class AttnUpBlock2D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            prev_output_channel: int,
            out_channels: int,
            temb_channels: int,
            num_layers: int = 1,
            resnet_act_fn: str = "swish",
            resnet_groups: int = 32,
            attention_head_dim: int = 1,
            upsample_type: str = "conv",
    ):
        super().__init__()
        resnets = []
        attentions = []

        if attention_head_dim is None:
            attention_head_dim = out_channels

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    groups=resnet_groups,
                    non_linearity=resnet_act_fn,
                )
            )
            attentions.append(
                Attention(
                    out_channels,
                    heads=out_channels // attention_head_dim,
                    dim_heads=attention_head_dim,
                    norm_num_groups=resnet_groups,
                    bias=True,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if upsample_type == "conv":
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, out_channels=out_channels)])
        else:
            self.upsamplers = None

    def forward(
            self,
            hidden_states: torch.Tensor,
            res_hidden_states_tuple: Tuple[torch.Tensor, ...],
            temb: Optional[torch.Tensor] = None,
            upsample_size: Optional[int] = None,
    ) -> torch.Tensor:

        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


class UNetMidBlock2D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            temb_channels: int,
            num_layers: int = 1,
            resnet_act_fn: str = "swish",
            resnet_groups: int = 32,
            attn_groups: Optional[int] = None,
            add_attention: bool = True,
            attention_head_dim: int = 1
    ):
        super().__init__()

        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        self.add_attention = add_attention

        if attn_groups is None:
            attn_groups = resnet_groups

        # one resnet block
        resnets = [
            ResnetBlock2D(
                in_channels,
                in_channels,
                temb_channels,
                resnet_groups,
                resnet_act_fn,
            )
        ]

        attentions = []
        for i in range(num_layers):
            attentions.append(
                Attention(
                    in_channels,
                    heads=in_channels // attention_head_dim,
                    dim_heads=attention_head_dim,
                    bias=True,
                    norm_num_groups=attn_groups,
                ) if add_attention else None
            )

        resnets.append(
            ResnetBlock2D(
                in_channels,
                in_channels,
                temb_channels,
                resnet_groups,
                resnet_act_fn,
            )
        )
        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)

    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                hidden_states = attn(hidden_states)
            hidden_states = resnet(hidden_states, temb)
        return hidden_states
