import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def nonlinearity(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

        self.scale = in_channels ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(x)
        q, k, v = self.q(h_), self.k(h_), self.v(h_)

        # Reshape and compute attention
        b, c, h, w = q.shape
        q, k, v = [t.view(b, c, -1) for t in (q, k, v)]
        q = q.permute(0, 2, 1)

        attn = torch.bmm(q, k) * self.scale
        attn = F.softmax(attn, dim=2)

        # Attend to values
        h_ = torch.bmm(v, attn.permute(0, 2, 1))
        h_ = h_.view(b, c, h, w)

        return x + self.proj_out(h_)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = None, conv_shortcut: bool = False, dropout: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=self.out_channels, eps=1e-6, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)

        self.nin_shortcut = (nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0) if in_channels != out_channels else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        x = self.nin_shortcut(x)

        return x + h


class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks, attn_resolutions, in_channels, resolution, z_channels, double_z=True,
                 **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.ch_mult = ch_mult
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
                if curr_res in attn_resolutions: attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res //= 2
            self.down.append(down)

        # middle
        self.mid = nn.ModuleDict({
            'block_1': ResnetBlock(in_channels=block_in, out_channels=block_in),
            'attn_1': AttnBlock(block_in),
            'block_2': ResnetBlock(in_channels=block_in, out_channels=block_in)
        })

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels if double_z else z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks, attn_resolutions, in_channels, resolution, z_channels, give_pre_end=False,
                 **ignorekwargs):
        super().__init__()
        self.last_z_shape = None
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[-1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print(f"Working with z of shape {self.z_shape} = {np.prod(self.z_shape)} dimensions.")

        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.ModuleDict({
            'block_1': ResnetBlock(block_in, block_in,),
            'attn_1': AttnBlock(block_in),
            'block_2': ResnetBlock(block_in, block_in)
        })

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
                if curr_res in attn_resolutions: attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res *= 2
            self.up.insert(0, up)

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        self.last_z_shape = z.shape

        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if self.up[i_level].attn:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0: h = self.up[i_level].upsample(h)

        if self.give_pre_end: return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(F.pad(x, (0, 1, 0, 1), mode="constant", value=0))
        return x
