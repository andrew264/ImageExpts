import torch
from torch import nn
from transformers.activations import get_activation


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels, time_embed_dim, act_fn="gelu"):
        super(TimestepEmbedding, self).__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        self.act_fn = get_activation(act_fn)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=True)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.act_fn(x)
        x = self.linear_2(x)
        return x


class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool = True):
        super(Timesteps, self).__init__()
        self.num_channels = num_channels
        self.max_period = torch.tensor(10000.0)
        self.flip_sin_to_cos = flip_sin_to_cos

    def forward(self, timesteps: torch.Tensor, ) -> torch.Tensor:
        assert len(timesteps.shape) == 1, f"Expected 1D tensor, got {timesteps.shape}"

        half_dim = self.num_channels // 2
        exponent = -torch.log(self.max_period) * torch.arange(0, half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / half_dim
        emb = torch.exp(exponent)
        emb = emb.unsqueeze(0) * timesteps.unsqueeze(1).float()

        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        if self.flip_sin_to_cos:
            emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

        if self.num_channels % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1, 0, 0))

        return emb
