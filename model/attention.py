from typing import Optional

from torch import nn


class Attention(nn.Module):
    def __init__(
            self,
            query_dim,
            heads: int = 8,
            dim_heads: int = 64,
            bias: bool = False,
            norm_num_groups: Optional[int] = None,

    ):
        super(Attention, self).__init__()
        self.heads = heads
        self.dim_heads = dim_heads
        self.query_dim = query_dim

        if query_dim % heads != 0:
            raise ValueError(f"Query dimension {query_dim} must be divisible by number of heads {heads}")

        self.scale = dim_heads ** -0.5
        self.hidden_dim = heads * dim_heads

        self.to_qkv = nn.Linear(query_dim, self.hidden_dim * 3, bias=bias)
        self.to_out = nn.Linear(self.hidden_dim, query_dim, bias=bias)

        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(num_channels=query_dim, num_groups=norm_num_groups, eps=1e-6, affine=True)
        else:
            self.group_norm = None

    def forward(self, hidden_states):
        batch_size, channel, height, width = hidden_states.shape
        residual = hidden_states

        hidden_states = hidden_states.view(batch_size, channel, height * width)

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states).transpose(1, 2)

        qkv = self.to_qkv(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, -1, self.heads, self.dim_heads).transpose(1, 2)
        k = k.view(batch_size, -1, self.heads, self.dim_heads).transpose(1, 2)
        v = v.view(batch_size, -1, self.heads, self.dim_heads).transpose(1, 2)

        attn = nn.functional.scaled_dot_product_attention(q, k, v, scale=self.scale)

        attn = attn.transpose(1, 2).reshape(batch_size, -1, self.heads * self.dim_heads)
        attn = self.to_out(attn).view(batch_size, channel, height, width)

        return residual + attn
