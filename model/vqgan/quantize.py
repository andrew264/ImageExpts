import torch
import torch.nn as nn
from torch import Tensor

class VectorQuantizer(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    def __init__(self, n_e, e_dim, beta = 1.):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)


    def forward(self, z: Tensor):
        z = z.permute(0, 2, 3, 1).contiguous()
        bz = z.shape[0]
        z_flattened = z.view(bz, -1, self.e_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 - 2 e * z + e^2 
        d = z_flattened.pow(2).sum(dim=-1, keepdim=True) - \
            2 * z_flattened @ self.embedding.weight.t() + \
            self.embedding.weight.t().pow(2).sum(dim=0, keepdim=True)

        min_encoding_indices = torch.argmin(d, dim=-1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # mse
        loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss, (None, None, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q
