from __future__ import annotations
from functools import partial
from contextlib import nullcontext
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import Module
from torch import tensor, Tensor
from torch import autocast

from einops import rearrange, pack, unpack

# helper functions

def default(*args):
    for arg in args:
        if arg is not None:
            return arg
    return None

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# main class

class FSQ(Module):
    def __init__(
        self,
        levels: list[int],
        dim: int | None = None,
        output_dim: int | None = None,
        num_codebooks = 1,
        keep_num_codebooks_dim: bool | None = None,
        scale: float | None = None,
        allowed_dtypes: tuple[torch.dtype, ...] = (torch.float32, torch.float64),
        channel_first = False,
        projection_has_bias = True,
        return_indices = True,
        force_quantization_f32 = True,
        preserve_symmetry = False,
        noise_dropout = 0.,
    ):
        super().__init__()

        _levels = tensor(levels, dtype=torch.int32)
        self.register_buffer('_levels', _levels, persistent = False)

        _basis = torch.cumprod(tensor([1] + levels[:-1]), dim = 0, dtype=torch.int32)
        self.register_buffer('_basis', _basis, persistent = False)

        self.scale = scale

        self.preserve_symmetry = preserve_symmetry
        self.noise_dropout = noise_dropout

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)
        
        _actual_output_dim = default(output_dim, self.dim)

        self.channel_first = channel_first

        has_input_projection = self.dim != effective_codebook_dim
        self.project_in = nn.Linear(self.dim, effective_codebook_dim, bias = projection_has_bias) if has_input_projection else nn.Identity()
        
        has_output_projection = _actual_output_dim != effective_codebook_dim
        self.project_out = nn.Linear(effective_codebook_dim, _actual_output_dim, bias = projection_has_bias) if has_output_projection else nn.Identity()

        self.return_indices = return_indices

        if self.return_indices:
            self.codebook_size = self._levels.prod().item()
            implicit_codebook_normalized = self._indices_to_codes(torch.arange(self.codebook_size))
            self.register_buffer('implicit_codebook', implicit_codebook_normalized, persistent = False)

        self.allowed_dtypes = allowed_dtypes
        self.force_quantization_f32 = force_quantization_f32
    
    @staticmethod
    def round_ste(z: Tensor) -> Tensor:
        """ round with straight through gradients. """
        zhat = z.round()
        return z + (zhat - z).detach()

    @staticmethod
    def floor_ste(z: Tensor) -> Tensor:
        """ floor with straight through gradients. """
        zhat = z.floor()
        return z + (zhat - z).detach()
    
    def bound(self, z: Tensor, eps = 1e-3) -> Tensor:
        """ Bound `z`, an array of shape (..., d). """
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        bounded_z = (z + shift).tanh() * half_l - offset
        half_width = self._levels // 2
        return self.round_ste(bounded_z) / half_width


    def symmetry_preserving_bound(self, z: Tensor) -> Tensor:
        """ QL(x) = 2 / (L - 1) * [(L - 1) * (tanh(x) + 1) / 2 + 0.5] - 1 """
        levels_minus_1 = (self._levels - 1)
        scale = 2. / levels_minus_1
        bracket = (levels_minus_1 * (z.tanh() + 1) / 2.) + 0.5
        bracket = self.floor_ste(bracket)
        return scale * bracket - 1.

    def quantize(self, z: Tensor) -> Tensor:
        """ Quantizes z, returns quantized zhat, same shape as z. """
        bound_fn = self.symmetry_preserving_bound if self.preserve_symmetry else self.bound

        bounded_z = bound_fn(z)

        # determine where to add a random offset elementwise
        # if using noise dropout

        if not self.training or self.noise_dropout == 0.:
            return bounded_z

        offset_mask = torch.bernoulli(torch.full_like(bounded_z, self.noise_dropout)).bool()
        offset = torch.rand_like(bounded_z) - 0.5
        bounded_z = torch.where(offset_mask, bounded_z + offset, bounded_z)

        return bounded_z

    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        if self.preserve_symmetry:
            return (zhat_normalized + 1.) / (2. / (self._levels - 1))

        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width
    
    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        if self.preserve_symmetry:
            return zhat * (2. / (self._levels - 1)) - 1.

        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def _indices_to_codes(self, indices: Tensor) -> Tensor:
        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes

    def indices_to_level_indices(self, indices: Tensor) -> Tensor:
        """ Converts indices to indices at each level, perhaps needed for a transformer with factorized embeddings """
        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        return codes_non_centered

    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        """ Converts a `code` to an index in the codebook. """
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim = -1).round().to(dtype=torch.int32)

    def indices_to_codes(self, indices: Tensor) -> Tensor:
        """ Inverse of `codes_to_indices`. """
        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        codes = self._indices_to_codes(indices)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, '... c d -> ... (c d)')

        codes = self.project_out(codes)

        if is_img_or_video or self.channel_first:
            codes = rearrange(codes, 'b ... d -> b d ...')

        return codes

    def forward(self, z: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension
        c - number of codebook dim
        """

        is_img_or_video = z.ndim >= 4
        need_move_channel_last = is_img_or_video or self.channel_first

        # standardize image or video into (batch, seq, dimension)

        if need_move_channel_last:
            z = rearrange(z, 'b d ... -> b ... d')
            z, ps = pack_one(z, 'b * d')

        assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'

        z = self.project_in(z)

        z = rearrange(z, 'b n (c d) -> b n c d', c = self.num_codebooks)

        # whether to force quantization step to be full precision or not

        force_f32 = self.force_quantization_f32
        quantization_context = partial(autocast, 'cuda', enabled = False) if force_f32 else nullcontext

        with quantization_context():
            orig_dtype = z.dtype

            if force_f32 and orig_dtype not in self.allowed_dtypes:
                z = z.float()

            codes = self.quantize(z)

            # returning indices could be optional

            indices = None

            if self.return_indices:
                indices = self.codes_to_indices(codes)

            codes = rearrange(codes, 'b n c d -> b n (c d)')

            codes = codes.to(orig_dtype)

        # project out

        out = self.project_out(codes)

        # reconstitute image or video dimensions

        if need_move_channel_last:
            out = unpack_one(out, ps, 'b * d')
            out = rearrange(out, 'b ... d -> b d ...')

            if indices is not None:
                indices = unpack_one(indices, ps, 'b * c')

        if not self.keep_num_codebooks_dim and self.return_indices and indices is not None:
            indices = rearrange(indices, '... 1 -> ...')

        # return quantized output and indices

        return out, indices