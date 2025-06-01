import math
from typing import Any, List, Tuple, Union, Optional

import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
from torch import Tensor
from torch.nn import functional as F

from model import VQModel

def exists(x: Optional[Any]) ->bool: return x is not None

# Helper function for padding dimensions
def _pad_to_multiple(n, mult):
    """Calculates the smallest multiple of `mult` greater than or equal to `n`."""
    return math.ceil(n / mult) * mult

class ImageTokenizer:
    # https://github.com/facebookresearch/chameleon/blob/main/chameleon/inference/image_tokenizer.py
    def __init__(
        self,
        model: Optional[VQModel] = None,
        device: str | torch.device | None = None,
        target_height: Optional[int] = None,
        target_width: Optional[int] = None,
    ):
        self._vq_model = model
        self._dtype = None
        self._device = device
        self.target_height = target_height
        self.target_width = target_width

        if self._vq_model is not None:
            if not device:
                devices = {p.device for p in self._vq_model.parameters()}
                assert len(devices) == 1, "Model parameters must be on a single device"
                self._device = devices.pop()
            else:
                self._vq_model.to(self._device)

            self._vq_model.eval()
            dtypes = {p.dtype for p in self._vq_model.parameters()}
            assert len(dtypes) == 1, "Model parameters must have a single dtype"
            self._dtype = dtypes.pop()

            if hasattr(self._vq_model, 'encoder') and hasattr(self._vq_model.encoder, 'ch_mult'):
                num_downsamplings = len(self._vq_model.encoder.ch_mult) - 1
                self.downsample_factor = 2 ** num_downsamplings
            else:
                # Fallback or raise error if config structure changes
                print("Warning: Could not determine downsample_factor from model config, assuming 16.")
                self.downsample_factor = 16
            # --- ---

        else:
            self.downsample_factor = 16
            print("Warning: No VQModel provided, assuming downsample_factor=16 for padding.")


        # Normalization remains the same
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.to_tensor = transforms.PILToTensor()
        self.to_dtype = transforms.ToDtype(self._dtype if self._dtype else torch.float32, scale=True)


    @staticmethod
    def _whiten_transparency(img: Image.Image) -> Image.Image:
        if img.mode == "RGB": return img
        vals_rgba = np.array(img)
        if img.mode != "RGBA":
            return img.convert("RGB")

        # Ensure RGBA before proceeding
        if vals_rgba.shape[2] != 4:
            return img.convert("RGB") # Or handle appropriately

        alpha = vals_rgba[:, :, 3:] / 255.0
        vals_rgb = (1 - alpha) * 255 + alpha * vals_rgba[:, :, :3]
        return Image.fromarray(np.uint8(vals_rgb), "RGB")

    def _preprocess_image(self, img: Image.Image) -> Tuple[Tensor, Tuple[int, int]]:
        """
        Resizes, pads, and preprocesses a single image.

        Returns:
            - Padded tensor ready for VQGAN.
            - image size (w, h).
        """
        img = self._whiten_transparency(img)
        w_orig, h_orig = img.size

        # --- Determine target size before padding ---

        if self.target_height and self.target_width:
            # Resize to specific target dimensions
            h_new, w_new = self.target_height, self.target_width
        elif self.target_height or self.target_width:
            # Scale longest side to target, preserve aspect ratio
            scale_w = (self.target_width / w_orig) if self.target_width else 1.0
            scale_h = (self.target_height / h_orig) if self.target_height else 1.0
            scale = min(scale_w, scale_h) if self.target_width and self.target_height else max(scale_w, scale_h)
            w_new = int(round(w_orig * scale))
            h_new = int(round(h_orig * scale))
        else:
            # Use original size
            w_new, h_new = w_orig, h_orig 

        # Ensure dimensions are at least the downsample factor
        w_new = max(w_new, self.downsample_factor)
        h_new = max(h_new, self.downsample_factor)

        img_resized = img.resize((w_new, h_new), Image.Resampling.LANCZOS)
        w_unpadded, h_unpadded = img_resized.size # size *before* padding

        # --- Pad to multiple of downsample_factor ---
        w_padded = _pad_to_multiple(w_unpadded, self.downsample_factor)
        h_padded = _pad_to_multiple(h_unpadded, self.downsample_factor)

        pad_left = (w_padded - w_unpadded) // 2
        pad_right = w_padded - w_unpadded - pad_left
        pad_top = (h_padded - h_unpadded) // 2
        pad_bottom = h_padded - h_unpadded - pad_top

        tensor_resized = self.to_tensor(img_resized) # Shape (C, H, W)

        # Pad (using F.pad for tensors)
        padding = (pad_left, pad_right, pad_top, pad_bottom)
        padded_tensor = F.pad(tensor_resized, padding, mode='constant', value=0) # Pad with 0 before scaling/norm

        # Apply dtype conversion and normalization
        processed_tensor = self.normalize(self.to_dtype(padded_tensor))

        return processed_tensor, (w_padded, h_padded)


    @torch.no_grad()
    def img_tokens_from_pil(
        self,
        images: Union[Image.Image | List[Image.Image]]
    ) -> Tuple[Tensor, Tensor]:
        """
        Encodes PIL image(s) into discrete tokens and returns grid shape info.

        Args:
            images: A single PIL Image or a list of PIL Images.

        Returns:
            A tuple containing:
            - img_toks (Tensor): Flattened image tokens (B, N).
            - grid_size (Tensor): Token grid dimensions for each image (B, 2), where each row is [h_tokens, w_tokens].
        """
        if not self._vq_model:
            raise ValueError("VQModel must be provided during initialization to tokenize images.")
        if not isinstance(images, list): images = [images]

        processed_tensors = []
        padded_sizes = []
        for img in images:
            tensor, pad_size = self._preprocess_image(img)
            processed_tensors.append(tensor)
            padded_sizes.append(pad_size)

        vqgan_input = torch.stack(processed_tensors, dim=0).to(self._device, dtype=self._dtype)

        h = self._vq_model.quant_conv(self._vq_model.encoder(vqgan_input))
        _, _, (_, _, img_toks_flat) = self._vq_model.quantize(h)

        # Calculate grid sizes
        grid_sizes_list = []
        for w_pad, h_pad in padded_sizes:
            w_tokens = w_pad // self.downsample_factor
            h_tokens = h_pad // self.downsample_factor
            grid_sizes_list.append([w_tokens, h_tokens])

        grid_size_tensor = torch.tensor(grid_sizes_list, dtype=torch.long, device=img_toks_flat.device)

        # Sanity check
        if img_toks_flat.ndim == 2:
            expected_n = grid_size_tensor[:, 0] * grid_size_tensor[:, 1]
            assert torch.all(img_toks_flat.shape[1] == expected_n), \
                f"Token count mismatch: {img_toks_flat.shape[1]} vs calculated {expected_n.tolist()}"

        return img_toks_flat, grid_size_tensor

    @staticmethod
    def _pil_from_chw_tensor(chw_tensor: Tensor) -> List[Image.Image]:
        """Converts batch of CHW tensors ([-1, 1]) to list of PIL Images."""
        # Clamp, denormalize (from [-1, 1] to [0, 1]), scale to [0, 255], change layout, convert to numpy/PIL
        normalized = torch.clamp(chw_tensor.detach().cpu(), -1.0, 1.0).add(1.0).div(2.0) # Range [0, 1]
        hwc_byte = (normalized.permute(0, 2, 3, 1) * 255.0).to(torch.uint8).numpy()
        return [Image.fromarray(img, mode="RGB") for img in hwc_byte]

    @torch.no_grad()
    def pil_from_img_toks(
        self,
        img_tensor: Tensor,
        grid_size: Tensor
    ) -> List[Image.Image]:
        """
        Decodes flat image tokens back to PIL Images, handling aspect ratio and cropping.

        Args:
            img_tensor (Tensor): Flattened image tokens (B, N).
            grid_size (Tensor): Token grid dimensions for each image (B, 2), [w_tokens, h_tokens].

        Returns:
            List[Image.Image]: The reconstructed PIL images.
        """
        if not self._vq_model:
            raise ValueError("VQModel must be provided during initialization to detokenize images.")

        bz = img_tensor.shape[0]

        pixels_list = []
        for i in range(bz):
            tokens_i = img_tensor[i]
            w_tokens, h_tokens = grid_size[i].tolist()

            num_tokens_expected = w_tokens * h_tokens
            assert tokens_i.shape[0] == num_tokens_expected, \
                f"Item {i}: Expected {num_tokens_expected} tokens, got {tokens_i.shape[0]}"

            # lookup codebook
            # The shape required by get_codebook_entry's permute is (B, H, W, C)
            # We process one image at a time here, so B=1 conceptually for shape
            shape_bhwc = (1, h_tokens, w_tokens, self._vq_model.quantize.e_dim)
            # Unsqueeze tokens_i to add batch dim: (1, N)
            codebook_entry = self._vq_model.quantize.get_codebook_entry(tokens_i.unsqueeze(0), shape=shape_bhwc)

            # Decode
            pixels = self._vq_model.decode(codebook_entry)
            pixels_list.append(pixels)

        # Stack results back into a batch and convert to PIL
        final_pixels = torch.cat(pixels_list, dim=0)
        return self._pil_from_chw_tensor(final_pixels)
    