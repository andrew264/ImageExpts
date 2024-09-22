from typing import List, Union

import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
from torch import Tensor

from model import VQModel


class ImageTokenizer:
    # https://github.com/facebookresearch/chameleon/blob/main/chameleon/inference/image_tokenizer.py
    def __init__(self, model: VQModel = None, device: str | torch.device | None = None,):
        self._vq_model = model
        self._dtype = None

        if self._vq_model is not None:
            if device is None:
                devices = {p.device for p in self._vq_model.parameters()}
                assert len(devices) == 1
                device = devices.pop()
            else:
                self._vq_model.to(device)
            self._vq_model.eval()
            dtypes = {p.dtype for p in self._vq_model.parameters()}
            assert len(dtypes) == 1
            self._dtype = dtypes.pop()
        self._device = device

        target_image_size = 512
        self.transform = transforms.Compose([
            transforms.Resize(target_image_size, transforms.InterpolationMode.LANCZOS),
            transforms.PILToTensor(),
            transforms.CenterCrop(target_image_size),
            transforms.ToDtype(self._dtype if self._dtype else torch.float32, scale=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    @staticmethod
    def _whiten_transparency(img: Image.Image) -> Image.Image:
        if img.mode == "RGB": return img
        vals_rgba = np.array(img)
        if img.mode != "RGBA": return img.convert("RGB")

        alpha = vals_rgba[:, :, 3:] / 255.0
        vals_rgb = (1 - alpha) * 255 + alpha * vals_rgba[:, :, :3]
        return Image.fromarray(np.uint8(vals_rgb), "RGB")

    def _vqgan_input_from(self, imgs: List[Image.Image]) -> Tensor:
        return torch.stack([self.transform(self._whiten_transparency(img)) for img in imgs], dim=0)

    def img_tokens_from_pil(self, images: Union[Image.Image | List[Image.Image]]) -> Tensor:
        if not isinstance(images, list): images = [images]
        vqgan_input = self._vqgan_input_from(images).to(self._device, dtype=self._dtype)
        _, _, [_, _, img_toks] = self._vq_model.encode(vqgan_input)
        return img_toks

    @staticmethod
    def _pil_from_chw_tensor(chw_tensor: Tensor) -> List[Image.Image]:
        normalized_chw_tensor = torch.clamp(chw_tensor.detach().cpu(), -1.0, 1.0).add(1).div(2)
        normalized_chw_tensor = (normalized_chw_tensor.permute(0, 2, 3, 1) * 255).to(torch.uint8)
        return [Image.fromarray(img.numpy(), mode="RGB") for img in normalized_chw_tensor]

    def pil_from_img_toks(self, img_tensor: Tensor) -> List[Image.Image]:
        emb_dim = self._vq_model.quantize.embedding.weight.shape[-1]
        if img_tensor.ndim == 3: img_tensor = img_tensor.unsqueeze(0)
        bz, num_tokens = img_tensor.size()
        out_size = int(num_tokens ** 0.5)
        codebook_entry = self._vq_model.quantize.get_codebook_entry(img_tensor, (bz, out_size, out_size, emb_dim))
        pixels = self._vq_model.decode(codebook_entry)
        return self._pil_from_chw_tensor(pixels)
