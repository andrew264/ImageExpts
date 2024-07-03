import numpy as np
import torch
from PIL import Image
from torch import Tensor

from model import VQModel


class ImageTokenizer:
    # https://github.com/facebookresearch/chameleon/blob/main/chameleon/inference/image_tokenizer.py
    def __init__(
            self,
            model: VQModel = None,
            device: str | torch.device | None = None,
    ):
        self._vq_model = model or VQModel()

        if device is None:
            devices = {p.device for p in self._vq_model.parameters()}
            assert len(devices) == 1
            device = devices.pop()
        else:
            self._vq_model.to(device)
        self._device = device

        dtypes = {p.dtype for p in self._vq_model.parameters()}
        assert len(dtypes) == 1
        self._dtype = dtypes.pop()

    @staticmethod
    def _whiten_transparency(img: Image.Image) -> Image.Image:
        if img.mode == "RGB":
            return img

        vals_rgba = np.array(img.convert("RGBA"))
        if not np.any(vals_rgba[:, :, 3] < 255):
            return img.convert("RGB")

        alpha = vals_rgba[:, :, 3] / 255.0
        vals_rgb = (1 - alpha[:, :, np.newaxis]) * 255 + alpha[:, :, np.newaxis] * vals_rgba[:, :, :3]
        return Image.fromarray(vals_rgb.astype(np.uint8), "RGB")

    @staticmethod
    def _vqgan_input_from(img: Image.Image, target_image_size: int = 512) -> Tensor:
        scale = target_image_size / min(img.size)
        new_size = tuple(round(scale * dim) for dim in img.size)
        img = img.resize(new_size, Image.LANCZOS)

        x0, y0 = [(dim - target_image_size) // 2 for dim in img.size]
        img = img.crop((x0, y0, x0 + target_image_size, y0 + target_image_size))

        np_img = np.array(img) / 127.5 - 1  # Normalize to [-1, 1]
        return torch.from_numpy(np_img).permute(2, 0, 1).float().unsqueeze(0)

    def img_tokens_from_pil(self, image: Image.Image) -> Tensor:
        image = self._whiten_transparency(image)
        vqgan_input = self._vqgan_input_from(image).to(self._device, dtype=self._dtype)
        _, _, [_, _, img_toks] = self._vq_model.encode(vqgan_input)
        return img_toks

    @staticmethod
    def _pil_from_chw_tensor(chw_tensor: Tensor) -> Image.Image:
        normalized_chw_tensor = torch.clamp(chw_tensor.detach().cpu(), -1.0, 1.0).add(1).div(2)
        hwc_array = (normalized_chw_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(hwc_array, mode="RGB")

    def pil_from_img_toks(self, img_tensor: Tensor) -> Image.Image:
        emb_dim = self._vq_model.quantize.embedding.weight.shape[-1]
        codebook_entry = self._vq_model.quantize.get_codebook_entry(
            img_tensor, (1, 32, 32, emb_dim)
        )
        pixels = self._vq_model.decode(codebook_entry)
        return self._pil_from_chw_tensor(pixels[0])
