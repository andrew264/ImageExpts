from typing import Any, List, Tuple, Union, Optional

import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image, ImageOps
from PIL.Image import Resampling
from torch import Tensor

from model import VQModel

def exists(x: Optional[Any]) ->bool: return x is not None


class ImageTokenizer:
    # https://github.com/facebookresearch/chameleon/blob/main/chameleon/inference/image_tokenizer.py
    def __init__(self, model: VQModel = None, resolution: int = 512, device: str | torch.device | None = None,):
        self._vq_model = model
        self._dtype = None

        if exists(model):
            if not exists(device):
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

        self.transform = transforms.Compose([
            transforms.Resize(resolution, transforms.InterpolationMode.LANCZOS),
            transforms.PILToTensor(),
            transforms.CenterCrop(resolution),
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

class TiledImageTokenizer:
    def __init__(self, model: Optional[VQModel] = None, max_height=512, tile_size=64, tile_separator: Optional[int]=None, row_separator: Optional[int]=None,
                 device: Optional[Union[str, torch.device]] = None):
        self._vq_model = model
        self._dtype = None
        self.max_h = max_height
        self.multiple = tile_size
        self.tile_size = (tile_size, tile_size)
        self._tokens_per_tile: Optional[int] = None
        if exists(row_separator) and exists(tile_separator): assert row_separator != tile_separator, "row_separator can't be the same as tile_separator"
        if not exists(row_separator): print('row_separator is not set; image tiles will be stacked vertically')

        if exists(model):
            device = self._setup_device(device)
            self._vq_model.to(device).eval()
            self._dtype = next(self._vq_model.parameters()).dtype
        
        self._device = device

        self.tile_separator = tile_separator
        self.row_separator = row_separator
        self.transform = transforms.Compose([
            transforms.ToDtype(self._dtype or torch.float32, scale=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def _setup_device(self, device: Optional[Union[str, torch.device]]) -> torch.device:
        if not exists(device):
            devices = {p.device for p in self._vq_model.parameters()}
            assert len(devices) == 1, "Model parameters are on different devices!"
            return devices.pop()
        return torch.device(device)

    @property
    def tokens_per_tile(self) -> int:
        if not exists(self._tokens_per_tile):
            random_img = torch.rand((1, 3, *self.tile_size), dtype=self._dtype, device=self._device)
            _, _, [_, _, img_toks] = self._vq_model.encode(random_img)
            self._tokens_per_tile = img_toks.size(-1)
        return self._tokens_per_tile

    def _preprocess_image(self, img: Image.Image) -> tuple[Tensor, int]:
        if img.height > self.max_h:
            ratio = self.max_h / img.height
            img = img.resize((int(img.width * ratio), self.max_h), resample=Resampling.LANCZOS)

        new_dims = ((img.width + self.multiple - 1) // self.multiple * self.multiple,
                    (img.height + self.multiple - 1) // self.multiple * self.multiple)
        img = ImageOps.pad(img, new_dims, color=(0, 0, 0))

        tiles = [self._process_tile(np.array(img.crop((j, i, j + self.tile_size[0], i + self.tile_size[1]))))
                 for i in range(0, img.height, self.tile_size[1])
                 for j in range(0, img.width, self.tile_size[0])]
        return self.transform(torch.stack(tiles).permute(0, 3, 1, 2)).to(device=self._device), img.width // self.tile_size[0]

    def _process_tile(self, tile: np.ndarray) -> Tensor:
        if tile.shape[-1] == 4:
            alpha = tile[:, :, 3:] / 255.0
            tile = (1 - alpha) * 255 + alpha * tile[:, :, :3]
        return torch.from_numpy(tile.astype(np.uint8))

    def img_tokens_from_pil(self, images: Union[Image.Image, List[Image.Image]]) -> Tensor:
        if not isinstance(images, list): images = [images]
        tokens_list = []
        for img in images:
            img_tensor, tiles_per_row = self._preprocess_image(img)
            _, _, [_, _, img_toks] = self._vq_model.encode(img_tensor)
            tokens_list.append(self._add_separators(img_toks, tiles_per_row))
        return torch.stack(tokens_list)
    
    def _add_separators(self, tokens: Tensor, tiles_per_row: int) -> Tensor:
        num_tiles, features_per_tile = tokens.shape
        num_rows = (num_tiles + tiles_per_row - 1) // tiles_per_row
        total_length = (num_tiles * features_per_tile
                    + (num_tiles if exists(self.tile_separator) else 0)
                    + (num_rows - 1 if exists(self.row_separator) else 0))
        out = torch.empty(total_length, dtype=tokens.dtype, device=tokens.device)
        idx = 0
        for i in range(num_tiles):
            out[idx:idx+features_per_tile] = tokens[i]
            idx+=features_per_tile
            if exists(self.tile_separator):
                out[idx] = self.tile_separator
                idx += 1
            if exists(self.row_separator) and (i+1) % tiles_per_row == 0 and i < num_tiles -1:
                out[idx] = self.row_separator
                idx+=1
        return out[:idx]
    
    def _remove_separators(self, tokens: Tensor) -> Tuple[Tensor, int]:
        tiles_per_row = None
        if exists(self.row_separator):
            first_row_sep_idx = (tokens==self.row_separator).nonzero(as_tuple=True)[0]
            if len(first_row_sep_idx)> 0:
                first_row_sep_idx = first_row_sep_idx[0].item()
                non_separator_count = sum((tokens[:first_row_sep_idx] != self.tile_separator) & (tokens[:first_row_sep_idx] != self.row_separator))
                tiles_per_row = non_separator_count // self.tokens_per_tile

        mask = torch.ones_like(tokens, dtype=torch.bool)
        if exists(self.tile_separator): mask &= (tokens != self.tile_separator)
        if exists(self.row_separator): mask &= (tokens != self.row_separator)
        new_tokens = tokens[mask]
        return new_tokens.reshape(-1, self.tokens_per_tile), tiles_per_row.item() if exists(tiles_per_row) else 1

    def stich_tiles(self, tiles: List[Image.Image], tiles_per_row: int) -> Image.Image:
        stitched_img = Image.new('RGB', (tiles_per_row * self.tile_size[0], len(tiles) // tiles_per_row * self.tile_size[1]))
        for idx, tile in enumerate(tiles):
            row, col = divmod(idx, tiles_per_row)
            stitched_img.paste(tile, (col * self.tile_size[0], row * self.tile_size[1]))
        return stitched_img

    def pil_from_img_toks(self, img_tensor: Tensor) -> List[Image.Image]:
        if img_tensor.ndim == 3: img_tensor = img_tensor.unsqueeze(0)

        emb_dim = self._vq_model.quantize.embedding.weight.shape[-1]
        img_list = []
        for img in img_tensor:
            img, tiles_per_row = self._remove_separators(img)
            bz, num_tokens = img.size()
            out_size = int(num_tokens ** 0.5)

            codebook_entry = self._vq_model.quantize.get_codebook_entry(img, (bz, out_size, out_size, emb_dim))
            pixels = self._vq_model.decode(codebook_entry)
            img_list.append(self.stich_tiles(self._pil_from_chw_tensor(pixels), tiles_per_row=tiles_per_row))
        return img_list

    @staticmethod
    def _pil_from_chw_tensor(chw_tensor: Tensor) -> List[Image.Image]:
        normalized_tensor = torch.clamp(chw_tensor.cpu().detach(), -1.0, 1.0).add(1).div(2)
        img_data = (normalized_tensor.permute(0, 2, 3, 1) * 255).to(torch.uint8)
        return [Image.fromarray(img.numpy(), mode="RGB") for img in img_data]
