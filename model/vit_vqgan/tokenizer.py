from typing import Optional, Union, List, Tuple

import torch
from torch import Tensor
import torchvision.transforms.v2 as transforms
from PIL import Image

from model.vit_vqgan.model import ViTVQGAN


class ImageTokenizer:
    def __init__(
        self,
        model: Optional[ViTVQGAN] = None,
        target_height: Optional[int] = None,
        target_width: Optional[int] = None,
        patch_size: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.model = model
        self.target_height = target_height
        self.target_width = target_width

        if model:
            self.patch_size = getattr(model.config, 'patch_size', 16) if patch_size is None else patch_size
            if dtype is None:
                try:
                    self.dtype = next(model.parameters()).dtype
                except (StopIteration, AttributeError):
                    self.dtype = torch.float32
            else:
                self.dtype = dtype
            self.device = next(model.parameters()).device if device is None and hasattr(model, 'parameters') and list(model.parameters()) else \
                          (torch.device(device) if isinstance(device, str) else device) if device else \
                          torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self.model and isinstance(self.model, ViTVQGAN):
                self.model.eval()
                self.model.to(self.device)

        else:
            if patch_size is None:
                raise ValueError("patch_size must be defined if model is not provided.")
            self.patch_size = patch_size
            self.dtype = dtype if dtype is not None else torch.float32
            self.device = (torch.device(device) if isinstance(device, str) else device) if device else \
                           torch.device("cuda" if torch.cuda.is_available() else "cpu")


        if self.target_height and self.target_height % self.patch_size != 0:
            print(f"Warning: target_height ({self.target_height}) is not a multiple of patch_size ({self.patch_size}). May lead to cropping or padding issues if not handled by model.")
        if self.target_width and self.target_width % self.patch_size != 0:
            print(f"Warning: target_width ({self.target_width}) is not a multiple of patch_size ({self.patch_size}). May lead to cropping or padding issues if not handled by model.")


    def _calculate_target_dims(self, w_orig: int, h_orig: int) -> Tuple[int, int]:
        if self.target_height and self.target_width:
            h_new, w_new = self.target_height, self.target_width
        elif self.target_height or self.target_width:
            scale = 1.0
            if self.target_width and self.target_height:
                scale = min(self.target_width / w_orig, self.target_height / h_orig)
            elif self.target_width:
                scale = self.target_width / w_orig
            elif self.target_height:
                scale = self.target_height / h_orig
            
            w_new = int(round(w_orig * scale))
            h_new = int(round(h_orig * scale))
        else:
            w_new, h_new = w_orig, h_orig

        w_new = max(w_new, self.patch_size)
        h_new = max(h_new, self.patch_size)

        w_new = (w_new // self.patch_size) * self.patch_size
        h_new = (h_new // self.patch_size) * self.patch_size
        
        w_new = max(w_new, self.patch_size)
        h_new = max(h_new, self.patch_size)

        return h_new, w_new

    def preprocess_image(self, img: Image.Image) -> Tensor:
        if img.mode == 'RGBA' and self.model and getattr(self.model.config, 'num_channels', 3) == 3:
            img = img.convert('RGB')
        elif img.mode != 'RGB' and img.mode != 'L' and img.mode != 'RGBA':
            img = img.convert('RGB')
        elif img.mode == 'RGB' and self.model and getattr(self.model.config, 'num_channels', 4) == 4:
            img = img.convert('RGBA')

        w_orig, h_orig = img.size
        h_new, w_new = self._calculate_target_dims(w_orig, h_orig)

        preprocess_transforms = transforms.Compose([
            transforms.Resize((h_new, w_new), interpolation=transforms.InterpolationMode.LANCZOS, antialias=True),
            transforms.PILToTensor(),
            transforms.ToDtype(self.dtype, scale=True),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        return preprocess_transforms(img)

    def postprocess_tensor(self, tensor: Tensor) -> Image.Image:
        if tensor.ndim == 4 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        
        unnormalized_tensor = (tensor * 0.5) + 0.5
        
        # Scale to [0, 255]
        if unnormalized_tensor.is_floating_point():
            scaled_tensor = unnormalized_tensor * 255.0
        else:
            scaled_tensor = unnormalized_tensor

        clamped_tensor = torch.clamp(scaled_tensor, 0, 255)
        uint8_tensor = clamped_tensor.to(torch.uint8)
        
        to_pil_transform = transforms.ToPILImage()
        pil_image = to_pil_transform(uint8_tensor.cpu())
        return pil_image

    @torch.no_grad()
    def tokenize(self, images: Union[Image.Image, List[Image.Image]]) -> Tuple[List[Tensor], List[Tuple[int, int]]]:
        if self.model is None or not isinstance(self.model, ViTVQGAN):
            raise RuntimeError("A ViTVQGAN model instance must be provided for tokenization.")

        if not isinstance(images, list):
            images = [images]

        processed_tensors = [self.preprocess_image(img).to(self.device) for img in images]

        all_same_shape = True
        if len(processed_tensors) > 1:
            first_shape = processed_tensors[0].shape
            for t in processed_tensors[1:]:
                if t.shape != first_shape:
                    all_same_shape = False
                    break
        
        all_indices: List[Tensor] = []
        all_grid_hw: List[Tuple[int, int]] = []

        if all_same_shape and processed_tensors:
            batch = torch.stack(processed_tensors)
            encoded_features, grid_hw_for_batch = self.model.encoder(batch)
            _, batch_indices = self.model.quantize(encoded_features)

            for i in range(batch_indices.shape[0]):
                all_indices.append(batch_indices[i].cpu())
                all_grid_hw.append(grid_hw_for_batch)
        else:
            for tensor_img in processed_tensors:
                tensor_img_batch = tensor_img.unsqueeze(0)
                encoded_features, grid_hw_single = self.model.encoder(tensor_img_batch)
                _, indices_single = self.model.quantize(encoded_features)
                
                all_indices.append(indices_single.squeeze(0).cpu())
                all_grid_hw.append(grid_hw_single)
        
        return all_indices, all_grid_hw

    @torch.no_grad()
    def detokenize(self, 
                   indices_list: List[Tensor], 
                   grid_hw_list: List[Tuple[int, int]]
                  ) -> List[Image.Image]:
        if self.model is None or not isinstance(self.model, ViTVQGAN):
            raise RuntimeError("A ViTVQGAN model instance must be provided for detokenization.")
        if len(indices_list) != len(grid_hw_list):
            raise ValueError("Length of indices_list and grid_hw_list must match.")

        pil_images: List[Image.Image] = []
        
        all_grid_hw_same = False
        if grid_hw_list:
            first_grid_hw = grid_hw_list[0]
            all_grid_hw_same = all(ghw == first_grid_hw for ghw in grid_hw_list)

        if all_grid_hw_same and indices_list:
            device_indices = [idx.to(self.device) for idx in indices_list]
            try:
                batch_indices = torch.stack(device_indices)
                common_grid_hw = grid_hw_list[0]
                reconstructed_batch = self.model.decode(batch_indices, common_grid_hw, is_indices=True)
                
                for i in range(reconstructed_batch.shape[0]):
                    pil_images.append(self.postprocess_tensor(reconstructed_batch[i]))
            except RuntimeError as e:
                print(f"Batch detokenization failed ({e}), falling back to individual processing.")
                all_grid_hw_same = False

        if not all_grid_hw_same or not pil_images:
            if pil_images:
                pil_images = []
            for i in range(len(indices_list)):
                idx_tensor = indices_list[i].to(self.device)
                grid_hw = grid_hw_list[i]
                
                idx_tensor_batch = idx_tensor.unsqueeze(0)
                
                reconstructed_single = self.model.decode(idx_tensor_batch, grid_hw, is_indices=True)
                pil_images.append(self.postprocess_tensor(reconstructed_single.squeeze(0)))
                
        return pil_images