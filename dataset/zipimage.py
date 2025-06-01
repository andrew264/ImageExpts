import zipfile
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms


class ZipImageDataset(Dataset):
    def __init__(self, zip_path, resolution, dtype=None):
        self.zip_path = zip_path
        self.resolution = resolution
        self._dtype = dtype if dtype is not None else torch.float32

        self.valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
        self.zf = zipfile.ZipFile(self.zip_path, 'r')
        self.file_list = [
            info for info in self.zf.infolist()
            if not info.is_dir() and info.filename.lower().endswith(self.valid_exts)
        ]

        self.to_tensor = transforms.PILToTensor()
        self.to_dtype = transforms.ToDtype(self._dtype, scale=True)
        self.normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        self.resize = transforms.Resize((resolution, resolution), antialias=True)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        info = self.file_list[index]
        with self.zf.open(info) as file:
            image = Image.open(file).convert("RGBA")

        image = self.resize(image)
        tensor = self.to_tensor(image)
        tensor = self.to_dtype(tensor)
        tensor = self.normalize(tensor)
        return tensor

    def __del__(self):
        self.zf.close()