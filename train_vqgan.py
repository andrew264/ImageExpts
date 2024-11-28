import os
import io

import h5py
import lightning as L
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
import torchvision.transforms.v2 as transforms

from model import VQModel
from model.vqgan.image_tokenizer import TiledImageTokenizer

torch.set_float32_matmul_precision('high')

class HDF5ImageDataset(Dataset):
    def __init__(self, file_path: str):
        self.path = file_path
        with h5py.File(self.path, 'r') as hf:
            self.keys = list(hf.keys())

        self.transform = transforms.Compose([transforms.ToDtype(torch.bfloat16, scale=True),
                                             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    
    def __len__(self): return len(self.keys)
    
    def jpeg_bytes_to_tensor(self, jpeg_bytes: bytes) -> torch.Tensor:
        img = Image.open(io.BytesIO(jpeg_bytes))
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1)
        return self.transform(img_tensor)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        with h5py.File(self.path, 'r') as hf:
            jpeg_bytes: bytes = hf[self.keys[idx]][()]
        return self.jpeg_bytes_to_tensor(jpeg_bytes)
    
class Debug(L.Callback):
    def __init__(self):
        self.tokenizer: TiledImageTokenizer = None
        self.image = Image.open('data/test.png')
    def setup(self, trainer, pl_module, stage):
        self.tokenizer = TiledImageTokenizer(pl_module, max_height=512, tile_size=128, row_separator=-1, tile_separator=-2)
    def on_train_batch_end(self, trainer: L.Trainer, pl_module, outputs, batch, batch_idx):
        step = batch_idx
        if step % 5000 == 0:
            tokens = self.tokenizer.img_tokens_from_pil(self.image)
            with open('data/tokens.txt', mode='a', encoding='utf-8') as f: f.write(f"step_{step}: " + str(tokens[0].tolist()) + '\n')
            img = self.tokenizer.pil_from_img_toks(tokens)[0]
            img.save(f'data/test_step_{step}.png')
            
def train(model: L.LightningModule, dataloader: DataLoader):
    checkpoint_callback = ModelCheckpoint(dirpath="weights/vqgan", filename="weights", save_weights_only=True, every_n_train_steps=5000)
    trainer = L.Trainer(max_epochs=1, precision="bf16-true", callbacks=[checkpoint_callback, Debug()])
    trainer.fit(model, dataloader)
    sd = model.state_dict()
    for key in list(sd.keys()):
        if key.startswith('loss'): sd.pop(key)
    torch.save(sd, "weights/vqgan.pth")


if __name__ == '__main__':
    dd_config = dict(double_z=False, z_channels=256, resolution=128, in_channels=3, out_ch=3, ch=128, ch_mult=[1, 1, 2, 2, 4], num_res_blocks=2, attn_resolutions=[],)
    loss_config = dict(disc_conditional=False, disc_in_channels=3, disc_start=100000000, disc_weight=0.8, codebook_weight=1.0,)
    model = VQModel(dd_config, lossconfig=loss_config, n_embed=2**13, grad_accum_steps=1).bfloat16().cuda()
    # model.forward = torch.compile(model=model.forward,)
    print(model)

    path = "./vid_images.h5"
    dataset = HDF5ImageDataset(path)

    dataloader = DataLoader(dataset, batch_size=16, num_workers=4, shuffle=True,)
    # if os.path.exists('./weights/vqgan.ckpt'):
    #     model.load_state_dict(torch.load('./weights/vqgan.ckpt', map_location='cuda', weights_only=True)['state_dict'], strict=False)
    #     print('loaded weights')
    train(model, dataloader)
