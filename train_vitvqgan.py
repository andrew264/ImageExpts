import torch
from torch import nn
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from dataset.zipimage import ZipImageDataset
from model.vit_vqgan.config import ViTVQGANConfig
from model.vit_vqgan.model import ViTVQGAN

torch.set_float32_matmul_precision('medium')


def count_params(model: nn.Module) -> float:
    total_params = sum(p.numel() for p in model.parameters())
    return round(total_params / 1_000_000, 2)


def train(model: L.LightningModule, dataloader: DataLoader):
    checkpoint_callback = ModelCheckpoint(
        dirpath="weights/vitvqgan", filename="weights", save_weights_only=True, every_n_train_steps=1000)
    trainer = L.Trainer(max_epochs=1, precision="bf16-true",
                        callbacks=[checkpoint_callback])
    trainer.fit(model, dataloader)
    sd = model.state_dict()
    for key in list(sd.keys()):
        if key.startswith('loss'):
            sd.pop(key)
    torch.save(sd, "weights/vitvqgan.pth")


if __name__ == '__main__':
    model = ViTVQGAN(config=ViTVQGANConfig()).bfloat16()
    print(model)
    path = "/home/andrew264/PycharmProjects/ImageExperiments/weights/vitvqgan/weights-v6.ckpt"
    sd = torch.load(path, weights_only=True)['state_dict']
    model.load_state_dict(sd, strict=False)
    print(count_params(model), 'M parameters')
    ds = ZipImageDataset('/mnt/d/stickers/train.zip', 512, torch.bfloat16)
    dataloader = DataLoader(ds, batch_size=8, num_workers=1, shuffle=True,)

    train(model, dataloader)
