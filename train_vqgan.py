import os

import datasets
import lightning as L
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from model import VQModel


def train(model: L.LightningModule, dataloader: DataLoader):
    trainer = L.Trainer(max_epochs=1)
    trainer.fit(model, dataloader)
    sd = model.state_dict()
    sd.pop('loss')
    torch.save(sd, "weights/vqgan.pth")


if __name__ == '__main__':
    model = VQModel(lossconfig=dict(
        disc_conditional=False,
        disc_in_channels=3,
        disc_start=10000,
        disc_weight=0.8,
        codebook_weight=1.0,
    ))
    print(model)
    dataset = datasets.load_dataset('/home/andrew264/datasets/imagenet-1k', streaming=True, trust_remote_code=True,
                                    split='train')
    preprocess = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


    def transform(examples):
        return torch.stack([preprocess(image["image"].convert("RGB")) for image in examples])


    dataloader = DataLoader(dataset, batch_size=2, num_workers=4, collate_fn=transform)
    if os.path.exists('./weights/vqgan.pth'):
        model.load_state_dict(torch.load('./weights/vqgan.pth', map_location='cuda'), strict=False)
    train(model, dataloader)
