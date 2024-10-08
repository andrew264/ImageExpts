import os.path

import lightning as L
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from model import UNet2DConfig, UNet2DModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16
torch.set_float32_matmul_precision('high')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model, dataloader):
    trainer = L.Trainer(max_epochs=50, precision='bf16-true', gradient_clip_val=1.0, accumulate_grad_batches=1)
    trainer.fit(model, dataloader)
    torch.save(model.state_dict(), "weights/unet-2d.pth")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    cfg = UNet2DConfig()
    model = UNet2DModel(config=cfg).to(device=device, dtype=dtype)
    print(model)
    print(f"Model has {count_parameters(model):,} trainable parameters.")
    dataset = load_dataset("huggan/smithsonian_butterflies_subset", split="train")
    preprocess = transforms.Compose(
        [
            transforms.Resize((cfg.sample_size, cfg.sample_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


    def transform(examples):
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        return {"images": images}


    dataset.set_transform(transform)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    if os.path.exists('./weights/unet-2d.pth'):
        sd = torch.load('./weights/unet-2d.pth', map_location='cuda')
        model.load_state_dict(sd)
    train(model, dataloader)
