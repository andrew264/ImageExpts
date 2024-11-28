import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import lightning as L
import torchvision.transforms as transforms


from model.siglip import SiglipModel, SiglipConfig

class MNISTDataset(Dataset):
    def __init__(self, image_size: int = 28, split: str = 'train'):
        self.dataset = load_dataset("ylecun/mnist", split=split)
        num_channels = len(self.dataset[0]['image'].getbands())
        self.transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * num_channels, std=[0.5] * num_channels)
    ])
        
    def __getitem__(self, idx: int) -> torch.Tensor:
        data = self.dataset[idx]
        image = data['image']
        label = data['label']

        if image.mode == 'RGBA':
            image = image.convert('RGB')

        return {'input_ids': torch.tensor([label]), 'pixel_values': self.transform(image), 'attention_mask': torch.tensor([1])}
    
    def __len__(self): return len(self.dataset)

def train(model: L.LightningModule,):
    trainer = L.Trainer(accelerator='gpu', max_epochs=1, precision='bf16-true')
    train_d = MNISTDataset(image_size=224)
    train_dl = DataLoader(train_d, batch_size=16, shuffle=True,)
    test_d = MNISTDataset(image_size=224, split='test')
    test_dl = DataLoader(test_d, batch_size=16,)
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=test_dl)
    sd = model.state_dict()
    torch.save(sd, "weights/siglip/siglip.pt")

if __name__ == '__main__':
    config: SiglipConfig = {
        "text_config": {
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "vocab_size": 10,
        "max_position_embeddings": 1,
        "attention_dropout": 0.0,
        "layer_norm_eps": 1e-6,
        },
    "vision_config": {
        "image_size": 224,
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "patch_size": 16,
        "num_channels": 1,
        "attention_dropout": 0.0,
        "layer_norm_eps": 1e-6,
        }
    }
    model = SiglipModel(config=config).cuda()
    model.apply(model._init_weights)
    # model.forward = torch.compile(model.forward)
    print(model)
    train(model)
