import os

import datasets
import lightning as L
import torch
from PIL import Image
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint

from model import VQModel
from model.vqgan.image_tokenizer import ImageTokenizer

torch.set_float32_matmul_precision('high')

class Debug(L.Callback):
    def __init__(self):
        self.tokenizer: ImageTokenizer = None
        self.image = Image.open('data/test.png')
    def setup(self, trainer, pl_module, stage):
        self.tokenizer = ImageTokenizer(pl_module)
    def on_train_batch_end(self, trainer: L.Trainer, pl_module, outputs, batch, batch_idx):
        step = batch_idx
        if step > 0 and step % 500 == 0:
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
    dd_config = dict(double_z=False, z_channels=256, resolution=512, in_channels=3, out_ch=3, ch=128, ch_mult=[1, 1, 2, 2, 4], num_res_blocks=2, attn_resolutions=[],)
    l_cfg = dict(disc_conditional=False, disc_in_channels=3, disc_start=100000, disc_weight=0.8, codebook_weight=1.0,)
    model = VQModel(dd_config, n_embed=2**14, lossconfig=l_cfg, grad_accum_steps=4).bfloat16().cuda()
    # model.forward = torch.compile(model=model.forward,)
    print(model)

    path = "/home/andrew264/datasets/text-to-image-2M/"
    files = [path + f for f in os.listdir(path)]
    dataset = datasets.load_dataset("webdataset", data_files={"train": files}, split='train', streaming=True)
    tokenizer = ImageTokenizer()

    def transform(examples): return tokenizer._vqgan_input_from([e["jpg"] for e in examples]).bfloat16()


    dataloader = DataLoader(dataset, batch_size=1, num_workers=4, collate_fn=transform)
    if os.path.exists('./weights/vqgan.pth'):
        model.load_state_dict(torch.load('./weights/vqgan.pth', map_location='cuda', weights_only=True), strict=False)
    train(model, dataloader)
