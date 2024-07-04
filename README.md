# Image Experiments
## Why does this repository exist?
This repository is a collection of experiments that I have done with images. So I can learn how to manipulate images and the various deep learning architectures that are used to generate images.

## What does this repository contain?
### 1. [UNet 2D](model/unet_2d/unet_2d.py)
A basic diffusion model for generating images.

### 2. [Vector Quantized Generative Adversarial Networks (VQGAN)](model/vqgan/vqgan.py)
A model that learns to represent images into discrete tokens. Can be used for image tokenization. Has training code for both the Generator and the Discriminator in PyTorch.

## What is the purpose of this repository?
This is my playground for experimenting with images. I will be adding more models and experiments as I learn more about other architectures and techniques.
The final goal is to add multimodality in [Smol-LM](https://github.com/andrew264/Smol-LM)

## Notes
- The code is not optimized for performance. It is written in a way that is easy for me to experiment with.
- Improvements and suggestions are welcome. Feel free to open an issue or a pull request.
- Experiments with Audio is done over on [AudioExpts](https://github.com/andrew264/AudioExpts)