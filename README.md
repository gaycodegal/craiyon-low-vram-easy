# Run Craiyon locally (formerly dalle-mini)

This script runs dalle-mini:mega locally for low VRAM devices by
only running one step and one model at a time on your GPU.
If multiple GPUs are detected I'm pretty sure this script
will attempt to use both of them, which may cause issues if one
of them is very bad.

## Usage
### Memory Issues
If this command says it fails to allocate enough memory your computer isn't
powerful enough to run with GPU. Either run it on CPU (hide CUDA from this
program) or use the dalle-mini:mini-1:v0 model instead.

WARNING: Please be aware this script is meant for lower end devices and does not
parallelize multiple instances onto the same GPU because that would lead to
OOM (out of memory) on such lower end devices

### Example use

python3 run-mega-low-vram.py --prompt "frog under a bridge" --prompt "frog on a mushroom" -n 2
python3 run-mega-low-vram.py --prompt "the Eiffel tower landing on the moon"

## Install
download to separate folders with files properly named

`./mega-1-fp16`

https://wandb.ai/dalle-mini/dalle-mini/artifacts/DalleBart_model/mega-1-fp16/latest/files

`./vqgan_imagenet_f16_16384`

https://huggingface.co/dalle-mini/vqgan_imagenet_f16_16384/tree/main

## CUDA installation for linux
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
both `cuda nvidia-cudnn` packages required

NOTE: There are post install steps for CUDA, which includes rebooting your
computer during part of them

!pip install -q dalle-mini
!pip install -q git+https://github.com/patil-suraj/vqgan-jax.git

## Credit
The ideas to run this locally were taken from suggestions from
https://github.com/dlivitz/dalle-playground/tree/main/backend
