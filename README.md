# Run Craiyon locally (formerly dalle-mini)

This script runs dalle-mini:mega locally for low VRAM devices by
only running one step and one model at a time on your GPU.
If multiple GPUs are detected I'm pretty sure this script
will attempt to use both of them, which may cause issues if one
of them is very bad.

WARNING: You may only be able to have basic programs like a text editor and terminal open at the same time as you run this script. I don't know if it can handle having a browser open at the same time on a low end device. I have 16GB RAM and a 1060 mobile GPU and it was taxing most of my GPU, many of my computer cores, and 12GB resting RAM.

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

### Filenames

The script will name files {n}-{uuid4}.png. I inserted a uuid4 so that multiple
runs of this script won't overwrite previous runs. This code takes a while
to run, so like losing work is really bad. I haven't coded in anything to
recover intermediary states if it crashes before it can use vqgan to decode
the images, but theoretically you could pickle the memory and write that
to disk so you can recover any work lost. N is a number from 1->N for the number of images produced

## Install
### Download Models
download to separate folders with files properly named

`./mega-1-fp16`

https://wandb.ai/dalle-mini/dalle-mini/artifacts/DalleBart_model/mega-1-fp16/latest/files

`./vqgan_imagenet_f16_16384`

https://huggingface.co/dalle-mini/vqgan_imagenet_f16_16384/tree/main

### CUDA installation for linux
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
both `cuda nvidia-cudnn` packages required

NOTE: There are post install steps for CUDA, which includes rebooting your
computer during part of them

### pip installation
Note: use python 3. pip --version must say python 3.xxxx
!pip install -q dalle-mini
!pip install -q git+https://github.com/patil-suraj/vqgan-jax.git

## Credit
The original source script came from [this APACHE 2.0 licensed jupyter notebook][inference_pipeline]
by borisdayma

The ideas to run this with low VRAM were taken from suggestions from
https://github.com/dlivitz/dalle-playground/tree/main/backend

## License
You just have to credit the original author of craiyon with its APACHE 2.0 license

[inference_pipeline]: https://github.com/borisdayma/dalle-mini/blob/main/tools/inference/inference_pipeline.ipynb