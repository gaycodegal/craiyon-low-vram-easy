#!/usr/bin/python3
"""
Run the dalle model for low VRAM devices like nvidia 1060 mobile (laptop) gpu. 

This script runs dalle-mini:mega locally for low VRAM devices by
only running one step and one model at a time on your GPU.
If multiple GPUs are detected I'm pretty sure this script
will attempt to use both of them, which may cause issues if one
of them is very bad.

The script will name files 1-{uuid4}.png. I inserted a uuid4 so that multiple
runs of this script won't overwrite previous runs. This code takes a while
to run, so like losing work is really bad. I haven't coded in anything to
recover intermediary states if it crashes before it can use vqgan to decode
the images, but theoretically you could pickle the memory and write that
to disk so you can recover any work lost.

Please see README.md for more information.

download to separate folders with files properly named
./mega-1-fp16
https://wandb.ai/dalle-mini/dalle-mini/artifacts/DalleBart_model/mega-1-fp16/latest/files
./vqgan_imagenet_f16_16384
https://huggingface.co/dalle-mini/vqgan_imagenet_f16_16384/tree/main

CUDA installation linux:
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
both `cuda nvidia-cudnn` packages required

NOTE: There are post install steps for CUDA, which includes rebooting your
computer during part of them

!pip install -q dalle-mini
!pip install -q git+https://github.com/patil-suraj/vqgan-jax.git

If this command says it fails to allocate enough memory your computer isn't
powerful enough to run with GPU. Either run it on CPU (hide CUDA from this
program) or use the dalle-mini:mini-1:v0 model instead.

WARNING: Please be aware this script is meant for lower end devices and does not
parallelize multiple instances onto the same GPU because that would lead to
OOM (out of memory) on such lower end devices
"""

import argparse
parser = argparse.ArgumentParser(description=__doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
parser.add_argument('-p', '--prompt', type=str, action="append",
                    help='The sentences this model will convert into images. each prompt requires you to type out --prompt="your prompt here"',
                    default=[])

parser.add_argument('-n', '--n-predictions', type=int, dest='predictions', default=8,
                    help='The number of images the model will update for each prompt. If you have 3 prompts and 8 predictions per prompt, you will receive 24 images.')

# listen technically it can load from wandb, but for me I found that it tried to download the model every time??
# I put the images in ./mega-1-fp16
parser.add_argument('--model', type=str, default='./mega-1-fp16',
                    help='The directory the text to image model (DalleBART model) is stored in. Download https://wandb.ai/dalle-mini/dalle-mini/artifacts/DalleBart_model/mega-1-fp16/latest/files into a folder, with the files named how they appear online, and give the path to the enclosing folder')

# see above comment for why i make you download this manually
parser.add_argument('--vqgan', type=str, default='./vqgan_imagenet_f16_16384',
                    help='The directory the vqgan model (decoder) is stored in. Download https://huggingface.co/dalle-mini/vqgan_imagenet_f16_16384/tree/main into a folder, with the files named how they appear online, and give the path to the enclosing folder')

parser.add_argument('-o', '--output-dir', dest="output", type=str, default='./images',
                    help='The directory the vqgan model (decoder) is stored in. Download https://huggingface.co/dalle-mini/vqgan_imagenet_f16_16384/tree/main into a folder, with the files named how they appear online, and give the path to the enclosing folder')

args = parser.parse_args()
print("WARNING: Consider closing other programs when running this script")


images_dir = args.output
from pathlib import Path
Path(images_dir).mkdir(parents=True, exist_ok=True)

# number of predictions per prompt
n_predictions = args.predictions

# Model references
# dalle-mega
DALLE_MODEL = args.model  # can be wandb artifact or 🤗 Hub or local folder or google bucket

def isPath(s):
    return s.startswith("../") or s.startswith("./") or s.startswith("/")
if not isPath(DALLE_MODEL):
    DALLE_MODEL = "./" + DALLE_MODEL
prompts = args.prompt
if len(prompts) == 0:
    prompts = ["sunset over a lake in the mountains", "the Eiffel tower landing on the moon"]
DALLE_COMMIT_ID = None

# if the notebook crashes too often you can use dalle-mini instead by uncommenting below line
#DALLE_MODEL = "./mini"

VQGAN_REPO = args.vqgan
# VQGAN model
if not isPath(VQGAN_REPO):
    VQGAN_REPO = "./" + VQGAN_REPO
VQGAN_COMMIT_ID = None#"e93a26e7707683d349bf5d5c41c5b0ef69b677a9"

print(f"\nPrompts: {prompts}\n")
print(f"N_predictions: {n_predictions}\n")

import jax
import jax.numpy as jnp

# check how many devices are available
jax.local_device_count()
# Load models & tokenizer
from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel
from transformers import CLIPProcessor, FlaxCLIPModel

# Load dalle-mini
model, params = DalleBart.from_pretrained(
    DALLE_MODEL, revision=DALLE_COMMIT_ID, dtype=jnp.float16, _do_init=False
)

# Load VQGAN
vqgan, vqgan_params = VQModel.from_pretrained(
    VQGAN_REPO, revision=VQGAN_COMMIT_ID, _do_init=False
)
from flax.jax_utils import replicate

params = replicate(params)
#vqgan_params = replicate(vqgan_params)
from functools import partial

# model inference
@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
def p_generate(
    tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale
):
    return model.generate(
        **tokenized_prompt,
        prng_key=key,
        params=params,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        condition_scale=condition_scale,
    )


# decode image
@partial(jax.pmap, axis_name="batch")
def p_decode(indices, params):
    return vqgan.decode_code(indices, params=params)
import random

# create a random key
seed = random.randint(0, 2**32 - 1)
key = jax.random.PRNGKey(seed)
from dalle_mini import DalleBartProcessor

processor = DalleBartProcessor.from_pretrained(DALLE_MODEL, revision=DALLE_COMMIT_ID)


# We can customize generation parameters (see https://huggingface.co/blog/how-to-generate)
gen_top_k = None
gen_top_p = None
temperature = None
cond_scale = 10.0
from flax.training.common_utils import shard_prng_key
import numpy as np
from PIL import Image
from tqdm.notebook import trange

# generate images
encoded_images_array = []
nameCounter=0
import uuid
j = 0

def run_one_prompt(prompt):
    global j, key
    tokenized_prompts = processor([prompt])
    tokenized_prompt = replicate(tokenized_prompts)
    for i in trange(max(n_predictions // jax.device_count(), 1)):
        j += 1
        # get a new key
        key, subkey = jax.random.split(key)
        # generate images
        encoded_images_array.append(p_generate(
            tokenized_prompt,
            shard_prng_key(subkey),
            params,
            gen_top_k,
            gen_top_p,
            temperature,
            cond_scale,
        ))
        print("done round {}".format(j))

for prompt in prompts:
    run_one_prompt(prompt)
del params
j = 0
vqgan_params = replicate(vqgan_params)
for prompt in prompts:
    for i in trange(max(n_predictions // jax.device_count(), 1)):
        # remove BOS
        encoded_images = encoded_images_array[j].sequences[..., 1:]
        # decode images
        decoded_images = p_decode(encoded_images, vqgan_params)
        decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
        for decoded_img in decoded_images:
            img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
            nameCounter += 1
            img.save("{}/{}-{}.png".format(images_dir, nameCounter, uuid.uuid4()))
            print("saved {}".format(j))
        j+=1
del vqgan_params
