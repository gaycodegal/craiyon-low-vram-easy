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

parser.add_argument('--parallel-count', dest="parallel_count", type=str,
                    help='How many prompts to process in parallel',
                    default=None)

parser.add_argument('--parallel-vqgan', dest="parallel_vqgan", type=bool,
                    action=argparse.BooleanOptionalAction,
                    help='Should we attempt to run vqgan and dalle at the same time. This will cause less powerful machines to crash',
                    default=False)

parser.add_argument('--include-model-in-image-metadata', dest="write_model_name", type=bool,
                    action=argparse.BooleanOptionalAction,
                    help='Opts in to saving the model name in the image metadata, which is whatever you passed to --model, and may include sensitive path information',
                    default=False)

parser.add_argument('--top-k', type=int, dest='k', default=None,
                    help='Turns on Top-K sampling. Probability is only distributed amongst the K most likely next symbols instead of all possibilities. https://huggingface.co/blog/how-to-generate')

parser.add_argument('--top-p', type=float, dest='p', default=None,
                    help='Turns on Top-p sampling. This is like a dynamic form of Top-k. P should be a probability between 0 and 1 if turned on. For instance -p=0.92 will redistribute probabilities between the largest probability options that add up to probability P. https://huggingface.co/blog/how-to-generate')

parser.add_argument('--temperature', type=float, default=None,
                    help='Modifies the probability distribution. Couldn\'t figure out how to explain. Hugging face says try 0.7? https://huggingface.co/blog/how-to-generate')

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


parallel_count = args.parallel_count
if parallel_count == None:
    parallel_count = 1
elif parallel_count.lower() == "max":
    parallel_count = len(prompts)
else:
    parallel_count = min(int(parallel_count), len(prompts))

    
print(f"\nPrompts: {prompts}\n")
print(f"N_predictions: {n_predictions}\n")
print(f"total images to process: {n_predictions * len(prompts)}\n")

import jax
import jax.numpy as jnp

# check how many devices are available
print(f"devices available: {jax.local_device_count()}\n")

# Load models & tokenizer
from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel
from transformers import CLIPProcessor, FlaxCLIPModel

# Load dalle-mini
from flax.jax_utils import replicate

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

from dalle_mini import DalleBartProcessor

from flax.training.common_utils import shard_prng_key
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from tqdm.notebook import trange

def makePngInfo(prompt, seed):
    """Add some alt text and common information to the image.

    Lets be honest, most websites will ignore this data,
    and strip it from the image. BUT it may one day be useful to you
    Also, you're probably only running this script if you are
    using the mega-1-fp16 model, so I have that defaulted for now
    """
    metadata = PngInfo()
    # generally true that it will be the mega model.
    # don't expose the file path name to the model in case its sensitive
    # Textual information chunks - the metadata in PNG
    # would have used exif but the python lib for that with PNG support is unmaintained.
    model_name = "dalle-mini:mega-1-fp16"
    if args.write_model_name:
        model_name = "dalle-mini:{}".format(DALLE_MODEL)
    # "Description" is a common key, should be known to other programs
    # https://dev.exiv2.org/projects/exiv2/wiki/The_Metadata_in_PNG_files
    metadata.add_text("Description", "An image generated using the phrase \"{}\" generated by the {} model".format(prompt, model_name))
    # short one line caption
    metadata.add_text("Title", prompt)
    # common field for warning of content nature. People don't like
    # blurry faces generated by AI sometimes
    metadata.add_text("Warning", "AI generated content")
    # device used to create the image
    
    metadata.add_text("Source", "model={} seed={} top-k={} top-p={} temperature={} cond_scale={}".format(
        model_name,
        seed,
        gen_top_k,
        gen_top_p,
        temperature,
        cond_scale))
    return metadata
    

def run_one_prompt_set(sliced_prompts):
    tokenized_prompts = processor(sliced_prompts)
    tokenized_prompt = replicate(tokenized_prompts)
    global j, key
    for i in trange(max(n_predictions // jax.device_count(), 1)):
        j += 1
        # get a new key
        key, subkey = jax.random.split(key)
        # generate images
        encoded_images = p_generate(
            tokenized_prompt,
            shard_prng_key(subkey),
            params,
            gen_top_k,
            gen_top_p,
            temperature,
            cond_scale,
        )
        print("done round {}".format(j))
        if args.parallel_vqgan:
            j -= 1
            save_one_prompt_set(sliced_prompts, encoded_images, subkey)
        else:
            key_array.append(subkey)
            encoded_images_array.append(encoded_images)



def save_one_prompt_set(prompt_set, encoded_images = None, key = None):
    global j, nameCounter
    # remove BOS
    if encoded_images is None:
        encoded_images = encoded_images_array[j].sequences[..., 1:]
    else:
        encoded_images = encoded_images.sequences[..., 1:]
    if key is None:
        key = key_array[j]
    # decode images
    decoded_images = p_decode(encoded_images, vqgan_params)
    decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
    for (di, decoded_img) in enumerate(decoded_images):
        img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
        nameCounter += 1
        metadata = makePngInfo(prompt_set[di], key)
        img.save("{}/{}-{}.png".format(images_dir, nameCounter, uuid.uuid4()), pnginfo=metadata)
        print("saved {}".format(nameCounter))
    j+=1

model, params = None, None
vqgan, vqgan_params = None, None
processor = None
# We can customize generation parameters (see https://huggingface.co/blog/how-to-generate)
gen_top_k = args.k
gen_top_p = args.p
temperature = args.temperature
cond_scale = 10.0

# generate images
encoded_images_array = []
key_array = []
nameCounter=0
import uuid
j = 0

print("generating key")

# create a random key
seed = random.randint(0, 2**32 - 1)
key = jax.random.PRNGKey(seed)
print("key created")

def init_dalle_bart():
    global params
    global model
    global processor
    global j
    j = 0
    processor = DalleBartProcessor.from_pretrained(DALLE_MODEL, revision=DALLE_COMMIT_ID)
            
    model, params = DalleBart.from_pretrained(
        DALLE_MODEL, revision=DALLE_COMMIT_ID, dtype=jnp.float16, _do_init=False
    )
    
    params = replicate(params)

def init_vqgan():
    global j
    global vqgan
    global vqgan_params
    j = 0
    # Load VQGAN
    vqgan, vqgan_params = VQModel.from_pretrained(
        VQGAN_REPO, revision=VQGAN_COMMIT_ID, _do_init=False
    )
    # image generation does not use the vqgan encoder
    del vqgan_params["encoder"]
    vqgan_params = replicate(vqgan_params)

def delete_params():
    global params
    del params

def delete_vqgan_params():
    global vqgan_params
    del vqgan_params

import math

def slice_prompts(i):
    p_start = i * parallel_count
    p_end = min(len(prompts), (i + 1) * parallel_count)
    sliced_prompts = prompts[p_start : p_end]
    return sliced_prompts

def main():
    global j
    # generate images
    init_dalle_bart()
    if args.parallel_vqgan:
        init_vqgan()

    runs = math.ceil(len(prompts) / parallel_count)
    for i in range(runs):
        sliced_prompts = slice_prompts(i)
        print("now computing", sliced_prompts)
        run_one_prompt_set(sliced_prompts)

    delete_params()
    if not args.parallel_vqgan:
        # save images
        init_vqgan()
        for i in range(runs):
            sliced_prompts = slice_prompts(i)
            for k in trange(max(n_predictions // jax.device_count(), 1)):
                save_one_prompt_set(sliced_prompts)
    delete_vqgan_params()

if __name__ == "__main__":
    main()
