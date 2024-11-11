from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL
from diffusers.utils import load_image, make_image_grid
from controlnet_aux.midas import MidasDetector
import torch
import numpy as np
from skimage.filters import gaussian
from pypher.pypher import psf2otf
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import skimage.io as io
import torchvision
# Retrieve the current directory path
import os
cwd = os.getcwd()

def fspecial_gaussian_2d(size, sigma):
    kernel = np.zeros(tuple(size))
    kernel[size[0]//2, size[1]//2] = 1
    kernel = gaussian(kernel, sigma)
    return kernel/np.sum(kernel)

# load adapter
adapter = T2IAdapter.from_pretrained(
  "TencentARC/t2i-adapter-depth-midas-sdxl-1.0", torch_dtype=torch.float16, varient="fp16"
).to("cuda")

# load euler_a scheduler
model_id = 'stabilityai/stable-diffusion-xl-base-1.0'
euler_a = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
vae=AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
    model_id, vae=vae, adapter=adapter, scheduler=euler_a, torch_dtype=torch.float16, variant="fp16", 
).to("cuda")

midas_depth = MidasDetector.from_pretrained(
  "valhalla/t2iadapter-aux-models", filename="dpt_large_384.pt", model_type="dpt_large"
).to("cuda")

url = os.path.join(cwd, "images/demo01.jpg")
image = load_image(url)
image = midas_depth(
  image, detect_resolution=512, image_resolution=1024
)

prompt = "Low Visibility in Urban Fog, 4k photo, highly detailed"
negative_prompt = ""

gen_images = pipe(
  prompt=prompt,
  negative_prompt=negative_prompt,
  image=image,
  num_inference_steps=30,
  adapter_conditioning_scale=1,
  guidance_scale=7.5,  
).images[0]

gen_images.save('generated_image.png')

# Adding gaussian noise into the generated image
sigma = 10

gen_images_np = np.array(gen_images).astype(float)/255

filtSize = np.ceil(9 * sigma).astype(int)
lp = fspecial_gaussian_2d((filtSize, filtSize), sigma)
cFT = psf2otf(lp, gen_images_np.shape[:2])

# Blur image with kernel
blur = np.zeros_like(gen_images_np)

# Apply kernel filter to img to retrieve blur
for channel in [0, 1, 2]:
    blur[..., channel] = np.real(ifft2(fft2(gen_images_np[:, :, channel]) * cFT))

io.imsave("blur_image.png", np.clip(blur * 255, a_min=0, \
                                      a_max=255.).astype(np.uint8))