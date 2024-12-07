from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL
from diffusers.utils import load_image
from controlnet_aux.midas import MidasDetector
import torch
import re
import os
import argparse

## Retrieve the current directory path
cwd = os.getcwd()

parser = argparse.ArgumentParser(description='Generate images using Stable Diffusion XL Adapter')
parser.add_argument('--dataset', type=str, required=True,
                      help='Path to the dataset file containing image paths')
parser.add_argument('--save-path', type=str, required=True,
                      help='Path to save the generated images')

# Function to retrieve image number from path
def get_image_number(image_path):
    match = re.search(r'/rgb/(\d+)\.jpg', image_path)
    if match:
        return int(match.group(1))
    return None

def main():
    ## Retrieve the arguments
    args = parser.parse_args()
    # Check the availability of gpu and set the device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # load T2I-adapter for setting up the stable diffusion model later
    adapter = T2IAdapter.from_pretrained(
      "TencentARC/t2i-adapter-depth-midas-sdxl-1.0", torch_dtype=torch.float16, varient="fp16"
    ).to(DEVICE)

    # load euler_a scheduler and then the stable diffusion model
    model_id = 'stabilityai/stable-diffusion-xl-base-1.0'
    euler_a = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    vae=AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
        model_id, vae=vae, adapter=adapter, scheduler=euler_a, torch_dtype=torch.float16, variant="fp16", 
    ).to(DEVICE)

    # load the midas depth estimator
    midas_depth = MidasDetector.from_pretrained(
      "valhalla/t2iadapter-aux-models", filename="dpt_large_384.pt", model_type="dpt_large"
    ).to(DEVICE)

    # Iteratively read image paths from the dataset file and then draw 10 images
    # based on 10 different prompt according to each image
    try:
        with open(args.dataset, 'r') as file:
            image_paths = [line.strip() for line in file.readlines()]
    except FileNotFoundError:
        print(f"Error: Could not find file at {args.dataset}")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # All prompts
    prompts = ["Hazardous smog blanketing city streets with motion blur. One or more cars in motion show a huge motion blur effect to emphasize movement, while the surroundings and other stationary objects like buildings, street signs, and parked cars are sharp and clear.",
                "Low visibility in urban fog with motion blur. One or more cars in motion show a huge motion blur effect to emphasize movement, while the surroundings and other stationary objects like buildings, street signs, and parked cars are sharp and clear.",
                "Stranded in a car during intense hailstorm with motion blur. One or more cars in motion show a huge motion blur effect to emphasize movement, while the surroundings and other stationary objects like buildings, street signs, and parked cars are sharp and clear.",
                "Urban roads with extreme sun glare with motion blur. One or more cars in motion show a motion blur effect to emphasize movement, while the surroundings and other stationary objects like buildings, street signs, and parked cars are sharp and clear.",
                "Tropical cyclone's downpour with motion blur. One or more cars in motion show a huge motion blur effect to emphasize movement, while the surroundings and other stationary objects like buildings, street signs, and parked cars are sharp and clear.",
                "Heavy snowfall on a remote forest road with motion blur. One or more cars in motion show a huge motion blur effect to emphasize movement, while the surroundings and other stationary objects like buildings, street signs, and parked cars are sharp and clear.",
                "Frozen night road amid freezing rain with motion blur. One or more cars in motion show a huge motion blur effect to emphasize movement, while the surroundings and other stationary objects like buildings, street signs, and parked cars are sharp and clear.",
                "Lost in a desert amidst a violent sandstorm with motion blur. One or more cars in motion show a huge motion blur effect to emphasize movement, while the surroundings and other stationary objects like buildings, street signs, and parked cars are sharp and clear.",
                "Stranded in a flash flood during daylight with motion blur. One or more cars in motion show a huge motion blur effect to emphasize movement, while the surroundings and other stationary objects like buildings, street signs, and parked cars are sharp and clear.",
                "City with intense nighttime lights with motion blur. One or more cars in motion show a huge motion blur effect to emphasize movement, while the surroundings and other stationary objects like buildings, street signs, and parked cars are sharp and clear."]
    negative_prompt = "Static cars, sharp car details, clear car outlines, no motion blur, little motion blur, overly clean environment"

    # Iteratively read image paths from the dataset file and then draw 10 images
    # based on 10 different prompt according to each image
    for i, image_path in enumerate(image_paths):
      ## Load the image
      image = load_image(image_path)

      ## Record the original shape of the image
      original_shape = image.size

      ## Estimate the depth of the image
      image = midas_depth(image)

      ## Retrieve the image number for naming the generated images later
      image_number = get_image_number(image_path)

      # Using the stable diffusion model to generate images based on prompts
      for j, prompt in enumerate(prompts):
          gen_images = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            num_inference_steps=30,
            adapter_conditioning_scale=1,
            guidance_scale=7.5,
          ).images[0]

          ## Resize the generated image to the original shape
          gen_images = gen_images.resize(original_shape)

          # Save each generated image
          if image_number != None:
            gen_images.save(f"{args.save_path}/{image_number}-{j+1}.jpg")

if __name__ == "__main__":
    main()
