# Use Stable Diffusion Model to augment the dataset with Adverse Weather Conditions and Motion Blur Effects

## Overview
we randomly pick 2,000 images from KITTIv2 train set as the easy input images. As for each input image, we use ControlNet [Midas depth estimation model](https://github.com/isl-org/MiDaS) to generate the corresponding depth image. After that, by using [SDXL](https://github.com/CompVis/stable-diffusion) along with [T2I-adapters](https://github.com/TencentARC/T2I-Adapter) and the above depth image, we can generate challenge RGB images with the same depth structure as the easy input image and with additional adverse environmental conditions and motion blur effect. To encourage SDXL to add required challenge conditions, we employ prompts that include motion blur and adverse environment descriptions such as,
```
"Hazardous smog blanketing city streets with motion blur. One or more cars in motion show a huge motion blur effect to emphasize movement, while the surroundings and other stationary objects like buildings, street signs, and parked cars are sharp and clear."
```
combined with negative prompts for example,
```
"Static cars, sharp car details, clear car outlines, no motion blur, little motion blur, overly clean environment"
```
to avoid generating static or sharp details that are not characteristic of realistic motion blur. Sample generated images are shown in the folder `generated_images`.

## Usage

### Preparation
```
pip install -r requirements.txt
```

To generate images with the above process, you can run the following command in the terminal:
```
python3 main.py --dataset <path_to_dataset> --save-path <folder_path>
```
where `<path_to_dataset>` is the path to the dataset .txt file containing the paths to all input images and `<folder_path>` is the path to the folder where the generated images will be saved.

An example command is shown below:
```
python main.py --dataset example_dataset_rgb_path.txt --save-path ./generated_images
```