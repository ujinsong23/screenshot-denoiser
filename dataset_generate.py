import os
from os import path as osp
import numpy as np
from PIL import Image
from glob import glob
from skimage.transform import resize

DATA_PATH = "../dataset"
# patch_size = 512
num_patches = 2

image_paths = glob(f"{DATA_PATH}/raw/*.png")
image_paths += glob(f"{DATA_PATH}/raw/*.jpg")
image_paths += glob(f"{DATA_PATH}/raw/*.jpeg")

image_id = 0
image_paths.sort()
for image_path in image_paths:
    raw_image = Image.open(image_path)
    h, w = raw_image.size
    for i in range(num_patches):
        for j in range(num_patches):
            cropped_image = raw_image.crop((i * w // num_patches, j * h // num_patches, (i + 1) * w // num_patches, (j + 1) * h // num_patches))
            # cropped_image = resize(np.array(cropped_image), (patch_size, patch_size), anti_aliasing=True)
            # cropped_image = Image.fromarray((cropped_image * 255).astype(np.uint8))
            cropped_image.save(osp.join(f"{DATA_PATH}/original", f"{image_id:03d}.png"))
            image_id += 1
