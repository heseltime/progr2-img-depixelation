import os
import random
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

import matplotlib.pyplot as plt

from assignments.a2_ex1 import to_grayscale
from assignments.a2_ex2 import prepare_image

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class RandomImagePixelationDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        width_range: Tuple[int, int],
        height_range: Tuple[int, int],
        size_range: Tuple[int, int],
        dtype: Optional[type] = None,
    ) -> None:
        self.image_files = sorted(
            [
                os.path.abspath(os.path.join(root, name))
                for root, dirs, files in os.walk(image_dir)
                for name in files
                if name.lower().endswith(".jpg")
            ]
        )
        self.width_range = width_range
        self.height_range = height_range
        self.size_range = size_range
        if width_range[0] < 2 or height_range[0] < 2:
            raise ValueError("Minimum width and height must be at least 2.")
        if width_range[0] > width_range[1] or height_range[0] > height_range[1]:
            raise ValueError("Minimum value must be less than or equal to maximum value.")
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        random.seed(index)

        # see A7: 64x64 px specification
        transform = transforms.Compose(
            [
                transforms.Resize(size=(64,64),interpolation=TF.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size=(64,64)),
            ]
        )

        file_path = self.image_files[index]

        with Image.open(file_path) as img:
            img = transform(img)
            img_arr = np.asarray(img, dtype=self.dtype)
        
        img_width = img_arr.shape[1]
        img_height = img_arr.shape[0]

        gray_image_array = to_grayscale(img_arr)
        #print("<-->")
        #print(gray_image_array.shape)

        width = random.randint(self.width_range[0], self.width_range[1])
        height = random.randint(self.height_range[0], self.height_range[1])
        size = random.randint(self.size_range[0], self.size_range[1])

        x = random.randint(0, img_width - width) 
        y = random.randint(0, img_height - height) 

        pixelated_image, known_array, target_array \
            = prepare_image(gray_image_array, x, y, width, height, size)
        
        #print("<-->")
        #print(pixelated_image.shape)
        #print(known_array.shape)
        #print(target_array.shape)

        return pixelated_image, known_array, target_array, file_path
    

if __name__ == '__main__':
    #print(os.getcwd() + "\\assignments\\test_imgs")

    ds = RandomImagePixelationDataset(
        os.getcwd() + "\\assignments\\test_imgs", # for example
        width_range=(2, 64),
        height_range=(2, 64),
        size_range=(4, 16)
    )

    for pixelated_image, known_array, target_array, image_file in ds:
        fig, axes = plt.subplots(ncols=3)
        axes[0].imshow(pixelated_image[0], cmap="gray", vmin=0, vmax=255)
        axes[0].set_title("pixelated_image")
        axes[1].imshow(known_array[0], cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("known_array")
        axes[2].imshow(target_array[0], cmap="gray", vmin=0, vmax=255)
        axes[2].set_title("target_array")
        fig.suptitle(image_file)
        fig.tight_layout()
        plt.show()
