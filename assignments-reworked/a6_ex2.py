import os
from PIL import Image
from torch.utils.data import Dataset
from typing import Union, Tuple, Sequence
import torch
import torchvision.transforms as transforms

from a6_ex1 import random_augmented_image

class ImageDataset(Dataset):
    def __init__(self, image_dir: str):
        self.image_paths = self._find_image_paths(image_dir)
        
    def _find_image_paths(self, image_dir: str):
        image_paths = []
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.endswith(".jpg"):
                    image_paths.append(os.path.join(root, file))
        image_paths.sort()
        return image_paths
    
    def __getitem__(self, index: int) -> Tuple[Image.Image, int]:
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        return image, index
    
    def __len__(self) -> int:
        return len(self.image_paths)


class TransformedImageDataset(Dataset):
    def __init__(self, dataset: ImageDataset, image_size: Union[int, Sequence[int]]) -> None:
        self.dataset = dataset
        self.image_size = image_size
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        original_image, image_index = self.dataset[index]
        transformed_image = random_augmented_image(original_image, self.image_size, index)
        return transformed_image, image_index
    
    def __len__(self) -> int:
        return len(self.dataset)


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    imgs = ImageDataset(image_dir="images")
    transformed_imgs = TransformedImageDataset(imgs, image_size=300)
    for (original_img, index), (transformed_img, _) in zip(imgs, transformed_imgs):
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(original_img)
        axes[0].set_title("Original image")
        axes[1].imshow(transforms.functional.to_pil_image(transformed_img))
        axes[1].set_title("Transformed image")
        fig.suptitle(f"Image {index}")
        fig.tight_layout()
        plt.show()