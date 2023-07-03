from typing import Union, Sequence
from PIL import Image
import torch
import torchvision.transforms as transforms
import random

def random_augmented_image(
    image: Image,
    image_size: Union[int, Sequence[int]],
    seed: int
) -> torch.Tensor:
    
    random.seed(seed)
    
    # list of possible transformations, needed for random selection
    transform_candidates = [
        transforms.RandomRotation,
        transforms.RandomVerticalFlip,
        transforms.RandomHorizontalFlip,
        transforms.ColorJitter
    ]
    
    # specified transformation step 1
    transform_chain = [transforms.Resize(image_size)]
    
    # randomly select 2 transformations and apply = specified step 2
    selected_transforms = random.sample(transform_candidates, k=2)
    for transform_class in selected_transforms:
        transform_chain.append(transform_class())
    
    # remaining transformations
    transform_chain.extend([
        transforms.ToTensor(), # specified step 3
        torch.nn.Dropout() # specified step 4, .5 is default p, see
                            # https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
    ])
    
    # apply the transform chain on the input image
    transformed_image = image
    for transform in transform_chain:
        transformed_image = transform(transformed_image)
    
    # convert the transformed image to a tensor
    transformed_image_tensor = torch.tensor(transformed_image)
    
    return transformed_image_tensor

if __name__ == "__main__":
    from matplotlib import pyplot as plt

    with Image.open("test_image.jpg") as image:
        transformed_image = random_augmented_image(image, image_size=300, seed=3)
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(image)
        axes[0].set_title("Original image")
        axes[1].imshow(transforms.functional.to_pil_image(transformed_image))
        axes[1].set_title("Transformed image")
        fig.tight_layout()
        plt.show() 