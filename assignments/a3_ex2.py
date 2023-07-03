import torch
from typing import List, Tuple

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from a3_ex1 import RandomImagePixelationDataset
import os

def stack_with_padding(batch_as_list: List[Tuple]) -> Tuple:
    # extract pixelated images, known arrays, target arrays, and image files from the batch
    pixelated_images, known_arrays, target_arrays, image_files = zip(*batch_as_list)

    # find the maximum height and maximum width in the batch
    max_height = max(img.shape[0] for img in pixelated_images)
    max_width = max(img.shape[1] for img in pixelated_images)

    # stack the pixelated images by padding them with zeros
    stacked_pixelated_images = []
    for img in pixelated_images:
        pad_height = max_height - img.shape[0]
        pad_width = max_width - img.shape[1]
        padded_img = torch.nn.functional.pad(torch.Tensor(img).unsqueeze(0), (0, pad_width, 0, pad_height), mode='constant', value=0)
        stacked_pixelated_images.append(padded_img)
        print("appended")
    stacked_pixelated_images = torch.cat(stacked_pixelated_images, dim=0)

    # stack the known arrays by padding them with ones
    stacked_known_arrays = []
    for array in known_arrays:
        pad_height = max_height - array.shape[0]
        pad_width = max_width - array.shape[1]
        padded_array = torch.nn.functional.pad(torch.Tensor(array).unsqueeze(0), (0, pad_width, 0, pad_height), mode='constant', value=1)
        stacked_known_arrays.append(padded_array)
    stacked_known_arrays = torch.cat(stacked_known_arrays, dim=0)

    # convert target arrays to PyTorch tensor and store them in a list
    target_arrays = [torch.Tensor(array) for array in target_arrays]

    # return the stacked pixelated images, stacked known arrays, target arrays, and image files
    return stacked_pixelated_images, stacked_known_arrays, target_arrays, image_files

if __name__ == '__main__':
    ds = RandomImagePixelationDataset(
        os.getcwd() + "\\a3\imgs", # for example
        width_range=(50, 300),
        height_range=(50, 300),
        size_range=(10, 50)
    )
    dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=stack_with_padding)
    for (stacked_pixelated_images, stacked_known_arrays, target_arrays, image_files) in dl:
        fig, axes = plt.subplots(nrows=dl.batch_size, ncols=3)
        print(dl.batch_size)
        for i in range(dl.batch_size):
            axes[i, 0].imshow(stacked_pixelated_images[i][0], cmap="gray", vmin=0, vmax=255)
            axes[i, 1].imshow(stacked_known_arrays[i][0], cmap="gray", vmin=0, vmax=1)
            axes[i, 2].imshow(target_arrays[i][0], cmap="gray", vmin=0, vmax=255)
            fig.tight_layout()
            plt.show()
