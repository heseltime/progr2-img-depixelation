# -*- coding: utf-8 -*-
"""project/architectures.py

Author -- Jack Heseltine
Contact -- jack.heseltine@gmail.com
Date -- June 2023

###############################################################################

Utils file of example project.
"""

import os

from matplotlib import pyplot as plt


def plot(inputs, targets, predictions, path, update):
    """Plotting the inputs, targets and predictions to file `path`"""
    os.makedirs(path, exist_ok=True)
    fig, axes = plt.subplots(ncols=3, figsize=(15, 5))
    
    for i in range(len(inputs)):
        for ax, data, title in zip(axes, [inputs, targets, predictions], ["Input", "Target", "Prediction"]):
            ax.clear()
            ax.set_title(title)
            ax.imshow(data[i, 0], cmap="gray", interpolation="none")
            ax.set_axis_off()
        fig.savefig(os.path.join(path, f"{update:07d}_{i:02d}.png"), dpi=100)
    
    plt.close(fig)
