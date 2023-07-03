# -*- coding: utf-8 -*-
"""project/architectures.py

Author -- Jack Heseltine
Contact -- jack.heseltine@gmail.com
Date -- June 2023

###############################################################################

Main file of the 2023 project: writes evaluation results to results.txt
"""

import os

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from architectures import SimpleCNN, SimpleNetwork
from datasets import CIFAR10, RotatedImages
from utils import plot

from assignments.a3_ex1 import RandomImagePixelationDataset

def main():
    #device = torch.device('cpu' if torch.cuda.is_available() else 'gpu')
    device = torch.device('cpu')

    #model = SimpleNetwork(4096,128,4096)
    #model.load_state_dict(torch.load('results\\best_model.pt'))

    model = torch.load('results\\best_model.pt')
    # better to save and load only the state_dict, requiries modification in main.py

    model = model.to(device) # Set model to gpu
    model.eval()

    inputs = torch.randn(1, 5, 4096)
    # Desirialze here

    inputs = inputs.to(device) # You can move your input to gpu, torch defaults to cpu

    # Run forward pass
    with torch.no_grad():
        pred = model(inputs)

    # Process predictions
    pred = pred.detach().cpu().numpy() # remove from computational graph to cpu and as numpy

    print(pred)
    # Serialize here


if __name__ == "__main__":
    main()
