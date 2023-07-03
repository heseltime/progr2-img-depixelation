# -*- coding: utf-8 -*-
"""project/predict.py

Author -- Jack Heseltine
Contact -- jack.heseltine@gmail.com
Date -- June 2023

###############################################################################

Prediction for test input: to be serialized and submitted to the challenge server. Uses best_model.pt from training, has to be in folder "results".
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

from submission_serialization import serialize, deserialize
import pickle, dill

def predict():
    #device = torch.device('cpu' if torch.cuda.is_available() else 'gpu')
    device = torch.device('cpu')

    #model = SimpleNetwork(4096,128,4096)
    #model.load_state_dict(torch.load('results\\best_model.pt'))

    model = torch.load('results\\best_model.pt')
    # better to save and load only the state_dict, requiries modification in main.py

    model = model.to(device) # Set model to gpu
    model.eval()

    #inputs = torch.randn(1, 5, 4096)

    # Desirialize ... pkl-import here

    path = 'test_set.pkl'
    data = None
    #inputs = deserialize(path)
    #inputs = np.load(path, allow_pickle=True)
    with open(path, 'rb') as f:
        data = pickle.load(f)

    pixelated_images = data['pixelated_images']
    known_arrays = data['known_arrays']

    # print(len(known_arrays)) # 6635
    # print(len(pixelated_images)) # 6635
    # print(known_arrays[0].shape) # (1,64,64)
    # print(pixelated_images[0].shape) # (1,64,64)

    pixelated_images = [im.flatten() for im in pixelated_images]
    known_arrays = [im.flatten() for im in known_arrays]

    inputs = torch.tensor(pixelated_images, dtype=torch.float32)

    #print(inputs.shape) # torch.Size([6635, 4096])

    inputs = inputs.to(device) # You can move your input to gpu, torch defaults to cpu

    # Run forward pass
    with torch.no_grad():
        pred = model(inputs)

    # Process predictions
    pred = pred.detach().cpu().numpy() # remove from computational graph to cpu and as numpy

    print(pred.shape) # (6635, 4096)
    #print(pred)

    # submit just the predictions for the unknowns, convert to np.uint8 as we go
    submission = [np.array(im[kn == 0], dtype=np.uint8) for im, kn in zip(pred, known_arrays)]

    #print(len(submission)) # 6635
    #print(len(submission[0])) # 532 (example 1)
    #print(len(submission[1])) # 88 (example 2)

    # Serialize here
    serialize(submission, 'submission.pkl')

if __name__ == "__main__":
    predict()
