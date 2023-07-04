# -*- coding: utf-8 -*-
"""project/architectures.py

Author -- Jack Heseltine
Contact -- jack.heseltine@gmail.com
Date -- June 2023

###############################################################################

Architectures file of example project.
"""

import torch
import torch.nn as nn

class SimpleCNN(torch.nn.Module):
    def __init__(self, n_in_channels: int = 1, n_hidden_layers: int = 3, n_kernels: int = 32, kernel_size: int = 3):
        """Simple CNN with `n_hidden_layers`, `n_kernels`, and `kernel_size` as hyperparameters"""
        super().__init__()
        
        cnn = []
        for i in range(n_hidden_layers):
            cnn.append(torch.nn.Conv2d(
                in_channels=n_in_channels,
                out_channels=n_kernels,
                kernel_size=kernel_size,
                padding=int(kernel_size / 2)
            ))
            cnn.append(torch.nn.ReLU())
            n_in_channels = n_kernels
        self.hidden_layers = torch.nn.Sequential(*cnn)
        
        self.output_layer = torch.nn.Conv2d(
            in_channels=n_in_channels,
            out_channels=1,
            kernel_size=kernel_size,
            padding=int(kernel_size / 2)
        )
    
    def forward(self, x):
        """Apply CNN to input `x` of shape (N, n_channels, X, Y), where N=n_samples and X, Y are spatial dimensions"""
        cnn_out = self.hidden_layers(x)  # apply hidden layers (N, n_in_channels, X, Y) -> (N, n_kernels, X, Y)
        pred = self.output_layer(cnn_out)  # apply output layer (N, n_kernels, X, Y) -> (N, 1, X, Y)
        return pred


    
class SimpleNetwork(nn.Module):
    def __init__(
        self,
        input_neurons: int,
        hidden_neurons: int,
        output_neurons: int,
        activation_function: nn.Module = nn.ReLU()
    ):
        super(SimpleNetwork, self).__init__()
        
        self.input_layer = nn.Linear(input_neurons, hidden_neurons)
        self.hidden_layer1 = nn.Linear(hidden_neurons, hidden_neurons)
        self.hidden_layer2 = nn.Linear(hidden_neurons, hidden_neurons)
        self.output_layer = nn.Linear(hidden_neurons, output_neurons)
        self.activation_function = activation_function

        self.add_module("input_layer", self.input_layer)
        self.add_module("hidden_layer1", self.hidden_layer1)
        self.add_module("hidden_layer2", self.hidden_layer2)
        self.add_module("output_layer", self.output_layer)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation_function(self.input_layer(x))
        x = self.activation_function(self.hidden_layer1(x))
        x = self.activation_function(self.hidden_layer2(x))
        x = self.output_layer(x)
        return x