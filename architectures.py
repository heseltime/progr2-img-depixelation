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

class SimpleCNN(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        num_hidden_layers: int,
        use_batchnormalization: bool,
        num_classes: int,
        kernel_size: int = 3,
        activation_function: nn.Module = nn.ReLU()
    ):
        super(SimpleCNN, self).__init__()
        
        self.conv_layers = nn.ModuleList()
        self.use_batchnormalization = use_batchnormalization
        
        # Input convolutional layer
        self.conv_layers.append(
            nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=1)
        )

        if use_batchnormalization:
            self.conv_layers.append(nn.BatchNorm2d(hidden_channels))
        self.conv_layers.append(activation_function)
        
        # Hidden convolutional layers
        for i in range(num_hidden_layers):
            self.conv_layers.append(
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=1)
            )
            
            if use_batchnormalization:
                self.conv_layers.append(nn.BatchNorm2d(hidden_channels))
            self.conv_layers.append(activation_function)
        
        # Fully connected output layer
        #self.output_layer = nn.Linear(hidden_channels * 64 * 64, num_classes)

        # Create a dummy input with the expected input size
        # for instance, if your actual input images are 1x128x128, replace the 1 below with your actual batch size
        dummy_input = torch.zeros(1, input_channels, 64, 64)
        
        # Run a forward pass through the convolutional layers
        dummy_output = self.forward_conv_layers(dummy_input)

        # Calculate the number of features from the output size
        num_features = dummy_output.view(1, -1).size(1)
        
        # Initialize the fully connected layer with the actual number of features
        self.output_layer = nn.Linear(num_features, num_classes)
        
        # Register all the layers
        for i, layer in enumerate(self.conv_layers):
            self.add_module(f"conv_{i}", layer)
            
        self.add_module("output", self.output_layer)

    def forward_conv_layers(self, x):
        # Forward pass through convolutional layers
        for layer in self.conv_layers:
            x = layer(x)
        return x
        
    def forward(self, input_images: torch.Tensor):
        x = input_images
        
        # Forward pass through convolutional layers
        for layer in self.conv_layers:
            x = layer(x)
        
        # Reshape: should be hidden channels * 64 * 64 size
        x = x.view(x.size(0), -1)
        #print(x.shape) # for checking
        
        # Forward pass through the fully connected output layer
        output = self.output_layer(x)
        
        return output.view(output.size(0), -1)
    
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