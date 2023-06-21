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
        self.output_layer = nn.Linear(hidden_channels * 64 * 64, num_classes)
        
        # Register all the layers
        for i, layer in enumerate(self.conv_layers):
            self.add_module(f"conv_{i}", layer)
            
        self.add_module("output", self.output_layer)
        
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

if __name__ == "__main__":
    torch.random.manual_seed(0)
    network = SimpleCNN(3, 32, 3, True, 10, activation_function=nn.ELU())
    input = torch.randn(1, 3, 64, 64)
    output = network(input)
    print(output)
