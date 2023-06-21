import torch
import torch.nn as nn

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

if __name__ == "__main__":
    torch.random.manual_seed(0)
    simple_network = SimpleNetwork(10, 20, 5)
    input = torch.randn(1, 10) 
    # input = torch.randn(2, 10) # also works
    output = simple_network(input)
    print(output)