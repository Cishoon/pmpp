import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=10):
        super(MLP, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_size))
        # self.layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            # self.layers.append(nn.ReLU())
        
        # Output layer
        self.layers.append(nn.Linear(hidden_size, output_size))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

input_size = 78400 
hidden_size = 1280
output_size = 100
num_layers = 100

model = MLP(input_size, hidden_size, output_size, num_layers)