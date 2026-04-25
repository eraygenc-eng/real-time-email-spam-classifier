import torch
import torch.nn as nn

# istenilen mimari: 7000 input -> 256 neuron -> 128 neuron -> 1 output
# istenilen loss: BCEWithLogitsLoss()

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim=7000):
        super().__init__()

        self.layer1 = nn.Linear(input_dim, 256)
        self.dropout1 = nn.Dropout(p=0.3)

        self.layer2 = nn.Linear(256,128)
        self.dropout2 = nn.Dropout(p=0.3)

        self.layer3 = nn.Linear(128,1)

    def forward(self, x):
        x = self.layer1(x) # z1
        x = torch.relu(x) # a1
        x = self.dropout1(x)

        x = self.layer2(x) # z2
        x = torch.relu(x) # a2
        x = self.dropout2(x)

        x = self.layer3(x) # z3 -> output
        return x