import torch
from torch import nn

# mlp for mnist
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.fc2_drop = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = torch.relu(self.fc2(x))
        x = self.fc2_drop(x)
        x = self.fc3(x)
        return torch.nn.functional.log_softmax(x, dim=1)