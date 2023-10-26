import torch
from torch import nn

class CNN(nn.Module):
    
    def __init__(self, n_channels, height, width):
        super(CNN, self).__init__()
        print(n_channels, height, width)

        self.conv_layers = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        for m in self.conv_layers:
          if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)

        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 6 * 6, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

        for m in self.fc_layers:
          if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(-1, 64 * 6 * 6)
        x = self.fc_layers(x)
        return x