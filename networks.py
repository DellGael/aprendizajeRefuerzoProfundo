import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # Con padding=1 mantenemos (W - 3 + 2) / stride + 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        # Ahora stride=1 para conservar tamaño
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)

        self.downsample = None
        # Si cambia resolución o número de canales, ajustamos identidad
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, padding=0),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class ResNetExtractor(BaseFeaturesExtractor):
    def __init__(self, obs_space, features_dim=512):
        super().__init__(obs_space, features_dim)
        n_ch = obs_space.shape[0]
        self.body = nn.Sequential(
            ResidualBlock(n_ch, 32, stride=4),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64, stride=1),
            nn.AvgPool2d(kernel_size=2,stride=2),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.as_tensor(obs_space.sample()[None]).float()
            n_flatten = self.body(sample).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )
    def forward(self, obs):
        return self.linear(self.body(obs))
