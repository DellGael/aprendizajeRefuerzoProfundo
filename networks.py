import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class ResBlock(nn.Module):
    """Simple residual block: Conv → ReLU → Conv → sum shortcut."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + x, inplace=True)

class ImpalaResNetExtractor(BaseFeaturesExtractor):
    """
    Extractor inspirado en Impala-ResNet:
     - 3 bloques (ConvSequence) con Conv → ReLU → MaxPool → 2× ResBlock
     - Canales [32, 64, 64]
     - Global Average Pooling
     - Fully‑connected a features_dim
     - Inicialización He (Kaiming)
    """
    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_ch = observation_space.shape[0]  # 4

        def make_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                ResBlock(out_c),
                ResBlock(out_c),
            )

        # Tres secuencias
        self.layer1 = make_block(n_ch, 32)
        self.layer2 = make_block(32, 64)
        self.layer3 = make_block(64, 64)

        # Pool y cabeza FC
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(64, features_dim, bias=False),
            nn.BatchNorm1d(features_dim),
            nn.ReLU(inplace=True),
        )

        # Inicialización He
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Normalizar si está en [0,255]
        x = obs.float()
        if x.max() > 1.0:
            x = x / 255.0

        x = self.layer1(x)   # → [B,32,H/2,W/2]
        x = self.layer2(x)   # → [B,64,H/4,W/4]
        x = self.layer3(x)   # → [B,64,H/8,W/8]

        x = self.global_pool(x).reshape(x.size(0), -1)  # → [B,64]
        x = self.fc(x)                                  # → [B,features_dim]
        return x

