import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3):
        super().__init__()
        # Padding automático para conservar dimensión espacial cuando stride=1
        if kernel_size == 8:
            padding = 3
        elif kernel_size == 4:
            padding = 1
        else:
            padding = kernel_size // 2

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Si cambian canales o stride > 1, adaptamos identity
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # Si las dimensiones espaciales no coinciden, interpolamos identity
        if identity.shape != out.shape:
            identity = nn.functional.interpolate(
                identity,
                size=out.shape[2:],  # (H, W) de out
                mode='bilinear',
                align_corners=False
            )

        out += identity
        return self.relu(out)


class LSTMWrapper(nn.LSTM):
    """
    Subclase de nn.LSTM que solo devuelve el tensor de salida (out_seq),
    eliminando la tupla (out_seq, (h_n, c_n)). Así, torchsummary ve
    un único Tensor y no falla.
    """
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        batch_first=True,
        bidirectional=False
    ):
        # Llamamos al constructor original de nn.LSTM
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Aquí llamamos a nn.LSTM.forward, que devuelve (out_seq, (h_n, c_n))
        out_seq, _ = super().forward(x)
        return out_seq


class ResNetExtractor(BaseFeaturesExtractor):
    """
    Extractor para Atari + frame-stack (4×84×84), combinado con ResNet + LSTM.
    Cada uno de los 4 frames (cada “canal” en la entrada) se procesa
    independientemente con bloques residuales; luego aplanamos y creamos
    una secuencia de vectores unidimensionales que se mete en un LSTM.
    """
    def __init__(self, obs_space, features_dim: int = 512):
        super().__init__(obs_space, features_dim)

        # obs_space.shape == (4, 84, 84)
        seq_len, H, W = obs_space.shape
        assert seq_len > 1, "Se espera que obs_space.shape[0] sea > 1 (frame-stack)."
        in_channels = 1  # cada frame lo tratamos como canal único

        # ─── 1) Cuerpo convolucional residual para cada frame 1×84×84 ───
        self.residual_body = nn.Sequential(
            ResidualBlock(in_channels, 32, stride=4, kernel_size=8),
            ResidualBlock(32, 64, stride=2, kernel_size=4),
            ResidualBlock(64, 64, stride=1, kernel_size=3),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )

        # ─── 2) Medimos cuántos features salen de residual_body ───
        with torch.no_grad():
            dummy = torch.zeros((1, in_channels, H, W))
            n_flatten = self.residual_body(dummy).shape[1]  # p.ej. 64*4*4 = 1024

        # ─── 3) LSTMWrapper toma vectores de tamaño (n_flatten) y los convierte en features_dim ───
        self.lstm = LSTMWrapper(
            input_size=n_flatten,
            hidden_size=features_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        # ReLU final
        self.relu = nn.ReLU()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (batch_size, seq_len=4, 84, 84)
         1) Añadimos dim de canal → (B, seq_len, 1, 84, 84)
         2) Aplanamos batch+tiempo → (B*seq_len, 1, 84, 84)
         3) Pasamos cada frame por residual_body → (B*seq_len, n_flatten)
         4) Reconstruimos secuencia → (B, seq_len, n_flatten)
         5) LSTMWrapper → (B, seq_len, features_dim)
         6) Tomamos el último paso temporal → (B, features_dim), aplicamos ReLU y devolvemos
        """
        batch_size = obs.shape[0]
        seq_len = obs.shape[1]
        H = obs.shape[2]
        W = obs.shape[3]

        # 1) Añadimos dimensión de canal:
        #    (batch_size, seq_len, 1, H, W)
        x = obs.view(batch_size, seq_len, 1, H, W)

        # 2) Aplanamos batch+tiempo:
        #    (batch_size * seq_len, 1, H, W)
        x = x.reshape(batch_size * seq_len, 1, H, W)

        # 3) Cada frame pasa por residual_body → (B*seq_len, n_flatten)
        feat = self.residual_body(x)

        # 4) Reconstruimos secuencia → (batch_size, seq_len, n_flatten)
        feat = feat.reshape(batch_size, seq_len, -1)

        # 5) Alimentamos LSTMWrapper → (batch_size, seq_len, features_dim)
        out_seq = self.lstm(feat)

        # 6) Tomamos el último paso temporal:
        last = out_seq[:, -1, :]  # (batch_size, features_dim)

        # 7) ReLU final y devolvemos
        return self.relu(last)
