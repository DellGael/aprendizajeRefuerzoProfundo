import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.cuda.amp import autocast, GradScaler

# --- OPCIONES DE RENDIMIENTO GPU -------------------------------------
# Ajusta algoritmos internos de cuDNN según el tamaño de entrada
torch.backends.cudnn.benchmark = True

# --- BLOQUES BÁSICOS -------------------------------------------------

class SepConv(nn.Module):
    """Depthwise separable convolution + BN + ReLU."""
    def __init__(self, in_c, out_c):
        super().__init__()
        # Depthwise
        self.dw = nn.Conv2d(in_c, in_c, kernel_size=3, stride=1, padding=1,
                            groups=in_c, bias=False)
        self.bn1 = nn.BatchNorm2d(in_c)
        # Pointwise
        self.pw = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.dw(x)))
        x = self.relu(self.bn2(self.pw(x)))
        return x

def conv_bn_relu(in_c, out_c, k=3, s=1, p=1):
    """Conv2d → BatchNorm2d → ReLU"""
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )

class ResBlockGRU(nn.Module):
    """Residual block 3×3 → ReLU → 3×3 → sum shortcut for GRU model."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = conv_bn_relu(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return self.relu(out + x)

class ResBlockLSTM(nn.Module):
    """Residual block for LSTM model."""
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

class ResBlockSimple(nn.Module):
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

class GRUWrapper(nn.GRU):
    """GRU que solo devuelve la secuencia de salida."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_seq, _ = super().forward(x)
        return out_seq

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

# --- EXTRACTORES ---

class ImpalaResNetExtractor(BaseFeaturesExtractor):
    """
    Versión optimizada del extractor estilo Impala-ResNet:
     - SepConv en lugar de conv estándar
     - 1 ResBlock por bloque para aligerar
     - Global Average Pooling → Flatten
     - GRU para procesar stack temporal
    """
    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        seq_len, H, W = observation_space.shape
        assert seq_len > 1, "Se espera frame-stack (seq_len>1)."
        n_ch = seq_len  # tratamos cada frame como canal

        def make_block(in_c, out_c):
            return nn.Sequential(
                SepConv(in_c, out_c),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                ResBlockGRU(out_c),  # solo un bloque para mayor velocidad
            )

        # Cuerpo convolucional
        self.conv_body = nn.Sequential(
            make_block(n_ch, 32),     # → [B,32,H/2,W/2]
            make_block(32, 64),       # → [B,64,H/4,W/4]
            make_block(64, 64),       # → [B,64,H/8,W/8]
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),             # → [B,64]
        )

        # Determinar tamaño de salida del conv_body
        with torch.no_grad():
            dummy = torch.zeros((1, n_ch, H, W))
            n_flatten = self.conv_body(dummy).shape[1]

        # GRU temporal
        self.gru = GRUWrapper(
            input_size=n_flatten,
            hidden_size=features_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        self.relu = nn.ReLU()

        # Inicialización He/Kaiming
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: [B, seq_len, H, W]
         1) Lo interpretamos como canales: [B, seq_len, H, W]
         2) ConvBody → [B, n_flatten]
         3) Replicar features en secuencia: [B, seq_len, n_flatten]
         4) GRU → [B, seq_len, features_dim]
         5) Último paso + ReLU → [B, features_dim]
        """
        batch_size = obs.size(0)
        seq_len, H, W = obs.size(1), obs.size(2), obs.size(3)

        # 1) Cada frame como canal
        x = obs.view(batch_size, seq_len, H, W).float()
        if x.max() > 1.0:
            x = x / 255.0

        # 2) ConvBody
        feat = self.conv_body(x)  # [B, n_flatten]

        # 3) Repetir para GRU
        feat_seq = feat.unsqueeze(1).repeat(1, seq_len, 1)  # [B, seq_len, n_flatten]

        # 4) GRU
        out_seq = self.gru(feat_seq)

        # 5) Último estado temporal
        last = out_seq[:, -1, :]  # [B, features_dim]
        return self.relu(last)

class ResNetLSTMExtractor(BaseFeaturesExtractor):
    """
    Extractor para Atari + frame-stack (4×84×84), combinado con ResNet + LSTM.
    Cada uno de los 4 frames (cada "canal" en la entrada) se procesa
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
            ResBlockLSTM(in_channels, 32, stride=4, kernel_size=8),
            ResBlockLSTM(32, 64, stride=2, kernel_size=4),
            ResBlockLSTM(64, 64, stride=1, kernel_size=3),
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

class ImpalaResNetSimpleExtractor(BaseFeaturesExtractor):
    """
    Extractor inspirado en Impala-ResNet (SENCILLO):
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
                ResBlockSimple(out_c),
                ResBlockSimple(out_c),
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
