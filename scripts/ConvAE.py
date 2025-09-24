import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Encoder Module
# -----------------------------
class Encoder(nn.Module):
    def __init__(self, in_channels=31, R=64, d=11, use_batchnorm=False):
        super().__init__()
        layers = []

        # first conv: 3x3 kernel, in_channels -> R
        layers.append(nn.Conv2d(in_channels, R, kernel_size=3, padding=1))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(R))
        layers.append(nn.ReLU(inplace=True))

        # d-1 hidden conv layers: R -> R
        for _ in range(d-1):
            layers.append(nn.Conv2d(R, R, kernel_size=3, padding=1))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(R))
            layers.append(nn.ReLU(inplace=True))

        # output layer: R -> R (linear, no activation)
        layers.append(nn.Conv2d(R, R, kernel_size=3, padding=1))

        self.encoder = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.encoder(x)

# -----------------------------
# Decoder Module
# -----------------------------
class Decoder(nn.Module):
    def __init__(self, out_channels=31, R=64, d=11, use_batchnorm=False, output_activation='sigmoid'):
        super().__init__()
        layers = []

        # d hidden layers: R -> R
        for _ in range(d):
            layers.append(nn.Conv2d(R, R, kernel_size=3, padding=1))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(R))
            layers.append(nn.ReLU(inplace=True))

        # output layer: R -> out_channels
        layers.append(nn.Conv2d(R, out_channels, kernel_size=3, padding=1))
        if output_activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif output_activation == 'tanh':
            layers.append(nn.Tanh())
        # else linear (identity)

        self.decoder = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.decoder(x)

# -----------------------------
# ConvAutoencoder: joins encoder & decoder
# -----------------------------
class ConvAutoencoder(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        alpha = self.encoder(x)
        recon = self.decoder(alpha)
        return recon, alpha
# =============================
# Loss and Metrics
# =============================
def ae_loss_exact(recon, x, tau_w=1e-8, model=None):
    k = x.size(0)
    mse_term = 0.5 * F.mse_loss(recon, x, reduction='sum') / k
    wd_term = 0.0

    return mse_term + wd_term

def psnr(x_hat, x, max_val=1.0):
    mse = F.mse_loss(x_hat, x)
    return 10 * torch.log10(max_val**2 / mse)
