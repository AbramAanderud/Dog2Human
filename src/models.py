import torch
import torch.nn as nn


class Dog2HumanNet(nn.Module):
    """
    Simple conv encoder-decoder:
    input:  3 x 64 x 64 dog image (in [-1, 1])
    output: 3 x 64 x 64 human image (in [-1, 1])
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base_channels, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        out = self.decoder(z)
        return out


import torch.nn as nn
import torch


class PatchDiscriminator(nn.Module):
    """
    Pix2Pix-style PatchGAN discriminator.
    Takes concatenated [dog, human] images as input (6 channels total).
    Outputs a grid of real/fake scores (patches).
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        c = base_channels

        # Input channels: dog (3) + human (3) = 6
        input_channels = in_channels * 2

        def conv_block(in_c, out_c, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            # 64x64 -> 32x32
            conv_block(input_channels, c, normalize=False),
            # 32x32 -> 16x16
            conv_block(c, c * 2),
            # 16x16 -> 8x8
            conv_block(c * 2, c * 4),
            # 8x8 -> 4x4 (no stride 2 here, smaller patches)
            nn.Conv2d(c * 4, c * 8, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(c * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # Final 1-channel patch output
            nn.Conv2d(c * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 6, H, W]
        return self.model(x)
