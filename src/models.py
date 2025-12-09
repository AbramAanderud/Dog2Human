import torch
import torch.nn as nn


class Dog2HumanNet(nn.Module):
    """
    Simple encoder and decoder that maps a 3x64x64 dog image to a 3x64x64
    human image. Uses Tanh on the output so it matches [-1, 1] scaling.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
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


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down: bool = True):
        super().__init__()
        if down:
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.block(x)


class UNetDog2Human(nn.Module):
    """
    Simple U-Net style generator: 3x64x64 -> 3x64x64.
    """

    def __init__(self, in_channels=3, out_channels=3, base_channels=64):
        super().__init__()

        self.enc1 = ConvBlock(in_channels, base_channels, down=True)              # 3 -> 64, 64x64 -> 32x32
        self.enc2 = ConvBlock(base_channels, base_channels * 2, down=True)        # 64 -> 128, 32x32 -> 16x16
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4, down=True)    # 128 -> 256, 16x16 -> 8x8
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8, down=True)    # 256 -> 512, 8x8 -> 4x4

        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.dec4 = ConvBlock(base_channels * 8, base_channels * 4, down=False)   # 4x4 -> 8x8
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 2, down=False)   # (256+256) -> 128, 8x8 -> 16x16
        self.dec2 = ConvBlock(base_channels * 4, base_channels, down=False)       # (128+128) -> 64, 16x16 -> 32x32
        self.dec1 = ConvBlock(base_channels * 2, base_channels // 2, down=False)  # (64+64) -> 32, 32x32 -> 64x64

        self.final = nn.Sequential(
            nn.Conv2d(base_channels // 2, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),  
        )

    def forward(self, x):
        e1 = self.enc1(x)   
        e2 = self.enc2(e1) 
        e3 = self.enc3(e2)  
        e4 = self.enc4(e3)  

        b = self.bottleneck(e4)

        d4 = self.dec4(b)                 
        d4 = torch.cat([d4, e3], dim=1)   

        d3 = self.dec3(d4)               
        d3 = torch.cat([d3, e2], dim=1)   

        d2 = self.dec2(d3)               
        d2 = torch.cat([d2, e1], dim=1)  

        d1 = self.dec1(d2)               

        out = self.final(d1)
        return out



class PatchDiscriminator(nn.Module):
    """
    Pix2Pix-style PatchGAN discriminator.
    Takes concatenated [dog, human] images as input.
    Outputs a grid of real/fake scores.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        c = base_channels

        # Input channels are dog (3) + human (3) = 6
        input_channels = in_channels * 2

        def conv_block(in_c, out_c, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            conv_block(input_channels, c, normalize=False),
            conv_block(c, c * 2),
            conv_block(c * 2, c * 4),
            nn.Conv2d(c * 4, c * 8, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(c * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
