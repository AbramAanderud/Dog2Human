from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PerceptualLoss(nn.Module):
    """
    VGG16-based perceptual loss.

    Expects inputs in the SAME [-1, 1] space as your training images.
    Internally converts to ImageNet-normalized space before feeding VGG.
    """

    def __init__(self, device: torch.device, layer: str = "relu3_3", weight: float = 1.0):
        super().__init__()
        self.weight = weight

        # Load pretrained VGG16
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features

        # Map layer name to index in VGG features
        layer_map = {
            "relu1_2": 3,
            "relu2_2": 8,
            "relu3_3": 15,
            "relu4_3": 22,
        }
        if layer not in layer_map:
            raise ValueError(f"Unknown layer {layer}, choose from {list(layer_map.keys())}")

        cut_idx = layer_map[layer]
        self.vgg = nn.Sequential(*list(vgg.children())[: cut_idx + 1]).to(device)
        for p in self.vgg.parameters():
            p.requires_grad = False  # keep VGG frozen

        # ImageNet normalization params
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _to_imagenet_space(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert from [-1, 1] (used in training) to ImageNet-normalized space for VGG.
        """
        # [-1, 1] -> [0, 1]
        x = (x + 1.0) / 2.0
        # Normalize with ImageNet stats
        return (x - self.mean) / self.std

    def forward(self, x_fake: torch.Tensor, x_real: torch.Tensor) -> torch.Tensor:
        # Detach real just to be safe
        x_fake_in = self._to_imagenet_space(x_fake)
        x_real_in = self._to_imagenet_space(x_real.detach())

        feat_fake = self.vgg(x_fake_in)
        feat_real = self.vgg(x_real_in)

        loss = F.l1_loss(feat_fake, feat_real)
        return self.weight * loss
