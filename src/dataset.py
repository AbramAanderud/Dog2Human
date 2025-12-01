import os
from glob import glob
from typing import Callable, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch


class DogHumanDataset(Dataset):
    """
    Expects:
        root/dogs/*.jpg
        root/humans/*.jpg
    Pairs dog[i] with human[i].
    """

    def __init__(self, root: str, image_size: int = 64, transform: Callable | None = None):
        self.root = root
        self.dog_paths = sorted(glob(os.path.join(root, "dogs", "Images", "*", "*")))
        self.human_paths = sorted(glob(os.path.join(root, "humans", "images_1000", "*")))

        if not self.dog_paths:
            raise RuntimeError(f"No dog images found in {os.path.join(root, 'dogs')}")
        if not self.human_paths:
            raise RuntimeError(f"No human images found in {os.path.join(root, 'humans')}")

        n = min(len(self.dog_paths), len(self.human_paths))
        self.dog_paths = self.dog_paths[:n]
        self.human_paths = self.human_paths[:n]

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = transform

    def __len__(self) -> int:
        return len(self.dog_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        dog_path = self.dog_paths[idx]
        human_path = self.human_paths[idx]

        dog_img = Image.open(dog_path).convert("RGB")
        human_img = Image.open(human_path).convert("RGB")

        dog_tensor = self.transform(dog_img)
        human_tensor = self.transform(human_img)

        return dog_tensor, human_tensor
