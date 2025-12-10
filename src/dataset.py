import os
from glob import glob
from typing import Callable, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch


class DogHumanDataset(Dataset):
    """
    Expects:
        root/dogs_cropped/*.jpg  (the pre cropped dog faces) 
        root/humans/thumbnails128x128/*.png (or .jpg)
    Pairs dog[i] with human[i].
    """

    def __init__(self, root: str, image_size: int = 64, transform: Callable | None = None):
        self.root = root
        valid_exts = {".jpg", ".jpeg", ".png"}

        # Cropped dog faces
        self.dog_paths = sorted(
            p for p in glob(os.path.join(root, "dogs_cropped", "**", "*"), recursive=True)
            if os.path.isfile(p) and os.path.splitext(p)[1].lower() in valid_exts
        )

        # Only use the square human thumbnails
        self.human_paths = sorted(
            p for p in glob(os.path.join(root, "humans", "thumbnails128x128", "**", "*"), recursive=True)
            if os.path.isfile(p) and os.path.splitext(p)[1].lower() in valid_exts
        )

        if not self.dog_paths:
            raise RuntimeError(f"No dog images found in {os.path.join(root, 'dogs_cropped')}")
        if not self.human_paths:
            raise RuntimeError(f"No human images found in {os.path.join(root, 'humans', 'thumbnails128x128')}")

        n = min(len(self.dog_paths), len(self.human_paths))
        self.dog_paths = self.dog_paths[:n]
        self.human_paths = self.human_paths[:n]

        print(f"Using {n} dog/human pairs")

        if transform is None:
            #roughly square 128x128 thumbnails
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),
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
