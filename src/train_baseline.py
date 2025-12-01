import os
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DogHumanDataset
from models import Dog2HumanNet


def main():
    data_root = "data"
    image_size = 64
    batch_size = 16
    num_epochs = 5
    lr = 2e-4
    num_workers = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = DogHumanDataset(root=data_root, image_size=image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = Dog2HumanNet().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    global_step = 0
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}", unit="batch")
        for dogs, humans in pbar:
            dogs = dogs.to(device)
            humans = humans.to(device)

            optimizer.zero_grad()
            outputs = model(dogs)
            loss = criterion(outputs, humans)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            global_step += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch} average loss: {epoch_loss:.4f}")

        ckpt_path = ckpt_dir / f"baseline_epoch_{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": epoch_loss,
            },
            ckpt_path,
        )
        print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
