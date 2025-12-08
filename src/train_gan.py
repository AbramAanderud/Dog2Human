import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from pathlib import Path
from tqdm import tqdm

from dataset import DogHumanDataset
from models import Dog2HumanNet, PatchDiscriminator


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)


def main():
    # --- config ---
    data_root = "data"
    image_size = 64
    batch_size = 16
    num_epochs = 20          # more training for GAN
    lr = 2e-4
    beta1 = 0.5
    beta2 = 0.999
    lambda_L1 = 100.0        # Pix2Pix style weight on L1 loss
    num_workers = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- dataset & dataloader ---
    dataset = DogHumanDataset(root=data_root, image_size=image_size)
    print(f"Found {len(dataset.dog_paths)} dog images")
    print(f"Found {len(dataset.human_paths)} human images")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # --- models ---
    G = Dog2HumanNet().to(device)
    D = PatchDiscriminator().to(device)

    G.apply(init_weights)
    D.apply(init_weights)

    # --- losses & optimizers ---
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()

    optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))

    ckpt_dir = Path("checkpoints_gan")
    ckpt_dir.mkdir(exist_ok=True)
    samples_dir = Path("samples_gan")
    samples_dir.mkdir(exist_ok=True)

    fixed_dogs, fixed_humans = next(iter(dataloader))
    fixed_dogs = fixed_dogs.to(device)
    fixed_humans = fixed_humans.to(device)

    def save_example(epoch):
        G.eval()
        with torch.no_grad():
            fake = G(fixed_dogs)
            # unnormalize
            dogs_unnorm = (fixed_dogs * 0.5) + 0.5
            fake_unnorm = (fake * 0.5) + 0.5
            dogs_unnorm = dogs_unnorm.clamp(0, 1)
            fake_unnorm = fake_unnorm.clamp(0, 1)
            # concat along width
            combined = torch.cat([dogs_unnorm, fake_unnorm], dim=3)
            from torchvision.utils import save_image
            out_path = samples_dir / f"epoch_{epoch}_samples.png"
            save_image(combined, out_path, nrow=4)
            print(f"Saved GAN samples to {out_path}")
        G.train()

    for epoch in range(1, num_epochs + 1):
        G.train()
        D.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}", unit="batch")

        running_loss_G = 0.0
        running_loss_D = 0.0

        for dogs, humans in pbar:
            dogs = dogs.to(device)
            humans = humans.to(device)
            batch_size_curr = dogs.size(0)

            # -------------------------
            # Train Discriminator D
            # -------------------------
            optimizer_D.zero_grad()

            # Real pairs (dog, human)
            real_pair = torch.cat([dogs, humans], dim=1)  # [B,6,H,W]
            pred_real = D(real_pair)
            target_real = torch.ones_like(pred_real, device=device)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake pairs (dog, G(dog)) - detach so G not updated here
            with torch.no_grad():
                fake_humans = G(dogs)
            fake_pair = torch.cat([dogs, fake_humans], dim=1)
            pred_fake = D(fake_pair)
            target_fake = torch.zeros_like(pred_fake, device=device)
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()

            # -------------------------
            # Train Generator G
            # -------------------------
            optimizer_G.zero_grad()

            fake_humans = G(dogs)
            fake_pair = torch.cat([dogs, fake_humans], dim=1)
            pred_fake_for_G = D(fake_pair)

            target_real_for_G = torch.ones_like(pred_fake_for_G, device=device)
            loss_G_GAN = criterion_GAN(pred_fake_for_G, target_real_for_G)
            loss_G_L1 = criterion_L1(fake_humans, humans) * lambda_L1
            loss_G = loss_G_GAN + loss_G_L1

            loss_G.backward()
            optimizer_G.step()

            running_loss_D += loss_D.item()
            running_loss_G += loss_G.item()

            pbar.set_postfix({
                "loss_D": f"{loss_D.item():.3f}",
                "loss_G": f"{loss_G.item():.3f}",
            })

        avg_loss_D = running_loss_D / len(dataloader)
        avg_loss_G = running_loss_G / len(dataloader)
        print(f"Epoch {epoch}: D_loss={avg_loss_D:.4f}, G_loss={avg_loss_G:.4f}")

        # Save checkpoint for generator & discriminator
        ckpt_path = ckpt_dir / f"gan_epoch_{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "G_state_dict": G.state_dict(),
            "D_state_dict": D.state_dict(),
            "optimizer_G": optimizer_G.state_dict(),
            "optimizer_D": optimizer_D.state_dict(),
            "loss_D": avg_loss_D,
            "loss_G": avg_loss_G,
        }, ckpt_path)
        print(f"Saved GAN checkpoint to {ckpt_path}")

        save_example(epoch)


if __name__ == "__main__":
    main()
