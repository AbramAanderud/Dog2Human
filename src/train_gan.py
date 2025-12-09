import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F 
from torchvision.utils import save_image

from perceptual import PerceptualLoss
from dataset import DogHumanDataset
from models import Dog2HumanNet, UNetDog2Human, PatchDiscriminator


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)


def main():
    data_root = "data"
    image_size = 64
    batch_size = 16
    num_epochs = 20
    lr = 2e-4
    beta1 = 0.5
    beta2 = 0.999
    lambda_L1 = 50.0
    lambda_perc = 1.0
    num_workers = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # dataset and dataloader
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

    # models
    use_unet = True
    G = Dog2HumanNet().to(device)
    D = PatchDiscriminator().to(device)
    

    if use_unet:
        print("Using UNetDog2Human generator (no warm-start).")
        G = UNetDog2Human().to(device)
        G.apply(init_weights)  # always from scratch
    else:
        print("Using Dog2HumanNet generator.")
        G = Dog2HumanNet().to(device)

        # warm-start G from baseline if available
        baseline_ckpt = Path("checkpoints/baseline_epoch_5.pt")
        if baseline_ckpt.exists():
            print(f"Loading pretrained generator from {baseline_ckpt}")
            ckpt = torch.load(baseline_ckpt, map_location=device, weights_only=True)
            G.load_state_dict(ckpt["model_state_dict"])
        else:
            print("No baseline checkpoint found, training G from scratch.")
            G.apply(init_weights)

    D = PatchDiscriminator().to(device)
    D.apply(init_weights)

    # losses and optimizers
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()
    perceptual_loss = PerceptualLoss(
        device=device,
        layer="relu3_3",
        weight=lambda_perc,
    )

    optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))

    ckpt_dir = Path("checkpoints_gan")
    ckpt_dir.mkdir(exist_ok=True)
    samples_dir = Path("samples_gan")
    samples_dir.mkdir(exist_ok=True)

    fixed_dogs, fixed_humans = next(iter(dataloader))
    fixed_dogs = fixed_dogs.to(device)
    fixed_humans = fixed_humans.to(device)

    def save_example(epoch: int) -> None:
        G.eval()
        with torch.no_grad():
            fake = G(fixed_dogs)
            dogs_unnorm = (fixed_dogs * 0.5) + 0.5
            fake_unnorm = (fake * 0.5) + 0.5
            dogs_unnorm = dogs_unnorm.clamp(0, 1)
            fake_unnorm = fake_unnorm.clamp(0, 1)
            combined = torch.cat([dogs_unnorm, fake_unnorm], dim=3)

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

            #  Train Discriminator
            optimizer_D.zero_grad()

            # Real pairs (dog, real human)
            real_pair = torch.cat([dogs, humans], dim=1)
            pred_real = D(real_pair)
            target_real = torch.ones_like(pred_real, device=device)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake pairs (dog, fake human)
            with torch.no_grad():
                fake_humans_detached = G(dogs)
            fake_pair = torch.cat([dogs, fake_humans_detached], dim=1)
            pred_fake = D(fake_pair)
            target_fake = torch.zeros_like(pred_fake, device=device)
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            loss_D.backward()
            optimizer_D.step()

            #  Train Generator
            optimizer_G.zero_grad()

            fake_humans = G(dogs)
            fake_pair_for_G = torch.cat([dogs, fake_humans], dim=1)
            pred_fake_for_G = D(fake_pair_for_G)

            # GAN loss: try to fool D
            target_real_for_G = torch.ones_like(pred_fake_for_G, device=device)
            loss_G_GAN = criterion_GAN(pred_fake_for_G, target_real_for_G)

            # Pixel L1 loss
            loss_G_L1 = criterion_L1(fake_humans, humans) * lambda_L1

            # Perceptual loss (VGG16 features)
            loss_G_perc = perceptual_loss(fake_humans, humans)

            # Total generator loss
            loss_G = loss_G_GAN + loss_G_L1 + loss_G_perc

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

