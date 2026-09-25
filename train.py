import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from tqdm import tqdm
import os # To check if files exist

from src.discriminator import Discriminator
from src.generator import Generator
from src.data_loader import get_data_loader

# --- Configuration ---
LOAD_MODEL = False       # Set to TRUE to resume training
START_EPOCH = 0        # The epoch number you are resuming FROM (e.g., 5)
NUM_EPOCHS = 30     # The total goal (e.g., train until 50)
SAVE_EVERY = 5          # Save checkpoints every X epochs

LEARNING_RATE = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 100
FEATURES_DISC = 64
FEATURES_GEN = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

def save_checkpoint(model, optimizer, filename):
    print(f"=> Saving checkpoint: {filename}")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print(f"=> Loading checkpoint: {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this, the optimizer might keep the old learning rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def train():
    loader = get_data_loader(batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)
    disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)
    gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)

    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    # --- RESUME LOGIC ---
    if LOAD_MODEL:
        # Try to load Generator
        gen_file = f"generator_epoch_{START_EPOCH}.pth"
        if os.path.isfile(gen_file):
            # Note: The previous code saved ONLY state_dict (raw weights), not the full dictionary.
            # We must handle both cases (Old format vs New format)
            try:
                # Try loading as a full checkpoint (New Format)
                load_checkpoint(gen_file, gen, opt_gen, LEARNING_RATE)
            except:
                # Fallback: Load as raw weights (Old Format from previous steps)
                print(f"=> Loading raw weights (Old Format) for Generator: {gen_file}")
                gen.load_state_dict(torch.load(gen_file, map_location=device))
        else:
            print(f"Warning: File {gen_file} not found. Starting Generator from scratch.")

        # Try to load Discriminator (You probably don't have this file yet)
        disc_file = f"discriminator_epoch_{START_EPOCH}.pth"
        if os.path.isfile(disc_file):
            try:
                load_checkpoint(disc_file, disc, opt_disc, LEARNING_RATE)
            except:
                disc.load_state_dict(torch.load(disc_file, map_location=device))
        else:
            print(f"Warning: File {disc_file} not found. Starting Discriminator from scratch (Normal for first resume).")

    # --- TRAINING LOOP ---
    # Start loop from START_EPOCH
    for epoch in range(START_EPOCH, NUM_EPOCHS):
        loop = tqdm(loader, leave=True)
        
        for batch_idx, (real, _) in enumerate(loop):
            real = real.to(device)
            current_batch_size = real.shape[0]
            
            # Train Discriminator
            noise = torch.randn(current_batch_size, Z_DIM, 1, 1).to(device)
            fake = gen(noise)

            disc.zero_grad()
            disc_real = disc(real).reshape(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake.detach()).reshape(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            loss_disc.backward()
            opt_disc.step()

            # Calculate Accuracy
            pred_real = (disc_real > 0.5).float()
            pred_fake = (disc_fake < 0.5).float()
            batch_acc = (pred_real.sum() + pred_fake.sum()) / (2 * current_batch_size)

            # Train Generator
            gen.zero_grad()
            output = disc(fake).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            loss_gen.backward()
            opt_gen.step()

            loop.set_description(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
            loop.set_postfix(loss_d=loss_disc.item(), loss_g=loss_gen.item(), acc=batch_acc.item())

        # SAVE BOTH MODELS
        if (epoch + 1) % SAVE_EVERY == 0:
            save_checkpoint(gen, opt_gen, f"generator_epoch_{epoch+1}.pth")
            save_checkpoint(disc, opt_disc, f"discriminator_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()