import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter # For visualization (optional but good practice)
from tqdm import tqdm # For progress bar

# Import our custom modules
from src.discriminator import Discriminator
from src.generator import Generator
from src.data_loader import get_data_loader

# --- Hyperparameters ---
LEARNING_RATE = 2e-4  # Standard DCGAN learning rate
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 100
NUM_EPOCHS = 5       # Start small to test. Increase to 50+ for real results.
FEATURES_DISC = 64
FEATURES_GEN = 64

# --- Device Setup ---
# Automatically detect GPU. If not found, use CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

def train():
    # 1. Initialize Data
    loader = get_data_loader(batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)

    # 2. Initialize Models
    disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)
    gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)

    # 3. Initialize Optimizers
    # Betas=(0.5, 0.999) is specific to DCGAN paper stability
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    # 4. Loss Function
    # Binary Cross Entropy (BCE) is standard for Real (1) vs Fake (0) classification
    criterion = nn.BCELoss()

    # Fixed noise for visualization (to see how the SAME noise improves over time)
    fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)

    # --- Training Loop ---
    for epoch in range(NUM_EPOCHS):
        # Tqdm gives us a nice progress bar in the terminal
        loop = tqdm(loader, leave=True)
        
        for batch_idx, (real, _) in enumerate(loop):
            real = real.to(device)
            noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1).to(device)
            fake = gen(noise)

            ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
            disc.zero_grad() # Clear gradients
            
            # Loss on Real Data (Target = 1)
            disc_real = disc(real).reshape(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            
            # Loss on Fake Data (Target = 0)
            disc_fake = disc(fake.detach()).reshape(-1) # .detach() prevents training G here
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            
            # Total Discriminator Loss & Backprop
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            loss_disc.backward()
            opt_disc.step()

            ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z)))
            gen.zero_grad()
            
            # We want Discriminator to output 1 (Real) for our fakes
            output = disc(fake).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            
            loss_gen.backward()
            opt_gen.step()

            # Update Progress Bar
            loop.set_description(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
            loop.set_postfix(loss_d=loss_disc.item(), loss_g=loss_gen.item())

        # (Optional) Save model checkpoints periodically
        if (epoch + 1) % 5 == 0:
            torch.save(gen.state_dict(), f"generator_epoch_{epoch+1}.pth")
            print(f"Saved Model checkpoint at epoch {epoch+1}")

if __name__ == "__main__":
    train()
