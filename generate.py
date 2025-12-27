import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

# Import the Generator architecture
from src.generator import Generator

# --- Configuration ---
# IMPORTANT: Update this filename to match the checkpoint you want to load
CHECKPOINT_PATH = "generator_epoch_5.pth" 

# These must match the training parameters exactly
Z_DIM = 100
CHANNELS_IMG = 3
FEATURES_GEN = 64
NUM_IMAGES_TO_GEN = 32  # How many images to create in the grid

# Device setup (allows loading GPU trained models onto CPU if needed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Generating on: {device}")

def generate_images():
    # 1. Instantiate the model architecture
    gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)

    # 2. Load trained weights
    try:
        # map_location ensures that if you trained on GPU but load on CPU, it works
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        gen.load_state_dict(checkpoint)
        print(f"Successfully loaded weights from {CHECKPOINT_PATH}")
    except FileNotFoundError:
        print(f"Error: Could not find file '{CHECKPOINT_PATH}'. Make sure you trained for at least 5 epochs.")
        return

    # 3. Set model to evaluation mode (Important for BatchNorm)
    gen.eval()

    # 4. Generate Noise Vectors
    # We don't need gradients for inference
    with torch.no_grad():
        noise = torch.randn(NUM_IMAGES_TO_GEN, Z_DIM, 1, 1).to(device)
        
        # 5. Generate Images (Forward pass)
        fake_images = gen(noise)

        # 6. Denormalize images
        # The generator outputs Tanh values [-1, 1].
        # We need to scale them back to [0, 1] for viewing.
        # Formula: (image + 1) / 2
        fake_images = (fake_images + 1) / 2

        # 7. Create a grid of images for visualization
        img_grid = vutils.make_grid(fake_images, padding=2, normalize=False)

        # Convert tensor to numpy array for matplotlib
        # .cpu() moves it to host memory, .permute changes shape from (C, H, W) to (H, W, C)
        img_grid_np = img_grid.cpu().numpy()
        plt.imshow(np.transpose(img_grid_np, (1, 2, 0)))
        plt.axis("off")
        plt.title("Generated Fake Images")
        
        # Save the output to a file
        output_filename = "final_generated_results.png"
        plt.savefig(output_filename)
        print(f"Success! Generated images saved to '{output_filename}'")
        # Uncomment the next line if you want a window to pop up showing the image
        # plt.show() 

if __name__ == "__main__":
    generate_images()