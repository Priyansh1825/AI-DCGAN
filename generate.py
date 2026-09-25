import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

# Import the Generator architecture
from src.generator import Generator

# --- Configuration ---
# UPDATE THIS to match your saved file (e.g., generator_epoch_20.pth)
CHECKPOINT_PATH = "generator_epoch_20.pth" 

Z_DIM = 100
CHANNELS_IMG = 3
FEATURES_GEN = 64
NUM_IMAGES_TO_GEN = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Generating on: {device}")

def generate_images():
    gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)

    # --- UPDATED LOADING LOGIC ---
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        
        # Check if the file is the "New Format" (Dictionary) or "Old Format" (Raw Weights)
        if "state_dict" in checkpoint:
            # New Format: Extract the weights from the dictionary
            print("Detected 'Resume-Ready' checkpoint format.")
            gen.load_state_dict(checkpoint["state_dict"])
        else:
            # Old Format: The file IS the weights
            print("Detected 'Raw Weights' checkpoint format.")
            gen.load_state_dict(checkpoint)
            
        print(f"Successfully loaded weights from {CHECKPOINT_PATH}")
        
    except FileNotFoundError:
        print(f"Error: Could not find file '{CHECKPOINT_PATH}'. Check the filename!")
        return
    except RuntimeError as e:
        print(f"Error loading model: {e}")
        print("Tip: Make sure your Generator architecture in src/generator.py matches exactly what you trained with.")
        return
    # -----------------------------

    gen.eval()

    with torch.no_grad():
        noise = torch.randn(NUM_IMAGES_TO_GEN, Z_DIM, 1, 1).to(device)
        fake_images = gen(noise)

        # Denormalize: Scale from [-1, 1] back to [0, 1]
        fake_images = (fake_images + 1) / 2

        # Create grid
        img_grid = vutils.make_grid(fake_images, padding=2, normalize=False)
        
        # Convert to numpy for plotting
        img_grid_np = img_grid.cpu().numpy()
        
        # Setup the plot
        plt.figure(figsize=(10, 10))
        plt.imshow(np.transpose(img_grid_np, (1, 2, 0)))
        plt.axis("off")
        plt.title("Generated Images")
        
        output_filename = "final_generated_results.png"
        plt.savefig(output_filename)
        print(f"Success! Result saved to '{output_filename}'")
        # plt.show() # Uncomment if you want a pop-up window

if __name__ == "__main__":
    generate_images()