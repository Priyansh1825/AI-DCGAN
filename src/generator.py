import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=100, channels_img=3, features_g=64):
        """
        Args:
            z_dim: Dimension of the noise vector (Latent Space).
            channels_img: 3 for RGB.
            features_g: Base number of filters.
        """
        super(Generator, self).__init__()
        
        self.gen = nn.Sequential(
            # Input: N x z_dim x 1 x 1
            # Layer 1: Upsample noise to 4x4
            self._block(z_dim, features_g * 16, kernel_size=4, stride=1, padding=0),
            # Output: N x 1024 x 4 x 4
            
            # Layer 2: Upsample to 8x8
            self._block(features_g * 16, features_g * 8, kernel_size=4, stride=2, padding=1),
            # Output: N x 512 x 8 x 8
            
            # Layer 3: Upsample to 16x16
            self._block(features_g * 8, features_g * 4, kernel_size=4, stride=2, padding=1),
            # Output: N x 256 x 16 x 16
            
            # Layer 4: Upsample to 32x32
            self._block(features_g * 4, features_g * 2, kernel_size=4, stride=2, padding=1),
            # Output: N x 128 x 32 x 32
            
            # Final Layer: Upsample to 64x64
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x 3 x 64 x 64
            nn.Tanh(), # Normalize output to [-1, 1]
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        Helper function for Generator Block:
        ConvTranspose2d -> BatchNorm -> ReLU
        """
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, 
                out_channels, 
                kernel_size, 
                stride, 
                padding, 
                bias=False # Bias false because of BatchNorm
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.gen(x)

# --- Verification Block ---
if __name__ == "__main__":
    # Simulate a batch of random noise vectors (Batch=8, z_dim=100, 1, 1)
    z_dim = 100
    z = torch.randn((8, z_dim, 1, 1))
    
    model = Generator(z_dim=z_dim)
    gen_images = model(z)
    
    print(f"Input Noise Shape: {z.shape}")
    print(f"Generated Image Shape: {gen_images.shape}")
    
    # Check if output is N x 3 x 64 x 64
    if gen_images.shape == (8, 3, 64, 64):
        print("Success! Generator architecture is correct.")
    else:
        print("Error in dimensions.")