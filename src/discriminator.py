import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels_img=3, features_d=64):
        """
        Args:
            channels_img: 3 for RGB images.
            features_d: Base number of filters (grows deeper into the network).
        """
        super(Discriminator, self).__init__()
        
        self.disc = nn.Sequential(
            # Input: N x 3 x 64 x 64
            # Layer 1: No BatchNorm in the first layer of Discriminator (Standard DCGAN practice)
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1), 
            nn.LeakyReLU(0.2), 
            # Output: N x 64 x 32 x 32
            
            # Layer 2
            self._block(features_d, features_d * 2, kernel_size=4, stride=2, padding=1),
            # Output: N x 128 x 16 x 16
            
            # Layer 3
            self._block(features_d * 2, features_d * 4, kernel_size=4, stride=2, padding=1),
            # Output: N x 256 x 8 x 8
            
            # Layer 4
            self._block(features_d * 4, features_d * 8, kernel_size=4, stride=2, padding=1),
            # Output: N x 512 x 4 x 4
            
            # Final Layer: Output a single value (Real vs Fake)
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
            # Output: N x 1 x 1 x 1
            
            nn.Sigmoid(), # Squash output between 0 and 1
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        Helper function to create a Convolutional Block:
        Conv2d -> BatchNorm -> LeakyReLU
        """
        return nn.Sequential(
            nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size, 
                stride, 
                padding, 
                bias=False # Bias is False because BatchNorm layers include a bias term
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)

# --- Verification Block ---
if __name__ == "__main__":
    # Create a random noise tensor simulating an image batch (BatchSize=8, RGB, 64x64)
    x = torch.randn((8, 3, 64, 64))
    model = Discriminator()
    preds = model(x)
    
    print(f"Input Shape: {x.shape}")
    print(f"Output Shape: {preds.shape}")
    
    # Check if output is N x 1 x 1 x 1
    if preds.shape == (8, 1, 1, 1):
        print("Success! Discriminator architecture is correct.")
    else:
        print("Error in dimensions.")