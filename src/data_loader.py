import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_data_loader(batch_size=128, image_size=64):
    """
    Downloads CIFAR-10 dataset and creates a PyTorch DataLoader.
    
    Args:
        batch_size (int): Number of images per training batch.
        image_size (int): Size to resize images to (DCGAN standard is usually 64x64).
    
    Returns:
        DataLoader: The iterable data object.
    """
    
    # Define transformations:
    # 1. Resize: Upscale 32x32 CIFAR images to 64x64 (Better for DCGAN stability)
    # 2. ToTensor: Convert images to PyTorch tensors (0-1 range)
    # 3. Normalize: Scale data to [-1, 1] using (image - 0.5) / 0.5
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])

    # Download Training Data
    # 'root' is where data will be stored (./data folder)
    dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )

    # Create DataLoader
    # shuffle=True is crucial for GANs to prevent the model from memorizing order
    # num_workers=2 uses parallel processing to load data faster (Set to 0 if on Windows and getting errors)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0  # Set to 0 for maximum Windows compatibility
    )
    
    return loader

# --- Verification Block (Only runs if you run this file directly) ---
if __name__ == "__main__":
    print("Testing Data Loader...")
    train_loader = get_data_loader()
    
    # Fetch one batch of images to check shape
    # iter() creates an iterator, next() grabs the first batch
    images, labels = next(iter(train_loader))
    
    print(f"Batch Shape: {images.shape}")
    print("Success! Data loaded correctly.")
    # Expected Output: torch.Size([128, 3, 64, 64]) 
    # (128 images, 3 color channels, 64 height, 64 width)