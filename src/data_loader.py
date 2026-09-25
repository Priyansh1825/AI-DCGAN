import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

def get_data_loader(batch_size=128, image_size=64):
    """
    Loads custom images from the 'custom_data' folder.
    """
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)), # Force resize to 64x64
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])

    # Check if data exists
    if not os.path.exists("./custom_data"):
        raise FileNotFoundError("Folder 'custom_data' not found. Run download_data.py first!")

    # Use ImageFolder. 
    # It expects structure: ./custom_data/class_name/image.jpg
    dataset = torchvision.datasets.ImageFolder(
        root='./custom_data', 
        transform=transform
    )

    print(f"Loaded Custom Dataset with {len(dataset)} images.")

    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0 # Keep 0 for Windows
    )
    
    return loader