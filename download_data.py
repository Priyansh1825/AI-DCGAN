import shutil
import os
from bing_image_downloader import downloader

def download_images(query, limit=100):
    # Define the output directory
    # structure: custom_data/images
    base_dir = "custom_data"
    
    # Clean up old data to avoid duplicates (Optional)
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
        
    # Download images
    # This library creates a folder with the 'query' name automatically.
    # We will rename it later to match our required structure.
    print(f"Downloading {limit} images of '{query}'...")
    
    downloader.download(
        query, 
        limit=limit, 
        output_dir=base_dir, 
        adult_filter_off=True, 
        force_replace=False, 
        timeout=10, 
        verbose=True
    )

    # --- FIX FOLDER STRUCTURE FOR PYTORCH ---
    # PyTorch needs: custom_data/images/img1.jpg
    # Bing downloader creates: custom_data/query_name/img1.jpg
    
    source_folder = os.path.join(base_dir, query)
    target_folder = os.path.join(base_dir, "images")
    
    # Rename the folder created by Bing to 'images'
    if os.path.exists(source_folder):
        os.rename(source_folder, target_folder)
        print(f"Success! Data organized into: {target_folder}")
    else:
        print("Error: Download folder not found.")

if __name__ == "__main__":
    download_images("sports cars", limit=50)