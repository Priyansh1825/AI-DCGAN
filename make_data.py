import os
import random
from PIL import Image, ImageDraw # This comes with torchvision

def create_dummy_data(num_images=100):
    # 1. Define folder structure
    # Structure must be: custom_data/images/img1.jpg
    root_dir = "custom_data"
    class_dir = os.path.join(root_dir, "images")

    # Create folders if they don't exist
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
    
    print(f"Generating {num_images} synthetic images in '{class_dir}'...")

    for i in range(num_images):
        # Create a random color image (64x64)
        img = Image.new('RGB', (64, 64), color=(
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        ))
        
        # Draw a random rectangle on it (so it has a 'shape')
        draw = ImageDraw.Draw(img)
        draw.rectangle(
            [random.randint(0, 32), random.randint(0, 32), random.randint(32, 64), random.randint(32, 64)],
            fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        )

        # Save the file
        img.save(os.path.join(class_dir, f"fake_img_{i}.jpg"))

    print("Success! Data created. You can now run train.py.")

if __name__ == "__main__":
    create_dummy_data()