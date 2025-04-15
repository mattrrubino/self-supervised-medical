import sys
import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

parent_dir = os.path.abspath("..")
sys.path.append(parent_dir)
from pretext_2d import jigsawify


def image_to_tensor(image_path):
    # Load image
    image = Image.open(image_path).convert("RGB")  # ensure it's RGB

    # Define the transform to convert image to tensor
    transform = transforms.ToTensor()

    # Apply transform
    tensor = transform(image)
 
    return tensor

def reform_image(patches, output_path):
    C, H, W = patches[0].shape

    # Reshape into grid and concatenate
    rows = []
    for i in range(num_patches):
        row_patches = [patches[i * num_patches + j] for j in range(num_patches)]
        row = np.concatenate(row_patches, axis=2)  # concat along width
        rows.append(row)

    full_image = np.concatenate(rows, axis=1)  # concat rows along height

    # Rearrange to (H, W, C) for saving
    full_image = np.transpose(full_image, (1, 2, 0))

    if full_image.max() <= 1.0:
        full_image = (full_image * 255).astype(np.uint8)
    else:
        full_image = full_image.astype(np.uint8)

    img = Image.fromarray(full_image)
    img.save(output_path)

# Example usage
if __name__ == "__main__":
    image_path = "input/bubble.png"  # Replace with your image path
    output_path = "output/output.png"
    num_patches = 4
    permutation = [list(np.random.permutation(num_patches**2))]
    jitter = 20 # use negative jitter to make the patches overlap
    is_training = True

    image_tensor = image_to_tensor(image_path)
    patches, label = jigsawify(image_tensor,is_training, num_patches, jitter, permutation)
    print("label: ", label)
    reform_image(patches, output_path)
    