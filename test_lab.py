import torch
from model_lab import ColorizationCNN
from PIL import Image
import os
import numpy as np
from skimage import color
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model
print("Loading model...")
model = ColorizationCNN().to(device)
try:
    model.load_state_dict(torch.load("color_model_lab.pth", map_location=device))
    print("✓ Model loaded successfully!")
except FileNotFoundError:
    print(" Error: color_model_lab.pth not found. Please train the model first using train_lab.py")
    exit()

model.eval()

# Test folder
test_folder = "images"
output_folder = "outputs"
os.makedirs(output_folder, exist_ok=True)

# Check if test folder exists
if not os.path.exists(test_folder):
    print(f" Error: '{test_folder}' folder not found!")
    exit()

# Get list of images
image_files = [f for f in os.listdir(test_folder) 
               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

if len(image_files) == 0:
    print(f" No images found in '{test_folder}' folder!")
    exit()

print(f"\nFound {len(image_files)} images to colorize")
print("="*50)

for i, img_name in enumerate(image_files, 1):
    try:
        path = os.path.join(test_folder, img_name)
        
        # Load and preprocess
        img = Image.open(path).convert("RGB")
        original_size = img.size
        img = img.resize((256, 256), Image.BILINEAR)
        img_np = np.array(img) / 255.0  # Normalize to [0, 1]
        
        # Convert RGB to Lab color space
        lab = color.rgb2lab(img_np)
        L = lab[:, :, 0] / 100.0  # Normalize L channel to [0, 1]
        
        # Prepare input tensor
        L_tensor = torch.from_numpy(L).unsqueeze(0).unsqueeze(0).float().to(device)
        
        # Predict ab channels
        with torch.no_grad():
            ab_pred = model(L_tensor).cpu().numpy()[0]  # Shape: [2, H, W]
        
        # Denormalize ab channels back to [-128, 127]
        ab_pred = ab_pred.transpose((1, 2, 0)) * 128.0
        
        # Clip values to valid range
        ab_pred = np.clip(ab_pred, -128, 127)
        
        # Reconstruct Lab image
        lab_output = np.zeros((256, 256, 3), dtype=np.float64)
        lab_output[:, :, 0] = L * 100.0  # Denormalize L back to [0, 100]
        lab_output[:, :, 1:] = ab_pred
        
        # Convert Lab to RGB
        rgb_output = color.lab2rgb(lab_output)
        rgb_output = np.clip(rgb_output, 0, 1)
        
        # Convert to uint8 and save
        output_img = (rgb_output * 255).astype(np.uint8)
        output_pil = Image.fromarray(output_img)
        
        # Resize back to original size if needed
        if original_size != (256, 256):
            output_pil = output_pil.resize(original_size, Image.BILINEAR)
        
        # Save colorized image
        output_path = os.path.join(output_folder, f"colorized_{img_name}")
        output_pil.save(output_path)
        
        print(f"[{i}/{len(image_files)}] ✓ Colorized: {img_name}")
        
        # Create comparison image for first 5 images
        if i <= 5:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Grayscale input
            axes[0].imshow(L, cmap='gray')
            axes[0].set_title('Grayscale Input (L channel)', fontsize=12)
            axes[0].axis('off')
            
            # Original image
            axes[1].imshow(img_np)
            axes[1].set_title('Original Image', fontsize=12)
            axes[1].axis('off')
            
            # Colorized output
            axes[2].imshow(rgb_output)
            axes[2].set_title('Colorized Output', fontsize=12)
            axes[2].axis('off')
            
            plt.suptitle(f'Colorization Result: {img_name}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            comparison_path = os.path.join(output_folder, f"comparison_{i}_{img_name}")
            plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"     → Comparison saved: comparison_{i}_{img_name}")
    
    except Exception as e:
        print(f"[{i}/{len(image_files)}]  Error processing {img_name}: {str(e)}")
        continue

print("="*50)
print(f"\n Colorization complete!")
print(f" Output folder: {output_folder}/")
print(f" Colorized images: colorized_*.jpg/png")
print(f" Comparison images: comparison_*.jpg/png (first 5 images)")
