import torch
from model_lab import ColorizationCNN
from PIL import Image
import numpy as np
from skimage import color
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*60)
print("Model Debug Script")
print("="*60)

# Load model
try:
    model = ColorizationCNN().to(device)
    model.load_state_dict(torch.load("color_model_lab.pth", map_location=device))
    model.eval()
    print(" Model loaded successfully")
except:
    print(" Could not load model. Train first!")
    exit()

test_folder = "images"
test_files = [f for f in os.listdir(test_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if len(test_files) == 0:
    print(" No test images found!")
    exit()

img_path = os.path.join(test_folder, test_files[0])
print(f"\nTesting with: {test_files[0]}")

# Load image
img = Image.open(img_path).convert("RGB")
img = img.resize((256, 256))
img_np = np.array(img) / 255.0

# Convert to Lab
lab = color.rgb2lab(img_np)
L = lab[:, :, 0] / 100.0
ab_true = lab[:, :, 1:] / 128.0

print(f"\nGround Truth Statistics:")
print(f"  a channel: min={ab_true[:,:,0].min():.3f}, max={ab_true[:,:,0].max():.3f}, mean={ab_true[:,:,0].mean():.3f}")
print(f"  b channel: min={ab_true[:,:,1].min():.3f}, max={ab_true[:,:,1].max():.3f}, mean={ab_true[:,:,1].mean():.3f}")

L_tensor = torch.from_numpy(L).unsqueeze(0).unsqueeze(0).float().to(device)

with torch.no_grad():
    ab_pred = model(L_tensor).cpu().numpy()[0]

print(f"\n Model Prediction Statistics:")
print(f"  a channel: min={ab_pred[0].min():.3f}, max={ab_pred[0].max():.3f}, mean={ab_pred[0].mean():.3f}")
print(f"  b channel: min={ab_pred[1].min():.3f}, max={ab_pred[1].max():.3f}, mean={ab_pred[1].mean():.3f}")

# Check if model is predicting correctly
if ab_pred[0].std() < 0.01 and ab_pred[1].std() < 0.01:
    print("\n  WARNING: Model is predicting nearly constant values!")
    print("   This means the model hasn't learned properly.")
    print("   Solution: Retrain with the updated training script.")

if abs(ab_pred[0].mean()) > 0.7 or abs(ab_pred[1].mean()) > 0.7:
    print("\n WARNING: Model is predicting extreme color values!")
    print("   This will cause images to be too red/green/blue.")
    print("   Solution: Retrain with gradient clipping and lower learning rate.")

# Reconstruct image
ab_pred_unnorm = ab_pred.transpose((1, 2, 0)) * 128.0
lab_output = np.zeros((256, 256, 3))
lab_output[:, :, 0] = L * 100.0
lab_output[:, :, 1:] = ab_pred_unnorm

rgb_output = color.lab2rgb(lab_output)
rgb_output = np.clip(rgb_output, 0, 1)

# Visualize
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Row 1: Images
axes[0, 0].imshow(L, cmap='gray')
axes[0, 0].set_title('Grayscale Input')
axes[0, 0].axis('off')

axes[0, 1].imshow(img_np)
axes[0, 1].set_title('Original Image')
axes[0, 1].axis('off')

axes[0, 2].imshow(rgb_output)
axes[0, 2].set_title('Model Output')
axes[0, 2].axis('off')

# Row 2: ab channels
axes[1, 0].imshow(ab_true[:,:,0], cmap='RdBu_r', vmin=-1, vmax=1)
axes[1, 0].set_title('True a channel')
axes[1, 0].axis('off')

axes[1, 1].imshow(ab_pred[0], cmap='RdBu_r', vmin=-1, vmax=1)
axes[1, 1].set_title('Predicted a channel')
axes[1, 1].axis('off')

axes[1, 2].imshow(ab_pred[1] - ab_true[:,:,1], cmap='RdBu_r', vmin=-1, vmax=1)
axes[1, 2].set_title('Prediction Error')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('debug_output.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Debug visualization saved as 'debug_output.png'")

# Final recommendation
print("\n" + "="*60)
print("RECOMMENDATIONS:")
print("="*60)

if ab_pred[0].std() < 0.01:
    print("Model not learning - RETRAIN with updated scripts")
elif abs(ab_pred[0].mean()) > 0.7:
    print(" Extreme predictions - RETRAIN with gradient clipping")
elif abs(ab_pred[0].mean() - ab_true[:,:,0].mean()) > 0.3:
    print(" Model biased - train longer or add more diverse data")
else:
    print("✓ Model seems OK - but may need more training epochs")

print("\nTo fix:")
print("1. Delete old color_model_lab.pth")
print("2. Replace model_lab.py with the fixed version")
print("3. Replace train_lab.py with the fixed version")
print("4. Run: python train_lab.py")
print("5. Wait for at least 50-100 epochs")
print("="*60)