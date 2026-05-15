import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_loader_lab import LabDataset
from model_lab import ColorizationCNN
import os
import numpy as np

# Config
num_epochs = 100
batch_size = 8
learning_rate = 0.0001  # Lower learning rate
train_folder = "images"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
print(f"Training configuration:")
print(f"  - Epochs: {num_epochs}")
print(f"  - Batch size: {batch_size}")
print(f"  - Learning rate: {learning_rate}")
print(f"  - Training folder: {train_folder}")

# Check if training folder exists and has images
if not os.path.exists(train_folder):
    print(f" Error: '{train_folder}' folder not found!")
    exit()

# Dataset and Loader
dataset = LabDataset(train_folder, size=256)
print(f"\n✓ Loaded {len(dataset)} images from '{train_folder}'")

if len(dataset) < 50:
    print(f"  Warning: Dataset is very small ({len(dataset)} images). Consider adding more images for better results.")

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Model
model = ColorizationCNN().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model initialized with {total_params:,} parameters")

# Loss and Optimizer
criterion_mse = nn.MSELoss()
criterion_l1 = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

# Learning rate scheduler (removed verbose parameter)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

os.makedirs("outputs", exist_ok=True)

print("\n" + "="*60)
print("Starting Training...")
print("="*60)

best_loss = float('inf')
current_lr = learning_rate

for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    running_mse = 0.0
    running_l1 = 0.0
    
    for batch_idx, (L, ab) in enumerate(loader):
        L, ab = L.to(device), ab.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(L)
        
        # Combined loss: MSE + L1 for better stability
        loss_mse = criterion_mse(output, ab)
        loss_l1 = criterion_l1(output, ab)
        loss = loss_mse + 0.1 * loss_l1
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        running_mse += loss_mse.item()
        running_l1 += loss_l1.item()
    
    # Calculate average losses
    avg_loss = running_loss / len(loader)
    avg_mse = running_mse / len(loader)
    avg_l1 = running_l1 / len(loader)
    
    # Print progress
    print(f"Epoch [{epoch:3d}/{num_epochs}] | Loss: {avg_loss:.4f} | MSE: {avg_mse:.4f} | L1: {avg_l1:.4f}")
    
    # Learning rate scheduling
    old_lr = current_lr
    scheduler.step(avg_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    # Print learning rate changes
    if old_lr != current_lr:
        print(f"  → Learning rate reduced from {old_lr:.6f} to {current_lr:.6f}")
    
    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "color_model_lab.pth")
        print(f"  → Best model saved! (Loss: {best_loss:.4f})")
    
    # Save checkpoint every 25 epochs
    if epoch % 25 == 0:
        checkpoint_path = f"color_model_lab_epoch{epoch}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"  → Checkpoint saved: {checkpoint_path}")
    
    # Validation check every 10 epochs
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            sample_L, sample_ab = next(iter(loader))
            sample_L = sample_L.to(device)
            sample_output = model(sample_L)
            
            # Check if output is reasonable
            mean_a = sample_output[:, 0, :, :].mean().item()
            mean_b = sample_output[:, 1, :, :].mean().item()
            std_a = sample_output[:, 0, :, :].std().item()
            std_b = sample_output[:, 1, :, :].std().item()
            
            print(f"  → Sample stats: a_mean={mean_a:.3f}, b_mean={mean_b:.3f}, a_std={std_a:.3f}, b_std={std_b:.3f}")
            
            # Warning if predictions are extreme
            if abs(mean_a) > 0.8 or abs(mean_b) > 0.8:
                print(f"   Warning: Extreme color predictions detected!")
        
        model.train()

print("\n" + "="*60)
print("✅ Training Complete!")
print(f"Best Loss: {best_loss:.4f}")
print(f"Model saved as: color_model_lab.pth")
print("="*60)
