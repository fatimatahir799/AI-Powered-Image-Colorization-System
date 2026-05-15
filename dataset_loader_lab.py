import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from skimage import color

class LabDataset(Dataset):
    def __init__(self, folder, size=256):
        self.files = [os.path.join(folder, f) for f in os.listdir(folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.size = size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        # Load and resize
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.size, self.size), Image.BILINEAR)
        img_np = np.array(img) / 255.0  
        
        # Convert RGB to Lab using skimage 
        lab = color.rgb2lab(img_np)
        
        # Split channels
        L = lab[:, :, 0]  # L channel: 0 to 100
        ab = lab[:, :, 1:]  # ab channels: -128 to 127
        
        # Normalize L to [0, 1]
        L = L / 100.0
        
        # Normalize ab to [-1, 1]
        ab = ab / 128.0
        
        # Convert to tensors
        L_tensor = torch.from_numpy(L).unsqueeze(0).float()  
        ab_tensor = torch.from_numpy(ab.transpose((2, 0, 1))).float() 
        
        return L_tensor, ab_tensor