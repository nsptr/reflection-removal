# utils/dataset.py
import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ReflectionDataset(Dataset):
    def __init__(self, reflect_dir, clean_dir):
        self.reflect_dir = reflect_dir
        self.clean_dir = clean_dir
        self.image_pairs = []
        
        print(f"Searching for images in: {reflect_dir}")
        
        for root, dirs, files in os.walk(reflect_dir):
            for file in files:
                if file.endswith('.png') or file.endswith('.jpg'):
                    mixture_path = os.path.join(root, file)
                    clean_path = os.path.join(clean_dir, file)
                    
                    if os.path.exists(clean_path):
                        self.image_pairs.append((mixture_path, clean_path))
        
        print(f"Found {len(self.image_pairs)} image pairs")
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        mixture_path, clean_path = self.image_pairs[idx]
        
        # 이미지 로드
        mixture_img = Image.open(mixture_path).convert('RGB')
        clean_img = Image.open(clean_path).convert('RGB')
        
        # Transform 적용
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        mixture_img = transform(mixture_img)
        clean_img = transform(clean_img)
        
        return {
            'reflection': mixture_img,
            'clean': clean_img,
            'paths': (mixture_path, clean_path)  # 디버깅을 위해 경로도 반환
        }