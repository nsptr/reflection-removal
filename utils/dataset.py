# utils/dataset.py
import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ReflectionDataset(Dataset):
    def __init__(self, reflect_dir, clean_dir, transform=None):
        self.reflect_dir = reflect_dir
        self.clean_dir = clean_dir
        
        # 기본 transform 설정
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                  std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform
            
        # 이미지 파일 리스트 가져오기
        self.image_files = sorted(list(reflect_dir.glob("*.jpg")))
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 파일 경로
        reflect_path = self.image_files[idx]
        # clean 이미지는 같은 이름을 가진 파일
        clean_path = self.clean_dir / reflect_path.name
        
        # 이미지 로드
        reflect_img = Image.open(reflect_path).convert('RGB')
        clean_img = Image.open(clean_path).convert('RGB')
        
        # Transform 적용
        if self.transform:
            reflect_img = self.transform(reflect_img)
            clean_img = self.transform(clean_img)
            
        return {
            'reflection': reflect_img,
            'clean': clean_img,
            'filename': reflect_path.name
        }