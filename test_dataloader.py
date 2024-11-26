# test_dataloader.py
from torch.utils.data import DataLoader
from config import Config
from utils.dataset import ReflectionDataset

def test_dataloader():
    # 데이터셋 생성
    dataset = ReflectionDataset(
        reflect_dir=Config.REFLECT_DIR,
        clean_dir=Config.CLEAN_DIR
    )
    
    # 데이터 로더 생성
    dataloader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS
    )
    
    # 데이터 확인
    for batch in dataloader:
        reflect_imgs = batch['reflection']
        clean_imgs = batch['clean']
        filenames = batch['filename']
        
        print("Batch information:")
        print(f"Reflection images shape: {reflect_imgs.shape}")
        print(f"Clean images shape: {clean_imgs.shape}")
        print(f"Filenames: {filenames}")
        break

if __name__ == "__main__":
    test_dataloader()