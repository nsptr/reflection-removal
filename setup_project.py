# setup_project.py
import os

def create_project_structure():
    # Main directories
    directories = [
        'data/train/reflected',
        'data/train/clean',
        'data/val/reflected',
        'data/val/clean',
        'models',
        'utils',
        'checkpoints',
        'logs'
    ]
    
    # Create directories
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    # Create empty Python files
    files = [
        'models/__init__.py',
        'models/swin_transformer.py',
        'models/diffusion.py',
        'utils/__init__.py',
        'utils/dataset.py',
        'utils/training.py',
        'config.py',
        'train.py',
        'requirements.txt'
    ]
    
    # Create files
    for file_path in files:
        with open(file_path, 'a') as f:
            pass
        print(f"Created file: {file_path}")

if __name__ == "__main__":
    create_project_structure()