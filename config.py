# config.py
import torch
from pathlib import Path

class Config:
    # Paths
    PROJECT_ROOT = Path("/data/reflection")
    DATA_ROOT = Path("/data/reflection/data/real")
    REFLECT_DIR = DATA_ROOT / "blended"
    CLEAN_DIR = DATA_ROOT / "transmission_layer"
    
    # Directory for saving
    CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
    LOG_DIR = PROJECT_ROOT / "logs"
    
    # Model Configuration
    IN_CHANNELS = 3
    OUT_CHANNELS = 3
    HIDDEN_DIM = 96
    NUM_HEADS = 3
    WINDOW_SIZE = 8
    DEPTHS = [2, 2, 6, 2]
    PATCH_SIZE = 4
    # Print model dimensions for debugging
    @classmethod
    def print_dims(cls):
        print(f"Initial hidden dim: {cls.HIDDEN_DIM}")
        for i in range(len(cls.DEPTHS)):
            dim = cls.HIDDEN_DIM * (2 ** i)
            print(f"Stage {i} dim: {dim}")
    # Training Configuration
    DEVICE = "cuda" #if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 4
    NUM_EPOCHS = 100
    LEARNING_RATE = 2e-5
    NUM_WORKERS = 4
    
    # Diffusion Configuration
    TIMESTEPS = 1000
    BETA_START = 0.0001
    BETA_END = 0.02
    
    # Wandb Configuration
    WANDB_PROJECT = "reflection-removal"
    WANDB_ENTITY = "bigchoi3449"  # your wandb username
    WANDB_ENABLED = True