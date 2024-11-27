# train.py
import torch
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import math

from config import Config
from models.swin_transformer import SwinTransformerBlock
from models.diffusion import ConditionedDiffusionModel
from utils.dataset import ReflectionDataset
from utils.training import Trainer

class SinusoidalPositionEmbeddings(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ReflectionRemovalModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        
        dims = [96, 192, 384, 768]
        depths = [2, 2, 6, 2]
        num_heads = [3, 6, 12, 24]
        window_size = 8
        
        # print("\nModel dimensions:")
        # print(f"Initial hidden dim: {dims[0]}")
        # for i, dim in enumerate(dims):
        #     print(f"Stage {i} dim: {dim}")
        # print("\n")
        
        # Initial projection
        self.patch_embed = torch.nn.Conv2d(6, dims[0], 1)
        
        # Time embedding
        self.time_mlp = torch.nn.Sequential(
            SinusoidalPositionEmbeddings(config.HIDDEN_DIM),
            torch.nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM),
            torch.nn.GELU()
        )
        
        # Encoder path
        self.encoder_blocks = torch.nn.ModuleList([
            StageBlock(dims[0], depths[0], num_heads[0], window_size, config, is_encoder=True),
            StageBlock(dims[1], depths[1], num_heads[1], window_size, config, is_encoder=True),
            StageBlock(dims[2], depths[2], num_heads[2], window_size, config, is_encoder=True),
            StageBlock(dims[3], depths[3], num_heads[3], window_size, config, is_encoder=True)
        ])
        
        # Middle block
        self.middle_block = StageBlock(dims[3], depths[3], num_heads[3], window_size, config)
        
        # Decoder path
        self.decoder_blocks = torch.nn.ModuleList([
            StageBlock(dims[2], depths[2], num_heads[2], window_size, config, is_decoder=True),
            StageBlock(dims[1], depths[1], num_heads[1], window_size, config, is_decoder=True),
            StageBlock(dims[0], depths[0], num_heads[0], window_size, config, is_decoder=True)
        ])
        
        # Final layer
        self.final_layer = torch.nn.Conv2d(96, 3, 1)
        
    def forward(self, x, condition, t):
        # print("\nForward pass:")
        
        # Input processing
        x = torch.cat([x, condition], dim=1)
        x = self.patch_embed(x)
        t_emb = self.time_mlp(t)
        # print(f"After initial processing: {x.shape}")
        
        # Encoder
        skip_connections = []
        for i, block in enumerate(self.encoder_blocks):
            x = block(x, t_emb)
            if i < len(self.encoder_blocks) - 1:  # 마지막 블록의 출력은 저장하지 않음
                skip_connections.append(x)
                # print(f"Saved skip connection {i}: {x.shape}")
        
        # Middle
        x = self.middle_block(x, t_emb)
        # print(f"After middle block: {x.shape}")
        
        # Decoder with skip connections
        for i, (block, skip) in enumerate(zip(self.decoder_blocks, reversed(skip_connections))):
            # print(f"\nProcessing decoder block {i}")
            # print(f"Current feature shape: {x.shape}")
            # print(f"Skip connection shape: {skip.shape}")
            x = torch.cat([x, skip], dim=1)
            # print(f"After concatenation: {x.shape}")
            x = block(x, t_emb)
            # print(f"After decoder block: {x.shape}")
        
        # Final layer
        x = self.final_layer(x)
        # print(f"Final output: {x.shape}")
        
        return x

class StageBlock(torch.nn.Module):
    def __init__(self, dim, depth, num_heads, window_size, config,
                 is_encoder=False, is_decoder=False):
        super().__init__()
        self.dim = dim
        self.is_encoder = is_encoder
        self.is_decoder = is_decoder
        
        # print(f"Initializing StageBlock: dim={dim}, {'encoder' if is_encoder else 'decoder' if is_decoder else 'middle'}")
        
        # Channel adjustment for decoder
        if is_decoder:
            if dim == 384:
                input_dim = 768 + 768  # 768(previous) + 768(skip)
            elif dim == 192:
                input_dim = 192 + 384  # 192(previous) + 384(skip)
            else:  # dim == 96
                input_dim = 96 + 192   # 96(previous) + 192(skip)
                
            self.channel_proj = torch.nn.Sequential(
                torch.nn.Conv2d(input_dim, dim, 1),
                torch.nn.GELU()
            )
            # print(f"Decoder channel projection: {input_dim} -> {dim}")
            
            # Spatial upsampling
            if dim == 96:  # 마지막 decoder block
                self.spatial_change = torch.nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2)
                # print(f"Decoder spatial change (final): {dim} -> {dim}")
            else:
                self.spatial_change = torch.nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)
                # print(f"Decoder spatial change: {dim} -> {dim // 2}")
        elif is_encoder and dim != 768:
            self.channel_proj = None
            self.spatial_change = torch.nn.Conv2d(dim, dim * 2, kernel_size=2, stride=2)
            # print(f"Encoder spatial change: {dim} -> {dim * 2}")
        else:
            self.channel_proj = None
            self.spatial_change = None
            # print("No spatial change")
        
        # Time embedding
        self.time_proj = torch.nn.Sequential(
            torch.nn.Linear(config.HIDDEN_DIM, dim),
            torch.nn.GELU()
        )
        
        # Transformer blocks
        self.blocks = torch.nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2
            ) for i in range(depth)
        ])
            
    def forward(self, x, t_emb):
        B, C, H, W = x.shape
        # print(f"StageBlock input shape: {x.shape}, {'encoder' if self.is_encoder else 'decoder' if self.is_decoder else 'middle'}")
        
        # Channel projection for decoder
        if self.channel_proj is not None:
            x = self.channel_proj(x)
            # print(f"After channel projection: {x.shape}")
        
        # Time embedding
        t = self.time_proj(t_emb)
        t = t.view(-1, self.dim, 1, 1).expand(-1, -1, H, W)
        x = x + t
        # print(f"After time embedding: {x.shape}")
        
        # Transformer blocks
        x_flat = x.flatten(2).transpose(1, 2)
        for block in self.blocks:
            x_flat = block(x_flat)
        x = x_flat.transpose(1, 2).reshape(B, -1, H, W)
        # print(f"After transformer blocks: {x.shape}")
        
        # Spatial change
        if self.spatial_change is not None:
            x = self.spatial_change(x)
            # print(f"After spatial change: {x.shape}")
        
        return x

def get_args():
    parser = argparse.ArgumentParser(description='Train Reflection Removal Model')
    parser.add_argument('--resume', type=str, help='path to checkpoint to resume from')
    parser.add_argument('--batch-size', type=int, default=Config.BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=Config.NUM_EPOCHS)
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE)
    return parser.parse_args()

def main():
    args = get_args()
    # print("\nModel dimensions:")
    Config.print_dims()
    # print("\n")
    # Update config with command line arguments
    Config.BATCH_SIZE = args.batch_size
    Config.NUM_EPOCHS = args.epochs
    Config.LEARNING_RATE = args.lr
    
    # Create datasets
    train_dataset = ReflectionDataset(
        reflect_dir=Config.REFLECT_DIR,
        clean_dir=Config.CLEAN_DIR,
        transform=None
    )
    
    # Split train dataset into train and validation sets
    total_size = len(train_dataset)
    train_size = int(0.9 * total_size)  # 90% for training
    val_size = total_size - train_size   # 10% for validation
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    # Initialize models
    model = ReflectionRemovalModel(Config).to(Config.DEVICE)
    
    # Initialize diffusion
    diffusion = ConditionedDiffusionModel(Config).to(Config.DEVICE)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        diffusion=diffusion,
        config=Config,
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f'Resumed from epoch {start_epoch}')
    
    # Start training
    trainer.train()

if __name__ == '__main__':
    main()