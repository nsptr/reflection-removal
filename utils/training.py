# utils/training.py
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import time
import logging
from datetime import datetime
import wandb

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Trainer:
    def __init__(self, model, diffusion, config, train_loader, val_loader=None):
        self.model = model
        self.diffusion = diffusion
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.device = config.DEVICE
        self.epochs = config.NUM_EPOCHS
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=0.01
        )
        
        # Setup learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs
        )
        
        # Create directories
        self.checkpoint_dir = Path(config.CHECKPOINT_DIR)
        self.log_dir = Path(config.LOG_DIR)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize wandb
        if config.WANDB_ENABLED:
            self.setup_wandb()
    
    def setup_wandb(self):
        """Initialize wandb project"""
        wandb.init(
            project=self.config.WANDB_PROJECT,
            entity=self.config.WANDB_ENTITY,
            config={
                "learning_rate": self.config.LEARNING_RATE,
                "epochs": self.config.NUM_EPOCHS,
                "batch_size": self.config.BATCH_SIZE,
                "model_dim": self.config.HIDDEN_DIM,
                "num_heads": self.config.NUM_HEADS,
                "window_size": self.config.WINDOW_SIZE,
                "timesteps": self.config.TIMESTEPS
            }
        )
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train(self):
        """Main training loop"""
        best_loss = float('inf')
        
        for epoch in range(1, self.epochs + 1):
            # Train for one epoch
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate(epoch) if self.val_loader else None
            
            # Log metrics
            metrics = {"train_loss": train_loss}
            if val_loss is not None:
                metrics["val_loss"] = val_loss
            
            if self.config.WANDB_ENABLED:
                wandb.log(metrics)
            
            self.logger.info(f"Epoch {epoch} - Train Loss: {train_loss:.6f}" + 
                           (f" - Val Loss: {val_loss:.6f}" if val_loss is not None else ""))
            
            # Save checkpoint
            if val_loss is not None and val_loss < best_loss:
                best_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
            
            if epoch % 10 == 0:
                self.save_checkpoint(epoch)
            
            # Update learning rate
            self.scheduler.step()
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.NUM_EPOCHS}"):
            clean_imgs = batch['clean'].to(self.device)
            reflect_imgs = batch['reflection'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # 랜덤 타임스텝 생성
            t = torch.randint(0, self.config.TIMESTEPS, (clean_imgs.shape[0],), device=self.device)
            
            # loss_type 인자 제거
            loss = self.diffusion.p_losses(
                denoise_model=self.model,
                x_start=clean_imgs,
                condition=reflect_imgs,
                t=t
            )
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                clean_imgs = batch['clean'].to(self.device)
                reflect_imgs = batch['reflection'].to(self.device)
                
                t = torch.randint(0, self.config.TIMESTEPS, (clean_imgs.shape[0],), device=self.device)
                
                # loss_type 인자 제거
                loss = self.diffusion.p_losses(
                    denoise_model=self.model,
                    x_start=clean_imgs,
                    condition=reflect_imgs,
                    t=t
                )
                
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        
        if is_best:
            path = self.checkpoint_dir / 'best_model.pth'
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint: {path}")
        
        if self.config.WANDB_ENABLED:
            wandb.save(str(path))