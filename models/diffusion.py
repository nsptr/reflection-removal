# models/diffusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

class DiffusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.DEVICE
        
        # Define beta schedule
        self.beta = self._linear_beta_schedule(
            timesteps=config.TIMESTEPS,
            beta_start=config.BETA_START,
            beta_end=config.BETA_END
        )
        
        # Pre-calculate diffusion parameters
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat)
        
    def _linear_beta_schedule(self, timesteps, beta_start, beta_end):
        return torch.linspace(beta_start, beta_end, timesteps, device=self.device)
    
    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    
    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process"""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alpha_hat_t = self._extract(self.sqrt_alpha_hat, t, x_start.shape)
        sqrt_one_minus_alpha_hat_t = self._extract(self.sqrt_one_minus_alpha_hat, t, x_start.shape)
        
        return sqrt_alpha_hat_t * x_start + sqrt_one_minus_alpha_hat_t * noise
    
    def p_losses(self, denoise_model, x_start, t, noise=None, loss_type="l2"):
        """Calculate loss for training"""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = denoise_model(x_noisy, t)
        
        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()
            
        return loss
    
    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        """Single step of reverse diffusion sampling"""
        betas_t = self._extract(self.beta, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alpha_hat, t, x.shape)
        sqrt_recip_alphas_t = self._extract(torch.sqrt(1.0 / self.alpha), t, x.shape)
        
        # Equation 11 in the paper
        # Use model to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self._extract(self.beta, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        """Complete reverse diffusion sampling"""
        device = next(model.parameters()).device
        b = shape[0]
        
        # Start from pure noise
        img = torch.randn(shape, device=device)
        imgs = []
        
        for i in tqdm(reversed(range(0, self.config.TIMESTEPS)), 
                     desc='sampling loop time step', total=self.config.TIMESTEPS):
            img = self.p_sample(
                model,
                img,
                torch.full((b,), i, device=device, dtype=torch.long),
                i
            )
            imgs.append(img.cpu())
        return imgs
    
    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=3):
        """Generate new samples"""
        return self.p_sample_loop(
            model,
            shape=(batch_size, channels, image_size, image_size)
        )

class ConditionedDiffusionModel(DiffusionModel):
    """Extends DiffusionModel for conditional generation (e.g., reflection removal)"""
    def __init__(self, config):
        super().__init__(config)
    
    def p_losses(self, denoise_model, x_start, condition, t, noise=None, loss_type="l2"):
        """Calculate loss for training with conditioning"""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = denoise_model(x_noisy, condition, t)
        
        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()
            
        return loss
    
    @torch.no_grad()
    def p_sample(self, model, x, condition, t, t_index):
        """Single step of reverse diffusion sampling with conditioning"""
        betas_t = self._extract(self.beta, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alpha_hat, t, x.shape)
        sqrt_recip_alphas_t = self._extract(torch.sqrt(1.0 / self.alpha), t, x.shape)
        
        model_output = model(x, condition, t)
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self._extract(self.beta, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise