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
        self.beta = torch.linspace(config.BETA_START, config.BETA_END, config.TIMESTEPS).to(self.device)
        
        # Pre-calculate diffusion parameters
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat)
        
    def _extract(self, a, t, x_shape):
        """Extract some coefficients at specified timesteps"""
        batch_size = x_shape[0]
        # t를 현재 디바이스로 이동
        t = t.to(a.device)
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process"""
        if noise is None:
            noise = torch.randn_like(x_start, device=x_start.device)
            
        # 모든 텐서가 같은 디바이스에 있도록 보장
        sqrt_alpha_hat_t = self._extract(self.sqrt_alpha_hat.to(x_start.device), t, x_start.shape)
        sqrt_one_minus_alpha_hat_t = self._extract(self.sqrt_one_minus_alpha_hat.to(x_start.device), t, x_start.shape)
        
        return sqrt_alpha_hat_t * x_start + sqrt_one_minus_alpha_hat_t * noise
    
    def p_losses(self, denoise_model, x_start, condition, t, noise=None):
        """Calculate loss with noise prediction and reconstruction"""
        if noise is None:
            noise = torch.randn_like(x_start, device=x_start.device)
        
        # 모든 입력을 동일한 디바이스로
        x_start = x_start.to(self.device)
        condition = condition.to(self.device)
        t = t.to(self.device)
        noise = noise.to(self.device)
        
        # Forward diffusion
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        # 노이즈 예측
        predicted_noise = denoise_model(x_noisy, condition, t)
        
        # 1. 노이즈 예측 loss (주요 loss)
        noise_loss = F.mse_loss(noise, predicted_noise)
        
        # 2. 이미지 복원 loss (보조 loss)
        # 예측된 노이즈로부터 원본 이미지 복원 시도
        alpha = self._extract(self.alpha, t, x_start.shape)
        alpha_hat = self._extract(self.alpha_hat, t, x_start.shape)
        
        # 복원된 이미지 계산
        pred_x0 = (x_noisy - predicted_noise * (1 - alpha).sqrt()) / alpha.sqrt()
        
        # 복원 loss
        recon_loss = F.l1_loss(x_start, pred_x0)
        
        # 최종 loss (가중치 조정 가능)
        total_loss = noise_loss + 0.01 * recon_loss
        
        return total_loss
    
    def _predict_x0_from_noise(self, x_t, t, noise):
        # 노이즈로부터 원본 이미지 예측
        alpha_hat = self._extract(self.alpha_hat, t, x_t.shape)
        sqrt_one_minus_alpha_hat = self._extract(self.sqrt_one_minus_alpha_hat, t, x_t.shape)
        
        return (x_t - sqrt_one_minus_alpha_hat * noise) / torch.sqrt(alpha_hat)
    
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
    
    def p_losses(self, denoise_model, x_start, condition, t, noise=None):
        """Calculate loss for training with conditioning"""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        # 기존 노이즈 예측 loss
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = denoise_model(x_noisy, condition, t)
        noise_loss = F.mse_loss(noise, predicted_noise)
        
        # 추가: 이미지 복원 loss
        # 예측된 노이즈를 사용하여 이미지 복원
        pred_x0 = self._predict_x0_from_noise(x_noisy, t, predicted_noise)
        reconstruction_loss = F.mse_loss(x_start, pred_x0)
        
        # 최종 loss는 두 loss의 조합
        total_loss = noise_loss + 0.1 * reconstruction_loss
        
        return total_loss
    
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