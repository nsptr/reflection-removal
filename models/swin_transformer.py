# models/swin_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = 4
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim=dim,
            window_size=(window_size, window_size),
            num_heads=num_heads
        )
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * self.mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * self.mlp_ratio, dim)
        )
    
    def forward(self, x):
        B, L, C = x.shape
        H = W = int(np.sqrt(L))
        
        shortcut = x
        x = self.norm1(x)
        x = x.reshape(B, H, W, C).contiguous()
        
        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        # Partition windows and apply attention
        x_windows, (H_pad, W_pad, pad_h, pad_w) = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.reshape(-1, self.window_size * self.window_size, C).contiguous()
        
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)
        
        # Merge windows
        attn_windows = attn_windows.reshape(-1, self.window_size, self.window_size, C).contiguous()
        shifted_x = window_reverse(attn_windows, self.window_size, H_pad, W_pad, pad_h, pad_w)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        x = x.reshape(B, H * W, C).contiguous()
        
        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        
        return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    
    # Check if padding is needed
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        H, W = H + pad_h, W + pad_w
    
    x = x.reshape(B, H // window_size, window_size, W // window_size, window_size, C).contiguous()
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.reshape(-1, window_size, window_size, C)
    return windows, (H, W, pad_h, pad_w)

def window_reverse(windows, window_size, H, W, pad_h=0, pad_w=0):
    B = int(windows.shape[0] / ((H * W) / (window_size * window_size)))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1).contiguous()
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.reshape(B, H, W, -1)
    
    if pad_h > 0 or pad_w > 0:
        x = x[:, :H-pad_h, :W-pad_w, :]
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Define the linear layers
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        
        # Define relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        
        # Get pair-wise relative position index
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        
    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        
        return x