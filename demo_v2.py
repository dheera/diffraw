"""
Differentiable RAW Processor v2
===============================
A sophisticated RAW processing pipeline using aesthetic models as judges.

Features:
- AgX-style filmic tone mapping with proper highlight rolloff
- HSL per-channel adjustments
- Split toning (shadows/highlights)
- Clarity/texture and Orton effect
- Skin tone protection
- LAION Aesthetic Predictor + optional CLIP for style
- Parameter bounds to keep edits photographically reasonable
"""

import math
import rawpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch.optim as optim
import argparse
import imageio
from pathlib import Path
import cv2

# We'll handle imports gracefully
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: CLIP not available. Style guidance disabled.")

#############################
# RAW Importer
#############################

class RawImporter:
    """
    Imports a RAW file using rawpy with minimal processing,
    preserving maximum dynamic range for our pipeline.
    """
    def __init__(self, target_size=(768, 768)):
        self.target_size = target_size

    def import_raw(self, file_path):
        with rawpy.imread(file_path) as raw:
            # Minimal processing - we want linear data
            rgb = raw.postprocess(
                output_bps=16,
                no_auto_bright=True,
                use_camera_wb=True,  # Use camera WB as starting point
                gamma=(1, 1),  # Linear output
                output_color=rawpy.ColorSpace.sRGB,
            )
        # Normalize to [0, 1] and convert to torch tensor (C, H, W)
        img = torch.from_numpy(rgb.astype(np.float32) / 65535.0).permute(2, 0, 1)
        img = transforms.functional.resize(img, self.target_size, antialias=True)
        return img


#############################
# Utility Functions
#############################

def rgb_to_hsl(rgb):
    """Convert RGB to HSL. Input shape: (B, 3, H, W), values in [0, 1]"""
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    
    max_c = torch.max(rgb, dim=1, keepdim=True)[0]
    min_c = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = max_c - min_c
    
    # Lightness
    l = (max_c + min_c) / 2
    
    # Saturation
    s = torch.where(
        delta < 1e-6,
        torch.zeros_like(delta),
        delta / (1 - torch.abs(2 * l - 1) + 1e-6)
    )
    
    # Hue
    h = torch.zeros_like(r)
    
    # When max is R
    mask_r = (max_c == r) & (delta > 1e-6)
    h = torch.where(mask_r, ((g - b) / (delta + 1e-6)) % 6, h)
    
    # When max is G
    mask_g = (max_c == g) & (delta > 1e-6)
    h = torch.where(mask_g, (b - r) / (delta + 1e-6) + 2, h)
    
    # When max is B
    mask_b = (max_c == b) & (delta > 1e-6)
    h = torch.where(mask_b, (r - g) / (delta + 1e-6) + 4, h)
    
    h = h / 6  # Normalize to [0, 1]
    h = h % 1  # Wrap around
    
    return torch.cat([h, s, l], dim=1)


def hsl_to_rgb(hsl):
    """Convert HSL to RGB. Input shape: (B, 3, H, W), H in [0,1], S in [0,1], L in [0,1]"""
    h, s, l = hsl[:, 0:1], hsl[:, 1:2], hsl[:, 2:3]
    
    c = (1 - torch.abs(2 * l - 1)) * s
    h_prime = h * 6
    x = c * (1 - torch.abs(h_prime % 2 - 1))
    
    zeros = torch.zeros_like(c)
    
    # Determine RGB based on hue sector
    rgb = torch.zeros_like(hsl)
    
    mask0 = (h_prime >= 0) & (h_prime < 1)
    mask1 = (h_prime >= 1) & (h_prime < 2)
    mask2 = (h_prime >= 2) & (h_prime < 3)
    mask3 = (h_prime >= 3) & (h_prime < 4)
    mask4 = (h_prime >= 4) & (h_prime < 5)
    mask5 = (h_prime >= 5) & (h_prime < 6)
    
    rgb[:, 0:1] = torch.where(mask0 | mask5, c, torch.where(mask1 | mask4, x, zeros))
    rgb[:, 1:2] = torch.where(mask1 | mask2, c, torch.where(mask0 | mask3, x, zeros))
    rgb[:, 2:3] = torch.where(mask3 | mask4, c, torch.where(mask2 | mask5, x, zeros))
    
    m = l - c / 2
    rgb = rgb + m
    
    return torch.clamp(rgb, 0, 1)


def soft_clamp(x, min_val, max_val, softness=0.1):
    """Soft clamp using sigmoid to keep gradients flowing"""
    mid = (min_val + max_val) / 2
    scale = (max_val - min_val) / 2
    return mid + scale * torch.tanh((x - mid) / (scale * softness + 1e-6))


#############################
# AgX-Style Filmic Tone Mapping
#############################

class AgXToneMapping(nn.Module):
    """
    AgX-inspired filmic tone mapping with:
    - Proper highlight rolloff (no harsh clipping)
    - Configurable contrast
    - Preserves color saturation better
    """
    def __init__(self):
        super().__init__()
        # Scene exposure adjustment (in stops)
        # Start at +1.5 stops to compensate for linear RAW being dark
        self.exposure = nn.Parameter(torch.tensor(1.5))
        # Contrast (slope at mid-gray)
        self.contrast = nn.Parameter(torch.tensor(0.0))
        # Gamma for final display transform
        self.gamma = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, x):
        # Apply exposure (in stops, so 2^exposure)
        exposure_factor = torch.pow(torch.tensor(2.0, device=x.device), 
                                    torch.clamp(self.exposure, -3, 4))
        x = x * exposure_factor
        
        # Simple gamma first to get into perceptual space
        # Linear RAW needs ~2.2 gamma to look normal
        base_gamma = 1.0 / 2.2
        x = torch.pow(x.clamp(min=1e-6), base_gamma)
        
        # Contrast adjustment (around mid-gray)
        contrast = 1.0 + torch.tanh(self.contrast) * 0.3  # [0.7, 1.3]
        x = 0.5 + (x - 0.5) * contrast
        
        # Gamma fine-tuning
        gamma_adjust = 1.0 + torch.tanh(self.gamma) * 0.2  # [0.8, 1.2]
        x = torch.pow(x.clamp(min=1e-6, max=1.0), gamma_adjust)
        
        return torch.clamp(x, 0, 1)


#############################
# HSL Adjustments
#############################

class HSLAdjustment(nn.Module):
    """
    Per-hue adjustments for 8 color ranges:
    Red, Orange, Yellow, Green, Cyan, Blue, Purple, Magenta
    
    Each range has: hue shift, saturation multiplier, luminance shift
    """
    def __init__(self):
        super().__init__()
        # 8 hue ranges, 3 adjustments each (hue_shift, sat_mult, lum_shift)
        self.hue_shifts = nn.Parameter(torch.zeros(8))
        self.sat_mults = nn.Parameter(torch.ones(8))
        self.lum_shifts = nn.Parameter(torch.zeros(8))
        
        # Hue centers (in [0, 1]): R, O, Y, G, C, B, P, M
        self.register_buffer('hue_centers', torch.tensor([
            0.0, 0.083, 0.167, 0.333, 0.5, 0.667, 0.75, 0.917
        ]))
        
    def forward(self, x):
        hsl = rgb_to_hsl(x)
        h, s, l = hsl[:, 0:1], hsl[:, 1:2], hsl[:, 2:3]
        
        # Compute weights for each hue range using soft assignment
        # Width of each hue range
        width = 0.08
        
        total_hue_shift = torch.zeros_like(h)
        total_sat_mult = torch.ones_like(s)
        total_lum_shift = torch.zeros_like(l)
        
        for i in range(8):
            center = self.hue_centers[i]
            # Handle wraparound for hue
            dist = torch.min(
                torch.abs(h - center),
                torch.min(torch.abs(h - center - 1), torch.abs(h - center + 1))
            )
            weight = torch.exp(-dist ** 2 / (2 * width ** 2))
            weight = weight * s  # Only affect saturated pixels
            
            total_hue_shift = total_hue_shift + weight * self.hue_shifts[i] * 0.1
            total_sat_mult = total_sat_mult + weight * (self.sat_mults[i] - 1)
            total_lum_shift = total_lum_shift + weight * self.lum_shifts[i] * 0.2
        
        # Apply adjustments
        h_new = (h + total_hue_shift) % 1
        s_new = torch.clamp(s * total_sat_mult, 0, 1)
        l_new = torch.clamp(l + total_lum_shift, 0, 1)
        
        hsl_new = torch.cat([h_new, s_new, l_new], dim=1)
        return hsl_to_rgb(hsl_new)


#############################
# Split Toning
#############################

class SplitToning(nn.Module):
    """
    Add color tints to shadows and highlights separately.
    Classic film look technique.
    """
    def __init__(self):
        super().__init__()
        # Shadow tint (HSL, but we'll use just HS and blend)
        self.shadow_hue = nn.Parameter(torch.tensor(0.6))  # Blue-ish default
        self.shadow_saturation = nn.Parameter(torch.tensor(-5.0))  # Start neutral (sigmoid(-5)≈0)
        
        # Highlight tint
        self.highlight_hue = nn.Parameter(torch.tensor(0.1))  # Warm default
        self.highlight_saturation = nn.Parameter(torch.tensor(-5.0))  # Start neutral (sigmoid(-5)≈0)
        
        # Balance (where shadows end and highlights begin)
        self.balance = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x):
        # Compute luminance
        luma = 0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]
        
        # Shadow and highlight masks
        balance = torch.sigmoid(self.balance)
        shadow_mask = 1 - torch.sigmoid((luma - balance) * 10)
        highlight_mask = torch.sigmoid((luma - balance) * 10)
        
        # Create tint colors
        def hue_to_rgb(hue):
            """Convert hue [0,1] to RGB unit vector"""
            h = hue * 6
            x_val = 1 - torch.abs(h % 2 - 1)
            r = torch.where(h < 1, torch.ones_like(h), 
                    torch.where(h < 2, x_val,
                    torch.where(h < 4, torch.zeros_like(h),
                    torch.where(h < 5, x_val, torch.ones_like(h)))))
            g = torch.where(h < 1, x_val,
                    torch.where(h < 3, torch.ones_like(h),
                    torch.where(h < 4, x_val, torch.zeros_like(h))))
            b = torch.where(h < 2, torch.zeros_like(h),
                    torch.where(h < 3, x_val,
                    torch.where(h < 5, torch.ones_like(h), x_val)))
            return torch.stack([r, g, b], dim=-1)
        
        shadow_rgb = hue_to_rgb(self.shadow_hue)
        highlight_rgb = hue_to_rgb(self.highlight_hue)
        
        # Apply tints
        shadow_sat = torch.sigmoid(self.shadow_saturation) * 0.3  # Max 30% tint
        highlight_sat = torch.sigmoid(self.highlight_saturation) * 0.3
        
        shadow_tint = shadow_rgb.view(1, 3, 1, 1) * shadow_sat
        highlight_tint = highlight_rgb.view(1, 3, 1, 1) * highlight_sat
        
        # Blend: add tint weighted by mask, preserving luminance
        x_tinted = x + shadow_mask * shadow_tint + highlight_mask * highlight_tint
        
        # Preserve original luminance
        new_luma = 0.2126 * x_tinted[:, 0:1] + 0.7152 * x_tinted[:, 1:2] + 0.0722 * x_tinted[:, 2:3]
        x_tinted = x_tinted * (luma / (new_luma + 1e-6))
        
        return torch.clamp(x_tinted, 0, 1)


#############################
# Clarity and Texture
#############################

class ClarityTexture(nn.Module):
    """
    Clarity: mid-frequency local contrast (like Lightroom)
    Texture: high-frequency detail enhancement
    """
    def __init__(self):
        super().__init__()
        self.clarity = nn.Parameter(torch.tensor(0.0))
        self.texture = nn.Parameter(torch.tensor(0.0))
        
        # Clarity uses larger kernel (mid-frequencies)
        self.clarity_kernel_size = 31
        self.clarity_sigma = 8
        
        # Texture uses smaller kernel (high-frequencies)
        self.texture_kernel_size = 7
        self.texture_sigma = 1.5
        
        # Pre-compute kernels
        self._init_kernels()
        
    def _init_kernels(self):
        def gaussian_kernel(size, sigma):
            grid = torch.arange(size).float() - size // 2
            gauss = torch.exp(-grid**2 / (2 * sigma * sigma))
            gauss = gauss / gauss.sum()
            kernel = gauss.unsqueeze(1) * gauss.unsqueeze(0)
            return kernel.unsqueeze(0).unsqueeze(0)
        
        self.register_buffer('clarity_kernel', 
                           gaussian_kernel(self.clarity_kernel_size, self.clarity_sigma))
        self.register_buffer('texture_kernel',
                           gaussian_kernel(self.texture_kernel_size, self.texture_sigma))
    
    def forward(self, x):
        # Work on luminance to avoid color shifts
        luma = 0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]
        
        # Clarity: enhance mid-frequencies
        clarity_pad = self.clarity_kernel_size // 2
        luma_blur_clarity = F.conv2d(luma, self.clarity_kernel, padding=clarity_pad)
        mid_freq = luma - luma_blur_clarity
        clarity_amount = torch.tanh(self.clarity) * 0.5  # [-0.5, 0.5]
        
        # Texture: enhance high-frequencies
        texture_pad = self.texture_kernel_size // 2
        luma_blur_texture = F.conv2d(luma, self.texture_kernel, padding=texture_pad)
        high_freq = luma - luma_blur_texture
        texture_amount = torch.tanh(self.texture) * 0.3  # [-0.3, 0.3]
        
        # Apply to luminance
        luma_enhanced = luma + clarity_amount * mid_freq + texture_amount * high_freq
        
        # Apply luminance change to RGB while preserving color ratios
        ratio = (luma_enhanced + 1e-6) / (luma + 1e-6)
        x_enhanced = x * ratio
        
        return torch.clamp(x_enhanced, 0, 1)


#############################
# Orton Effect
#############################

class OrtonEffect(nn.Module):
    """
    Orton effect: dreamy glow + sharpness
    Named after Michael Orton's film technique
    """
    def __init__(self):
        super().__init__()
        self.amount = nn.Parameter(torch.tensor(-5.0))  # Start disabled (sigmoid(-5)≈0)
        self.glow_size = 51
        self.glow_sigma = 15
        
        # Pre-compute glow kernel
        grid = torch.arange(self.glow_size).float() - self.glow_size // 2
        gauss = torch.exp(-grid**2 / (2 * self.glow_sigma * self.glow_sigma))
        gauss = gauss / gauss.sum()
        kernel = gauss.unsqueeze(1) * gauss.unsqueeze(0)
        self.register_buffer('glow_kernel', kernel.unsqueeze(0).unsqueeze(0))
        
    def forward(self, x):
        amount = torch.sigmoid(self.amount) * 0.4  # Max 40% effect
        
        if amount < 0.01:
            return x
        
        # Create glow layer (blur + brighten)
        glow_channels = []
        pad = self.glow_size // 2
        for c in range(3):
            channel = x[:, c:c+1]
            blurred = F.conv2d(channel, self.glow_kernel, padding=pad)
            glow_channels.append(blurred)
        glow = torch.cat(glow_channels, dim=1)
        
        # Brighten the glow slightly
        glow = glow * 1.2
        
        # Screen blend mode for dreamy effect
        result = 1 - (1 - x) * (1 - glow * amount)
        
        return torch.clamp(result, 0, 1)


#############################
# Vignette
#############################

class Vignette(nn.Module):
    """
    Optical vignette effect with controllable amount and shape
    """
    def __init__(self, image_size=(768, 768)):
        super().__init__()
        self.amount = nn.Parameter(torch.tensor(0.0))
        self.roundness = nn.Parameter(torch.tensor(0.0))  # 0 = circular, affects ellipse
        self.feather = nn.Parameter(torch.tensor(0.5))
        
        H, W = image_size
        y = torch.linspace(-1, 1, H)
        x = torch.linspace(-1, 1, W)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        self.register_buffer('xx', xx)
        self.register_buffer('yy', yy)
        
    def forward(self, x):
        amount = torch.tanh(self.amount) * 0.5  # [-0.5, 0.5]
        
        # Compute radial distance
        aspect = 1 + torch.tanh(self.roundness) * 0.3
        dist = torch.sqrt(self.xx**2 + (self.yy * aspect)**2)
        
        # Feathered falloff
        feather = 0.3 + torch.sigmoid(self.feather) * 0.7
        vignette_mask = 1 - torch.sigmoid((dist - feather) * 5) * amount
        
        vignette_mask = vignette_mask.unsqueeze(0).unsqueeze(0)
        return x * vignette_mask


#############################
# Skin Tone Protection
#############################

class SkinToneProtection(nn.Module):
    """
    Detects skin-tone-ish colors and constrains their adjustment
    to prevent unnatural skin colors.
    """
    def __init__(self):
        super().__init__()
        # Skin tone warmth adjustment (subtle)
        self.warmth = nn.Parameter(torch.tensor(0.0))
        # Skin tone smoothness (subtle blur on skin areas)
        self.smoothness = nn.Parameter(torch.tensor(-5.0))  # Start disabled (sigmoid(-5)≈0)
        
        # Skin detection kernel for smoothness
        kernel_size = 5
        sigma = 1.0
        grid = torch.arange(kernel_size).float() - kernel_size // 2
        gauss = torch.exp(-grid**2 / (2 * sigma * sigma))
        gauss = gauss / gauss.sum()
        kernel = gauss.unsqueeze(1) * gauss.unsqueeze(0)
        self.register_buffer('smooth_kernel', kernel.unsqueeze(0).unsqueeze(0))
        
    def detect_skin(self, x):
        """
        Detect skin tones using HSL ranges.
        Returns a soft mask [0, 1] where 1 = likely skin
        """
        hsl = rgb_to_hsl(x)
        h, s, l = hsl[:, 0:1], hsl[:, 1:2], hsl[:, 2:3]
        
        # Skin hue range: roughly 0-40 degrees (0-0.11 in [0,1])
        # With some orange/brown tones up to 50 degrees (0.14)
        hue_score = torch.exp(-((h - 0.05) ** 2) / (2 * 0.04 ** 2))
        
        # Skin typically has medium saturation (0.2-0.6)
        sat_score = torch.exp(-((s - 0.4) ** 2) / (2 * 0.2 ** 2))
        
        # Skin typically has medium-high lightness (0.3-0.8)
        lum_score = torch.exp(-((l - 0.55) ** 2) / (2 * 0.25 ** 2))
        
        skin_mask = hue_score * sat_score * lum_score
        return skin_mask
    
    def forward(self, x):
        skin_mask = self.detect_skin(x)
        
        # Apply subtle warmth to skin tones
        warmth = torch.tanh(self.warmth) * 0.1  # Very subtle
        warm_shift = torch.tensor([warmth * 0.05, warmth * 0.02, -warmth * 0.03], 
                                  device=x.device).view(1, 3, 1, 1)
        x_warm = x + skin_mask * warm_shift
        
        # Optional subtle smoothing on skin areas
        smoothness = torch.sigmoid(self.smoothness) * 0.3
        if smoothness > 0.01:
            smooth_channels = []
            pad = 2
            for c in range(3):
                channel = x_warm[:, c:c+1]
                smoothed = F.conv2d(channel, self.smooth_kernel, padding=pad)
                smooth_channels.append(smoothed)
            x_smooth = torch.cat(smooth_channels, dim=1)
            x_warm = x_warm + skin_mask * smoothness * (x_smooth - x_warm)
        
        return torch.clamp(x_warm, 0, 1)


#############################
# White Balance
#############################

class WhiteBalance(nn.Module):
    """
    Temperature and tint adjustment (like Lightroom)
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(0.0))  # Warm/cool
        self.tint = nn.Parameter(torch.tensor(0.0))  # Green/magenta
        
    def forward(self, x):
        temp = torch.tanh(self.temperature) * 0.3
        tint = torch.tanh(self.tint) * 0.2
        
        # Temperature: adjust blue-yellow axis
        # Tint: adjust green-magenta axis
        adjustment = torch.tensor([
            temp * 0.5,      # Red: warm adds red
            tint * -0.5,     # Green: tint affects green
            -temp * 0.5      # Blue: warm reduces blue
        ], device=x.device).view(1, 3, 1, 1)
        
        return torch.clamp(x + adjustment, 0, 1)


class SaturationVibrance(nn.Module):
    """
    Global saturation and vibrance controls.
    Vibrance boosts less-saturated colors more (protects already-saturated areas).
    """
    def __init__(self):
        super().__init__()
        self.saturation = nn.Parameter(torch.tensor(0.0))  # Global saturation
        self.vibrance = nn.Parameter(torch.tensor(0.0))    # Smart saturation
        
    def forward(self, x):
        # Compute luminance
        luma = 0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]
        
        # Current saturation level per pixel
        max_c = torch.max(x, dim=1, keepdim=True)[0]
        min_c = torch.min(x, dim=1, keepdim=True)[0]
        current_sat = (max_c - min_c) / (max_c + 1e-6)
        
        # Saturation adjustment: simple blend toward/away from gray
        sat_factor = 1.0 + torch.tanh(self.saturation) * 0.5  # [0.5, 1.5]
        
        # Vibrance: boost less saturated colors more
        # High current_sat -> low vibrance effect, low current_sat -> high vibrance effect
        vibrance_amount = torch.tanh(self.vibrance) * 0.5
        vibrance_factor = 1.0 + vibrance_amount * (1.0 - current_sat)
        
        # Combined factor
        total_factor = sat_factor * vibrance_factor
        
        # Apply: blend between luminance (gray) and original color
        x_adjusted = luma + total_factor * (x - luma)
        
        return torch.clamp(x_adjusted, 0, 1)


#############################
# Dehaze
#############################

class Dehaze(nn.Module):
    """
    Simple dehaze effect based on dark channel prior concept
    """
    def __init__(self):
        super().__init__()
        self.amount = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, x):
        amount = torch.tanh(self.amount) * 0.5  # [-0.5, 0.5]
        
        # Estimate atmospheric light (brightest region)
        dark_channel = torch.min(x, dim=1, keepdim=True)[0]
        
        # Simple dehaze: increase contrast and reduce haze
        # Positive amount = dehaze, negative = add haze
        if amount > 0:
            # Dehaze: stretch contrast
            x_dehazed = (x - dark_channel * amount) / (1 - amount + 1e-6)
        else:
            # Add haze: blend toward gray
            x_dehazed = x * (1 + amount) - amount * 0.5
        
        return torch.clamp(x_dehazed, 0, 1)


#############################
# Sharpening
#############################

class Sharpening(nn.Module):
    """
    Unsharp mask sharpening
    """
    def __init__(self):
        super().__init__()
        self.amount = nn.Parameter(torch.tensor(-5.0))  # Start disabled (sigmoid(-5)≈0)
        self.radius = 3
        
        # Gaussian kernel for blur
        kernel_size = 2 * self.radius + 1
        sigma = self.radius / 2
        grid = torch.arange(kernel_size).float() - self.radius
        gauss = torch.exp(-grid**2 / (2 * sigma * sigma))
        gauss = gauss / gauss.sum()
        kernel = gauss.unsqueeze(1) * gauss.unsqueeze(0)
        self.register_buffer('blur_kernel', kernel.unsqueeze(0).unsqueeze(0))
        
    def forward(self, x):
        amount = torch.sigmoid(self.amount) * 1.0  # [0, 1]
        
        if amount < 0.01:
            return x
        
        # Work on luminance
        luma = 0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]
        
        # Blur
        blurred = F.conv2d(luma, self.blur_kernel, padding=self.radius)
        
        # Unsharp mask
        sharp_luma = luma + amount * (luma - blurred)
        
        # Apply to RGB
        ratio = (sharp_luma + 1e-6) / (luma + 1e-6)
        x_sharp = x * torch.clamp(ratio, 0.5, 2.0)  # Limit ratio to prevent artifacts
        
        return torch.clamp(x_sharp, 0, 1)


#############################
# Complete Processor v2
#############################

class DifferentiableProcessorV2(nn.Module):
    """
    Complete differentiable RAW processing pipeline with professional adjustments.
    """
    def __init__(self, image_size=(768, 768)):
        super().__init__()
        
        # Processing modules in order
        self.white_balance = WhiteBalance()
        self.tone_mapping = AgXToneMapping()
        self.saturation_vibrance = SaturationVibrance()
        self.hsl = HSLAdjustment()
        self.split_toning = SplitToning()
        self.clarity_texture = ClarityTexture()
        self.dehaze = Dehaze()
        self.skin_protection = SkinToneProtection()
        self.orton = OrtonEffect()
        self.vignette = Vignette(image_size=image_size)
        self.sharpening = Sharpening()
        
    def forward(self, x):
        # Scene-referred adjustments
        x = self.white_balance(x)
        x = self.tone_mapping(x)  # Scene to display transform
        
        # Display-referred adjustments
        x = self.saturation_vibrance(x)  # Global color intensity
        x = self.hsl(x)
        x = self.split_toning(x)
        x = self.clarity_texture(x)
        x = self.dehaze(x)
        x = self.skin_protection(x)
        x = self.orton(x)
        x = self.vignette(x)
        x = self.sharpening(x)
        
        return torch.clamp(x, 0, 1)
    
    def get_parameter_summary(self):
        """Return a human-readable summary of current parameters"""
        summary = []
        summary.append(f"White Balance: temp={self.white_balance.temperature.item():.2f}, tint={self.white_balance.tint.item():.2f}")
        summary.append(f"Tone Mapping: exposure={self.tone_mapping.exposure.item():.2f}, contrast={self.tone_mapping.contrast.item():.2f}, gamma={self.tone_mapping.gamma.item():.2f}")
        summary.append(f"Saturation: {self.saturation_vibrance.saturation.item():.2f}, Vibrance: {self.saturation_vibrance.vibrance.item():.2f}")
        summary.append(f"Clarity: {self.clarity_texture.clarity.item():.2f}, Texture: {self.clarity_texture.texture.item():.2f}")
        summary.append(f"Dehaze: {self.dehaze.amount.item():.2f}")
        summary.append(f"Orton: {self.orton.amount.item():.2f}")
        summary.append(f"Vignette: {self.vignette.amount.item():.2f}")
        summary.append(f"Sharpening: {self.sharpening.amount.item():.2f}")
        return "\n".join(summary)


#############################
# Aesthetic Predictor
#############################

class AestheticPredictor(nn.Module):
    """
    LAION Aesthetic Predictor - a small MLP on top of CLIP features
    that predicts aesthetic scores.
    """
    def __init__(self, input_dim=768):
        super().__init__()
        # Match the exact architecture from the pretrained weights
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        return self.layers(x)


def load_aesthetic_predictor(device):
    """
    Load the LAION aesthetic predictor weights.
    Downloads if not present.
    """
    import urllib.request
    import os
    
    cache_dir = Path.home() / ".cache" / "aesthetic_predictor"
    cache_dir.mkdir(parents=True, exist_ok=True)
    weights_path = cache_dir / "sac+logos+ava1-l14-linearMSE.pth"
    
    if not weights_path.exists():
        print("Downloading aesthetic predictor weights...")
        url = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth"
        urllib.request.urlretrieve(url, weights_path)
        print("Download complete.")
    
    # Load model
    predictor = AestheticPredictor(input_dim=768)  # ViT-L/14 has 768 dim
    state_dict = torch.load(weights_path, map_location=device)
    predictor.load_state_dict(state_dict)
    predictor.to(device)
    predictor.eval()
    
    return predictor


#############################
# Multi-Objective Optimizer
#############################

class ProcessorOptimizerV2:
    """
    Optimizes the processor using multiple objectives:
    1. Aesthetic score (primary)
    2. Optional CLIP style guidance
    3. Exposure/histogram sanity
    4. Color naturalness
    """
    def __init__(self, processor, device='cuda' if torch.cuda.is_available() else 'cpu',
                 use_clip_style=True):
        self.processor = processor.to(device)
        self.device = device
        self.use_clip_style = use_clip_style
        
        # Load CLIP (ViT-L/14 for aesthetic predictor compatibility)
        print("Loading CLIP ViT-L/14...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=device)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Load aesthetic predictor
        print("Loading aesthetic predictor...")
        self.aesthetic_predictor = load_aesthetic_predictor(device)
        for param in self.aesthetic_predictor.parameters():
            param.requires_grad = False
        
        # CLIP normalization
        self.clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        self.clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
        
    def compute_aesthetic_score(self, image):
        """Compute aesthetic score for an image"""
        # Resize for CLIP
        image_resized = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
        image_norm = (image_resized - self.clip_mean) / self.clip_std
        
        with torch.no_grad():
            features = self.clip_model.encode_image(image_norm)
            features = features / features.norm(dim=-1, keepdim=True)
        
        # Aesthetic predictor expects float32
        score = self.aesthetic_predictor(features.float())
        return score
    
    def compute_clip_similarity(self, image, text_features):
        """Compute CLIP similarity between image and text"""
        image_resized = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
        image_norm = (image_resized - self.clip_mean) / self.clip_std
        
        image_features = self.clip_model.encode_image(image_norm)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        similarity = torch.cosine_similarity(image_features, text_features)
        return similarity
    
    def compute_exposure_loss(self, image):
        """
        Penalize under/over exposure.
        Target: mean luminance around 0.45-0.55, with good spread.
        """
        luma = 0.2126 * image[:, 0] + 0.7152 * image[:, 1] + 0.0722 * image[:, 2]
        
        # Mean should be around 0.5
        mean_loss = (luma.mean() - 0.5) ** 2
        
        # Should have reasonable contrast (std around 0.15-0.25)
        std = luma.std()
        std_loss = (std - 0.2) ** 2
        
        # Penalize clipping
        clip_loss = (luma > 0.99).float().mean() + (luma < 0.01).float().mean()
        
        return mean_loss + std_loss + clip_loss * 0.5
    
    def compute_color_naturalness_loss(self, image):
        """
        Penalize unnatural colors (oversaturated, neon, etc.)
        But don't penalize normal saturation levels - we want colorful images!
        """
        hsl = rgb_to_hsl(image)
        s = hsl[:, 1]
        
        # Only penalize VERY high saturation (>0.95 is neon territory)
        oversat_loss = F.relu(s - 0.95).mean() * 2.0
        
        # Also penalize very LOW saturation (washed out)
        undersat_loss = F.relu(0.15 - s.mean()) * 2.0
        
        # Penalize extreme color channel imbalance (prevents strong color casts)
        # But allow some imbalance for creative color grading
        r_mean = image[:, 0].mean()
        g_mean = image[:, 1].mean()
        b_mean = image[:, 2].mean()
        channel_balance = ((r_mean - g_mean) ** 2 + (g_mean - b_mean) ** 2 + (b_mean - r_mean) ** 2)
        # Only penalize if very imbalanced
        channel_loss = F.relu(channel_balance - 0.01) * 0.5
        
        return oversat_loss + undersat_loss + channel_loss
    
    def optimize(self, raw_image, text_prompt=None, num_steps=200,
                 aesthetic_weight=0.5, style_weight=0.2, 
                 exposure_weight=0.2, naturalness_weight=0.1,
                 live_view=False, live_view_scale=1.0):
        """
        Optimize the processor parameters.
        
        Args:
            raw_image: Input image tensor (C, H, W)
            text_prompt: Optional style prompt for CLIP guidance
            num_steps: Number of optimization steps
            aesthetic_weight: Weight for aesthetic score loss
            style_weight: Weight for CLIP style loss
            exposure_weight: Weight for exposure/histogram loss
            naturalness_weight: Weight for color naturalness loss
            live_view: If True, show live preview window during optimization
            live_view_scale: Scale factor for the preview window (e.g., 0.5 for half size)
        """
        raw_image = raw_image.unsqueeze(0).to(self.device)
        
        # Encode text prompt if provided
        text_features = None
        if text_prompt and self.use_clip_style:
            text_tokens = clip.tokenize([text_prompt]).to(self.device)
            with torch.no_grad():
                text_features = self.clip_model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Optimizer with different learning rates for different param groups
        optimizer = optim.AdamW(self.processor.parameters(), lr=0.05, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_steps, eta_min=0.001)
        
        best_loss = float('inf')
        best_state = None
        
        print(f"\nOptimizing with prompt: '{text_prompt}'")
        print(f"Weights: aesthetic={aesthetic_weight}, style={style_weight}, "
              f"exposure={exposure_weight}, naturalness={naturalness_weight}")
        print("-" * 60)
        
        # Setup live view window if requested
        if live_view:
            cv2.namedWindow('DiffRAW Live', cv2.WINDOW_NORMAL)
            print("Live view enabled. Press 'q' to stop early, any other key to continue.")
        
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # Process image
            processed = self.processor(raw_image)
            
            # Compute losses
            losses = {}
            
            # Aesthetic loss (we want to maximize, so negate)
            aesthetic_score = self.compute_aesthetic_score(processed)
            losses['aesthetic'] = -aesthetic_score.mean() * aesthetic_weight
            
            # Style loss (CLIP similarity)
            if text_features is not None:
                style_sim = self.compute_clip_similarity(processed, text_features)
                losses['style'] = (1 - style_sim.mean()) * style_weight
            
            # Exposure loss
            losses['exposure'] = self.compute_exposure_loss(processed) * exposure_weight
            
            # Naturalness loss
            losses['naturalness'] = self.compute_color_naturalness_loss(processed) * naturalness_weight
            
            # Total loss
            total_loss = sum(losses.values())
            
            # Backward and optimize
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.processor.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Track best
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_state = {k: v.clone() for k, v in self.processor.state_dict().items()}
            
            # Logging
            if (step + 1) % 20 == 0 or step == 0:
                loss_str = ", ".join([f"{k}={v.item():.4f}" for k, v in losses.items()])
                aes_score = aesthetic_score.item()
                print(f"Step {step+1:3d}/{num_steps}: total={total_loss.item():.4f}, "
                      f"aesthetic_score={aes_score:.2f}, {loss_str}")
            
            # Live view update
            if live_view:
                with torch.no_grad():
                    # Get current processed image
                    display_img = processed.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    display_img = (display_img * 255).astype(np.uint8)
                    # Convert RGB to BGR for OpenCV
                    display_img = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
                    # Scale if requested
                    if live_view_scale != 1.0:
                        h, w = display_img.shape[:2]
                        new_size = (int(w * live_view_scale), int(h * live_view_scale))
                        display_img = cv2.resize(display_img, new_size)
                    # Add text overlay with current step and aesthetic score
                    text = f"Step {step+1}/{num_steps} | Aesthetic: {aes_score:.2f}"
                    cv2.putText(display_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(display_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.imshow('DiffRAW Live', display_img)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\nEarly stop requested by user.")
                        break
        
        # Restore best state
        if best_state is not None:
            self.processor.load_state_dict(best_state)
        
        # Close live view window
        if live_view:
            cv2.destroyAllWindows()
        
        print("-" * 60)
        print("Optimization complete!")
        print("\nFinal parameters:")
        print(self.processor.get_parameter_summary())
        
        return self.processor


#############################
# Main
#############################

def main():
    parser = argparse.ArgumentParser(
        description="Differentiable RAW processor v2 with aesthetic optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_v2.py photo.ARW --prompt "cinematic, moody, film look"
  python demo_v2.py photo.CR2 --prompt "bright, airy, natural light" --steps 300
  python demo_v2.py photo.NEF --prompt "dramatic sunset, warm tones" --aesthetic-weight 0.6
        """
    )
    parser.add_argument("input_file", type=str, help="Path to input RAW file")
    parser.add_argument("--prompt", type=str, default="professional photograph, beautiful lighting, natural colors",
                        help="Text prompt describing desired style")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: input_processed.jpg)")
    parser.add_argument("--steps", type=int, default=200,
                        help="Number of optimization steps")
    parser.add_argument("--size", type=int, default=768,
                        help="Processing size (square)")
    parser.add_argument("--aesthetic-weight", type=float, default=0.6,
                        help="Weight for aesthetic score")
    parser.add_argument("--style-weight", type=float, default=0.2,
                        help="Weight for CLIP style guidance")
    parser.add_argument("--exposure-weight", type=float, default=0.15,
                        help="Weight for exposure sanity")
    parser.add_argument("--naturalness-weight", type=float, default=0.05,
                        help="Weight for color naturalness")
    parser.add_argument("--no-clip-style", action="store_true",
                        help="Disable CLIP style guidance (use aesthetic only)")
    parser.add_argument("--live", action="store_true",
                        help="Show live preview window during optimization")
    parser.add_argument("--live-scale", type=float, default=1.0,
                        help="Scale factor for live preview window (e.g., 0.5 for half size)")
    
    args = parser.parse_args()
    
    # Determine output path
    if args.output is None:
        input_path = Path(args.input_file)
        args.output = str(input_path.parent / f"{input_path.stem}_processed.jpg")
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Import RAW
    print(f"\nLoading RAW file: {args.input_file}")
    importer = RawImporter(target_size=(args.size, args.size))
    raw_image = importer.import_raw(args.input_file)
    raw_image = raw_image.to(device)
    print(f"Image loaded: {raw_image.shape}")
    
    # Create processor
    processor = DifferentiableProcessorV2(image_size=(args.size, args.size))
    
    # Create optimizer
    optimizer_module = ProcessorOptimizerV2(
        processor, 
        device=device,
        use_clip_style=not args.no_clip_style
    )
    
    # Optimize
    optimized_processor = optimizer_module.optimize(
        raw_image, 
        text_prompt=args.prompt,
        num_steps=args.steps,
        aesthetic_weight=args.aesthetic_weight,
        style_weight=args.style_weight,
        exposure_weight=args.exposure_weight,
        naturalness_weight=args.naturalness_weight,
        live_view=args.live,
        live_view_scale=args.live_scale
    )
    
    # Process final image
    print("\nGenerating final output...")
    with torch.no_grad():
        processed_image = optimized_processor(raw_image.unsqueeze(0))
    
    # Save
    processed_np = (processed_image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    imageio.imwrite(args.output, processed_np, quality=95)
    print(f"Output saved to: {args.output}")
    
    # Also save a comparison image
    comparison_path = Path(args.output).parent / f"{Path(args.output).stem}_comparison.jpg"
    with torch.no_grad():
        # Simple baseline processing for comparison
        baseline = raw_image.unsqueeze(0)
        baseline = torch.pow(baseline, 1/2.2)  # Simple gamma
        baseline_np = (baseline.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    
    # Side by side
    comparison = np.concatenate([baseline_np, processed_np], axis=1)
    imageio.imwrite(str(comparison_path), comparison, quality=95)
    print(f"Comparison saved to: {comparison_path}")


if __name__ == "__main__":
    main()

