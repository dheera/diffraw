"""
Differentiable Image Processing Filters
=======================================
PyTorch modules for RAW/image processing with learnable parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


#############################
# White Balance
#############################

class WhiteBalance(nn.Module):
    """Temperature and tint adjustment (like Lightroom)"""
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(0.0))  # Warm/cool
        self.tint = nn.Parameter(torch.tensor(0.0))  # Green/magenta
        
    def forward(self, x):
        temp = torch.tanh(self.temperature) * 0.3
        tint = torch.tanh(self.tint) * 0.2
        
        adjustment = torch.tensor([
            temp * 0.5,      # Red: warm adds red
            tint * -0.5,     # Green: tint affects green
            -temp * 0.5      # Blue: warm reduces blue
        ], device=x.device).view(1, 3, 1, 1)
        
        return torch.clamp(x + adjustment, 0, 1)


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
        self.exposure = nn.Parameter(torch.tensor(1.5))  # In stops
        self.contrast = nn.Parameter(torch.tensor(0.0))
        self.gamma = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, x):
        # Apply exposure (in stops, so 2^exposure)
        exposure_factor = torch.pow(torch.tensor(2.0, device=x.device), 
                                    torch.clamp(self.exposure, -3, 4))
        x = x * exposure_factor
        
        # Simple gamma first to get into perceptual space
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
# Saturation & Vibrance
#############################

class SaturationVibrance(nn.Module):
    """
    Global saturation and vibrance controls.
    Vibrance boosts less-saturated colors more (protects already-saturated areas).
    """
    def __init__(self):
        super().__init__()
        self.saturation = nn.Parameter(torch.tensor(0.0))
        self.vibrance = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, x):
        luma = 0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]
        
        max_c = torch.max(x, dim=1, keepdim=True)[0]
        min_c = torch.min(x, dim=1, keepdim=True)[0]
        current_sat = (max_c - min_c) / (max_c + 1e-6)
        
        sat_factor = 1.0 + torch.tanh(self.saturation) * 0.5
        vibrance_amount = torch.tanh(self.vibrance) * 0.5
        vibrance_factor = 1.0 + vibrance_amount * (1.0 - current_sat)
        
        total_factor = sat_factor * vibrance_factor
        x_adjusted = luma + total_factor * (x - luma)
        
        return torch.clamp(x_adjusted, 0, 1)


#############################
# HSL Adjustments
#############################

class HSLAdjustment(nn.Module):
    """
    Per-hue adjustments for 8 color ranges:
    Red, Orange, Yellow, Green, Cyan, Blue, Purple, Magenta
    """
    def __init__(self):
        super().__init__()
        self.hue_shifts = nn.Parameter(torch.zeros(8))
        self.sat_mults = nn.Parameter(torch.ones(8))
        self.lum_shifts = nn.Parameter(torch.zeros(8))
        
        self.register_buffer('hue_centers', torch.tensor([
            0.0, 0.083, 0.167, 0.333, 0.5, 0.667, 0.75, 0.917
        ]))
        
    def forward(self, x):
        hsl = rgb_to_hsl(x)
        h, s, l = hsl[:, 0:1], hsl[:, 1:2], hsl[:, 2:3]
        
        width = 0.08
        
        total_hue_shift = torch.zeros_like(h)
        total_sat_mult = torch.ones_like(s)
        total_lum_shift = torch.zeros_like(l)
        
        for i in range(8):
            center = self.hue_centers[i]
            dist = torch.min(
                torch.abs(h - center),
                torch.min(torch.abs(h - center - 1), torch.abs(h - center + 1))
            )
            weight = torch.exp(-dist ** 2 / (2 * width ** 2))
            weight = weight * s
            
            total_hue_shift = total_hue_shift + weight * self.hue_shifts[i] * 0.1
            total_sat_mult = total_sat_mult + weight * (self.sat_mults[i] - 1)
            total_lum_shift = total_lum_shift + weight * self.lum_shifts[i] * 0.2
        
        h_new = (h + total_hue_shift) % 1
        s_new = torch.clamp(s * total_sat_mult, 0, 1)
        l_new = torch.clamp(l + total_lum_shift, 0, 1)
        
        hsl_new = torch.cat([h_new, s_new, l_new], dim=1)
        return hsl_to_rgb(hsl_new)


#############################
# Split Toning
#############################

class SplitToning(nn.Module):
    """Add color tints to shadows and highlights separately."""
    def __init__(self):
        super().__init__()
        self.shadow_hue = nn.Parameter(torch.tensor(0.6))
        self.shadow_saturation = nn.Parameter(torch.tensor(0.0))
        self.highlight_hue = nn.Parameter(torch.tensor(0.1))
        self.highlight_saturation = nn.Parameter(torch.tensor(0.0))
        self.balance = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x):
        luma = 0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]
        
        balance = torch.sigmoid(self.balance)
        shadow_mask = 1 - torch.sigmoid((luma - balance) * 10)
        highlight_mask = torch.sigmoid((luma - balance) * 10)
        
        def hue_to_rgb(hue):
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
        
        shadow_sat = torch.sigmoid(self.shadow_saturation) * 0.3
        highlight_sat = torch.sigmoid(self.highlight_saturation) * 0.3
        
        shadow_tint = shadow_rgb.view(1, 3, 1, 1) * shadow_sat
        highlight_tint = highlight_rgb.view(1, 3, 1, 1) * highlight_sat
        
        x_tinted = x + shadow_mask * shadow_tint + highlight_mask * highlight_tint
        
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
        
        self.clarity_kernel_size = 31
        self.clarity_sigma = 8
        self.texture_kernel_size = 7
        self.texture_sigma = 1.5
        
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
        luma = 0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]
        
        clarity_pad = self.clarity_kernel_size // 2
        luma_blur_clarity = F.conv2d(luma, self.clarity_kernel, padding=clarity_pad)
        mid_freq = luma - luma_blur_clarity
        clarity_amount = torch.tanh(self.clarity) * 0.5
        
        texture_pad = self.texture_kernel_size // 2
        luma_blur_texture = F.conv2d(luma, self.texture_kernel, padding=texture_pad)
        high_freq = luma - luma_blur_texture
        texture_amount = torch.tanh(self.texture) * 0.3
        
        luma_enhanced = luma + clarity_amount * mid_freq + texture_amount * high_freq
        
        ratio = (luma_enhanced + 1e-6) / (luma + 1e-6)
        x_enhanced = x * ratio
        
        return torch.clamp(x_enhanced, 0, 1)


#############################
# Dehaze
#############################

class Dehaze(nn.Module):
    """Simple dehaze effect based on dark channel prior concept"""
    def __init__(self):
        super().__init__()
        self.amount = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, x):
        amount = torch.tanh(self.amount) * 0.5
        
        dark_channel = torch.min(x, dim=1, keepdim=True)[0]
        
        # Dehaze or add haze depending on sign
        x_dehazed = torch.where(
            amount > 0,
            (x - dark_channel * amount) / (1 - amount + 1e-6),
            x * (1 + amount) - amount * 0.5
        )
        
        return torch.clamp(x_dehazed, 0, 1)


#############################
# Orton Effect
#############################

class OrtonEffect(nn.Module):
    """Orton effect: dreamy glow + sharpness"""
    def __init__(self):
        super().__init__()
        self.amount = nn.Parameter(torch.tensor(0.0))
        self.glow_size = 51
        self.glow_sigma = 15
        
        grid = torch.arange(self.glow_size).float() - self.glow_size // 2
        gauss = torch.exp(-grid**2 / (2 * self.glow_sigma * self.glow_sigma))
        gauss = gauss / gauss.sum()
        kernel = gauss.unsqueeze(1) * gauss.unsqueeze(0)
        self.register_buffer('glow_kernel', kernel.unsqueeze(0).unsqueeze(0))
        
    def forward(self, x):
        amount = torch.sigmoid(self.amount) * 0.4
        
        if amount < 0.01:
            return x
        
        glow_channels = []
        pad = self.glow_size // 2
        for c in range(3):
            channel = x[:, c:c+1]
            blurred = F.conv2d(channel, self.glow_kernel, padding=pad)
            glow_channels.append(blurred)
        glow = torch.cat(glow_channels, dim=1)
        
        glow = glow * 1.2
        result = 1 - (1 - x) * (1 - glow * amount)
        
        return torch.clamp(result, 0, 1)


#############################
# Vignette
#############################

class Vignette(nn.Module):
    """Optical vignette effect with controllable amount and shape"""
    def __init__(self, image_size=(768, 768)):
        super().__init__()
        self.amount = nn.Parameter(torch.tensor(0.0))
        self.roundness = nn.Parameter(torch.tensor(0.0))
        self.feather = nn.Parameter(torch.tensor(0.5))
        
        H, W = image_size
        y = torch.linspace(-1, 1, H)
        x = torch.linspace(-1, 1, W)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        self.register_buffer('xx', xx)
        self.register_buffer('yy', yy)
        
    def forward(self, x):
        amount = torch.tanh(self.amount) * 0.5
        
        aspect = 1 + torch.tanh(self.roundness) * 0.3
        dist = torch.sqrt(self.xx**2 + (self.yy * aspect)**2)
        
        feather = 0.3 + torch.sigmoid(self.feather) * 0.7
        vignette_mask = 1 - torch.sigmoid((dist - feather) * 5) * amount
        
        vignette_mask = vignette_mask.unsqueeze(0).unsqueeze(0)
        return x * vignette_mask


#############################
# Sharpening
#############################

class Sharpening(nn.Module):
    """Unsharp mask sharpening"""
    def __init__(self):
        super().__init__()
        self.amount = nn.Parameter(torch.tensor(0.0))
        self.radius = 3
        
        kernel_size = 2 * self.radius + 1
        sigma = self.radius / 2
        grid = torch.arange(kernel_size).float() - self.radius
        gauss = torch.exp(-grid**2 / (2 * sigma * sigma))
        gauss = gauss / gauss.sum()
        kernel = gauss.unsqueeze(1) * gauss.unsqueeze(0)
        self.register_buffer('blur_kernel', kernel.unsqueeze(0).unsqueeze(0))
        
    def forward(self, x):
        amount = torch.sigmoid(self.amount) * 1.0
        
        if amount < 0.01:
            return x
        
        luma = 0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]
        blurred = F.conv2d(luma, self.blur_kernel, padding=self.radius)
        sharp_luma = luma + amount * (luma - blurred)
        
        ratio = (sharp_luma + 1e-6) / (luma + 1e-6)
        x_sharp = x * torch.clamp(ratio, 0.5, 2.0)
        
        return torch.clamp(x_sharp, 0, 1)


#############################
# Skin Tone Protection
#############################

class SkinToneProtection(nn.Module):
    """Detects skin tones and applies subtle adjustments"""
    def __init__(self):
        super().__init__()
        self.warmth = nn.Parameter(torch.tensor(0.0))
        self.smoothness = nn.Parameter(torch.tensor(0.0))
        
        kernel_size = 5
        sigma = 1.0
        grid = torch.arange(kernel_size).float() - kernel_size // 2
        gauss = torch.exp(-grid**2 / (2 * sigma * sigma))
        gauss = gauss / gauss.sum()
        kernel = gauss.unsqueeze(1) * gauss.unsqueeze(0)
        self.register_buffer('smooth_kernel', kernel.unsqueeze(0).unsqueeze(0))
        
    def detect_skin(self, x):
        hsl = rgb_to_hsl(x)
        h, s, l = hsl[:, 0:1], hsl[:, 1:2], hsl[:, 2:3]
        
        hue_score = torch.exp(-((h - 0.05) ** 2) / (2 * 0.04 ** 2))
        sat_score = torch.exp(-((s - 0.4) ** 2) / (2 * 0.2 ** 2))
        lum_score = torch.exp(-((l - 0.55) ** 2) / (2 * 0.25 ** 2))
        
        skin_mask = hue_score * sat_score * lum_score
        return skin_mask
    
    def forward(self, x):
        skin_mask = self.detect_skin(x)
        
        warmth = torch.tanh(self.warmth) * 0.1
        warm_shift = torch.tensor([warmth * 0.05, warmth * 0.02, -warmth * 0.03], 
                                  device=x.device).view(1, 3, 1, 1)
        x_warm = x + skin_mask * warm_shift
        
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
# Complete Processor
#############################

class DifferentiableProcessor(nn.Module):
    """Complete differentiable RAW processing pipeline with professional adjustments."""
    def __init__(self, image_size=(768, 768)):
        super().__init__()
        
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
        x = self.white_balance(x)
        x = self.tone_mapping(x)
        x = self.saturation_vibrance(x)
        x = self.hsl(x)
        x = self.split_toning(x)
        x = self.clarity_texture(x)
        x = self.dehaze(x)
        x = self.skin_protection(x)
        x = self.orton(x)
        x = self.vignette(x)
        x = self.sharpening(x)
        
        return torch.clamp(x, 0, 1)

