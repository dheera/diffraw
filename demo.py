import math
import rawpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import clip  # OpenAI’s CLIP package; install via pip if needed
import torch.optim as optim
import argparse
import imageio

#############################
# RAW Importer
#############################

class RawImporter:
    """
    Imports a RAW file using rawpy, performs a simple demosaic/processing,
    and converts it into a high-precision torch tensor. Optionally, the image is resized/cropped.
    """
    def __init__(self, target_size=(768, 768)):
        self.target_size = target_size

    def import_raw(self, file_path):
        with rawpy.imread(file_path) as raw:
            # Use rawpy's postprocessing to get an initial demosaiced version.
            rgb = raw.postprocess(output_bps=16, no_auto_bright=True, use_camera_wb=False)
        # Normalize to [0, 1] and convert to torch tensor (C, H, W)
        img = torch.from_numpy(rgb.astype(np.float32) / 65535.0).permute(2, 0, 1)
        img = transforms.functional.resize(img, self.target_size)
        return img

#############################
# Differentiable Adjustments
#############################

class WhiteBalanceAdjustment(nn.Module):
    """
    Applies channel-wise gain adjustments.
    """
    def __init__(self):
        super().__init__()
        # One gain per channel (initialize at 1)
        self.gains = nn.Parameter(torch.ones(3))

    def forward(self, x):
        # x shape: (B, C, H, W); apply per-channel multiplication
        return x * self.gains.view(1, 3, 1, 1)


class ToneCurveAdjustment(nn.Module):
    """
    A simple tone-mapping using a gamma adjustment.
    """
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # Apply gamma correction (assumes input in [0,1])
        return torch.pow(x, self.gamma)


class ShadowHighlightEnhancement(nn.Module):
    """
    Enhances shadows and reduces highlights by computing a soft mask.
    """
    def __init__(self):
        super().__init__()
        self.shadow_boost = nn.Parameter(torch.tensor(0.0))
        self.highlight_reduce = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        shadow_mask = 1 - torch.sigmoid((x - 0.5) * 10)
        highlight_mask = torch.sigmoid((x - 0.5) * 10)
        x_adjusted = x + self.shadow_boost * shadow_mask - self.highlight_reduce * highlight_mask
        return torch.clamp(x_adjusted, 0, 1)


class GlobalBrightnessContrastAdjustment(nn.Module):
    """
    Applies a global brightness and contrast adjustment.
    """
    def __init__(self):
        super().__init__()
        self.brightness = nn.Parameter(torch.tensor(0.0))
        self.contrast = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x_adjusted = self.contrast * (x - 0.5) + 0.5 + self.brightness
        return torch.clamp(x_adjusted, 0, 1)


class LocalContrastEnhancement(nn.Module):
    """
    Enhances local contrast using a simple local (Gaussian) average.
    """
    def __init__(self, kernel_size=15, sigma=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.padding = kernel_size // 2
        grid = torch.arange(kernel_size).float() - kernel_size // 2
        gauss = torch.exp(-grid**2 / (2 * sigma * sigma))
        gauss = gauss / gauss.sum()
        kernel = gauss.unsqueeze(1) * gauss.unsqueeze(0)
        self.register_buffer('kernel', kernel.unsqueeze(0).unsqueeze(0))
        self.enhance_amount = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        channels = []
        for c in range(x.shape[1]):
            channel = x[:, c:c+1, :, :]
            local_mean = F.conv2d(channel, self.kernel, padding=self.padding)
            enhanced = channel + self.enhance_amount * (channel - local_mean)
            channels.append(enhanced)
        return torch.cat(channels, dim=1)


class SoftnessFilter(nn.Module):
    """
    A softness (blur) filter that can be mixed with the original image.
    """
    def __init__(self, kernel_size=15, sigma=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.padding = kernel_size // 2
        grid = torch.arange(kernel_size).float() - kernel_size // 2
        gauss = torch.exp(-grid**2 / (2 * sigma * sigma))
        gauss = gauss / gauss.sum()
        kernel = gauss.unsqueeze(1) * gauss.unsqueeze(0)
        self.register_buffer('kernel', kernel.unsqueeze(0).unsqueeze(0))
        self.softness = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        blurred_channels = []
        for c in range(x.shape[1]):
            channel = x[:, c:c+1, :, :]
            blurred = F.conv2d(channel, self.kernel, padding=self.padding)
            blurred_channels.append(blurred)
        blurred = torch.cat(blurred_channels, dim=1)
        return self.softness * blurred + (1 - self.softness) * x


class GradientNDAdjustment(nn.Module):
    """
    Applies a gradient neutral density effect (like a graduated ND filter).
    """
    def __init__(self, image_size=(768, 768)):
        super().__init__()
        self.intensity = nn.Parameter(torch.tensor(0.0))
        self.rotation = nn.Parameter(torch.tensor(0.0))
        self.hardness = nn.Parameter(torch.tensor(1.0))
        self.image_size = image_size
        H, W = image_size
        ys = torch.linspace(-1, 1, H)
        xs = torch.linspace(-1, 1, W)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        self.register_buffer('grid_x', grid_x)
        self.register_buffer('grid_y', grid_y)

    def forward(self, x):
        theta = self.rotation * math.pi / 180.0
        grid_rot = self.grid_x * math.cos(theta) + self.grid_y * math.sin(theta)
        mask = torch.sigmoid(-self.hardness * grid_rot)
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = mask.expand(x.size(0), x.size(1), -1, -1)
        return torch.clamp(x * (1 - self.intensity * mask), 0, 1)

#############################
# Differentiable Processor
#############################

class DifferentiableProcessor(nn.Module):
    """
    Combines all the adjustment modules into one trainable processing pipeline.
    """
    def __init__(self, image_size=(768, 768)):
        super().__init__()
        self.white_balance = WhiteBalanceAdjustment()
        self.tone_curve = ToneCurveAdjustment()
        self.shadow_highlight = ShadowHighlightEnhancement()
        self.global_brightness_contrast = GlobalBrightnessContrastAdjustment()
        self.local_contrast = LocalContrastEnhancement()
        self.softness = SoftnessFilter()
        self.gradient_nd = GradientNDAdjustment(image_size=image_size)

    def forward(self, x):
        x = self.white_balance(x)
        x = self.tone_curve(x)
        x = self.shadow_highlight(x)
        x = self.global_brightness_contrast(x)
        x = self.local_contrast(x)
        x = self.softness(x)
        x = self.gradient_nd(x)
        return torch.clamp(x, 0, 1)

#############################
# Processor Optimization with CLIP
#############################

class ProcessorOptimizer:
    """
    Uses a frozen vision-language model (CLIP) to steer the adjustments.
    Given a text prompt, it optimizes the parameters of a DifferentiableProcessor
    so that the processed image maximally aligns with the text.
    """
    def __init__(self, processor, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.processor = processor.to(device)
        self.device = device
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.optimizer = optim.Adam(self.processor.parameters(), lr=1e-3)

    def optimize(self, raw_image, text_prompt, num_steps=100):
        # Ensure a batch dimension and move to device
        raw_image = raw_image.unsqueeze(0).to(self.device)
        text_tokens = clip.tokenize([text_prompt]).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        for step in range(num_steps):
            self.optimizer.zero_grad()
            processed = self.processor(raw_image)
            processed_resized = F.interpolate(processed, size=(224, 224), mode='bilinear', align_corners=False)
            clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1, 3, 1, 1)
            clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1, 3, 1, 1)
            processed_norm = (processed_resized - clip_mean) / clip_std
            image_features = self.clip_model.encode_image(processed_norm)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            loss = 1 - torch.cosine_similarity(image_features, text_features).mean()
            loss.backward()
            self.optimizer.step()
            print(f"Step {step+1}/{num_steps}, Loss: {loss.item():.4f}")
        return self.processor

#############################
# Example Usage with Command Line Arguments
#############################

def main():
    parser = argparse.ArgumentParser(description="Differentiable RAW processor demo")
    parser.add_argument("input_file", type=str, help="Path to input RAW file")
    parser.add_argument("--prompt", type=str, default="A cool, dreamy, scene",
                        help="Text prompt for processing")
    parser.add_argument("--output", type=str, default="processed_output.jpg",
                        help="Path to save the processed image (jpg, tiff, etc.)")
    args = parser.parse_args()

    importer = RawImporter(target_size=(768, 768))
    raw_image = importer.import_raw(args.input_file)
    
    # Determine device and move raw image to the same device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    raw_image = raw_image.to(device)

    processor = DifferentiableProcessor(image_size=(768, 768))
    optimizer_module = ProcessorOptimizer(processor, device=device)
    
    # Optimize the processor using the provided or default text prompt.
    optimized_processor = optimizer_module.optimize(raw_image, args.prompt, num_steps=500)
    
    # Process the image using the optimized processor.
    processed_image = optimized_processor(raw_image.unsqueeze(0).to(device))
    
    # Convert processed image to a displayable/savable format.
    processed_np = (processed_image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    
    # Save the image using imageio.
    imageio.imwrite(args.output, processed_np)
    print(f"Processing complete. Output saved to {args.output}")

if __name__ == "__main__":
    main()

