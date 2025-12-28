"""
Differentiable RAW Processor v3
===============================
Uses a diffusion model as a perceptual critic while keeping
fixed, controllable edit operations.

Key insight: Instead of letting the diffusion model generate pixels
(which causes hallucinations), we use it to score "does this edited
image look like a good photo matching this prompt?"

This combines:
- demo_v2's controllable edit pipeline (exposure, contrast, HSL, etc.)
- Diffusion model's learned understanding of good photos
- Score Distillation Sampling (SDS) loss for optimization
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

# Import the processor from demo_v2
from demo_v2 import (
    RawImporter,
    DifferentiableProcessorV2,
    rgb_to_hsl,
)

# Diffusers imports
from diffusers import StableDiffusionPipeline, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer


class DiffusionCritic:
    """
    Uses a diffusion model as a perceptual critic.
    
    The key idea: if we add noise to an image and the diffusion model
    can easily denoise it toward a text prompt, the image is "on distribution"
    for that prompt. If it struggles, the image doesn't match.
    
    We use the Score Distillation Sampling (SDS) loss:
    - Add noise to the edited image
    - Get the diffusion model's predicted noise
    - The gradient of this prediction w.r.t. the image tells us
      how to make the image more "prompt-like"
    """
    
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        print(f"Loading diffusion critic on {device}...")
        
        # Load just the components we need (not full pipeline)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
        
        # Load UNet and VAE
        from diffusers import AutoencoderKL, UNet2DConditionModel
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        
        # Scheduler for noise levels
        self.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        
        # Move to device and set precision
        dtype = torch.float16 if device == 'cuda' else torch.float32
        self.dtype = dtype
        
        self.text_encoder = self.text_encoder.to(device, dtype=dtype)
        self.vae = self.vae.to(device, dtype=dtype)
        self.unet = self.unet.to(device, dtype=dtype)
        
        # Freeze all diffusion model parameters
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        
        # VAE scaling factor
        self.vae_scale = self.vae.config.scaling_factor
        
        print("Diffusion critic loaded!")
    
    def encode_prompt(self, prompt, negative_prompt=""):
        """Encode text prompt to embeddings"""
        # Tokenize
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        uncond_input = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        
        # Concatenate for classifier-free guidance
        return torch.cat([uncond_embeddings, text_embeddings])
    
    def encode_image(self, image):
        """
        Encode image to latent space.
        Input: (B, 3, H, W) in [0, 1]
        Output: latent tensor
        """
        # Normalize to [-1, 1]
        image = image * 2 - 1
        
        # Encode - use .mean instead of .sample() to keep gradients flowing!
        latent_dist = self.vae.encode(image.to(self.dtype)).latent_dist
        latent = latent_dist.mean  # Deterministic, differentiable
        latent = latent * self.vae_scale
        
        return latent
    
    def compute_sds_loss(self, image, text_embeddings, 
                         min_step=0.02, max_step=0.98,
                         guidance_scale=100.0,
                         num_samples=4):
        """
        Compute Score Distillation Sampling loss with improvements to prevent
        the typical SDS "washing out" problem.
        
        Uses lower timesteps (less noise) and negative guidance to preserve details.
        """
        batch_size = image.shape[0]
        
        # Encode image to latent
        latent = self.encode_image(image)
        
        num_train_timesteps = self.scheduler.config.num_train_timesteps
        
        # Average over multiple timesteps to reduce variance
        total_grad = torch.zeros_like(latent, dtype=torch.float32)
        
        for _ in range(num_samples):
            # Use LOWER timesteps - less noise = more detail preservation
            # High timesteps cause the "wash out" effect
            t = torch.randint(
                int(num_train_timesteps * min_step),
                int(num_train_timesteps * max_step),
                (batch_size,),
                device=self.device,
            ).long()
            
            # Add noise
            noise = torch.randn_like(latent)
            noisy_latent = self.scheduler.add_noise(latent, noise, t)
            
            # Predict noise with diffusion model
            latent_model_input = torch.cat([noisy_latent] * 2)
            t_input = torch.cat([t] * 2)
            
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input.to(self.dtype),
                    t_input,
                    encoder_hidden_states=text_embeddings.to(self.dtype),
                ).sample
            
            # Classifier-free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            
            # KEY CHANGE: Use the DIFFERENCE only, not full CFG
            # This gives us "direction toward prompt" without over-smoothing
            noise_direction = noise_pred_text - noise_pred_uncond
            
            # Scale by guidance but also by timestep (less influence at low noise)
            alpha_t = self.scheduler.alphas_cumprod[t].view(-1, 1, 1, 1).to(self.device)
            
            # Accumulate gradient - just the directional component
            grad = guidance_scale * noise_direction.float() * (1 - alpha_t)
            total_grad = total_grad + grad
        
        # Average the gradients
        total_grad = total_grad / num_samples
        
        # The actual SDS loss (for backprop through the image)
        loss = F.mse_loss(latent.float(), (latent - total_grad).detach())
        
        return loss


class ProcessorOptimizerV3:
    """
    Optimizes the processor using diffusion model as critic.
    
    Combines:
    - Diffusion SDS loss (learned aesthetic prior)
    - Exposure/histogram sanity
    - Color naturalness
    """
    
    def __init__(self, processor, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.processor = processor.to(device)
        self.device = device
        
        # Load diffusion critic
        self.critic = DiffusionCritic(device=device)
        
    def compute_exposure_loss(self, image):
        """Penalize under/over exposure"""
        luma = 0.2126 * image[:, 0] + 0.7152 * image[:, 1] + 0.0722 * image[:, 2]
        
        # Mean should be around 0.45
        mean_loss = (luma.mean() - 0.45) ** 2
        
        # Should have reasonable contrast
        std = luma.std()
        std_loss = (std - 0.18) ** 2
        
        # Penalize clipping
        clip_loss = (luma > 0.98).float().mean() + (luma < 0.02).float().mean()
        
        return mean_loss + std_loss + clip_loss * 0.5
    
    def compute_color_loss(self, image, original_saturation=None, target_saturation=None):
        """Penalize unnatural colors and saturation changes"""
        hsl = rgb_to_hsl(image)
        s = hsl[:, 1]
        current_sat = s.mean()
        
        # If we have a target saturation (from prompt analysis), guide toward it
        if target_saturation is not None:
            sat_loss = (current_sat - target_saturation) ** 2 * 10.0
        else:
            # Default: preserve original saturation
            if original_saturation is not None:
                # Penalize deviation from original
                sat_loss = (current_sat - original_saturation) ** 2 * 5.0
            else:
                sat_loss = torch.tensor(0.0, device=image.device)
        
        # Always penalize extreme saturation (>0.95)
        oversat_loss = F.relu(s - 0.95).mean()
        
        return sat_loss + oversat_loss
    
    def compute_saturation_multiplier(self, prompt):
        """
        Use CLIP to compute saturation multiplier by comparing prompt similarity
        to "desaturated muted faded colors" vs "vibrant saturated vivid colors".
        Returns a float: <1 for desaturation, >1 for boost, ~1 for neutral.
        """
        # Encode the user prompt and reference prompts
        prompts = [
            prompt,
            "desaturated, muted, faded colors, low saturation",
            "vibrant, saturated, vivid colors, high saturation",
        ]
        
        with torch.no_grad():
            tokens = self.critic.tokenizer(
                prompts,
                padding="max_length",
                max_length=self.critic.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(self.device)
            
            embeddings = self.critic.text_encoder(tokens)[0]  # (3, seq_len, dim)
            
            # Mean pool across sequence
            embeddings = embeddings.mean(dim=1)  # (3, dim)
            embeddings = F.normalize(embeddings.float(), dim=-1)
            
            user_emb = embeddings[0]
            desat_emb = embeddings[1]
            sat_emb = embeddings[2]
            
            # Cosine similarity
            desat_sim = (user_emb @ desat_emb).item()
            sat_sim = (user_emb @ sat_emb).item()
        
        # Convert similarities to multiplier
        # Higher desat_sim -> lower multiplier, higher sat_sim -> higher multiplier
        diff = sat_sim - desat_sim  # Positive = more saturated, negative = more desaturated
        multiplier = 1.0 + diff * 2.0  # Scale factor
        
        print(f"CLIP saturation analysis: desat_sim={desat_sim:.3f}, sat_sim={sat_sim:.3f}")
        return max(0.3, min(1.5, multiplier))
    
    def optimize(self, raw_image, prompt, 
                 negative_prompt="blurry, low quality, oversaturated, undersaturated",
                 num_steps=100,
                 sds_weight=1.0,
                 exposure_weight=0.3,
                 color_weight=0.1,
                 guidance_scale=30.0,
                 sds_samples=4,
                 live_view=False,
                 live_view_scale=1.0):
        """
        Optimize processor parameters using diffusion critic.
        """
        raw_image = raw_image.unsqueeze(0).to(self.device)
        
        # Compute original image saturation to preserve it
        with torch.no_grad():
            orig_processed = self.processor(raw_image)
            orig_hsl = rgb_to_hsl(orig_processed)
            original_saturation = orig_hsl[:, 1].mean()
            print(f"Original saturation: {original_saturation.item():.3f}")
        
        # Compute target saturation from prompt
        sat_multiplier = self.compute_saturation_multiplier(prompt)
        target_saturation = torch.clamp(original_saturation * sat_multiplier, 0.1, 0.8)
        print(f"Saturation multiplier: {sat_multiplier:.2f} -> target: {target_saturation.item():.3f}")
        
        # Encode prompt once
        print("Encoding prompt...")
        text_embeddings = self.critic.encode_prompt(prompt, negative_prompt)
        
        # Optimizer - higher LR since SDS gradients can be weak
        optimizer = optim.AdamW(self.processor.parameters(), lr=0.1, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_steps, eta_min=0.01)
        
        best_loss = float('inf')
        best_state = None
        
        print(f"\nOptimizing with prompt: '{prompt}'")
        print(f"Weights: sds={sds_weight}, exposure={exposure_weight}, color={color_weight}")
        print(f"Guidance scale: {guidance_scale}")
        print("-" * 60)
        
        # Setup live view
        if live_view:
            cv2.namedWindow('DiffRAW v3', cv2.WINDOW_NORMAL)
        
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # Process image
            processed = self.processor(raw_image)
            
            # Resize for diffusion model (512x512)
            processed_resized = F.interpolate(processed, size=(512, 512), 
                                             mode='bilinear', align_corners=False)
            
            # Compute losses
            losses = {}
            
            # SDS loss from diffusion model
            # Use LOW timesteps to prevent wash-out (key insight!)
            sds_loss = self.critic.compute_sds_loss(
                processed_resized, 
                text_embeddings,
                guidance_scale=guidance_scale,
                min_step=0.02,
                max_step=0.15,  # Very low noise = preserve details
                num_samples=sds_samples,
            )
            losses['sds'] = sds_loss * sds_weight
            
            # Exposure loss
            losses['exposure'] = self.compute_exposure_loss(processed) * exposure_weight
            
            # Color loss (with saturation guidance from prompt)
            losses['color'] = self.compute_color_loss(processed, original_saturation, target_saturation) * color_weight
            
            # Total
            total_loss = sum(losses.values())
            
            # Backward
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
            if (step + 1) % 10 == 0 or step == 0:
                loss_str = ", ".join([f"{k}={v.item():.4f}" for k, v in losses.items()])
                print(f"Step {step+1:3d}/{num_steps}: {loss_str}")
            
            # Live view (every step)
            if live_view:
                with torch.no_grad():
                    display_img = processed.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    display_img = (display_img * 255).astype(np.uint8)
                    display_img = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
                    if live_view_scale != 1.0:
                        h, w = display_img.shape[:2]
                        new_size = (int(w * live_view_scale), int(h * live_view_scale))
                        display_img = cv2.resize(display_img, new_size)
                    cv2.putText(display_img, f"Step {step+1}/{num_steps}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.imshow('DiffRAW v3', display_img)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\nEarly stop requested.")
                        break
        
        if live_view:
            cv2.destroyAllWindows()
        
        # Restore best
        if best_state is not None:
            self.processor.load_state_dict(best_state)
        
        print("-" * 60)
        print("Optimization complete!")
        print("\nFinal parameters:")
        print(self.processor.get_parameter_summary())
        
        return self.processor


def main():
    parser = argparse.ArgumentParser(
        description="Differentiable RAW processor v3 with diffusion critic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_v3.py photo.ARW --prompt "cinematic, moody, film look"
  python demo_v3.py photo.CR2 --prompt "bright, airy, professional portrait"
  python demo_v3.py photo.NEF --prompt "dramatic landscape, golden hour" --guidance 75
        """
    )
    parser.add_argument("input_file", type=str, help="Path to input RAW file")
    parser.add_argument("--prompt", type=str, required=True, help="Style description")
    parser.add_argument("--negative-prompt", type=str,
                        default="blurry, low quality, oversaturated, undersaturated, ugly",
                        help="What to avoid")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    parser.add_argument("--steps", type=int, default=100, help="Optimization steps")
    parser.add_argument("--size", type=int, default=768, help="Processing size")
    parser.add_argument("--sds-weight", type=float, default=5.0, help="SDS loss weight")
    parser.add_argument("--exposure-weight", type=float, default=0.3, help="Exposure loss weight")
    parser.add_argument("--color-weight", type=float, default=0.5, help="Color loss weight")
    parser.add_argument("--guidance", type=float, default=30.0, help="CFG guidance scale")
    parser.add_argument("--sds-samples", type=int, default=4, help="SDS samples per step (more=stable, slower)")
    parser.add_argument("--live", action="store_true", help="Show live preview")
    parser.add_argument("--live-scale", type=float, default=1.0, help="Live preview scale")
    
    args = parser.parse_args()
    
    # Output path
    if args.output is None:
        input_path = Path(args.input_file)
        args.output = str(input_path.parent / f"{input_path.stem}_v3.jpg")
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load RAW
    print(f"\nLoading: {args.input_file}")
    importer = RawImporter(target_size=(args.size, args.size))
    raw_image = importer.import_raw(args.input_file)
    raw_image = raw_image.to(device)
    print(f"Image shape: {raw_image.shape}")
    
    # Create processor
    processor = DifferentiableProcessorV2(image_size=(args.size, args.size))
    
    # Create optimizer with diffusion critic
    optimizer_module = ProcessorOptimizerV3(processor, device=device)
    
    # Optimize
    optimized_processor = optimizer_module.optimize(
        raw_image,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_steps=args.steps,
        sds_weight=args.sds_weight,
        exposure_weight=args.exposure_weight,
        color_weight=args.color_weight,
        guidance_scale=args.guidance,
        sds_samples=args.sds_samples,
        live_view=args.live,
        live_view_scale=args.live_scale,
    )
    
    # Generate output
    print("\nGenerating final output...")
    with torch.no_grad():
        processed_image = optimized_processor(raw_image.unsqueeze(0))
    
    # Save
    processed_np = (processed_image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    imageio.imwrite(args.output, processed_np, quality=95)
    print(f"Output saved to: {args.output}")
    
    # Comparison
    comparison_path = Path(args.output).parent / f"{Path(args.output).stem}_comparison.jpg"
    with torch.no_grad():
        baseline = raw_image.unsqueeze(0)
        baseline = torch.pow(baseline.clamp(min=1e-6), 1/2.2)
        baseline_np = (baseline.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    comparison = np.concatenate([baseline_np, processed_np], axis=1)
    imageio.imwrite(str(comparison_path), comparison, quality=95)
    print(f"Comparison saved to: {comparison_path}")


if __name__ == "__main__":
    main()

