#!/usr/bin/env python3
"""
Manual RAW Editor
=================
A simple tkinter GUI for manually adjusting RAW processing parameters.
All filters are PyTorch modules from filters.py.
"""

import argparse
import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
from PIL import Image, ImageTk
import torch
import rawpy
from torchvision import transforms

from filters import (
    WhiteBalance, AgXToneMapping, SaturationVibrance, HSLAdjustment,
    SplitToning, ClarityTexture, Dehaze, OrtonEffect, Vignette,
    Sharpening, SkinToneProtection, DifferentiableProcessor
)


class RawEditor:
    def __init__(self, root, image_path=None, preview_size=800):
        self.root = root
        self.root.title("DiffRAW Editor")
        self.preview_size = preview_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Raw image data
        self.raw_image = None
        self.display_image = None
        
        # Create processor
        self.processor = DifferentiableProcessor(image_size=(preview_size, preview_size))
        self.processor.to(self.device)
        self.processor.eval()
        
        # Build UI
        self._build_ui()
        
        # Load image if provided
        if image_path:
            self.load_image(image_path)
    
    def _build_ui(self):
        # Main layout: left panel (controls), right panel (image)
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - scrollable controls
        self.control_frame = ttk.Frame(self.main_frame, width=300)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        
        # Canvas with scrollbar for controls
        self.control_canvas = tk.Canvas(self.control_frame, width=280)
        self.scrollbar = ttk.Scrollbar(self.control_frame, orient=tk.VERTICAL, 
                                        command=self.control_canvas.yview)
        self.scrollable_frame = ttk.Frame(self.control_canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))
        )
        
        self.control_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.control_canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Enable mousewheel scrolling
        self.control_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.control_canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.control_canvas.bind_all("<Button-5>", self._on_mousewheel)
        
        # Right panel - image display
        self.image_frame = ttk.Frame(self.main_frame)
        self.image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.image_label = ttk.Label(self.image_frame, text="Load an image to begin")
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Add controls
        self._add_controls()
        
        # Menu bar
        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open RAW...", command=self.open_file)
        filemenu.add_command(label="Save JPEG...", command=self.save_file)
        filemenu.add_separator()
        filemenu.add_command(label="Reset All", command=self.reset_all)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=filemenu)
        self.root.config(menu=menubar)
    
    def _on_mousewheel(self, event):
        if event.num == 4:
            self.control_canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.control_canvas.yview_scroll(1, "units")
        else:
            self.control_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def _add_controls(self):
        """Add all the slider controls"""
        self.sliders = {}
        
        # Load button
        load_btn = ttk.Button(self.scrollable_frame, text="Open RAW File...", command=self.open_file)
        load_btn.pack(fill=tk.X, pady=5)
        
        # Group: White Balance
        self._add_section("White Balance")
        self._add_slider("temperature", "Temperature", -1.0, 1.0, 0.0, 
                        "white_balance", "temperature")
        self._add_slider("tint", "Tint", -1.0, 1.0, 0.0,
                        "white_balance", "tint")
        
        # Group: Tone
        self._add_section("Tone")
        self._add_slider("exposure", "Exposure", -2.0, 4.0, 1.5,
                        "tone_mapping", "exposure")
        self._add_slider("contrast", "Contrast", -1.0, 1.0, 0.0,
                        "tone_mapping", "contrast")
        self._add_slider("gamma", "Gamma", -1.0, 1.0, 0.0,
                        "tone_mapping", "gamma")
        
        # Group: Color
        self._add_section("Color")
        self._add_slider("saturation", "Saturation", -1.0, 1.0, 0.0,
                        "saturation_vibrance", "saturation")
        self._add_slider("vibrance", "Vibrance", -1.0, 1.0, 0.0,
                        "saturation_vibrance", "vibrance")
        
        # Group: Detail
        self._add_section("Detail")
        self._add_slider("clarity", "Clarity", -1.0, 1.0, 0.0,
                        "clarity_texture", "clarity")
        self._add_slider("texture", "Texture", -1.0, 1.0, 0.0,
                        "clarity_texture", "texture")
        self._add_slider("sharpening", "Sharpening", -5.0, 2.0, 0.0,
                        "sharpening", "amount")
        
        # Group: Effects
        self._add_section("Effects")
        self._add_slider("dehaze", "Dehaze", -1.0, 1.0, 0.0,
                        "dehaze", "amount")
        self._add_slider("orton", "Orton/Glow", -5.0, 2.0, 0.0,
                        "orton", "amount")
        self._add_slider("vignette", "Vignette", -1.0, 1.0, 0.0,
                        "vignette", "amount")
        
        # Group: Split Toning
        self._add_section("Split Toning")
        self._add_slider("shadow_hue", "Shadow Hue", 0.0, 1.0, 0.6,
                        "split_toning", "shadow_hue")
        self._add_slider("shadow_sat", "Shadow Saturation", -5.0, 2.0, 0.0,
                        "split_toning", "shadow_saturation")
        self._add_slider("highlight_hue", "Highlight Hue", 0.0, 1.0, 0.1,
                        "split_toning", "highlight_hue")
        self._add_slider("highlight_sat", "Highlight Saturation", -5.0, 2.0, 0.0,
                        "split_toning", "highlight_saturation")
        self._add_slider("split_balance", "Balance", 0.0, 1.0, 0.5,
                        "split_toning", "balance")
        
        # Group: Skin
        self._add_section("Skin Tones")
        self._add_slider("skin_warmth", "Skin Warmth", -1.0, 1.0, 0.0,
                        "skin_protection", "warmth")
        self._add_slider("skin_smooth", "Skin Smoothness", -5.0, 2.0, 0.0,
                        "skin_protection", "smoothness")
        
        # Reset button
        reset_btn = ttk.Button(self.scrollable_frame, text="Reset All", command=self.reset_all)
        reset_btn.pack(fill=tk.X, pady=10)
        
        # Save button
        save_btn = ttk.Button(self.scrollable_frame, text="Save JPEG...", command=self.save_file)
        save_btn.pack(fill=tk.X, pady=5)
    
    def _add_section(self, title):
        """Add a section header"""
        frame = ttk.Frame(self.scrollable_frame)
        frame.pack(fill=tk.X, pady=(15, 5))
        label = ttk.Label(frame, text=title, font=('TkDefaultFont', 10, 'bold'))
        label.pack(anchor=tk.W)
        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)
    
    def _add_slider(self, name, label, min_val, max_val, default, module_name, param_name):
        """Add a slider control"""
        frame = ttk.Frame(self.scrollable_frame)
        frame.pack(fill=tk.X, pady=2)
        
        # Label
        lbl = ttk.Label(frame, text=label, width=18)
        lbl.pack(side=tk.LEFT)
        
        # Value label
        val_var = tk.StringVar(value=f"{default:.2f}")
        val_lbl = ttk.Label(frame, textvariable=val_var, width=6)
        val_lbl.pack(side=tk.RIGHT)
        
        # Use tk.Scale instead of ttk.Scale for better callback support
        slider = tk.Scale(frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL,
                         resolution=0.01, showvalue=False, length=150,
                         command=lambda v, mn=module_name, p=param_name, vv=val_var: 
                                 self._on_slider_change(float(v), mn, p, vv))
        slider.set(default)
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.sliders[name] = {
            'slider': slider,
            'module_name': module_name,
            'param': param_name,
            'default': default,
            'val_var': val_var
        }
    
    def _on_slider_change(self, value, module_name, param_name, val_var):
        """Handle slider value change"""
        # Get the current module from processor (handles processor recreation)
        module = getattr(self.processor, module_name)
        
        # Update the parameter
        with torch.no_grad():
            param = getattr(module, param_name)
            param.fill_(value)
        
        # Update value label
        val_var.set(f"{value:.2f}")
        
        # Update preview
        self.update_preview()
    
    def load_image(self, path):
        """Load a RAW image file"""
        try:
            with rawpy.imread(path) as raw:
                rgb = raw.postprocess(
                    output_bps=16,
                    no_auto_bright=True,
                    use_camera_wb=True,
                    gamma=(1, 1),
                    output_color=rawpy.ColorSpace.sRGB,
                )
            
            # Convert to tensor
            img = torch.from_numpy(rgb.astype(np.float32) / 65535.0).permute(2, 0, 1)
            img = transforms.functional.resize(img, (self.preview_size, self.preview_size), 
                                               antialias=True)
            self.raw_image = img.unsqueeze(0).to(self.device)
            
            # Recreate processor with correct size (for vignette)
            h, w = self.raw_image.shape[2], self.raw_image.shape[3]
            old_state = {k: v.clone() for k, v in self.processor.state_dict().items() 
                        if 'xx' not in k and 'yy' not in k}
            self.processor = DifferentiableProcessor(image_size=(h, w))
            self.processor.to(self.device)
            
            # Restore parameters
            new_state = self.processor.state_dict()
            for k, v in old_state.items():
                if k in new_state and new_state[k].shape == v.shape:
                    new_state[k] = v
            self.processor.load_state_dict(new_state)
            self.processor.eval()
            
            self.root.title(f"DiffRAW Editor - {path}")
            self.update_preview()
            
        except Exception as e:
            print(f"Error loading image: {e}")
            import traceback
            traceback.print_exc()

    def update_preview(self):
        """Update the image preview"""
        if self.raw_image is None:
            return
        
        with torch.no_grad():
            processed = self.processor(self.raw_image)
            
            # Convert to PIL Image
            img_np = (processed.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            
            # Resize to fit window if needed
            display_size = min(self.preview_size, 900)
            pil_img.thumbnail((display_size, display_size), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            self.display_image = ImageTk.PhotoImage(pil_img)
            self.image_label.configure(image=self.display_image)
    
    def open_file(self):
        """Open file dialog to load a RAW image"""
        filetypes = [
            ("RAW files", "*.arw *.ARW *.cr2 *.CR2 *.cr3 *.CR3 *.nef *.NEF *.dng *.DNG *.raf *.RAF *.orf *.ORF *.rw2 *.RW2"),
            ("All files", "*.*")
        ]
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            self.load_image(path)
    
    def save_file(self):
        """Save the processed image"""
        if self.raw_image is None:
            return
        
        filetypes = [("JPEG", "*.jpg"), ("PNG", "*.png")]
        path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=filetypes)
        if path:
            with torch.no_grad():
                processed = self.processor(self.raw_image)
                img_np = (processed.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np)
                pil_img.save(path, quality=95)
                print(f"Saved to {path}")
    
    def reset_all(self):
        """Reset all sliders to defaults"""
        for name, info in self.sliders.items():
            info['slider'].set(info['default'])
            info['val_var'].set(f"{info['default']:.2f}")
            with torch.no_grad():
                module = getattr(self.processor, info['module_name'])
                param = getattr(module, info['param'])
                param.fill_(info['default'])
        self.update_preview()


def main():
    parser = argparse.ArgumentParser(description="Manual RAW Editor")
    parser.add_argument("input_file", nargs="?", default=None, help="RAW file to open")
    parser.add_argument("--size", type=int, default=800, help="Preview size")
    args = parser.parse_args()
    
    root = tk.Tk()
    root.geometry("1200x900")
    
    editor = RawEditor(root, image_path=args.input_file, preview_size=args.size)
    
    root.mainloop()


if __name__ == "__main__":
    main()

