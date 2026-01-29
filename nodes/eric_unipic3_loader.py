"""
Eric UniPic3 Loader
Load UniPic3 pipeline with custom transformer weights.

Author: Eric Hiss (GitHub: EricRollei)
License: MIT
"""

import os
import torch
from typing import Tuple, Optional

from .eric_unipic3_utils import (
    get_pipeline_cache,
    clear_pipeline_cache,
    get_default_paths,
)


def scan_transformer_folders() -> list:
    """Scan for available transformer folders."""
    paths = get_default_paths()
    available = []
    
    # Check each default path
    for name, path in paths.items():
        if name.startswith("transformer_") and os.path.isdir(path):
            # Check for model files
            has_model = any(
                f.endswith(('.safetensors', '.bin'))
                for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f))
            )
            if has_model:
                variant = name.replace("transformer_", "")
                available.append(f"{variant}: {path}")
    
    # Also check H:\Testing for any Unipic folders
    test_dir = r"H:\Testing"
    if os.path.isdir(test_dir):
        for item in os.listdir(test_dir):
            if "unipic" in item.lower():
                item_path = os.path.join(test_dir, item)
                if os.path.isdir(item_path):
                    # Check transformer subfolder
                    transformer_path = os.path.join(item_path, "transformer")
                    ema_path = os.path.join(item_path, "ema_transformer")
                    
                    for path in [transformer_path, ema_path]:
                        if os.path.isdir(path):
                            entry = f"custom: {path}"
                            if entry not in available:
                                available.append(entry)
    
    if not available:
        available = ["teacher: H:\\Testing\\Unipic3\\transformer"]
    
    return available


class EricUniPic3Loader:
    """
    Load the UniPic3 pipeline.
    
    UniPic3 uses the QwenImageEditPlusPipeline architecture with a custom
    transformer trained for multi-image composition. This node loads the
    pipeline with your choice of transformer variant.
    
    Variants:
    - teacher: Full 50-step diffusion, highest quality
    - dmd: 8-step DMD distillation, fast inference
    - consistency: ≤8-step consistency model, fast inference
    """
    
    CATEGORY = "Eric UniPic3"
    FUNCTION = "load_pipeline"
    RETURN_TYPES = ("UNIPIC3_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    
    @classmethod
    def INPUT_TYPES(cls):
        transformer_options = scan_transformer_folders()
        default_paths = get_default_paths()
        
        return {
            "required": {
                "variant": (["teacher", "dmd", "consistency"], {
                    "default": "teacher",
                    "tooltip": "Model variant: teacher (50 steps), dmd (8 steps), consistency (≤8 steps)"
                }),
                "base_pipeline_path": ("STRING", {
                    "default": default_paths["base_pipeline"],
                    "tooltip": "Path to Qwen-Image-Edit-2511 or similar base pipeline with VAE/text encoder"
                }),
            },
            "optional": {
                "transformer_path_override": ("STRING", {
                    "default": "",
                    "tooltip": "Override transformer path (leave empty to use variant default)"
                }),
                "precision": (["bf16", "fp16", "fp32"], {
                    "default": "bf16",
                    "tooltip": "Model precision (bf16 recommended for RTX 40/50 series)"
                }),
                "device": (["cuda", "cuda:0", "cuda:1", "cpu"], {
                    "default": "cuda",
                    "tooltip": "Device to load model on"
                }),
                "keep_in_vram": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep model in VRAM between runs (faster but uses memory)"
                }),
            }
        }
    
    def load_pipeline(
        self,
        variant: str,
        base_pipeline_path: str,
        transformer_path_override: str = "",
        precision: str = "bf16",
        device: str = "cuda",
        keep_in_vram: bool = True,
    ) -> Tuple:
        """
        Load the UniPic3 pipeline.
        
        Args:
            variant: Model variant (teacher/dmd/consistency)
            base_pipeline_path: Path to base pipeline with VAE/text encoder
            transformer_path_override: Override path for transformer
            precision: Model precision
            device: Target device
            keep_in_vram: Whether to cache the pipeline
            
        Returns:
            Tuple containing the pipeline wrapper
        """
        from diffusers import QwenImageEditPlusPipeline
        from diffusers.models import QwenImageTransformer2DModel
        
        # Determine transformer path
        if transformer_path_override and os.path.isdir(transformer_path_override):
            transformer_path = transformer_path_override
        else:
            default_paths = get_default_paths()
            transformer_key = f"transformer_{variant}"
            transformer_path = default_paths.get(transformer_key)
            
            if not transformer_path or not os.path.isdir(transformer_path):
                raise ValueError(f"Transformer path not found for variant '{variant}': {transformer_path}")
        
        print(f"[EricUniPic3] Loading variant: {variant}")
        print(f"[EricUniPic3] Transformer path: {transformer_path}")
        print(f"[EricUniPic3] Base pipeline: {base_pipeline_path}")
        
        # Check cache
        cache = get_pipeline_cache()
        if (cache["pipeline"] is not None and 
            cache["variant"] == variant and 
            cache["transformer_path"] == transformer_path):
            print("[EricUniPic3] Using cached pipeline")
            return ({"pipeline": cache["pipeline"], "variant": variant},)
        
        # Clear existing cache if loading different model
        if cache["pipeline"] is not None:
            print("[EricUniPic3] Clearing cached pipeline (loading different variant)")
            clear_pipeline_cache()
        
        # Set precision
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        dtype = dtype_map.get(precision, torch.bfloat16)
        
        # Load transformer from UniPic3 weights
        print(f"[EricUniPic3] Loading transformer from: {transformer_path}")
        transformer = QwenImageTransformer2DModel.from_pretrained(
            transformer_path,
            torch_dtype=dtype,
            local_files_only=True,
        )
        
        # Load base pipeline (VAE, text encoder, scheduler, etc.)
        print(f"[EricUniPic3] Loading base pipeline from: {base_pipeline_path}")
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            base_pipeline_path,
            transformer=transformer,
            torch_dtype=dtype,
            local_files_only=True,
        )
        
        # Move to device
        pipeline = pipeline.to(device)
        
        # Enable VAE tiling for high-res decode
        pipeline.vae.enable_tiling()
        
        # Cache if requested
        if keep_in_vram:
            cache["pipeline"] = pipeline
            cache["variant"] = variant
            cache["transformer_path"] = transformer_path
        
        print(f"[EricUniPic3] Pipeline loaded successfully on {device}")
        print(f"[EricUniPic3] Transformer parameters: {sum(p.numel() for p in transformer.parameters()) / 1e9:.2f}B")
        
        return ({"pipeline": pipeline, "variant": variant},)


class EricUniPic3Unload:
    """
    Unload the UniPic3 pipeline from VRAM.
    
    Use this node to free GPU memory when done with UniPic3.
    """
    
    CATEGORY = "Eric UniPic3"
    FUNCTION = "unload"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "trigger": ("*", {
                    "tooltip": "Connect any output here to trigger unload after that node completes"
                }),
            }
        }
    
    def unload(self, trigger=None):
        """Unload the pipeline and free VRAM."""
        if clear_pipeline_cache():
            return ("Pipeline unloaded successfully",)
        else:
            return ("No pipeline was loaded",)
