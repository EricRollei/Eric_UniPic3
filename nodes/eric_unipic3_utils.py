"""
Eric UniPic3 Utilities
Shared utilities for UniPic3 nodes.

Author: Eric Hiss (GitHub: EricRollei)
License: MIT
"""

import os
import gc
import torch
import numpy as np
from PIL import Image
from typing import Optional, List, Tuple, Dict, Any

# Global pipeline cache
_UNIPIC3_PIPELINE_CACHE: Dict[str, Any] = {
    "pipeline": None,
    "variant": None,
    "transformer_path": None,
}


def get_pipeline_cache() -> Dict[str, Any]:
    """Get the global pipeline cache."""
    return _UNIPIC3_PIPELINE_CACHE


def clear_pipeline_cache() -> bool:
    """Clear the cached UniPic3 pipeline and free VRAM."""
    global _UNIPIC3_PIPELINE_CACHE
    
    if _UNIPIC3_PIPELINE_CACHE["pipeline"] is not None:
        print("[EricUniPic3] Unloading UniPic3 pipeline...")
        
        pipeline = _UNIPIC3_PIPELINE_CACHE["pipeline"]
        if hasattr(pipeline, 'transformer'):
            del pipeline.transformer
        if hasattr(pipeline, 'text_encoder'):
            del pipeline.text_encoder
        if hasattr(pipeline, 'vae'):
            del pipeline.vae
        del pipeline
        
        _UNIPIC3_PIPELINE_CACHE["pipeline"] = None
        _UNIPIC3_PIPELINE_CACHE["variant"] = None
        _UNIPIC3_PIPELINE_CACHE["transformer_path"] = None
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        print("[EricUniPic3] Pipeline unloaded, VRAM freed.")
        return True
    else:
        print("[EricUniPic3] No pipeline was loaded.")
        return False


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a ComfyUI image tensor to PIL Image.
    
    Args:
        tensor: Image tensor [H, W, C] with values 0-1
        
    Returns:
        PIL Image in RGB or RGBA mode
    """
    img_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
    
    if img_np.shape[-1] == 4:
        return Image.fromarray(img_np, mode="RGBA")
    elif img_np.shape[-1] == 3:
        return Image.fromarray(img_np, mode="RGB")
    else:
        return Image.fromarray(img_np.squeeze(), mode="L")


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """
    Convert a PIL Image to ComfyUI image tensor.
    
    Args:
        image: PIL Image
        
    Returns:
        Image tensor [H, W, C] with values 0-1
    """
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")
    
    img_np = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(img_np)


def images_to_tensor_batch(images: List[Image.Image]) -> torch.Tensor:
    """
    Convert a list of PIL Images to a batched ComfyUI tensor.
    
    Args:
        images: List of PIL Images
        
    Returns:
        Batched tensor [B, H, W, C]
    """
    tensors = [pil_to_tensor(img) for img in images]
    return torch.stack(tensors, dim=0)


def get_default_paths() -> Dict[str, str]:
    """
    Get default paths for UniPic3 components.
    """
    return {
        "transformer_teacher": r"H:\Testing\Unipic3\transformer",
        "transformer_dmd": r"H:\Testing\Unipic3-DMD\ema_transformer",
        "transformer_consistency": r"H:\Testing\Unipic3-Consistency-Model\ema_transformer",
        "base_pipeline": r"H:\Training\Qwen-Image-Edit-2511",
    }


def get_recommended_steps(variant: str) -> int:
    """
    Get recommended inference steps for each variant.
    
    Args:
        variant: Model variant (teacher/dmd/consistency)
        
    Returns:
        Recommended number of steps
    """
    steps_map = {
        "teacher": 50,
        "dmd": 8,
        "consistency": 8,
    }
    return steps_map.get(variant, 50)


def calculate_dimensions(
    width: int,
    height: int,
    max_pixels: int = 1024 * 1024,
    divisor: int = 16
) -> Tuple[int, int]:
    """
    Calculate output dimensions respecting pixel budget and divisibility.
    
    Args:
        width: Desired width
        height: Desired height
        max_pixels: Maximum total pixels (default 1024*1024)
        divisor: Dimension must be divisible by this (default 16)
        
    Returns:
        Tuple of (adjusted_width, adjusted_height)
    """
    current_pixels = width * height
    
    if current_pixels > max_pixels:
        scale = (max_pixels / current_pixels) ** 0.5
        width = int(width * scale)
        height = int(height * scale)
    
    width = (width // divisor) * divisor
    height = (height // divisor) * divisor
    
    width = max(divisor, width)
    height = max(divisor, height)
    
    return width, height


def prepare_image_for_pipeline(image: Image.Image) -> Image.Image:
    """
    Prepare a single image for the pipeline (convert to RGB).
    
    Args:
        image: PIL Image
        
    Returns:
        RGB PIL Image
    """
    if image.mode == "RGB":
        return image
    elif image.mode == "RGBA":
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        return bg
    else:
        return image.convert("RGB")


def collect_input_images(
    image1: Optional[Image.Image] = None,
    image2: Optional[Image.Image] = None,
    image3: Optional[Image.Image] = None,
    image4: Optional[Image.Image] = None,
    image5: Optional[Image.Image] = None,
    image6: Optional[Image.Image] = None,
) -> List[Image.Image]:
    """
    Collect non-None images into a list.
    
    Args:
        image1-image6: Optional input images
        
    Returns:
        List of images (None values filtered out)
    """
    images = [image1, image2, image3, image4, image5, image6]
    return [prepare_image_for_pipeline(img) for img in images if img is not None]
