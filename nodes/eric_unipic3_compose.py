"""
Eric UniPic3 Compose
Multi-image composition (HOI) using UniPic3.

UniPic3 Model Credits:
- Developed by Skywork AI (https://github.com/SkyworkAI/UniPic)
- Paper: "Skywork UniPic 3.0" (arXiv:2601.15664)
- Authors: Hongyang Wei, Hongbo Liu, Zidong Wang, et al.

ComfyUI Nodes Author: Eric Hiss (GitHub: EricRollei)
License: MIT
"""

import torch
from typing import Tuple, Optional

from .eric_unipic3_utils import (
    tensor_to_pil,
    pil_to_tensor,
    prepare_image_for_pipeline,
)


class EricUniPic3Compose:
    """
    Compose multiple images into a unified scene using UniPic3.
    
    This is UniPic3's signature capability - Human-Object Interaction (HOI)
    composition. It takes 2-6 input images and composes them into a 
    coherent scene where elements naturally interact.
    
    Best use cases:
    - Virtual try-on: Person + clothing items
    - Product photography: Product + background + props
    - Scene composition: Subject + environment + objects
    
    The model handles:
    - Natural spatial relationships
    - Realistic occlusions (clothing folds, object overlaps)
    - Proper lighting and shadows
    - Consistent style across all elements
    
    Image order matters! Typically:
    - image1: Main subject (person, primary object)
    - image2: Primary item to add (main clothing, key object)
    - image3-6: Additional items, accessories, background
    
    Example prompts:
    - "The person wearing the jacket"
    - "Put the dress and shoes on the model"
    - "The subject holding the guitar in front of the background"
    - "Professional photo of person in the outfit with accessories"
    """
    
    CATEGORY = "Eric UniPic3"
    FUNCTION = "compose"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("UNIPIC3_PIPELINE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "The person wearing the clothing",
                    "tooltip": "Describe how to compose the images"
                }),
                "image1": ("IMAGE", {
                    "tooltip": "Primary subject (e.g., person)"
                }),
                "image2": ("IMAGE", {
                    "tooltip": "Item to compose (e.g., clothing, object)"
                }),
            },
            "optional": {
                "image3": ("IMAGE", {
                    "tooltip": "Additional item"
                }),
                "image4": ("IMAGE", {
                    "tooltip": "Additional item"
                }),
                "image5": ("IMAGE", {
                    "tooltip": "Additional item"
                }),
                "image6": ("IMAGE", {
                    "tooltip": "Additional item"
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "What to avoid"
                }),
                "steps": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Inference steps (50 for teacher, 8 for dmd/consistency)"
                }),
                "true_cfg_scale": ("FLOAT", {
                    "default": 4.0,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": "True CFG scale (main quality control)"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed"
                }),
            }
        }
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return kwargs.get("seed", 0)
    
    def compose(
        self,
        pipeline: dict,
        prompt: str,
        image1: torch.Tensor,
        image2: torch.Tensor,
        image3: Optional[torch.Tensor] = None,
        image4: Optional[torch.Tensor] = None,
        image5: Optional[torch.Tensor] = None,
        image6: Optional[torch.Tensor] = None,
        negative_prompt: str = "",
        steps: int = 50,
        true_cfg_scale: float = 4.0,
        seed: int = 0,
    ) -> Tuple[torch.Tensor]:
        """Compose multiple images into a scene."""
        pipe = pipeline["pipeline"]
        variant = pipeline["variant"]
        
        # Convert all images
        def convert(tensor):
            if tensor is None:
                return None
            if tensor.dim() == 4:
                pil = tensor_to_pil(tensor[0])
            else:
                pil = tensor_to_pil(tensor)
            return prepare_image_for_pipeline(pil)
        
        # Build image list (required images first)
        all_images = [
            convert(image1),
            convert(image2),
        ]
        
        # Add optional images
        for img in [image3, image4, image5, image6]:
            converted = convert(img)
            if converted is not None:
                all_images.append(converted)
        
        print(f"[EricUniPic3] Compose (HOI)")
        print(f"[EricUniPic3] Variant: {variant}, Steps: {steps}, CFG: {true_cfg_scale}")
        print(f"[EricUniPic3] Composing {len(all_images)} images")
        for i, img in enumerate(all_images):
            print(f"[EricUniPic3]   Image {i+1}: {img.size}")
        print(f"[EricUniPic3] Prompt: {prompt[:80]}...")
        
        device = next(pipe.transformer.parameters()).device
        generator = torch.Generator(device=device).manual_seed(seed)
        
        # Run pipeline with image list
        with torch.inference_mode():
            output = pipe(
                prompt=prompt,
                image=all_images,
                negative_prompt=negative_prompt if negative_prompt else " ",
                num_inference_steps=steps,
                true_cfg_scale=true_cfg_scale,
                generator=generator,
                num_images_per_prompt=1,
            )
        
        result = output.images[0]
        result_tensor = pil_to_tensor(result).unsqueeze(0)
        
        print(f"[EricUniPic3] Composition complete: {result_tensor.shape}")
        
        return (result_tensor,)
