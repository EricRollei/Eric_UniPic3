"""
Eric UniPic3 Image Edit
Edit a single image with optional reference images using UniPic3.

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


class EricUniPic3ImageEdit:
    """
    Edit a single image using UniPic3 with optional reference images.
    
    UniPic3 excels at semantic image editing. The primary image is what
    you want to edit. Optional reference images can provide:
    
    - Style reference: "Edit in the style of the reference"
    - Content to add: "Add the hat from reference to the person"
    - Replacement elements: "Replace the shirt with the one from reference"
    - Context/background: "Place the subject in the scene from reference"
    
    For composing multiple subjects together (e.g., person + multiple
    clothing items), use the Compose (HOI) node instead.
    
    Examples:
    - Single image only: "Change the background to sunset"
    - With clothing ref: "Put the jacket from reference on the person"
    - With style ref: "Edit this photo in the painterly style of reference"
    """
    
    CATEGORY = "Eric UniPic3"
    FUNCTION = "edit"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("UNIPIC3_PIPELINE",),
                "image": ("IMAGE", {
                    "tooltip": "Primary image to edit"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Edit this image to...",
                    "tooltip": "Describe the edit to apply"
                }),
            },
            "optional": {
                "ref_image1": ("IMAGE", {
                    "tooltip": "Reference image for style, content, or elements"
                }),
                "ref_image2": ("IMAGE", {
                    "tooltip": "Additional reference image"
                }),
                "ref_image3": ("IMAGE", {
                    "tooltip": "Additional reference image"
                }),
                "ref_image4": ("IMAGE", {
                    "tooltip": "Additional reference image"
                }),
                "ref_image5": ("IMAGE", {
                    "tooltip": "Additional reference image"
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
    
    def edit(
        self,
        pipeline: dict,
        image: torch.Tensor,
        prompt: str,
        ref_image1: Optional[torch.Tensor] = None,
        ref_image2: Optional[torch.Tensor] = None,
        ref_image3: Optional[torch.Tensor] = None,
        ref_image4: Optional[torch.Tensor] = None,
        ref_image5: Optional[torch.Tensor] = None,
        negative_prompt: str = "",
        steps: int = 50,
        true_cfg_scale: float = 4.0,
        seed: int = 0,
    ) -> Tuple[torch.Tensor]:
        """Edit an image with optional references."""
        pipe = pipeline["pipeline"]
        variant = pipeline["variant"]
        
        # Convert primary image
        if image.dim() == 4:
            primary_pil = tensor_to_pil(image[0])
        else:
            primary_pil = tensor_to_pil(image)
        primary_pil = prepare_image_for_pipeline(primary_pil)
        
        # Convert reference images
        def convert_ref(tensor):
            if tensor is None:
                return None
            if tensor.dim() == 4:
                pil = tensor_to_pil(tensor[0])
            else:
                pil = tensor_to_pil(tensor)
            return prepare_image_for_pipeline(pil)
        
        refs = [
            convert_ref(ref_image1),
            convert_ref(ref_image2),
            convert_ref(ref_image3),
            convert_ref(ref_image4),
            convert_ref(ref_image5),
        ]
        
        # Build image list: primary first, then references
        all_images = [primary_pil] + [r for r in refs if r is not None]
        
        print(f"[EricUniPic3] Image Edit")
        print(f"[EricUniPic3] Variant: {variant}, Steps: {steps}, CFG: {true_cfg_scale}")
        print(f"[EricUniPic3] Images: 1 primary + {len(all_images)-1} references")
        print(f"[EricUniPic3] Prompt: {prompt[:80]}...")
        
        device = next(pipe.transformer.parameters()).device
        generator = torch.Generator(device=device).manual_seed(seed)
        
        # Run pipeline - single image or list depending on ref count
        with torch.inference_mode():
            if len(all_images) == 1:
                output = pipe(
                    prompt=prompt,
                    image=primary_pil,
                    negative_prompt=negative_prompt if negative_prompt else " ",
                    num_inference_steps=steps,
                    true_cfg_scale=true_cfg_scale,
                    generator=generator,
                    num_images_per_prompt=1,
                )
            else:
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
        
        print(f"[EricUniPic3] Edit complete: {result_tensor.shape}")
        
        return (result_tensor,)
