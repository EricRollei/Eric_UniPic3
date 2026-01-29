"""
Eric UniPic3 Nodes
ComfyUI node package for UniPic3 multi-image composition and editing.

UniPic3 is a unified multimodal framework for:
- Single-image editing (1 image + prompt)
- Multi-image composition (2-6 images + prompt)
- Human-Object Interaction (HOI) composition

NOTE: UniPic3 does NOT support text-to-image generation.
For T2I, use UniPic1 or UniPic2 instead.

Model variants:
- Teacher: High quality, 50 steps
- DMD: Fast inference, 8 steps
- Consistency: Fast inference, ≤8 steps (geometric alignment focused)

Author: Eric Hiss (GitHub: EricRollei)
License: MIT
"""

from .nodes.eric_unipic3_loader import EricUniPic3Loader, EricUniPic3Unload
from .nodes.eric_unipic3_edit import EricUniPic3ImageEdit
from .nodes.eric_unipic3_compose import EricUniPic3Compose

# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "EricUniPic3Loader": EricUniPic3Loader,
    "EricUniPic3Unload": EricUniPic3Unload,
    "EricUniPic3ImageEdit": EricUniPic3ImageEdit,
    "EricUniPic3Compose": EricUniPic3Compose,
}

# Display names for the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "EricUniPic3Loader": "Eric UniPic3 Load Model",
    "EricUniPic3Unload": "Eric UniPic3 Unload Model",
    "EricUniPic3ImageEdit": "Eric UniPic3 Image Edit",
    "EricUniPic3Compose": "Eric UniPic3 Compose (HOI)",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
