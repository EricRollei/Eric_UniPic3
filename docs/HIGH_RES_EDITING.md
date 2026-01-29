# High Resolution Editing: Removing the 1MP Limit

## Table of Contents

- [Summary](#summary)
- [The Problem](#the-problem)
- [The Discovery](#the-discovery)
- [Where the Limit is Enforced](#where-the-limit-is-enforced)
- [The Fix (Simple)](#the-fix-simple)
- [Proposed Enhancement: Preserve Input Resolution](#proposed-enhancement-preserve-input-resolution)
- [Resolution Recommendations](#resolution-recommendations)
- [VRAM Considerations](#vram-considerations)
- [Dimension Alignment](#dimension-alignment)
- [Technical Background](#technical-background)
- [Reverting the Changes](#reverting-the-changes)
- [Future Considerations](#future-considerations)
- [Related Links](#related-links)

---

## Summary

The Qwen-Image-Edit and UniPic3 models are artificially limited to 1MP (1024x1024) output resolution by hardcoded constants in the diffusers pipeline code. **The models themselves can handle much higher resolutions** - up to 17MP according to user testing.

This document explains the issue, the fix, and proposes a smarter approach for upstream contribution.

---

## The Problem

When using Qwen-Image-Edit (2509, 2511) or UniPic3 for image editing, input images are automatically downscaled to ~1MP regardless of their original resolution. This means:

- A 12MP phone photo gets reduced to 1MP
- All fine details outside the edit region are lost
- Output is always limited to ~1024x1024 (or equivalent aspect ratio)

---

## The Discovery

### GitHub Issue #9481

A user raised this issue with ComfyUI:

> **"Is 1MP fixed resizing necessary for the TextEncodeQwenImageEdit node?"**
> 
> "Currently, the TextEncodeQwenImageEdit node automatically rescales input images so that their total pixel count is fixed to around 1 million. Is it technically necessary for this processing step to always rescale inputs to ~1 million pixels?"
>
> "**The base Qwen-Image model works well in the range of 1.5M-17M pixels. Qwen-Image-Edit also works properly within this range without particular issues. In fact, image quality tends to improve as resolution increases.**"

Source: https://github.com/Comfy-Org/ComfyUI/issues/9481

### Qwen-Image-Edit-2511 Supports Higher Resolution

The newer Qwen-Image-Edit-2511 model officially supports **2560x2560 (~6.5MP)** output:

> "With native whopping 2560x2560 pixels image output capability and with only 12 steps it is next level."

Source: https://github.com/QwenLM/Qwen-Image/issues/241

### UniPic3 Uses Same Architecture

UniPic3 is built on the exact same pipeline as Qwen-Image-Edit:

> "Our model architecture follows that of Qwen-Image, which incorporates Qwen2.5-VL as the condition encoder, employs a VAE as the image tokenizer, and utilizes MMDiT as the backbone diffusion model."

Source: https://arxiv.org/html/2601.15664

---

## Where the Limit is Enforced

The 1MP limit is hardcoded in the **diffusers** library, not in the models themselves.

### Location
```
{python_env}/Lib/site-packages/diffusers/pipelines/qwenimage/
```

### Files Affected

| File | Limit Location |
|------|----------------|
| `pipeline_qwenimage_edit_plus.py` | Line ~42: `VAE_IMAGE_SIZE = 1024 * 1024` |
| `pipeline_qwenimage_edit.py` | In `__call__`: `calculate_dimensions(1024 * 1024, ...)` |
| `pipeline_qwenimage_edit_inpaint.py` | In `__call__`: `calculate_dimensions(1024 * 1024, ...)` |

---

## The Fix (Simple)

This is the quick fix - just raise the hardcoded limit.

### Edit 1: `pipeline_qwenimage_edit_plus.py`

This affects: **UniPic3, Qwen-Image-Edit-2509, Qwen-Image-Edit-2511**

Find near line 42:
```python
CONDITION_IMAGE_SIZE = 384 * 384
VAE_IMAGE_SIZE = 1024 * 1024
```

Change to:
```python
CONDITION_IMAGE_SIZE = 384 * 384
# VAE_IMAGE_SIZE = 1024 * 1024  # Original 1MP limit
VAE_IMAGE_SIZE = 8 * 1024 * 1024  # PATCHED: 8MP max for high-res editing
```

Also find in the `__call__` method (around line 450):
```python
calculated_width, calculated_height, _ = calculate_dimensions(1024 * 1024, image_size[0] / image_size[1])
```

Change to use the constant:
```python
calculated_width, calculated_height, _ = calculate_dimensions(VAE_IMAGE_SIZE, image_size[0] / image_size[1])
```

### Edit 2: `pipeline_qwenimage_edit.py`

This affects: **Basic Qwen-Image-Edit**

Find in the `__call__` method (around line 580):
```python
calculated_width, calculated_height, _ = calculate_dimensions(1024 * 1024, image_size[0] / image_size[1])
```

Change to:
```python
calculated_width, calculated_height, _ = calculate_dimensions(8 * 1024 * 1024, image_size[0] / image_size[1])
```

### Edit 3: `pipeline_qwenimage_edit_inpaint.py`

This affects: **Qwen-Image-Edit Inpainting**

Find the same pattern and change `1024 * 1024` to `8 * 1024 * 1024`.

---

## Proposed Enhancement: Preserve Input Resolution

The simple fix above always fills UP TO the budget. A smarter approach would **preserve input resolution** (properly aligned) with a cap at the maximum.

### Current Behavior (After Simple Fix)

```python
# Always fills UP TO the budget, regardless of input size
calculated_width, calculated_height = calculate_dimensions(VAE_IMAGE_SIZE, ratio)
```

- 2MP input -> 8MP output (unnecessarily upscaled)
- 6MP input -> 8MP output (upscaled)
- 0.5MP input -> 8MP output (massively upscaled, may hallucinate detail)

### Proposed Behavior

```python
# Preserve input size, but cap at maximum for VRAM safety
input_pixels = image_size[0] * image_size[1]
target_pixels = min(input_pixels, VAE_IMAGE_SIZE)
calculated_width, calculated_height = calculate_dimensions(target_pixels, ratio)
```

- 2MP input -> 2MP output (preserved)
- 6MP input -> 6MP output (preserved)
- 12MP input -> 8MP output (capped)
- 0.5MP input -> 0.5MP output (preserved, no hallucination)

### Why This Is Better

1. **Quality**: The model was trained at ~1MP. While it generalizes well to higher resolutions, asking it to upscale a tiny input to 8MP requires hallucinating detail it cannot reliably generate.

2. **Efficiency**: No wasted computation upscaling images that don't need it.

3. **Predictability**: Output size relates logically to input size.

4. **VRAM Safety**: Cap prevents OOM on very large inputs.

### Full Implementation

In `pipeline_qwenimage_edit_plus.py`, find the dimension calculation in `__call__` (around line 450):

```python
# Current code
calculated_width, calculated_height, _ = calculate_dimensions(
    VAE_IMAGE_SIZE, image_size[0] / image_size[1]
)
```

Replace with:

```python
# Preserve input resolution, cap at VAE_IMAGE_SIZE
input_pixels = image_size[0] * image_size[1]
target_pixels = min(input_pixels, VAE_IMAGE_SIZE)
calculated_width, calculated_height, _ = calculate_dimensions(
    target_pixels, image_size[0] / image_size[1]
)
```

Apply the same pattern to `pipeline_qwenimage_edit.py` and `pipeline_qwenimage_edit_inpaint.py`.

### Optional: Expose as Pipeline Parameter

For maximum flexibility, the pipeline could accept an optional `target_pixels` parameter:

```python
def __call__(
    self,
    ...
    target_pixels: Optional[int] = None,  # None = preserve input, capped at VAE_IMAGE_SIZE
):
    ...
    input_pixels = image_size[0] * image_size[1]
    if target_pixels is None:
        target_pixels = min(input_pixels, VAE_IMAGE_SIZE)
    else:
        target_pixels = min(target_pixels, VAE_IMAGE_SIZE)  # Always cap for safety
    
    calculated_width, calculated_height, _ = calculate_dimensions(
        target_pixels, image_size[0] / image_size[1]
    )
```

This allows:
- **Default**: Preserve input size (capped at max)
- **Explicit**: `target_pixels=4*1024*1024` to force 4MP output
- **Node-level control**: Without patching diffusers for each use case

---

## Resolution Recommendations

| Setting | Pixels | Notes |
|---------|--------|-------|
| `1024 * 1024` | 1MP | Original default |
| `2 * 1024 * 1024` | 2MP | Safe for most GPUs |
| `4 * 1024 * 1024` | 4MP | Good balance |
| `6 * 1024 * 1024` | 6MP | Matches Qwen-Edit-2511 native |
| `8 * 1024 * 1024` | 8MP | High-end GPUs |

---

## VRAM Considerations

### Tested Configuration
- **GPU**: NVIDIA RTX PRO 6000 Blackwell (96GB VRAM)
- **Result**: ~70% VRAM usage at 8MP budget with VAE tiling enabled

### VAE Tiling Required for High Resolution

Without VAE tiling, the VAE decode step can OOM even on high-VRAM GPUs. Enable tiling after pipeline load:

```python
pipeline.vae.enable_tiling()
```

This processes the latent decode in chunks, dramatically reducing peak VRAM during decode.

### VRAM Scaling (Approximate)

| Resolution | Estimated VRAM (with tiling) |
|------------|------------------------------|
| 1MP | ~12GB |
| 2MP | ~18GB |
| 4MP | ~28GB |
| 8MP | ~45GB |

These are rough estimates. Actual usage depends on model variant, number of reference images, and batch size.

---

## Dimension Alignment

The pipeline uses a divisor of **32** for dimension alignment:

```python
def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return width, height
```

This comes from `vae_scale_factor * 2` = 8 x 2 x 2 = 32 (accounting for VAE compression and latent packing).

---

## Technical Background

### Why the Limit Exists

The 1MP limit was likely set as a conservative default because:
1. The models were trained/benchmarked at 1024x1024
2. Higher resolutions require more VRAM
3. Multi-image composition (UniPic3) multiplies memory requirements

### Why Higher Resolutions Work

The underlying architecture has no hard resolution limit:
- **Qwen2.5-VL encoder**: Supports up to 16,384 tokens/image (~3584x3584 pixels)
- **MMDiT transformer**: Attention scales with sequence length, no fixed limit
- **VAE**: Standard architecture, no resolution constraint

The models were trained with resolution-aware techniques and can generalize beyond their training resolution.

---

## Reverting the Changes

To restore the original 1MP limit, change the values back to:
```python
VAE_IMAGE_SIZE = 1024 * 1024
```
or
```python
calculate_dimensions(1024 * 1024, ...)
```

---

## Future Considerations

1. **Diffusers updates** may overwrite these changes - check after updating
2. **VRAM usage** scales with resolution - monitor on lower-end GPUs
3. **Quality testing** - verify edits look correct at higher resolutions
4. **Upstream contribution** - propose the "preserve input resolution" approach to diffusers maintainers

---

## Related Links

- GitHub Issue: https://github.com/Comfy-Org/ComfyUI/issues/9481
- Qwen-Image-Edit-2511 Announcement: https://github.com/QwenLM/Qwen-Image/issues/241
- UniPic3 Paper: https://arxiv.org/abs/2601.15664
- Qwen-Image GitHub: https://github.com/QwenLM/Qwen-Image
- Hugging Face Diffusers: https://github.com/huggingface/diffusers

---

*Document created: January 2026*
*Author: Eric Hiss / Claude collaboration*
