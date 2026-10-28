"""Optional selective rendering of blue ballpoint handwriting.

This is an additional *recognition* view, not a row-ownership detector. The
original crop must still be OCR'd, and ambiguous physical ownership still
forces REVIEW. No data is inferred from adjacent rows or printed location codes.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps


def isolate_blue_ink(image: Image.Image, *, min_pixels: int = 25) -> Image.Image | None:
    """Make blue-dominant ink black on white; leave black printing out.

    Uses direct B-R and B-G channel dominance, rather than generic saturation
    (which would also preserve scanner artifacts, red marks and black type).
    Returns None on grayscale/monochrome or blue-free cells to avoid asking an
    OCR model to hallucinate BINs from empty images.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    selected = (
        (blue - red >= 20)
        & (blue - green >= 5)
        & (blue < 245)
    )
    # Pixels alone are not an adequate handwriting quality measure, but this
    # gate prevents noise and almost-white crops from becoming OCR votes.
    if int(np.count_nonzero(selected)) < min_pixels:
        return None

    mask = np.full(selected.shape, 255, dtype=np.uint8)
    mask[selected] = 0
    return Image.fromarray(mask, mode="L")


def blue_recognition_variant(image: Image.Image, variant: str) -> Image.Image | None:
    isolated = isolate_blue_ink(image)
    if isolated is None:
        return None
    if variant == "blue_core":
        return isolated.convert("RGB")
    if variant == "blue_3x_border":
        larger = isolated.resize(
            (isolated.width * 3, isolated.height * 3),
            Image.Resampling.LANCZOS,
        )
        return ImageOps.expand(
            larger, border=(24, 18, 24, 18), fill=255
        ).convert("RGB")
    raise ValueError(f"Unknown blue recognition variant {variant}")
