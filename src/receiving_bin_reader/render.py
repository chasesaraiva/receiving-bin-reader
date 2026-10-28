from __future__ import annotations

from pathlib import Path

import fitz
import numpy as np
from PIL import Image


def render_page(pdf_path: str | Path, page_number: int = 1, dpi: int = 180) -> Image.Image:
    """Render a 1-based PDF page into an RGB Pillow image."""
    if page_number < 1:
        raise ValueError("page_number must be 1-based and >= 1")

    pdf_path = Path(pdf_path)
    with fitz.open(pdf_path) as doc:
        if page_number > doc.page_count:
            raise ValueError(
                f"PDF has {doc.page_count} pages; requested page {page_number}."
            )
        page = doc[page_number - 1]
        scale = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)

    array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )

    if pix.n == 1:
        array = np.repeat(array, 3, axis=2)
    elif pix.n == 4:
        array = array[:, :, :3]

    return Image.fromarray(array, mode="RGB")
