from io import BytesIO

import cv2
import fitz
import numpy as np
from PIL import Image
import pytest

from receiving_bin_reader.batch import crop_pdf_batch
from test_batch import sample_pdf


def make_mixed_packet(path, tmp_path):
    """Receiving worksheet followed by a carrier-style page with two right rules."""
    source = tmp_path / "receiving_only.pdf"
    sample_pdf(source)
    output = fitz.open()
    with fitz.open(source) as original:
        output.insert_pdf(original, from_page=0, to_page=0)

    width, height = 1400, 1000
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.line(image, (80, 170), (1300, 170), (0, 0, 0), 3)
    for x in (1000, 1250):
        cv2.line(image, (x, 90), (x, 750), (0, 0, 0), 2)
    for idx in range(10):
        y = 220 + idx * 45
        cv2.putText(
            image, f"Carrier reference {idx + 1}",
            (350, y), cv2.FONT_HERSHEY_SIMPLEX,
            .65, (0, 0, 0), 2, cv2.LINE_AA,
        )
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    page = output.new_page(width=width, height=height)
    page.insert_image(fitz.Rect(0, 0, width, height), stream=buffer.getvalue())
    output.save(path)
    output.close()


def test_mixed_packet_fails_closed_by_default(tmp_path):
    packet = tmp_path / "mixed.pdf"
    make_mixed_packet(packet, tmp_path)
    with pytest.raises(RuntimeError, match="page 2"):
        crop_pdf_batch(packet, tmp_path / "all", dpi=72)


def test_selected_receiving_page_is_safe(tmp_path):
    packet = tmp_path / "mixed.pdf"
    make_mixed_packet(packet, tmp_path)
    result = crop_pdf_batch(packet, tmp_path / "selected", pages=[1], dpi=72)
    assert result["row_count"] == 3
    assert result["source_page_count"] == 2
    assert result["selected_pages"] == [1]
    assert result["skipped_pages"] == [2]
