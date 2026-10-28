import cv2
import numpy as np
from PIL import Image

from receiving_bin_reader.geometry import detect_page_geometry


def make_synthetic_page(row_count: int = 7) -> Image.Image:
    height = 1000
    width = 1400
    image = np.full((height, width, 3), 255, dtype=np.uint8)

    header_y = 170
    sku_right = 900
    bin_right = 1020
    bottles_right = 1140
    notes_right = 1300

    cv2.line(image, (80, header_y), (1300, header_y), (0, 0, 0), 3)

    for x in [sku_right, bin_right, bottles_right, notes_right]:
        cv2.line(image, (x, 90), (x, 750), (0, 0, 0), 2)

    y = 220
    for row in range(row_count):
        cv2.putText(
            image,
            f"Product {row + 1}  SKU{row + 1:03d}",
            (350, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"{row + 1}A{row + 1}",
            (925, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        y += 55

    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def test_geometry_detects_rows_and_bin_column():
    image = make_synthetic_page(7)
    geometry = detect_page_geometry(image)

    assert len(geometry.rows) == 7
    assert 880 <= geometry.sku_right <= 920
    assert 1000 <= geometry.bin_right <= 1040


def test_padding_does_not_change_core_ownership():
    image = make_synthetic_page(5)
    geometry = detect_page_geometry(image, vertical_padding_ratio=0.20)

    for row in geometry.rows:
        assert row.bin_padded.y1 <= row.core.y1
        assert row.bin_padded.y2 >= row.core.y2
        assert row.core_top_in_bin_crop >= 0
        assert row.core_bottom_in_bin_crop <= row.bin_padded.height


def test_carrier_style_two_rule_layout_fails_closed():
    """A non-NWC carrier page with only two right-side rules must not make false rows."""
    import pytest

    width, height = 1400, 1000
    array = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.line(array, (80, 170), (1300, 170), (0, 0, 0), 3)
    # Carrier-like layout: two surviving vertical rules and printed products.
    for x in (1000, 1250):
        cv2.line(array, (x, 90), (x, 750), (0, 0, 0), 2)
    for index in range(10):
        y = 220 + index * 45
        cv2.putText(
            array,
            f"Carrier item {index + 1}",
            (350, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    image = Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))
    with pytest.raises(ValueError, match="NWC receiving worksheet"):
        detect_page_geometry(image)
