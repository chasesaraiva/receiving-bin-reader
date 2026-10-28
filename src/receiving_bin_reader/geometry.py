from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .models import Box, PageGeometry, RowGeometry


def _group_indices(indices: np.ndarray, max_gap: int = 2) -> list[tuple[int, int]]:
    if len(indices) == 0:
        return []

    groups: list[tuple[int, int]] = []
    start = previous = int(indices[0])

    for raw in indices[1:]:
        value = int(raw)
        if value <= previous + max_gap:
            previous = value
        else:
            groups.append((start, previous))
            start = previous = value

    groups.append((start, previous))
    return groups


def _to_gray(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _detect_right_table_lines(gray: np.ndarray) -> list[tuple[int, int, int]]:
    height, width = gray.shape

    _, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(50, int(height * 0.05)))
    )
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    projection = (vertical > 0).sum(axis=0)

    xs = np.where(projection > height * 0.20)[0]
    raw = [
        (start, end, int(projection[start : end + 1].max()))
        for start, end in _group_indices(xs)
    ]

    right = [
        item
        for item in raw
        if item[0] > width * 0.50 and item[1] < width * 0.95
    ]

    # Real NWC worksheet scans vary: some present only THREE reliably
    # detectable right-side separators (SKU/BIN/Bottles), while others
    # present four or more. Requiring four silently rejected valid worksheets.
    # The sample carrier BOLs are portrait and have fewer column separators.
    # Keep conservative guards, but do not mistake one scan template for all.
    if width <= height * 1.16 or len(right) < 3:
        raise ValueError(
            "Page does not match a supported landscape NWC worksheet layout: "
            f"image {width}x{height}, {len(right)} right-side table separators. "
            "Use --pages for mixed receiving/carrier PDF packets."
        )

    # Typical first lines are the SKU/BIN and BIN/Bottles boundaries, about
    # 4-13% of image width apart. If the geometry differs, require review
    # rather than extract potentially incorrect carrier location codes.
    first, second, third = (item[0] for item in right[:3])
    if not (
        width * 0.55 <= first <= width * 0.80
        and width * 0.04 <= second - first <= width * 0.13
        and width * 0.04 <= third - second <= width * 0.15
    ):
        raise ValueError(
            "Page's right-side column geometry is not a recognized NWC "
            "receiving layout; select the worksheet pages explicitly."
        )

    return right


def _detect_header_bottom(gray: np.ndarray) -> int:
    height, width = gray.shape

    _, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(100, int(width * 0.12)), 1)
    )
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    projection = (horizontal > 0).sum(axis=1)

    ys = np.where(projection > width * 0.25)[0]
    groups = [
        (start, end, int(projection[start : end + 1].max()))
        for start, end in _group_indices(ys)
    ]

    candidates = [
        item
        for item in groups
        if height * 0.05 < item[1] < height * 0.35
    ]

    if not candidates:
        raise ValueError("Could not identify receiving-table header separator.")

    return max(candidates, key=lambda item: item[2])[1]


def _detect_printed_row_bands(
    gray: np.ndarray,
    header_bottom: int,
    sku_right: int,
) -> list[tuple[int, int, float]]:
    height, width = gray.shape

    x1 = int(width * 0.24)
    x2 = max(x1 + 50, sku_right - 10)

    roi = gray[header_bottom + 1 : int(height * 0.985), x1:x2]

    dark = (roi < 165).astype(np.uint8)
    projection = dark.sum(axis=1).astype(float)
    smooth = np.convolve(projection, np.ones(3) / 3.0, mode="same")

    activation = max(8.0, (x2 - x1) * 0.018)
    active_rows = np.where(smooth > activation)[0]

    bands: list[tuple[int, int, float]] = []
    for start, end in _group_indices(active_rows):
        if end - start + 1 < 5:
            continue

        peak = float(smooth[start : end + 1].max())
        bands.append(
            (
                start + header_bottom + 1,
                end + header_bottom + 1,
                peak,
            )
        )

    min_peak = (x2 - x1) * 0.15
    bands = [band for band in bands if band[2] >= min_peak]

    if not bands:
        raise ValueError("No printed product rows were detected.")

    if len(bands) >= 4:
        centers = [(top + bottom) // 2 for top, bottom, _ in bands]
        gaps = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
        smallest = sorted(gaps)[: max(2, int(len(gaps) * 0.70))]
        typical = float(np.median(smallest))

        keep = len(bands)
        for index, gap in enumerate(gaps):
            if index >= 1 and gap > max(100.0, typical * 4.0):
                keep = index + 1
                break

        bands = bands[:keep]

    return bands


def _make_row_boundaries(
    bands: list[tuple[int, int, float]],
    header_bottom: int,
    page_height: int,
) -> list[tuple[int, int, int]]:
    centers = [int(round((top + bottom) / 2.0)) for top, bottom, _ in bands]

    if len(centers) == 1:
        half_height = max(12, int(page_height * 0.012))
        return [
            (
                max(header_bottom + 1, centers[0] - half_height),
                min(page_height - 1, centers[0] + half_height),
                centers[0],
            )
        ]

    gaps = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
    typical = max(12, int(round(float(np.median(gaps)))))

    rows: list[tuple[int, int, int]] = []

    for index, center in enumerate(centers):
        if index == 0:
            top = max(header_bottom + 1, center - typical // 2)
        else:
            top = int(round((centers[index - 1] + center) / 2.0))

        if index == len(centers) - 1:
            bottom = min(page_height - 1, center + typical // 2)
        else:
            bottom = int(round((center + centers[index + 1]) / 2.0))

        rows.append((top, bottom, center))

    return rows


def detect_page_geometry(
    image: Image.Image,
    vertical_padding_ratio: float = 0.18,
) -> PageGeometry:
    gray = _to_gray(image)
    page_height, page_width = gray.shape

    right_lines = _detect_right_table_lines(gray)
    sku_right = right_lines[0][0]
    bin_right = right_lines[1][0]

    header_bottom = _detect_header_bottom(gray)
    bands = _detect_printed_row_bands(gray, header_bottom, sku_right)
    boundaries = _make_row_boundaries(bands, header_bottom, page_height)

    bin_x1 = min(page_width - 2, sku_right + 2)
    bin_x2 = max(bin_x1 + 2, bin_right - 2)

    rows: list[RowGeometry] = []

    for row_number, (top, bottom, _) in enumerate(boundaries, start=1):
        core_height = max(1, bottom - top)
        pad = max(2, int(round(core_height * vertical_padding_ratio)))

        padded_top = max(header_bottom + 1, top - pad)
        padded_bottom = min(page_height - 1, bottom + pad)

        rows.append(
            RowGeometry(
                row_number=row_number,
                core=Box(0, top, page_width, bottom),
                bin_padded=Box(bin_x1, padded_top, bin_x2, padded_bottom),
                core_top_in_bin_crop=top - padded_top,
                core_bottom_in_bin_crop=bottom - padded_top,
            )
        )

    return PageGeometry(
        page_width=page_width,
        page_height=page_height,
        header_bottom=header_bottom,
        sku_right=sku_right,
        bin_right=bin_right,
        rows=rows,
    )
