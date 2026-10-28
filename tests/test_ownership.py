from __future__ import annotations

import csv

import cv2
import numpy as np
from PIL import Image, ImageDraw

from receiving_bin_reader.consensus import RecognitionPass
from receiving_bin_reader.ownership import (
    detect_row_ownership_conflicts,
    ink_crosses_boundary,
)
from receiving_bin_reader.safe_reader import _read_manifest


def empty_bin_crop() -> Image.Image:
    return Image.new("RGB", (130, 90), "white")


def test_clean_boundaries_can_be_processed():
    image = empty_bin_crop()
    ImageDraw.Draw(image).text((15, 10), "1A4", fill=(0, 0, 0))
    ImageDraw.Draw(image).text((15, 67), "1B7", fill=(0, 0, 0))
    assert not ink_crosses_boundary(image, 45)


def test_ink_crossing_boundary_requires_review():
    image = empty_bin_crop()
    ImageDraw.Draw(image).line((45, 37, 45, 53), fill=(0, 0, 120), width=4)
    assert ink_crosses_boundary(image, 45)


def test_printed_horizontal_rule_does_not_look_like_crossing_ink():
    image = np.full((90, 130, 3), 255, dtype=np.uint8)
    cv2.line(image, (0, 45), (129, 45), (0, 0, 0), 2)
    assert not ink_crosses_boundary(Image.fromarray(image), 45)


def test_insufficient_boundary_visibility_fails_closed():
    assert ink_crosses_boundary(empty_bin_crop(), 1)


def test_conflict_blocks_unanimous_ocr_acceptance(tmp_path):
    (tmp_path / "bins").mkdir()
    (tmp_path / "rows").mkdir()

    first = empty_bin_crop()
    # First row's core bottom is global 50, at y=45 in padded crop.
    ImageDraw.Draw(first).line(
        (45, 38, 45, 53), fill=(0, 0, 120), width=4,
    )
    first.save(tmp_path / "bins" / "bin_001.png")
    empty_bin_crop().save(tmp_path / "bins" / "bin_002.png")
    empty_bin_crop().save(tmp_path / "rows" / "row_001.png")
    empty_bin_crop().save(tmp_path / "rows" / "row_002.png")
    fields = [
        "row", "core_top", "core_bottom", "bin_x1", "bin_x2",
        "bin_padded_top", "bin_padded_bottom", "row_crop", "bin_crop",
    ]
    records = [
        {
            "row": "1", "core_top": "30", "core_bottom": "50",
            "bin_x1": "0", "bin_x2": "130", "bin_padded_top": "5",
            "bin_padded_bottom": "95",
            "row_crop": "rows/row_001.png", "bin_crop": "bins/bin_001.png",
        },
        {
            "row": "2", "core_top": "50", "core_bottom": "70",
            "bin_x1": "0", "bin_x2": "130", "bin_padded_top": "45",
            "bin_padded_bottom": "135",
            "row_crop": "rows/row_002.png", "bin_crop": "bins/bin_002.png",
        },
    ]
    with (tmp_path / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    warnings = detect_row_ownership_conflicts(tmp_path, records)
    assert 1 in warnings and 2 in warnings

    class SureReader:
        def read_passes(self, image):
            return [
                RecognitionPass("one", "core", "1A4", 0.99),
                RecognitionPass("two", "3x_border", "1A4", 0.99),
            ]

    reads, rows = _read_manifest(tmp_path, {"1A4"}, SureReader())
    assert len(reads) == 2
    assert [item.status for item in reads] == ["REVIEW", "REVIEW"]
    assert [item.accepted_bins for item in reads] == [[], []]
    assert all("BIN_INK_CROSSES_BOUNDARY" in row[6] for row in rows)
    assert all("one:core=1A4" in row[2] for row in rows)


def test_clean_two_rows_retain_consensus(tmp_path):
    (tmp_path / "bins").mkdir()
    (tmp_path / "rows").mkdir()
    empty_bin_crop().save(tmp_path / "bins" / "bin_001.png")
    empty_bin_crop().save(tmp_path / "bins" / "bin_002.png")
    manifest = [
        {
            "row": "1", "core_top": "30", "core_bottom": "50",
            "bin_padded_top": "5", "bin_crop": "bins/bin_001.png",
            "row_crop": "rows/row_001.png",
        },
        {
            "row": "2", "core_top": "50", "core_bottom": "70",
            "bin_padded_top": "45", "bin_crop": "bins/bin_002.png",
            "row_crop": "rows/row_002.png",
        },
    ]
    assert detect_row_ownership_conflicts(tmp_path, manifest) == {}
