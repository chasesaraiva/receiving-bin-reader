"""Conservative recognition guard for handwritten BIN ink crossing printed rows.

Physical row geometry owns every BIN. This module does not reassign ink or
infer a value from neighboring rows. If dark strokes cross a printed-row
ownership boundary, BOTH rows require manual review, regardless of OCR votes.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def ink_crosses_boundary(
    crop: Image.Image,
    boundary_y: int,
    *,
    window: int = 5,
) -> bool:
    """Detect continuous ink near both sides of a boundary in one BIN crop.

    The crop includes margin across boundary_y. Horizontal printed grid rules
    and the vertical outer BIN separators are excluded. The method errs on
    the side of REVIEW; it is not handwriting recognition or segmentation.
    """
    rgb = np.asarray(crop.convert("RGB"))
    if rgb.size == 0:
        return True
    height, width = rgb.shape[:2]
    radius = min(window, boundary_y, height - boundary_y)
    if radius < 2 or width < 24:
        # Inadequate visibility across boundary -> cannot auto-approve.
        return True

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dark = (gray < 145).astype(np.uint8)

    # Remove long horizontal worksheet rules, not handwriting strokes.
    horizontal = cv2.morphologyEx(
        dark,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(20, int(width * 0.65)), 1)
        ),
    )
    strokes = (dark > 0) & (horizontal == 0)
    inset = max(3, int(round(width * 0.035)))
    strokes[:, :inset] = False
    strokes[:, -inset:] = False

    upper = strokes[boundary_y - radius:boundary_y]
    lower = strokes[boundary_y:boundary_y + radius]
    if upper.sum() < 4 or lower.sum() < 4:
        return False

    # Ink on either side must occupy nearby x coordinates; this avoids
    # overreacting to unrelated marks at opposite ends of a cell.
    x_upper = upper.any(axis=0).astype(np.uint8)
    x_lower = lower.any(axis=0)
    expanded = cv2.dilate(
        x_upper[np.newaxis, :], np.ones((1, 5), np.uint8)
    )[0].astype(bool)
    return int(np.count_nonzero(expanded & x_lower)) >= 3


def detect_row_ownership_conflicts(
    out_dir: str | Path,
    manifest_rows: list[dict[str, str]],
) -> dict[int, list[str]]:
    """Return human-readable review reasons keyed by physical row number.

    Only adjacent printed rows are compared. The earlier row's padded BIN
    crop already includes both sides of its lower boundary.
    """
    out_dir = Path(out_dir)
    warnings: dict[int, list[str]] = defaultdict(list)
    ordered = sorted(manifest_rows, key=lambda row: int(row["row"]))

    for upper, lower in zip(ordered, ordered[1:]):
        upper_id, lower_id = int(upper["row"]), int(lower["row"])
        if lower_id != upper_id + 1:
            raise ValueError("Manifest contains missing or nonconsecutive physical rows.")

        core_bottom = int(upper["core_bottom"])
        next_top = int(lower["core_top"])
        if abs(core_bottom - next_top) > 2:
            # A measurable gap in row geometry cannot guarantee assignment.
            reason = "ROW_BOUNDARY_GEOMETRY_GAP"
            warnings[upper_id].append(reason)
            warnings[lower_id].append(reason)
            continue

        with Image.open(out_dir / upper["bin_crop"]) as source:
            padded = source.convert("RGB")
        y = core_bottom - int(upper["bin_padded_top"])
        if ink_crosses_boundary(padded, y):
            warnings[upper_id].append(
                f"BIN_INK_CROSSES_BOUNDARY_WITH_ROW_{lower_id}"
            )
            warnings[lower_id].append(
                f"BIN_INK_CROSSES_BOUNDARY_WITH_ROW_{upper_id}"
            )

    return dict(warnings)
