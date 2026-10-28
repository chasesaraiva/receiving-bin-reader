from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean

from .crops import save_contact_sheet, save_crops, save_overlay
from .geometry import detect_page_geometry
from .models import RowRead
from .ocr import PaddleBinReader
from .render import render_page
from .validation import (
    load_approved_bins,
    split_bin_candidates,
    validate_candidates,
)


def crop_pdf(
    pdf_path: str | Path,
    out_dir: str | Path,
    page_number: int = 1,
    dpi: int = 180,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image = render_page(pdf_path, page_number=page_number, dpi=dpi)
    geometry = detect_page_geometry(image)

    row_paths, bin_paths = save_crops(image, geometry, out_dir)
    save_overlay(image, geometry, out_dir / "row_overlay.png")
    save_contact_sheet(image, geometry, out_dir / "contact_sheet.png")

    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "row",
                "core_top",
                "core_bottom",
                "bin_x1",
                "bin_x2",
                "bin_padded_top",
                "bin_padded_bottom",
                "row_crop",
                "bin_crop",
            ]
        )

        for row, row_path, bin_path in zip(geometry.rows, row_paths, bin_paths):
            writer.writerow(
                [
                    row.row_number,
                    row.core.y1,
                    row.core.y2,
                    row.bin_padded.x1,
                    row.bin_padded.x2,
                    row.bin_padded.y1,
                    row.bin_padded.y2,
                    row_path.relative_to(out_dir),
                    bin_path.relative_to(out_dir),
                ]
            )

    summary = {
        "pdf": str(pdf_path),
        "page": page_number,
        "dpi": dpi,
        "row_count": len(geometry.rows),
        "header_bottom": geometry.header_bottom,
        "sku_right": geometry.sku_right,
        "bin_right": geometry.bin_right,
    }

    (out_dir / "geometry.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return summary


def read_pdf(
    pdf_path: str | Path,
    approved_bins_path: str | Path,
    out_dir: str | Path,
    page_number: int = 1,
    dpi: int = 180,
    score_review_threshold: float = 0.70,
) -> list[RowRead]:
    out_dir = Path(out_dir)
    crop_pdf(pdf_path, out_dir, page_number=page_number, dpi=dpi)

    approved = load_approved_bins(approved_bins_path)
    reader = PaddleBinReader(score_threshold=0.0)

    manifest_rows = list(
        csv.DictReader((out_dir / "manifest.csv").open(encoding="utf-8"))
    )

    reads: list[RowRead] = []

    for manifest in manifest_rows:
        row_number = int(manifest["row"])
        bin_path = out_dir / manifest["bin_crop"]
        row_path = out_dir / manifest["row_crop"]

        from PIL import Image

        image = Image.open(bin_path).convert("RGB")

        core_top = int(manifest["core_top"]) - int(manifest["bin_padded_top"])
        core_bottom = int(manifest["core_bottom"]) - int(manifest["bin_padded_top"])

        tokens = reader.read(image)

        owned = [
            token
            for token in tokens
            if core_top <= token.box.center_y <= core_bottom
        ]

        raw_candidates: list[str] = []
        scores: list[float] = []

        for token in sorted(owned, key=lambda item: (item.box.y1, item.box.x1)):
            raw_candidates.extend(split_bin_candidates(token.text))
            scores.append(token.score)

        raw_candidates = list(dict.fromkeys(raw_candidates))

        accepted, rejected = validate_candidates(raw_candidates, approved)
        mean_score = mean(scores) if scores else 0.0

        status = "VALID"
        if not accepted or rejected or mean_score < score_review_threshold:
            status = "REVIEW"

        reads.append(
            RowRead(
                row_number=row_number,
                accepted_bins=accepted,
                raw_candidates=raw_candidates,
                rejected_candidates=rejected,
                mean_score=mean_score,
                status=status,
                row_crop=row_path,
                bin_crop=bin_path,
            )
        )

    with (out_dir / "review.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "row",
                "bin",
                "raw_ocr",
                "rejected",
                "ocr_score",
                "status",
                "row_crop",
                "bin_crop",
            ]
        )

        for read in reads:
            writer.writerow(
                [
                    read.row_number,
                    ", ".join(read.accepted_bins),
                    ", ".join(read.raw_candidates),
                    ", ".join(read.rejected_candidates),
                    f"{read.mean_score:.4f}",
                    read.status,
                    read.row_crop.relative_to(out_dir) if read.row_crop else "",
                    read.bin_crop.relative_to(out_dir) if read.bin_crop else "",
                ]
            )

    return reads
