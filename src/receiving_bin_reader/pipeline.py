from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean

from PIL import Image

from .consensus import decide_consensus
from .crops import save_contact_sheet, save_crops, save_overlay, save_review_contexts
from .geometry import detect_page_geometry
from .models import RowRead
from .ocr import PaddleConsensusReader
from .render import render_page
from .validation import load_approved_bins, normalize_bin


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
    context_paths = save_review_contexts(image, geometry, out_dir)
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
                "review_context",
                "row_crop",
                "bin_crop",
            ]
        )

        for row, row_path, bin_path, context_path in zip(
            geometry.rows, row_paths, bin_paths, context_paths
        ):
            writer.writerow(
                [
                    row.row_number,
                    row.core.y1,
                    row.core.y2,
                    row.bin_padded.x1,
                    row.bin_padded.x2,
                    row.bin_padded.y1,
                    row.bin_padded.y2,
                    context_path.relative_to(out_dir),
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


def _canonical_pass_text(text: str) -> str:
    return ",".join(
        part
        for raw in str(text or "").split(",")
        if (part := normalize_bin(raw))
    )


def read_pdf(
    pdf_path: str | Path,
    approved_bins_path: str | Path,
    out_dir: str | Path,
    page_number: int = 1,
    dpi: int = 180,
    score_review_threshold: float = 0.70,
) -> list[RowRead]:
    """
    Read handwritten BINs using conservative independent-model consensus.

    The score_review_threshold argument is retained for CLI compatibility, but
    acceptance is determined by exact Approved BIN membership plus agreement
    across two independent PaddleOCR recognition model families.
    """
    out_dir = Path(out_dir)
    crop_pdf(pdf_path, out_dir, page_number=page_number, dpi=dpi)

    approved = load_approved_bins(approved_bins_path)
    reader = PaddleConsensusReader()

    manifest_rows = list(
        csv.DictReader((out_dir / "manifest.csv").open(encoding="utf-8"))
    )

    reads: list[RowRead] = []
    audit_rows: list[list[str]] = []

    for manifest in manifest_rows:
        row_number = int(manifest["row"])
        bin_path = out_dir / manifest["bin_crop"]
        row_path = out_dir / manifest["row_crop"]

        image = Image.open(bin_path).convert("RGB")

        core_top = int(manifest["core_top"]) - int(manifest["bin_padded_top"])
        core_bottom = int(manifest["core_bottom"]) - int(manifest["bin_padded_top"])

        # Same crop policy that produced the benchmark result: preserve a small
        # handwriting margin, while deterministic row geometry still owns the row.
        top = max(0, core_top - 4)
        bottom = min(image.height, core_bottom + 4)
        recognition_crop = image.crop((0, top, image.width, bottom))

        passes = reader.read_passes(recognition_crop)
        decision = decide_consensus(passes, approved)

        canonical_candidates = list(
            dict.fromkeys(
                candidate
                for result in passes
                if (candidate := _canonical_pass_text(result.text))
            )
        )

        rejected_candidates = [
            candidate
            for candidate in canonical_candidates
            if not all(part in approved for part in candidate.split(","))
        ]

        scores = [result.score for result in passes if result.text]
        mean_score = mean(scores) if scores else 0.0

        reads.append(
            RowRead(
                row_number=row_number,
                accepted_bins=list(decision.bins),
                raw_candidates=canonical_candidates,
                rejected_candidates=rejected_candidates,
                mean_score=mean_score,
                status="VALID" if decision.status == "ACCEPT" else "REVIEW",
                row_crop=row_path,
                bin_crop=bin_path,
            )
        )

        audit_rows.append(
            [
                str(row_number),
                decision.status,
                decision.candidate,
                ", ".join(decision.supporting_models),
                decision.reason,
                " | ".join(
                    f"{result.model}/{result.variant}:{result.text}:{result.score:.4f}"
                    for result in passes
                ),
            ]
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

    with (out_dir / "ocr_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "row",
                "consensus_status",
                "candidate",
                "supporting_models",
                "reason",
                "all_passes",
            ]
        )
        writer.writerows(audit_rows)

    return reads
