from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean

from PIL import Image

from .consensus import decide_consensus
from .models import RowRead
from .ocr import PaddleConsensusReader, EXTENDED_VARIANTS
from .ownership import detect_row_ownership_conflicts
from .pipeline import crop_pdf
from .validation import load_approved_bins, split_bin_candidates


def _review_header() -> list[str]:
    return [
        "row",
        "bin",
        "ocr_passes",
        "rejected",
        "mean_ocr_score",
        "status",
        "decision_reason",
        "supporting_models",
        "review_context",
        "row_crop",
        "bin_crop",
    ]


def _ownership_crop(image: Image.Image, manifest: dict[str, str]) -> Image.Image:
    core_top = int(manifest["core_top"]) - int(manifest["bin_padded_top"])
    core_bottom = int(manifest["core_bottom"]) - int(manifest["bin_padded_top"])
    top = max(0, core_top - 4)
    bottom = min(image.height, core_bottom + 4)
    return image.crop((0, top, image.width, bottom))


def _read_manifest(
    out_dir: Path,
    approved: set[str],
    reader: PaddleConsensusReader,
) -> tuple[list[RowRead], list[list[str]]]:
    """Read the output of one deterministic crop operation."""
    with (out_dir / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))

    conflicts = detect_row_ownership_conflicts(out_dir, manifest_rows)
    reads: list[RowRead] = []
    output_rows: list[list[str]] = []

    for manifest in manifest_rows:
        row_number = int(manifest["row"])
        bin_path = out_dir / manifest["bin_crop"]
        row_path = out_dir / manifest["row_crop"]

        with Image.open(bin_path) as source:
            owned = _ownership_crop(source.convert("RGB"), manifest)
        passes = reader.read_passes(owned)
        decision = decide_consensus(passes, approved)

        raw_passes = [f"{p.model}:{p.variant}={p.text}" for p in passes]
        scores = [p.score for p in passes]
        rejected: list[str] = []

        for p in passes:
            parts = split_bin_candidates(p.text)
            if parts and not all(part in approved for part in parts):
                value = ",".join(parts)
                if value not in rejected:
                    rejected.append(value)

        warnings = conflicts.get(row_number, [])
        # Never approve a BIN with ambiguous physical ownership, even if
        # the OCR models unanimously agree on the text. Show OCR as hints.
        accepted = [] if warnings else list(decision.bins)
        avg_score = mean(scores) if scores else 0.0
        status = (
            "VALID" if decision.status == "ACCEPT" and not warnings
            else "REVIEW"
        )
        reason = (
            "; ".join(warnings) + "; " + decision.reason
            if warnings else decision.reason
        )

        reads.append(RowRead(
            row_number=row_number,
            accepted_bins=accepted,
            raw_candidates=raw_passes,
            rejected_candidates=rejected,
            mean_score=avg_score,
            status=status,
            row_crop=row_path,
            bin_crop=bin_path,
        ))

        output_rows.append([
            str(row_number),
            ", ".join(accepted),
            " | ".join(raw_passes),
            ", ".join(rejected),
            f"{avg_score:.4f}",
            status,
            reason,
            ", ".join(decision.supporting_models),
            manifest.get("review_context", ""),
            str(row_path.relative_to(out_dir)),
            str(bin_path.relative_to(out_dir)),
        ])
    return reads, output_rows


def read_pdf_consensus(
    pdf_path: str | Path,
    approved_bins_path: str | Path,
    out_dir: str | Path,
    page_number: int = 1,
    dpi: int = 180,
    extended_variants: bool = False,
) -> list[RowRead]:
    """
    Production-safe single-page reader. Retains the historical CLI format.
    No fuzzy BIN correction or automated inventory writes.
    """
    out_dir = Path(out_dir)
    crop_pdf(pdf_path, out_dir, page_number=page_number, dpi=dpi)

    approved = load_approved_bins(approved_bins_path)
    reader = (
        PaddleConsensusReader(variants=EXTENDED_VARIANTS)
        if extended_variants else PaddleConsensusReader()
    )
    reads, output_rows = _read_manifest(out_dir, approved, reader)

    with (out_dir / "review.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(_review_header())
        writer.writerows(output_rows)
    return reads
