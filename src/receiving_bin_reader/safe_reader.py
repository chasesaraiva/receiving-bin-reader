from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean

from PIL import Image

from .consensus import decide_consensus
from .models import RowRead
from .ocr import PaddleConsensusReader
from .pipeline import crop_pdf
from .validation import load_approved_bins, normalize_bin


def _ownership_crop(image: Image.Image, manifest: dict[str, str]) -> Image.Image:
    core_top = int(manifest["core_top"]) - int(manifest["bin_padded_top"])
    core_bottom = int(manifest["core_bottom"]) - int(manifest["bin_padded_top"])
    top = max(0, core_top - 4)
    bottom = min(image.height, core_bottom + 4)
    return image.crop((0, top, image.width, bottom))


def read_pdf_consensus(
    pdf_path: str | Path,
    approved_bins_path: str | Path,
    out_dir: str | Path,
    page_number: int = 1,
    dpi: int = 180,
) -> list[RowRead]:
    """
    Production-safe reader:
    crop geometry first, then four recognition-only passes, then exact
    cross-model consensus. No fuzzy BIN correction.
    """
    out_dir = Path(out_dir)
    crop_pdf(pdf_path, out_dir, page_number=page_number, dpi=dpi)

    approved = load_approved_bins(approved_bins_path)
    reader = PaddleConsensusReader()

    manifest_rows = list(
        csv.DictReader((out_dir / "manifest.csv").open(encoding="utf-8"))
    )

    reads: list[RowRead] = []
    output_rows: list[list[str]] = []

    for manifest in manifest_rows:
        row_number = int(manifest["row"])
        bin_path = out_dir / manifest["bin_crop"]
        row_path = out_dir / manifest["row_crop"]

        padded = Image.open(bin_path).convert("RGB")
        owned = _ownership_crop(padded, manifest)
        passes = reader.read_passes(owned)
        decision = decide_consensus(passes, approved)

        raw_passes = [
            f"{p.model}:{p.variant}={p.text}"
            for p in passes
        ]
        scores = [p.score for p in passes]
        rejected: list[str] = []

        for p in passes:
            parts = [
                normalize_bin(part)
                for part in str(p.text or "").split(",")
                if normalize_bin(part)
            ]
            if parts and not all(part in approved for part in parts):
                value = ",".join(parts)
                if value not in rejected:
                    rejected.append(value)

        accepted = list(decision.bins)
        avg_score = mean(scores) if scores else 0.0
        status = "VALID" if decision.status == "ACCEPT" else "REVIEW"

        reads.append(
            RowRead(
                row_number=row_number,
                accepted_bins=accepted,
                raw_candidates=raw_passes,
                rejected_candidates=rejected,
                mean_score=avg_score,
                status=status,
                row_crop=row_path,
                bin_crop=bin_path,
            )
        )

        output_rows.append(
            [
                str(row_number),
                ", ".join(accepted),
                " | ".join(raw_passes),
                ", ".join(rejected),
                f"{avg_score:.4f}",
                status,
                decision.reason,
                ", ".join(decision.supporting_models),
                str(row_path.relative_to(out_dir)),
                str(bin_path.relative_to(out_dir)),
            ]
        )

    with (out_dir / "review.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "row",
                "bin",
                "ocr_passes",
                "rejected",
                "mean_ocr_score",
                "status",
                "decision_reason",
                "supporting_models",
                "row_crop",
                "bin_crop",
            ]
        )
        writer.writerows(output_rows)

    return reads
