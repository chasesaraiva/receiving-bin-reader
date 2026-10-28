from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter

import fitz

from .pipeline import crop_pdf
from .safe_reader import _read_manifest, _review_header
from .ocr import PaddleConsensusReader
from .validation import load_approved_bins


def _page_count(pdf_path: str | Path) -> int:
    with fitz.open(pdf_path) as pdf:
        count = pdf.page_count
    if count == 0:
        raise ValueError("PDF contains no pages.")
    return count


def _page_output_dir(out_dir: Path, page: int) -> Path:
    return out_dir / "pages" / f"page_{page:04d}"


def crop_pdf_batch(
    pdf_path: str | Path, out_dir: str | Path, *, dpi: int = 180
) -> dict:
    """Crop every page; keep page and physical-row identities distinct."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    page_summaries = []
    all_manifest = []

    for page in range(1, _page_count(pdf_path) + 1):
        page_dir = _page_output_dir(out_dir, page)
        try:
            summary = crop_pdf(pdf_path, page_dir, page_number=page, dpi=dpi)
        except Exception as exc:
            raise RuntimeError(f"Could not crop page {page}: {exc}") from exc
        page_summaries.append(summary)
        with (page_dir / "manifest.csv").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                all_manifest.append({
                    "page": page,
                    "row": int(row["row"]),
                    **{
                        key: (
                            str((page_dir / row[key]).relative_to(out_dir))
                            if key in {"row_crop", "bin_crop"}
                            else row[key]
                        )
                        for key in row if key != "row"
                    },
                })

    if all_manifest:
        columns = list(all_manifest[0])
        with (out_dir / "manifest.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(all_manifest)

    result = {
        "pages": len(page_summaries),
        "row_count": sum(x["row_count"] for x in page_summaries),
        "page_summaries": page_summaries,
    }
    (out_dir / "batch_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def read_pdf_batch_consensus(
    pdf_path: str | Path,
    approved_bins_path: str | Path,
    out_dir: str | Path,
    *,
    dpi: int = 180,
) -> dict:
    """
    Batch reading uses exactly one OCR reader across every page.
    No automatic inventory writes and no unapproved BIN corrections.
    """
    out_dir = Path(out_dir)
    start = perf_counter()
    crop_result = crop_pdf_batch(pdf_path, out_dir, dpi=dpi)
    approved = load_approved_bins(approved_bins_path)
    reader = PaddleConsensusReader()

    review_rows = []
    page_stats = []
    for page in range(1, crop_result["pages"] + 1):
        page_dir = _page_output_dir(out_dir, page)
        try:
            reads, rows = _read_manifest(page_dir, approved, reader)
        except Exception as exc:
            raise RuntimeError(f"Could not recognize BINs on page {page}: {exc}") from exc

        # The per-page output remains self-contained for inspection.
        with (page_dir / "review.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(_review_header())
            writer.writerows(rows)

        for row in rows:
            row = list(row)
            # Crop paths in the root review file are relative to batch root.
            row[-2] = str((page_dir / row[-2]).relative_to(out_dir))
            row[-1] = str((page_dir / row[-1]).relative_to(out_dir))
            review_rows.append([page, *row])

        page_stats.append({
            "page": page,
            "rows": len(reads),
            "valid": sum(x.status == "VALID" for x in reads),
            "review": sum(x.status == "REVIEW" for x in reads),
        })

    with (out_dir / "review.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["page", *_review_header()])
        writer.writerows(review_rows)

    result = {
        "pages": crop_result["pages"],
        "row_count": len(review_rows),
        "valid": sum(x["valid"] for x in page_stats),
        "review": sum(x["review"] for x in page_stats),
        "elapsed_seconds": round(perf_counter() - start, 3),
        "page_stats": page_stats,
    }
    (out_dir / "batch_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result
