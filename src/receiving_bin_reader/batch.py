from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter

import fitz

from .pipeline import crop_pdf
from .safe_reader import _read_manifest, _review_header
from .ocr import PaddleConsensusReader, EXTENDED_VARIANTS, BLUE_VARIANTS
from .validation import load_approved_bins


def _page_count(pdf_path: str | Path) -> int:
    with fitz.open(pdf_path) as pdf:
        count = pdf.page_count
    if count == 0:
        raise ValueError("PDF contains no pages.")
    return count


def parse_page_selection(value: str) -> list[int]:
    """Parse 1-based PDF pages such as '1,3-5'; no duplicates or invalid ranges."""
    pages = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            raise ValueError("Empty item in page selection.")
        if "-" in item:
            a, sep, b = item.partition("-")
            if not a.isdecimal() or not b.isdecimal():
                raise ValueError(f"Invalid page range: {item!r}")
            start, stop = int(a), int(b)
            if start < 1 or stop < start:
                raise ValueError(f"Invalid page range: {item!r}")
            pages.extend(range(start, stop + 1))
        elif item.isdecimal() and int(item) >= 1:
            pages.append(int(item))
        else:
            raise ValueError(f"Invalid page number: {item!r}")
    if not pages or len(set(pages)) != len(pages):
        raise ValueError("Page selection must be nonempty and contain no duplicates.")
    return sorted(pages)


def _selected_pages(pdf_path: str | Path, pages: list[int] | None) -> tuple[int, list[int]]:
    total = _page_count(pdf_path)
    selected = list(range(1, total + 1)) if pages is None else sorted(pages)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Page selection must be nonempty and contain no duplicates.")
    if any(page < 1 or page > total for page in selected):
        raise ValueError(f"Requested pages {selected} exceed PDF page range 1-{total}.")
    return total, selected


def _page_output_dir(out_dir: Path, page: int) -> Path:
    return out_dir / "pages" / f"page_{page:04d}"


def crop_pdf_batch(
    pdf_path: str | Path,
    out_dir: str | Path,
    *,
    dpi: int = 180,
    pages: list[int] | None = None,
) -> dict:
    """
    Process all pages, or an explicit subset of receiving-worksheet pages.

    Supporting carrier BOLs must NOT be mistaken for receiving worksheets.
    Until the automatic page classifier is validated, use --pages for mixed PDFs.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total, selected = _selected_pages(pdf_path, pages)
    page_summaries = []
    all_manifest = []

    for page in selected:
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
                            if key in {"row_crop", "bin_crop", "review_context"}
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
        "pages": len(selected),
        "source_page_count": total,
        "selected_pages": selected,
        "skipped_pages": [p for p in range(1, total + 1) if p not in selected],
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
    pages: list[int] | None = None,
    extended_variants: bool = False,
    blue_ink_variants: bool = False,
) -> dict:
    """Load the OCR models once; keep page/row ownership and approval conservative."""
    out_dir = Path(out_dir)
    start = perf_counter()
    crop_result = crop_pdf_batch(pdf_path, out_dir, dpi=dpi, pages=pages)
    approved = load_approved_bins(approved_bins_path)
    if blue_ink_variants and extended_variants:
        raise ValueError("Choose only one experimental variant mode.")
    reader = (
        PaddleConsensusReader(variants=BLUE_VARIANTS)
        if blue_ink_variants else
        PaddleConsensusReader(variants=EXTENDED_VARIANTS)
        if extended_variants else PaddleConsensusReader()
    )

    review_rows = []
    page_stats = []
    for page in crop_result["selected_pages"]:
        page_dir = _page_output_dir(out_dir, page)
        try:
            reads, rows = _read_manifest(page_dir, approved, reader)
        except Exception as exc:
            raise RuntimeError(f"Could not recognize BINs on page {page}: {exc}") from exc

        with (page_dir / "review.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(_review_header())
            writer.writerows(rows)

        for row in rows:
            row = list(row)
            if row[-3]:
                row[-3] = str((page_dir / row[-3]).relative_to(out_dir))
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
        "source_page_count": crop_result["source_page_count"],
        "selected_pages": crop_result["selected_pages"],
        "skipped_pages": crop_result["skipped_pages"],
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
