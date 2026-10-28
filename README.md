# Receiving BIN Reader

A conservative open-source pipeline for receiving worksheets containing printed product rows and a handwritten warehouse BIN column.

**Row ownership is determined by printed page geometry, never by OCR.** Every physical row is cropped separately. Recognition uses two independent PaddleOCR model families and only accepts an exact, approved BIN when those models agree on one unambiguous candidate. Everything else is marked `REVIEW`.

## Installation

Requires Python 3.11, PyMuPDF, OpenCV, Pillow, and NumPy.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
# Optional OCR inference:
python -m pip install paddlepaddle==3.2.0
python -m pip install paddleocr==3.7.0
```

## Crop a single page (no OCR)

```bash
receiving-bin-reader crop "sample.pdf" --page 1 --out output/
```

Produces `manifest.csv`, `row_overlay.png`, `contact_sheet.png`, and individual `rows/` and `bins/` crops.

## Crop an entire multipage PDF (no OCR)

```bash
receiving-bin-reader crop "receiving-packet.pdf" --all-pages --out output/
```

The batch output has a combined `manifest.csv` with **page + physical row** columns, `batch_summary.json`, and page-specific images and manifests under `pages/page_0001/`, `pages/page_0002/`, etc. A failure on any page stops the batch with that page number (no silent omissions).

## Mixed packets containing carrier BOLs

Some multi-page deliveries combine authoritative NWC receiving worksheets with carrier BOLs, invoices, and manifests. **Only select pages containing receiving worksheets.** The reader does not yet automatically classify page types or extract carrier location codes.

For example, if receiving worksheets are on pages 1, 3, and 4:

```bash
receiving-bin-reader crop "mixed-packet.pdf" --pages 1,3-4 --out output/
```

The combined manifest and review identify the actual PDF page numbers and audit skipped pages in `batch_summary.json`. Invalid page ranges and geometry failures are errors, not silent fallbacks.

## Read handwritten BINs

Create `approved_bins.txt` with one approved BIN on each line (e.g. `1A1`, `1B7`, `BC`).

Single page:
```bash
receiving-bin-reader read "sample.pdf" --page 1 \
  --approved-bins approved_bins.txt --out output/
```

Entire packet of receiving worksheets (no carrier-only pages):
```bash
receiving-bin-reader read "receiving-packet.pdf" --all-pages \
  --approved-bins approved_bins.txt --out output/
```

Mixed packet with receiving worksheets on known pages:

```bash
receiving-bin-reader read "mixed-packet.pdf" --pages 1,3-4 \
  --approved-bins approved_bins.txt --out output/
```

Multipage processing **loads the two OCR models once** and reuses them across the packet. The combined `review.csv` has page and row identifiers, exact accepted BINs (if any), OCR passes, rejected candidates, status, review reason, and crop paths. Individual page review CSVs remain available inside each page folder.

**A `VALID` decision is not permission to update Inventory.** Review and human sign-off remain separate. Unapproved readings are never auto-corrected; ambiguity is never automatically resolved through fuzzy matching.

## Status and limitations

Stage 1 covers deterministic crop geometry, conservative OCR, review files, and synthetic multipage tests. The private handwriting benchmark has 17 fixtures; its initial conservative consensus accepted 10 correctly and sent seven for review, with zero incorrect acceptances in that small sample. These figures **are not a guarantee** on new PDFs or other handwriting.

The cropper assumes the receiving-table layout matches tested geometry. Validate real full PDFs, row counts, and end-to-end CPU runtime before connecting Google Drive/Sheets. A Google Sheets interface, SKU matching, inventory consolidation, vintage changes, and approvals are not implemented here.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
