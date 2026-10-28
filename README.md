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

Some multi-page deliveries combine authoritative NWC receiving worksheets with carrier BOLs, invoices, and manifests. **Only select pages containing receiving worksheets.** The reader does not yet automatically classify all page types or extract carrier location codes. A conservative landscape + three-column-line geometry check rejects some carrier layouts instead of falsely treating them as receiving worksheets, but **it is not a complete carrier detector**. Different NWC scans may have three, four, or more detectable vertical rules.

For example, if receiving worksheets are on pages 1, 3, and 4:

```bash
receiving-bin-reader crop "mixed-packet.pdf" --pages 1,3-4 --out output/
```

The combined manifest and review identify the actual PDF page numbers and audit skipped pages in `batch_summary.json`. Invalid page ranges and geometry failures are errors, not silent fallbacks. On a verified sample containing three receiving rows followed by a carrier BOL, the original geometric detector incorrectly treated 17 carrier text bands as product rows; the layout safeguard rejects that carrier page.

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

### Experimental handwriting improvement

A private 17-fixture benchmark compared eight image treatments. Adding two +/-3 degree rotated variants (using the original two independent OCR models) increased correctly accepted BIN entries from 10 to 11 with zero wrong acceptances in that small test. Other tested treatments did not increase coverage.

The original two-variant consensus remains the default. To evaluate the extra rotation safely:

```bash
receiving-bin-reader read "packet.pdf" --pages 1,3-4 \
  --approved-bins approved_bins.txt --extended-variants --out output/
```

This costs additional CPU inference and **is experimental**, not validated for unattended inventory updates.

### Handwriting crossing printed rows

The reader now checks dark ink on both sides of every adjacent physical-row
boundary, excluding long printed horizontal grid rules. If ink appears to cross
a boundary, **both rows are forced to `REVIEW`**, even when the OCR models
agree. The `decision_reason` column names the adjacent row; OCR passes remain
visible as suggestions but are not accepted. Every row also has a
`review_context` image showing its product plus neighboring printed rows,
complete BIN cells, and a target-row marker. Context images are **never**
passed to OCR; they are strictly for human review. This is intentionally conservative:
it prevents silent cross-row BIN assignments but may increase manual review.

**A `VALID` decision is not permission to update Inventory.** Review and human sign-off remain separate. Unapproved readings are never auto-corrected; ambiguity is never automatically resolved through fuzzy matching.

## Status and limitations

Stage 1 covers deterministic crop geometry, conservative OCR, review files, and synthetic multipage tests. The private handwriting benchmark has 17 fixtures; its initial conservative consensus accepted 10 correctly and sent seven for review, with zero incorrect acceptances in that small sample. These figures **are not a guarantee** on new PDFs or other handwriting.

The cropper assumes the receiving-table layout matches tested geometry. Validate real full PDFs, row counts, and end-to-end CPU runtime before connecting Google Drive/Sheets. A Google Sheets interface, SKU matching, inventory consolidation, vintage changes, and approvals are not implemented here.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).


### Experimental blue-ink isolation

For worksheets written in blue ballpoint, the reader can generate two extra
black-on-white recognition views containing only blue-dominant pixels.
Black printed text and table lines are removed from **these additional views**
but the original grayscale crops are always retained.

```bash
receiving-bin-reader read packet.pdf --pages 1 --blue-ink-variants \
  --approved-bins approved_bins.txt --out output/
```

The mode is **off by default** and must be evaluated against manually verified
receiving worksheets before use. A blue-only OCR candidate cannot become
`VALID` without support from an original-image OCR pass. Physical ink
crossing between rows overrides even unanimous recognition and forces
`REVIEW`. Black-written/printed BINs still use the ordinary OCR path.
