# Receiving BIN Reader

A small, open-source pipeline for receiving worksheets that contain:

- printed product rows
- a handwritten warehouse BIN column
- a stable table layout

The design deliberately separates **row ownership** from **handwriting recognition**:

1. Render the PDF page once.
2. Detect the printed receiving-table geometry deterministically.
3. Split the page into one physical product row per crop.
4. Make a slightly padded crop of only the BIN column.
5. Run OCR on each BIN crop.
6. Accept only exact values from an approved BIN list.
7. Write a review CSV and visual contact sheet.

The OCR engine is never allowed to decide which product owns a BIN. Geometry decides ownership first.

## Why this project starts small

This repository intentionally does **not** contain:

- Google Sheets writes
- inventory merging
- SKU alias logic
- vintage-change logic
- automatic approvals
- AI model fallbacks
- fuzzy BIN correction

Those are separate concerns and can be added after crop + BIN accuracy is proven.

## Requirements

Core cropper:

- Python 3.11
- PyMuPDF
- OpenCV
- Pillow
- NumPy

BIN OCR:

- PaddleOCR 3.7
- PaddlePaddle CPU 3.2

PaddleOCR 3.x supports CPU inference and exposes recognition text, confidence, and bounding boxes. The reader loads the OCR engine once and reuses it for the whole PDF batch.

## Install locally

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .

# Optional OCR engine:
python -m pip install paddlepaddle==3.2.0 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -m pip install "paddleocr==3.7.0"
```

## Crop-only validation

This is the first thing to run on a new worksheet style:

```bash
receiving-bin-reader crop \
  "sample.pdf" \
  --out output/
```

Outputs:

```text
output/
  manifest.csv
  row_overlay.png
  contact_sheet.png
  rows/
    row_001.png
    row_002.png
    ...
  bins/
    bin_001.png
    bin_002.png
    ...
```

No OCR is involved in this command.

## BIN OCR

Create a newline-delimited approved BIN file:

```text
1A1
1A2
1A3
...
BC
R1A
R1B
```

Then run:

```bash
receiving-bin-reader read \
  "sample.pdf" \
  --approved-bins approved_bins.txt \
  --out output/
```

The result `review.csv` contains:

- physical row number
- accepted BIN(s)
- raw OCR text
- OCR score
- status
- crop filenames

Unapproved OCR values are never changed to the nearest approved BIN.

## Confidence policy

The first release is intentionally conservative:

- `VALID`: at least one approved BIN was read and no unapproved BIN-like candidate remains.
- `REVIEW`: no approved BIN, mixed approved/unapproved candidates, or low OCR confidence.

`VALID` is **not** an instruction to write Inventory automatically. Human review remains a separate step.

## GitHub

The included CI workflow runs the deterministic geometry and validator tests on every push.

The Drive/GitHub automation layer is intentionally deferred until the core reader is benchmarked on real receiving PDFs. This prevents cloud plumbing from becoming entangled with OCR accuracy.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
