# Architecture

## Core principle

The pipeline has one responsibility at each stage.

```text
PDF
  ↓
render.py
  ↓
geometry.py
  ├─ physical row boundaries
  └─ BIN column boundaries
  ↓
crops.py
  ├─ full row crops
  └─ padded BIN crops
  ↓
ocr.py
  └─ raw OCR boxes/text/scores
  ↓
validation.py
  └─ exact Approved BIN validation
  ↓
pipeline.py
  ├─ manifest.csv
  ├─ review.csv
  ├─ contact sheet
  └─ row/BIN images
```

## Stage 1 — current repository

Goal: prove that row ownership and BIN recognition are accurate.

No Google Sheets writes.

Acceptance criteria:

1. Physical row count is correct.
2. Every crop contains exactly one printed product row.
3. Padded BIN crops do not change row ownership.
4. Approved BIN validation never fuzzy-corrects a value.
5. Known problem PDFs reproduce their physical row counts consistently.
6. End-to-end runtime stays practical on CPU.

## Stage 2 — GitHub Actions + Google Drive

After Stage 1 is proven:

```text
Google Drive PDF file ID
       ↓
GitHub repository_dispatch
       ↓
GitHub Actions runner
       ↓
download PDF with service account
       ↓
run receiving-bin-reader
       ↓
upload review artifact / result file
```

The repository remains open source. Company credentials and Drive IDs live only in GitHub Secrets/Variables.

## Stage 3 — Google Sheet review

After Stage 2 is proven:

- lightweight Apps Script submits Drive file IDs
- result JSON/CSV is loaded into `Inbound Review`
- human approval remains the final gate
- Inventory processing is a separate module

## Non-goals

The cropper must not:

- infer a nearby BIN
- merge product rows
- move a BIN to an adjacent row
- deduplicate SKUs
- decide product identity
- write Inventory
