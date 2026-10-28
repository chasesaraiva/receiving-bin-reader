# Architecture

```text
PDF -> render each page -> deterministic row geometry
    -> physical row and BIN crop (page + row identity preserved)
    -> two PaddleOCR recognition model families / two image variants
    -> exact approved BIN consensus -> VALID or REVIEW
    -> per-page + batch review CSV and visual contact sheets
```

## Stage 1: in this repository

- Crop individual pages or entire receiving-only PDFs with `--all-pages`; explicitly select receiving-worksheet pages within mixed BOL packets using `--pages 1,3-5`.
- Report source page count, selected PDF pages and skipped pages, rather than guessing whether carrier-only pages contain BINs.
- Keep row ownership independent from OCR and never transfer BIN values between rows.
- Inspect ink immediately above and below adjacent row boundaries; if strokes seem to cross, mark BOTH products REVIEW regardless of OCR agreement. This avoids accidental adjacent-row assignments without guessing new row ownership.
- Write page-specific crop artifacts, combined manifests, review CSV, and elapsed runtime.
- Reuse one OCR reader per batch to avoid repeated model initialization.
- Fail the batch if any page cannot be geometrically processed.
- Fail closed for non-NWC pages with fewer than four right-side table separators. This check is not a general classifier; use explicit `--pages` for mixed packets.
- Require exact approved-list membership and independent model agreement.
- Optional `--extended-variants` tests +/-3° rotation variants; the original consensus stays default until validated on larger fixture sets.
- **No** inventory writes, fuzzy BIN correction, SKU alias matching, or auto approval.

Synthetic two-page tests exercise page/row isolation, review aggregation, one-time reader initialization, and fail-closed page handling. Real scanned documents and handwritten fixtures still require validation.

## Stage 2: proposed Drive/GitHub Actions integration

```text
Google Drive PDF file ID
  -> GitHub repository_dispatch
  -> runner downloads PDF using a service account
  -> run full PDF reader
  -> upload review artifacts
```

Keep company credentials and Drive IDs in private secrets, not public source.

## Stage 3: proposed Google Sheet review

```text
Apps Script submits a file ID
  -> batch review CSV imported into Inbound Review
  -> user verifies flagged BINs and approves
  -> separate Inventory processing module handles SKU/vintage changes
```

No stage 2 or 3 automation should be enabled until row counts, OCR correctness, and throughput are validated on actual receiving packets.
