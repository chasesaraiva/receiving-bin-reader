from __future__ import annotations

import csv
from io import BytesIO

import fitz
from PIL import Image
import pytest

from receiving_bin_reader.batch import crop_pdf_batch, read_pdf_batch_consensus
from receiving_bin_reader.consensus import RecognitionPass
from receiving_bin_reader.geometry import detect_page_geometry
from receiving_bin_reader.render import render_page


def sample_pdf(path):
    # A deterministic two-page receiving worksheet with different row counts.
    import cv2
    import numpy as np
    doc = fitz.open()
    for count in (3, 5):
        height, width = 1000, 1400
        array = np.full((height, width, 3), 255, dtype=np.uint8)
        cv2.line(array, (80, 170), (1300, 170), (0, 0, 0), 3)
        for x in (900, 1020, 1140, 1300):
            cv2.line(array, (x, 90), (x, 750), (0, 0, 0), 2)
        for i in range(count):
            y = 220 + i * 55
            cv2.putText(array, f"Product {i+1} SKU{i+1:03d}",
                        (350, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(array, "1A4", (930, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 0, 0), 2, cv2.LINE_AA)
        img = Image.fromarray(array)
        stream = BytesIO()
        img.save(stream, format="PNG")
        page = doc.new_page(width=width, height=height)
        page.insert_image(fitz.Rect(0, 0, width, height),
                          stream=stream.getvalue())
    doc.save(path)
    doc.close()


def test_batch_crop_keeps_page_and_row_identity(tmp_path):
    pdf_path = tmp_path / "two.pdf"
    sample_pdf(pdf_path)
    out = tmp_path / "out"
    result = crop_pdf_batch(pdf_path, out, dpi=72)
    assert result["pages"] == 2
    assert result["row_count"] == 8
    with (out / "manifest.csv").open(newline="") as handle:
        manifest = list(csv.DictReader(handle))
    assert [(int(x["page"]), int(x["row"])) for x in manifest] == (
        [(1, x) for x in range(1, 4)] +
        [(2, x) for x in range(1, 6)]
    )
    assert all((out / x["bin_crop"]).is_file() for x in manifest)
    assert (out / "pages/page_0002/contact_sheet.png").is_file()
    assert all((out / x["review_context"]).is_file() for x in manifest)


def test_batch_reuses_reader_and_writes_page_qualified_review(tmp_path, monkeypatch):
    import receiving_bin_reader.batch as batch
    pdf_path = tmp_path / "two.pdf"
    sample_pdf(pdf_path)
    approved = tmp_path / "approved.txt"
    approved.write_text("1A4\n", encoding="utf-8")
    calls = []

    class StubReader:
        def __init__(self):
            calls.append("initialized")

        def read_passes(self, image):
            calls.append("read")
            return [
                RecognitionPass("one", "core", "1A4", 0.98),
                RecognitionPass("two", "3x_border", "1A4", 0.99),
            ]

    monkeypatch.setattr(batch, "PaddleConsensusReader", StubReader)
    out = tmp_path / "out"
    result = read_pdf_batch_consensus(pdf_path, approved, out, dpi=72)
    assert result["row_count"] == 8
    assert result["valid"] == 8
    assert result["review"] == 0
    assert calls == ["initialized"] + ["read"] * 8
    with (out / "review.csv").open(newline="") as handle:
        records = list(csv.DictReader(handle))
    assert len(records) == 8
    assert records[0]["page"] == "1"
    assert records[-1]["page"] == "2"
    assert records[-1]["row"] == "5"
    assert all((out / x["bin_crop"]).is_file() for x in records)
    assert all((out / x["review_context"]).is_file() for x in records)
    assert (out / "pages/page_0001/review.csv").is_file()


def test_page_is_not_silently_skipped_on_failure(tmp_path, monkeypatch):
    import receiving_bin_reader.batch as batch
    pdf_path = tmp_path / "two.pdf"
    sample_pdf(pdf_path)
    original = batch.crop_pdf

    def fail_second(pdf, out, page_number=1, dpi=180):
        if page_number == 2:
            raise ValueError("simulated geometry failure")
        return original(pdf, out, page_number=page_number, dpi=dpi)

    monkeypatch.setattr(batch, "crop_pdf", fail_second)
    with pytest.raises(RuntimeError, match="page 2"):
        crop_pdf_batch(pdf_path, tmp_path / "out", dpi=72)
