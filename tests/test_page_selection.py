import csv

import pytest

from receiving_bin_reader.batch import (
    crop_pdf_batch, parse_page_selection, read_pdf_batch_consensus,
)
from receiving_bin_reader.consensus import RecognitionPass
from test_batch import sample_pdf


@pytest.mark.parametrize(("value", "expected"), [
    ("1", [1]), ("1,3-5", [1, 3, 4, 5]), ("2,1", [1, 2]),
])
def test_page_selection_parser(value, expected):
    assert parse_page_selection(value) == expected


@pytest.mark.parametrize("bad", [
    "", "0", "1,", "a", "3-2", "1-0", "1,1", "1,2-3,3",
])
def test_reject_invalid_page_selections(bad):
    with pytest.raises(ValueError):
        parse_page_selection(bad)


def test_explicit_pages_skip_supporting_pages_safely(tmp_path):
    pdf = tmp_path / "two.pdf"
    sample_pdf(pdf)
    out = tmp_path / "out"
    result = crop_pdf_batch(pdf, out, dpi=72, pages=[2])
    assert result["source_page_count"] == 2
    assert result["pages"] == 1
    assert result["selected_pages"] == [2]
    assert result["skipped_pages"] == [1]
    assert result["row_count"] == 5
    with (out / "manifest.csv").open(newline="") as handle:
        records = list(csv.DictReader(handle))
    assert len(records) == 5
    assert all(x["page"] == "2" for x in records)


def test_out_of_range_page_rejected(tmp_path):
    pdf = tmp_path / "two.pdf"
    sample_pdf(pdf)
    with pytest.raises(ValueError, match="range"):
        crop_pdf_batch(pdf, tmp_path / "out", dpi=72, pages=[3])
