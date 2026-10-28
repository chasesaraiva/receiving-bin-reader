from receiving_bin_reader.validation import (
    normalize_bin,
    split_bin_candidates,
    validate_candidates,
)


def test_normalization_does_not_fuzzy_correct():
    assert normalize_bin("R 1 B") == "R1B"
    assert normalize_bin("RIB") == "RIB"
    assert normalize_bin("3 b 2") == "3B2"


def test_multiple_bins():
    assert split_bin_candidates("1A4, 1B15") == ["1A4", "1B15"]


def test_validation_is_exact():
    approved = {"1A4", "1B15", "R1B"}
    accepted, rejected = validate_candidates(
        ["1A4", "1B15", "RIB"],
        approved,
    )
    assert accepted == ["1A4", "1B15"]
    assert rejected == ["RIB"]
