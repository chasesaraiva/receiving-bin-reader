from receiving_bin_reader.consensus import (
    RecognitionPass,
    decide_consensus,
)


APPROVED = {"1A1", "1A4", "1B15", "3C18", "4A19"}


def test_two_independent_models_accept_exact_bin():
    decision = decide_consensus(
        [
            RecognitionPass("english", "core", "1A4"),
            RecognitionPass("server", "3x", "|1A4"),
        ],
        APPROVED,
    )
    assert decision.status == "ACCEPT"
    assert decision.bins == ("1A4",)


def test_two_variants_of_same_model_are_not_independent_votes():
    decision = decide_consensus(
        [
            RecognitionPass("server", "core", "4A19"),
            RecognitionPass("server", "3x", "4A19"),
        ],
        APPROVED,
    )
    assert decision.status == "REVIEW"


def test_no_fuzzy_correction():
    decision = decide_consensus(
        [
            RecognitionPass("english", "core", "IA1"),
            RecognitionPass("server", "3x", "IA1"),
        ],
        APPROVED,
    )
    assert decision.status == "REVIEW"
    assert decision.bins == ()


def test_multi_bin_requires_every_part_to_be_approved():
    decision = decide_consensus(
        [
            RecognitionPass("english", "core", "1A4,1B15"),
            RecognitionPass("server", "3x", "1A4,1B15"),
        ],
        APPROVED,
    )
    assert decision.status == "ACCEPT"
    assert decision.bins == ("1A4", "1B15")


def test_conflicting_corroborated_candidates_are_reviewed():
    decision = decide_consensus(
        [
            RecognitionPass("english", "core", "1A1"),
            RecognitionPass("server", "core", "1A1"),
            RecognitionPass("english", "3x", "4A19"),
            RecognitionPass("server", "3x", "4A19"),
        ],
        APPROVED,
    )
    assert decision.status == "REVIEW"
