from PIL import Image, ImageDraw

from receiving_bin_reader.ocr import (
    _recognition_variant, CONSENSUS_VARIANTS, EXTENDED_VARIANTS,
)
from receiving_bin_reader.consensus import RecognitionPass, decide_consensus


def test_extended_variants_are_opt_in():
    assert CONSENSUS_VARIANTS == ("core", "3x_border")
    assert EXTENDED_VARIANTS == (
        "core", "3x_border", "rotate_cw3", "rotate_ccw3"
    )


def test_small_angle_rotations_create_expected_rgb_canvas():
    image = Image.new("L", (150, 48), "white")
    ImageDraw.Draw(image).text((20, 12), "1A4,1B15", fill="black")
    for variant in EXTENDED_VARIANTS:
        result = _recognition_variant(image, variant)
        assert result.mode == "RGB"
        assert result.width >= image.width
        assert result.height >= image.height


def test_new_variants_cannot_vote_without_two_distinct_models():
    decision = decide_consensus(
        [
            RecognitionPass("english", "core", "1A4"),
            RecognitionPass("english", "rotate_cw3", "1A4"),
            RecognitionPass("english", "rotate_ccw3", "1A4"),
        ],
        {"1A4"},
    )
    assert decision.status == "REVIEW"


def test_conflicting_rotated_predictions_still_require_review():
    decision = decide_consensus(
        [
            RecognitionPass("english", "core", "1A4"),
            RecognitionPass("server", "rotate_cw3", "1A4"),
            RecognitionPass("english", "rotate_ccw3", "1A1"),
            RecognitionPass("server", "3x_border", "1A1"),
        ],
        {"1A4", "1A1"},
    )
    assert decision.status == "REVIEW"
