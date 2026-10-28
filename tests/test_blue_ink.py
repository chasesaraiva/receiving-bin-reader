from __future__ import annotations

from PIL import Image, ImageDraw

from receiving_bin_reader.blue_ink import isolate_blue_ink
from receiving_bin_reader.consensus import RecognitionPass, decide_consensus
from receiving_bin_reader.ocr import BLUE_VARIANTS, PaddleConsensusReader, _recognition_variant
from receiving_bin_reader.safe_reader import _blue_only_accept


def test_blue_mask_keeps_blue_ink_but_excludes_black_printing():
    image = Image.new("RGB", (200, 65), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 26, 48), fill="black")
    draw.line((45, 12, 130, 49), fill=(24, 22, 146), width=5)
    blue = isolate_blue_ink(image)
    assert blue is not None
    assert blue.getpixel((16, 24)) == 255
    assert blue.getpixel((80, 28)) == 0


def test_monochrome_printing_does_not_generate_blue_ocr_votes():
    image = Image.new("RGB", (180, 45), "white")
    ImageDraw.Draw(image).text((15, 15), "4A19", fill="black")
    assert isolate_blue_ink(image) is None
    assert _recognition_variant(image, "blue_core") is None
    assert _recognition_variant(image, "blue_3x_border") is None


def test_one_or_two_blue_pixels_are_not_ocr_evidence():
    image = Image.new("RGB", (90, 40), "white")
    image.putpixel((11, 11), (5, 5, 180))
    assert isolate_blue_ink(image) is None


def test_blue_only_prediction_cannot_be_accepted():
    passes = [
        RecognitionPass("english", "core", "1A1"),
        RecognitionPass("english", "blue_core", "1A4"),
        RecognitionPass("server", "blue_core", "1A4"),
    ]
    decision = decide_consensus(passes, {"1A1", "1A4"})
    assert decision.status == "ACCEPT"
    assert _blue_only_accept(decision, passes)


def test_one_original_variant_supports_blue_consensus():
    passes = [
        RecognitionPass("english", "core", "1A4"),
        RecognitionPass("english", "blue_core", "1A4"),
        RecognitionPass("server", "blue_core", "1A4"),
    ]
    decision = decide_consensus(passes, {"1A4"})
    assert decision.status == "ACCEPT"
    assert not _blue_only_accept(decision, passes)


def test_rejected_if_two_distinct_models_disagree_even_after_blue():
    passes = [
        RecognitionPass("english", "core", "1A4"),
        RecognitionPass("server", "core", "1A4"),
        RecognitionPass("english", "blue_core", "1B7"),
        RecognitionPass("server", "blue_core", "1B7"),
    ]
    assert decide_consensus(passes, {"1A4", "1B7"}).status == "REVIEW"


def test_blue_ocr_skips_blank_blue_crop(monkeypatch):
    from receiving_bin_reader import ocr
    calls = []
    class MockRecognition:
        def __init__(self, model_name, device):
            self.name = model_name
        def predict(self, input, batch_size):
            calls.append(batch_size)
            return [{"res": {"rec_text": "1A4", "rec_score": 0.98}} for _ in input]
    import types, sys
    monkeypatch.setitem(sys.modules, "paddleocr", types.SimpleNamespace(
        TextRecognition=MockRecognition
    ))
    reader = PaddleConsensusReader(
        model_names=("english", "server"), variants=BLUE_VARIANTS
    )
    output = reader.read_passes(Image.new("RGB", (100, 40), "white"))
    assert len(output) == 4   # 2 models x 2 ORIGINAL image variants
    assert all(not r.variant.startswith("blue_") for r in output)
    assert calls == [2, 2]
