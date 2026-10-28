from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .blue_ink import blue_recognition_variant
from .consensus import RecognitionPass
from .models import Box, OcrToken


DEFAULT_DETECTION_MODEL = "PP-OCRv5_mobile_det"
DEFAULT_RECOGNITION_MODEL = "en_PP-OCRv5_mobile_rec"

CONSENSUS_MODELS = (
    "en_PP-OCRv5_mobile_rec",
    "PP-OCRv5_server_rec",
)
CONSENSUS_VARIANTS = ("core", "3x_border")
EXTENDED_VARIANTS = CONSENSUS_VARIANTS + ("rotate_cw3", "rotate_ccw3")
BLUE_VARIANTS = CONSENSUS_VARIANTS + ("blue_core", "blue_3x_border")


def _walk_for_ocr_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        if "rec_texts" in value and ("rec_boxes" in value or "rec_polys" in value):
            return value
        for child in value.values():
            found = _walk_for_ocr_dict(child)
            if found is not None:
                return found

    if isinstance(value, (list, tuple)):
        for child in value:
            found = _walk_for_ocr_dict(child)
            if found is not None:
                return found

    return None


def _box_from_any(raw: Any) -> Box:
    array = np.asarray(raw, dtype=float)

    if array.ndim == 1 and array.size >= 4:
        x1, y1, x2, y2 = array[:4]
        return Box(int(x1), int(y1), int(x2), int(y2))

    if array.ndim == 2 and array.shape[1] >= 2:
        xs = array[:, 0]
        ys = array[:, 1]
        return Box(
            int(xs.min()),
            int(ys.min()),
            int(xs.max()),
            int(ys.max()),
        )

    return Box(0, 0, 0, 0)


def _extract_recognition_result(result: Any) -> tuple[str, float]:
    payload = getattr(result, "json", result)
    if callable(payload):
        payload = payload()

    if isinstance(payload, dict):
        data = payload.get("res", payload)
        return str(data.get("rec_text", "")), float(data.get("rec_score", 0.0))

    return "", 0.0


def _recognition_variant(image: Image.Image, variant: str) -> Image.Image | None:
    if variant.startswith("blue_"):
        return blue_recognition_variant(image, variant)

    gray = image.convert("L")

    if variant == "core":
        return gray.convert("RGB")

    if variant == "3x_border":
        scaled = gray.resize(
            (gray.width * 3, gray.height * 3),
            Image.Resampling.LANCZOS,
        )
        return ImageOps.expand(
            scaled,
            border=(24, 18, 24, 18),
            fill=255,
        ).convert("RGB")

    if variant in ("rotate_cw3", "rotate_ccw3"):
        # Research-only option: ±3° on the original grayscale crop, then
        # the identical 3x upscaling and border used in private benchmarks.
        degrees = -3 if variant == "rotate_cw3" else 3
        rotated = gray.rotate(
            degrees,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=255,
        )
        scaled = rotated.resize(
            (rotated.width * 3, rotated.height * 3),
            Image.Resampling.LANCZOS,
        )
        return ImageOps.expand(
            scaled, border=(24, 18, 24, 18), fill=255
        ).convert("RGB")

    raise ValueError(f"Unknown recognition variant: {variant}")


class PaddleConsensusReader:
    """
    Recognition-only reader for an already isolated handwritten BIN crop.

    Two independent PaddleOCR recognition models read two fixed image variants.
    Row ownership has already been decided by deterministic geometry.
    """

    def __init__(
        self,
        model_names: tuple[str, ...] = CONSENSUS_MODELS,
        variants: tuple[str, ...] = CONSENSUS_VARIANTS,
    ) -> None:
        try:
            from paddleocr import TextRecognition
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR is not installed. Install the optional OCR dependencies."
            ) from exc

        self.model_names = model_names
        self.variants = variants
        self._models = {
            name: TextRecognition(model_name=name, device="cpu")
            for name in model_names
        }

    def read_passes(self, image: Image.Image) -> list[RecognitionPass]:
        rendered = {
            variant: _recognition_variant(image, variant)
            for variant in self.variants
        }
        # No blue ink => NO blue OCR passes. Do not let a model hallucinate a
        # vote on a blank color-extracted image.
        prepared = {
            variant: np.asarray(result)
            for variant, result in rendered.items()
            if result is not None
        }
        active_variants = [variant for variant in self.variants if variant in prepared]

        passes: list[RecognitionPass] = []

        for model_name, model in self._models.items():
            if not active_variants:
                continue
            outputs = list(
                model.predict(
                    input=[prepared[variant] for variant in active_variants],
                    batch_size=len(active_variants),
                )
            )

            for variant, output in zip(active_variants, outputs):
                text, score = _extract_recognition_result(output)
                passes.append(
                    RecognitionPass(
                        model=model_name,
                        variant=variant,
                        text=text,
                        score=score,
                    )
                )

        return passes


class PaddleBinReader:
    """
    Legacy detector + recognizer adapter retained for diagnostics/smoke tests.

    Production consensus reading should use PaddleConsensusReader.
    """

    def __init__(
        self,
        score_threshold: float = 0.0,
        detection_model: str = DEFAULT_DETECTION_MODEL,
        recognition_model: str = DEFAULT_RECOGNITION_MODEL,
    ) -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR is not installed. Install the optional OCR dependencies."
            ) from exc

        self.score_threshold = score_threshold
        self.detection_model = detection_model
        self.recognition_model = recognition_model

        self._ocr = PaddleOCR(
            text_detection_model_name=detection_model,
            text_recognition_model_name=recognition_model,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def read(self, image: Image.Image) -> list[OcrToken]:
        array = np.asarray(image.convert("RGB"))

        results = list(
            self._ocr.predict(
                input=array,
                text_rec_score_thresh=self.score_threshold,
            )
        )

        if not results:
            return []

        payload = getattr(results[0], "json", results[0])
        ocr_dict = _walk_for_ocr_dict(payload)

        if not ocr_dict:
            return []

        texts = list(ocr_dict.get("rec_texts", []))
        scores = list(ocr_dict.get("rec_scores", []))
        boxes = ocr_dict.get("rec_boxes")
        if boxes is None:
            boxes = ocr_dict.get("rec_polys", [])

        tokens: list[OcrToken] = []

        for index, text in enumerate(texts):
            score = float(scores[index]) if index < len(scores) else 0.0
            raw_box = boxes[index] if index < len(boxes) else [0, 0, 0, 0]
            tokens.append(
                OcrToken(
                    text=str(text),
                    score=score,
                    box=_box_from_any(raw_box),
                )
            )

        return tokens
