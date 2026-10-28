from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from .models import Box, OcrToken


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


class PaddleBinReader:
    def __init__(self, score_threshold: float = 0.0) -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR is not installed. Install the optional OCR dependencies."
            ) from exc

        self.score_threshold = score_threshold
        self._ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang="en",
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
