from __future__ import annotations

import time

from PIL import Image, ImageDraw, ImageFont

from receiving_bin_reader.ocr import PaddleBinReader


def main() -> None:
    image = Image.new("RGB", (500, 120), "white")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            64,
        )
    except OSError:
        font = ImageFont.load_default()

    draw.text((25, 20), "1A10", fill="black", font=font)

    start = time.perf_counter()
    reader = PaddleBinReader()
    tokens = reader.read(image)
    elapsed = time.perf_counter() - start

    print("recognition model:", reader.recognition_model)
    print("detection model:", reader.detection_model)
    print("tokens:", [(token.text, token.score) for token in tokens])
    print(f"startup + first inference: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
