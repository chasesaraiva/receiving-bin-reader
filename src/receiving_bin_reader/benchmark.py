from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from PIL import Image

from .ocr import PaddleBinReader
from .validation import normalize_bin, split_bin_candidates


def _canonical(value: str) -> str:
    parts = [
        normalize_bin(part)
        for part in str(value or "").split(",")
        if normalize_bin(part)
    ]
    return ",".join(parts)


def run_benchmark(
    crop_output_dir: str | Path,
    truth_csv: str | Path,
    output_dir: str | Path | None = None,
) -> dict:
    crop_output_dir = Path(crop_output_dir)
    truth_csv = Path(truth_csv)
    output_dir = Path(output_dir) if output_dir else crop_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    truth: dict[int, str] = {}
    with truth_csv.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            truth[int(row["row"])] = _canonical(row["expected"])

    manifest = list(
        csv.DictReader((crop_output_dir / "manifest.csv").open(encoding="utf-8"))
    )

    reader = PaddleBinReader(score_threshold=0.0)
    start = time.perf_counter()

    results: list[dict] = []

    for item in manifest:
        row_number = int(item["row"])
        if row_number not in truth:
            continue

        bin_path = crop_output_dir / item["bin_crop"]
        image = Image.open(bin_path).convert("RGB")

        core_top = int(item["core_top"]) - int(item["bin_padded_top"])
        core_bottom = int(item["core_bottom"]) - int(item["bin_padded_top"])

        tokens = reader.read(image)
        owned = [
            token
            for token in tokens
            if core_top <= token.box.center_y <= core_bottom
        ]

        candidates: list[str] = []
        for token in sorted(owned, key=lambda token: (token.box.y1, token.box.x1)):
            candidates.extend(split_bin_candidates(token.text))

        candidates = list(dict.fromkeys(candidates))
        actual = ",".join(candidates)
        expected = truth[row_number]

        results.append(
            {
                "row": row_number,
                "expected": expected,
                "actual": actual,
                "exact_match": actual == expected,
                "raw_tokens": " | ".join(token.text for token in owned),
                "scores": " | ".join(f"{token.score:.4f}" for token in owned),
            }
        )

    elapsed = time.perf_counter() - start
    correct = sum(1 for row in results if row["exact_match"])
    total = len(results)

    with (output_dir / "benchmark_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()) if results else [
            "row", "expected", "actual", "exact_match", "raw_tokens", "scores"
        ])
        writer.writeheader()
        writer.writerows(results)

    summary = {
        "rows": total,
        "exact_matches": correct,
        "exact_accuracy": (correct / total) if total else 0.0,
        "elapsed_seconds": elapsed,
        "seconds_per_row": (elapsed / total) if total else 0.0,
        "recognition_model": reader.recognition_model,
        "detection_model": reader.detection_model,
    }

    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark PaddleOCR against known BIN truth without changing crop geometry."
    )
    parser.add_argument("crop_output_dir")
    parser.add_argument("--truth", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()

    summary = run_benchmark(
        args.crop_output_dir,
        args.truth,
        output_dir=args.out,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
