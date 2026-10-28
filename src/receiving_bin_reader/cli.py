from __future__ import annotations

import argparse
import json

from .pipeline import crop_pdf
from .safe_reader import read_pdf_consensus


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="receiving-bin-reader",
        description="Deterministic receiving-row cropper and BIN OCR.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    crop = subparsers.add_parser("crop", help="Crop physical rows; no OCR.")
    crop.add_argument("pdf")
    crop.add_argument("--out", required=True)
    crop.add_argument("--page", type=int, default=1)
    crop.add_argument("--dpi", type=int, default=180)

    read = subparsers.add_parser(
        "read",
        help="Crop rows, read BINs, and require exact cross-model consensus.",
    )
    read.add_argument("pdf")
    read.add_argument("--approved-bins", required=True)
    read.add_argument("--out", required=True)
    read.add_argument("--page", type=int, default=1)
    read.add_argument("--dpi", type=int, default=180)

    return parser


def main() -> None:
    args = _parser().parse_args()

    if args.command == "crop":
        result = crop_pdf(
            args.pdf,
            args.out,
            page_number=args.page,
            dpi=args.dpi,
        )
        print(json.dumps(result, indent=2))
        return

    reads = read_pdf_consensus(
        args.pdf,
        args.approved_bins,
        args.out,
        page_number=args.page,
        dpi=args.dpi,
    )

    print(
        json.dumps(
            {
                "row_count": len(reads),
                "valid": sum(1 for item in reads if item.status == "VALID"),
                "review": sum(1 for item in reads if item.status == "REVIEW"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
