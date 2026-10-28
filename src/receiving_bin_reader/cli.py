from __future__ import annotations

import argparse
import json

from .batch import crop_pdf_batch, parse_page_selection, read_pdf_batch_consensus
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
    cg = crop.add_mutually_exclusive_group()
    cg.add_argument("--page", type=int, default=1)
    cg.add_argument("--all-pages", action="store_true")
    cg.add_argument("--pages", help="Receiving worksheet pages only, e.g. 1,3-5.")
    crop.add_argument("--dpi", type=int, default=180)

    read = subparsers.add_parser(
        "read", help="Crop rows, read BINs, and require exact cross-model consensus."
    )
    read.add_argument("pdf")
    read.add_argument("--approved-bins", required=True)
    read.add_argument("--out", required=True)
    rg = read.add_mutually_exclusive_group()
    rg.add_argument("--page", type=int, default=1)
    rg.add_argument("--all-pages", action="store_true")
    rg.add_argument("--pages", help="Receiving worksheet pages only, e.g. 1,3-5.")
    read.add_argument("--dpi", type=int, default=180)
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()

    selection = None
    if args.pages:
        try:
            selection = parse_page_selection(args.pages)
        except ValueError as exc:
            parser.error(str(exc))

    if args.command == "crop":
        if args.all_pages or selection is not None:
            result = crop_pdf_batch(args.pdf, args.out, dpi=args.dpi, pages=selection)
        else:
            result = crop_pdf(args.pdf, args.out, page_number=args.page, dpi=args.dpi)
        print(json.dumps(result, indent=2))
        return

    if args.all_pages or selection is not None:
        result = read_pdf_batch_consensus(
            args.pdf, args.approved_bins, args.out,
            dpi=args.dpi, pages=selection
        )
        print(json.dumps(result, indent=2))
        return

    reads = read_pdf_consensus(
        args.pdf, args.approved_bins, args.out,
        page_number=args.page, dpi=args.dpi,
    )
    print(json.dumps({
        "row_count": len(reads),
        "valid": sum(item.status == "VALID" for item in reads),
        "review": sum(item.status == "REVIEW" for item in reads),
    }, indent=2))


if __name__ == "__main__":
    main()
