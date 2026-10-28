from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from .models import PageGeometry


def save_crops(
    image: Image.Image,
    geometry: PageGeometry,
    out_dir: str | Path,
) -> tuple[list[Path], list[Path]]:
    out_dir = Path(out_dir)
    rows_dir = out_dir / "rows"
    bins_dir = out_dir / "bins"
    rows_dir.mkdir(parents=True, exist_ok=True)
    bins_dir.mkdir(parents=True, exist_ok=True)

    row_paths: list[Path] = []
    bin_paths: list[Path] = []

    for row in geometry.rows:
        row_path = rows_dir / f"row_{row.row_number:03d}.png"
        bin_path = bins_dir / f"bin_{row.row_number:03d}.png"

        image.crop(
            (0, row.core.y1, geometry.bin_right, row.core.y2)
        ).save(row_path)

        bin_image = image.crop(
            (
                row.bin_padded.x1,
                row.bin_padded.y1,
                row.bin_padded.x2,
                row.bin_padded.y2,
            )
        )
        bin_image.save(bin_path)

        row_paths.append(row_path)
        bin_paths.append(bin_path)

    return row_paths, bin_paths


def save_overlay(
    image: Image.Image,
    geometry: PageGeometry,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)

    draw.line(
        (geometry.sku_right, geometry.header_bottom, geometry.sku_right, geometry.page_height),
        fill=(255, 0, 0),
        width=3,
    )
    draw.line(
        (geometry.bin_right, geometry.header_bottom, geometry.bin_right, geometry.page_height),
        fill=(255, 0, 0),
        width=3,
    )

    for row in geometry.rows:
        draw.line(
            (0, row.core.y1, geometry.bin_right, row.core.y1),
            fill=(0, 120, 255),
            width=2,
        )
        draw.text((8, row.core.y1 + 2), str(row.row_number), fill=(255, 0, 0))

    if geometry.rows:
        draw.line(
            (0, geometry.rows[-1].core.y2, geometry.bin_right, geometry.rows[-1].core.y2),
            fill=(0, 120, 255),
            width=2,
        )

    canvas.save(output_path)
    return output_path


def save_contact_sheet(
    image: Image.Image,
    geometry: PageGeometry,
    output_path: str | Path,
    width: int = 1500,
) -> Path:
    output_path = Path(output_path)

    entries: list[tuple[int, Image.Image]] = []
    total_height = 16

    for row in geometry.rows:
        crop = image.crop(
            (0, row.core.y1, geometry.bin_right, row.core.y2)
        ).convert("RGB")
        target_width = width - 110
        scale = target_width / crop.width
        target_height = max(42, int(round(crop.height * scale * 1.7)))
        resized = crop.resize((target_width, target_height), Image.Resampling.LANCZOS)
        entries.append((row.row_number, resized))
        total_height += target_height + 12

    sheet = Image.new("RGB", (width, total_height), "white")
    draw = ImageDraw.Draw(sheet)

    y = 16
    for row_number, crop in entries:
        draw.text((10, y + 8), f"ROW {row_number:03d}", fill="black")
        sheet.paste(crop, (100, y))
        y += crop.height + 12

    sheet.save(output_path)
    return output_path
