"""Create LOCAL-only privacy review contact sheets and an Office/media manifest.

The output is intentionally gitignored. Thumbnail sheets help a HUMAN inspect
faces, documents, addresses and worksite details; they do not certify privacy.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
OFFICE_EXTS = {".pptx", ".xlsx", ".docx", ".pdf"}
COLUMNS, ROWS = 6, 6
TILE_W, TILE_H = 264, 206


def create_review(root: Path, output: Path, include_derived: bool = False) -> tuple[int, int]:
    root = root.resolve()
    output = output.resolve()
    if not root.is_dir():
        raise ValueError(f"Input directory is missing: {root}")
    if output == root or root in output.parents:
        # It is safe for an output subfolder to exist, but must not recursively
        # include it as a source on reruns.
        pass
    output.mkdir(parents=True, exist_ok=True)
    images, offices = [], []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file() or output in path.parents:
            continue
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTS:
            if not include_derived and ("_filtered" in path.stem.lower() or "_stamped" in path.stem.lower()):
                continue
            images.append(path)
        elif suffix in OFFICE_EXTS:
            offices.append(path)

    manifest = output / "manual-review-checklist.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "relative_path", "type", "reviewed", "faces_or_names", "addresses_or_contacts",
                         "rights_confirmed", "action", "reviewer_notes"])
        for index, path in enumerate(images + offices, start=1):
            writer.writerow([index, path.relative_to(root).as_posix(),
                             "image" if path in images else "office_document",
                             "NO", "UNKNOWN", "UNKNOWN", "UNKNOWN", "REVIEW", ""])

    font = ImageFont.load_default()
    for page_number, offset in enumerate(range(0, len(images), COLUMNS * ROWS), start=1):
        sheet = Image.new("RGB", (COLUMNS * TILE_W, ROWS * TILE_H), "#182237")
        draw = ImageDraw.Draw(sheet)
        for local_index, path in enumerate(images[offset:offset + COLUMNS * ROWS]):
            row, column = divmod(local_index, COLUMNS)
            x, y = column * TILE_W, row * TILE_H
            try:
                with Image.open(path) as img:
                    thumb = ImageOps.exif_transpose(img).convert("RGB")
                    thumb.thumbnail((TILE_W - 18, TILE_H - 48))
                sheet.paste(thumb, (x + (TILE_W - thumb.width) // 2, y + 5))
            except (OSError, ValueError):
                draw.text((x + 12, y + 28), "UNREADABLE IMAGE", fill="#fca5a5", font=font)
            label = f"{offset + local_index + 1:04d} | {path.name[:31]}"
            draw.text((x + 8, y + TILE_H - 33), label, fill="white", font=font)
            draw.rectangle((x, y, x + TILE_W - 1, y + TILE_H - 1), outline="#425a79", width=1)
        sheet.save(output / f"contact-sheet-{page_number:02d}.jpg", quality=86)

    print(f"LOCAL MANUAL REVIEW: {len(images)} source images, {len(offices)} Office/PDF documents.")
    print(f"Private review checklist: {manifest}")
    print("No visual privacy clearance is implied. Inspect originals and document contents yourself.")
    return len(images), len(offices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("examples/reference/ExampleLocation"))
    parser.add_argument("--out", type=Path, default=Path(".privacy-review"))
    parser.add_argument("--include-derived", action="store_true")
    args = parser.parse_args()
    create_review(args.root, args.out, args.include_derived)
