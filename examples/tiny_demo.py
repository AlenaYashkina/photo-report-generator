"""Small, reproducible slide-renderer demonstration using wholly synthetic images.

Runs locally with the project's normal requirements; no worksite photographs or
production settings are required. Exercises the actual PPTX layout helpers.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Allow direct execution from the examples/ directory on a fresh checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def create_demo(output: Path) -> None:
    from PIL import Image, ImageDraw
    from pptx import Presentation
    from pptx.util import Inches

    with tempfile.TemporaryDirectory(prefix="photo-report-fixture-") as temp:
        root = Path(temp)
        # Prevent importing the app configuration from seeding the full example archive.
        os.environ["FOLDER_PATH"] = str(root)
        os.environ["TITLE_CONTENT"] = (
            "{activity}\\nSYNTHETIC PORTFOLIO DEMO — NOT AN ACTUAL WORK REPORT\\n"
            "Location: Fictional Sample Site\\nReporting period: {period}"
        )
        from photo_reports.pptx_creator import add_title_slide, insert_slides_adaptive

        items = []
        source_checksums = {}
        for index, label in enumerate(("BEFORE", "AFTER"), start=1):
            path = root / f"synthetic_{index}.png"
            img = Image.new("RGB", (1080, 700), (29, 53 + index * 27, 78 + index * 18))
            draw = ImageDraw.Draw(img)
            draw.rectangle((90, 90, 980, 610), outline=(190, 215, 234), width=11)
            draw.rectangle((185, 190, 370, 500), fill=(45 + index * 12, 119, 160))
            draw.text((410, 300), f"FICTIONAL {label} IMAGE", fill="white")
            img.save(path)
            source_checksums[path] = hashlib.sha256(path.read_bytes()).hexdigest()
            items.append((str(path), f"Example {index}: {label} (synthetic)"))

        presentation = Presentation()
        presentation.slide_width = Inches(13.33)
        presentation.slide_height = Inches(7.5)
        add_title_slide(presentation, "Example maintenance report", "01–02 Nov 2026")
        insert_slides_adaptive(presentation, items)
        output.parent.mkdir(parents=True, exist_ok=True)
        presentation.save(output)

        verify = Presentation(output)
        assert len(verify.slides) >= 2, "Expected title and photo slide"
        picture_count = sum(
            shape.shape_type == 13 for slide in verify.slides for shape in slide.shapes
        )
        assert picture_count == 2, f"Expected two embedded images, got {picture_count}"
        for path, digest in source_checksums.items():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, "Input image modified"
            shutil.copy2(path, output.parent / path.name)
        print(f"PASS: {output} — {len(verify.slides)} slides, {picture_count} images, sources unchanged")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/tiny_demo.pptx"))
    args = parser.parse_args()
    create_demo(args.output)
