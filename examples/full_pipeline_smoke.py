"""Try the real main.py reporting pipeline using a private temporary synthetic folder.

The output is checked for an actual PPTX with embedded photos and no modified
source images. No company photo files or configured production workspaces used.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw
from pptx import Presentation


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="photo-pipeline-check-") as temp:
        root = Path(temp) / "fictional-photo-workspace"
        photo_dir = root / "1 December" / "01.12.2024" / "1"
        photo_dir.mkdir(parents=True)
        files = []
        for n in (1, 2):
            p = photo_dir / f"fictional_{n}.jpg"
            picture = Image.new("RGB", (1000, 700), (30 + n * 35, 60, 90))
            draw = ImageDraw.Draw(picture)
            draw.rectangle((100, 110, 850, 590), outline=(255, 255, 255), width=6)
            draw.text((240, 300), f"FICTIONAL WORKSITE PHOTO {n}", fill="white")
            picture.save(p)
            files.append(p)
        before = {f: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
        env = dict(os.environ)
        env.update({
            "FOLDER_PATH": str(root),
            "REPORT_MODE": "daily",
            "START_DATE": "01.12.2024",
            "END_DATE": "01.12.2024",
            "GENERATE_ALL_REPORTS": "false",
            "RAW_IMAGE_MODE": "true",
            "MONTHLY_WORK_PLAN": "{}",
            "TITLE_CONTENT": "{activity}\\nFICTIONAL PORTFOLIO TEST\\nLocation: Sample Street\\nReporting period: {period}",
        })
        subprocess.run([sys.executable, "main.py"], check=True, env=env, cwd=Path(__file__).resolve().parents[1], timeout=180)
        presentations = list(root.rglob("*.pptx"))
        assert presentations, "Full pipeline did not generate any PowerPoint report"
        slide_count = 0
        images = 0
        for deck in presentations:
            pptx = Presentation(deck)
            slide_count += len(pptx.slides)
            images += sum(s.shape_type == 13 for slide in pptx.slides for s in slide.shapes)
        assert slide_count >= 1, "Generated PPTX has no slides"
        assert images >= 2, f"Expected two synthetic embedded photos, got {images}"
        for p, digest in before.items():
            assert hashlib.sha256(p.read_bytes()).hexdigest() == digest, f"Input was modified: {p}"
        print(f"PASS: Full reporting pipeline generated {len(presentations)} PPTX, {slide_count} slides, {images} embedded images; inputs unchanged")


if __name__ == "__main__":
    main()
