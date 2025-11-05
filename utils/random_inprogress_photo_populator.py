"""Populate in-progress (stage 2) folders with random work photos.

Source images::

    examples/source_photos/Works

Target tree defaults to the configured project workspace. The script copies
photos into every folder whose name matches a stage 2 title from
``WORK_PROFILES`` (e.g. ``2 Surface polished``). Unlike the general
populator, it never skips folders that already contain images; each run appends
an additional random photo. Images are not reused during a single execution.
"""
from __future__ import annotations

import random
import re
import shutil
from pathlib import Path
from typing import Iterable, List, Set

from configs.work_profiles import WORK_STAGE_TITLES
from configs.config import FOLDER_PATH

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "examples" / "source_photos" / "Works"
TARGET_ROOT = FOLDER_PATH

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DATE_COMPONENT = re.compile(r"\d{2}\.\d{2}\.\d{4}")

STAGE_TWO_TITLES: Set[str] = {
    str(titles[1]).strip()
    for titles in WORK_STAGE_TITLES.values()
    if len(titles) >= 2
}


def _collect_images(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Source folder not found: {root}")
    images = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not images:
        raise RuntimeError(f"No images discovered under {root}")
    random.shuffle(images)
    return images


def _has_date_ancestor(path: Path) -> bool:
    return any(DATE_COMPONENT.fullmatch(part) for part in path.parts)


def _stage_two_directories(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        if path.name.strip() not in STAGE_TWO_TITLES:
            continue
        if not _has_date_ancestor(path):
            continue
        yield path


def _copy_image(src: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    counter = 1
    while dest.exists():
        dest = dest_dir / f"{src.stem}_copy{counter}{src.suffix}"
        counter += 1
    shutil.copy2(src, dest)
    return dest


def populate() -> int:
    images = _collect_images(SOURCE_ROOT)
    moved = 0

    for directory in _stage_two_directories(TARGET_ROOT):
        if not images:
            break
        image = images.pop()
        _copy_image(image, directory)
        moved += 1
    return moved


if __name__ == "__main__":
    total = populate()
    print(f"Copied {total} photos into stage 2 folders.")
