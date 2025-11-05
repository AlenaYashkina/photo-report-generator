"""Populate numeric and stage (1 & 3) folders with random photos.

Source images are taken from::

    examples/source_photos/Days/<CN>

Each CN (1, 2, 3, ...) is expected to have its own folder. The script walks the
configured project workspace and copies a random, unused image into:
  - root CN folders named exactly as digits (e.g. ``1``)
  - stage folders whose names match stage 1 or stage 3 titles defined in
    ``WORK_PROFILES`` (e.g. ``1 Issue detected...``, ``3 Issue resolved``)

Rules:
  - Only one image per destination folder; folders that already contain images
    are skipped.
  - Images are copied (not moved) and never reused within the same run.
  - Stage matching ignores stage 2 titles entirely.
"""
from __future__ import annotations

import random
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from configs.work_profiles import WORK_STAGE_TITLES
from configs.config import FOLDER_PATH

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "examples" / "source_photos" / "Days"
TARGET_ROOT = FOLDER_PATH

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
NUMERIC_NAME = re.compile(r"^\d+$")


def _collect_stage_titles() -> Tuple[Set[str], Set[str]]:
    stage_one: Set[str] = set()
    stage_three: Set[str] = set()
    for titles in WORK_STAGE_TITLES.values():
        if len(titles) >= 1:
            stage_one.add(str(titles[0]).strip())
        if len(titles) >= 3:
            stage_three.add(str(titles[2]).strip())
    return stage_one, stage_three


STAGE_ONE_TITLES, STAGE_THREE_TITLES = _collect_stage_titles()
_normalised_stage_one = {title.lower() for title in STAGE_ONE_TITLES}
_normalised_stage_three = {title.lower() for title in STAGE_THREE_TITLES}


def _gather_images(root: Path) -> Dict[str, List[Path]]:
    if not root.exists():
        raise FileNotFoundError(f"Portfolio root not found: {root}")
    buckets: Dict[str, List[Path]] = {}
    for subdir in sorted(root.iterdir(), key=lambda p: p.name):
        if not subdir.is_dir():
            continue
        name = subdir.name.strip()
        if not NUMERIC_NAME.fullmatch(name):
            continue
        images = [
            path
            for path in subdir.glob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if images:
            random.shuffle(images)
            buckets[name] = images
    if not buckets:
        raise RuntimeError(f"No portfolio images discovered under {root}")
    return buckets


def _find_cn_token(path: Path) -> Optional[str]:
    for part in reversed(path.parts):
        stripped = part.strip()
        token = stripped.split()[0]
        if NUMERIC_NAME.fullmatch(token):
            return token
    return None


def _directory_has_images(directory: Path) -> bool:
    try:
        for child in directory.iterdir():
            if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS:
                return True
    except OSError:
        return False
    return False


def _target_directories(root: Path) -> Iterable[Tuple[str, Path]]:
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        name = str(path.name).strip()
        if name.isdigit() and NUMERIC_NAME.fullmatch(name):
            yield name, path
            continue
        normalised_name = name.strip().lower()
        if normalised_name in _normalised_stage_one or normalised_name in _normalised_stage_three:
            cn = _find_cn_token(path.parent)
            if cn:
                yield cn, path


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
    images_map = _gather_images(SOURCE_ROOT)
    used: Set[Path] = set()
    moved = 0

    for cn, directory in _target_directories(TARGET_ROOT):
        if _directory_has_images(directory):
            continue
        candidates = images_map.get(cn)
        if not candidates:
            continue
        while candidates:
            candidate = candidates.pop()
            if candidate in used:
                continue
            _copy_image(candidate, directory)
            used.add(candidate)
            moved += 1
            break
    return moved


if __name__ == "__main__":
    total = populate()
    print(f"Copied {total} photos into stage 1/3 and CN folders.")
