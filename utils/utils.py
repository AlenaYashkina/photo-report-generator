# utils\utils.py
import functools
import re
from datetime import timedelta
from pathlib import Path
from random import randint
from typing import Optional, List, Sequence

from PIL import Image

from configs.config import MIN_GAP_SEC, MAX_GAP_SEC
from configs.utils_config import STAGE_FOLDER_RE, DATE_RE, IMAGE_EXTENSIONS, SUFFIX_RE


@functools.lru_cache(maxsize=None)
def stage_dirs(group_path: Path) -> list[str]:
    dirs = [p.name for p in group_path.iterdir()
            if p.is_dir() and STAGE_FOLDER_RE.match(p.name)]
    return sorted(dirs, key=lambda name: int(name.split()[0]))


def split_parts(path: Path) -> List[str]:
    return re.split(r"[\\/]", str(path))


@functools.lru_cache(maxsize=65536)
def parse_date(parts: Sequence[str]) -> Optional[str]:
    for raw in reversed(parts):
        token = str(raw).strip()
        if not token:
            continue

        prefix_chars = []
        for ch in token:
            if ch.isdigit() or ch in ".-":
                prefix_chars.append(ch)
            else:
                break
        date_token = "".join(prefix_chars)
        if not date_token:
            continue

        m = DATE_RE.match(date_token)
        if not m:
            continue

        day_str, month_str, year_str = m.groups()

        if year_str is None and "-" in token and "." not in token:
            continue

        day = int(day_str)
        month = int(month_str)
        if not (1 <= day <= 31 and 1 <= month <= 12):
            continue

        if year_str is None:
            year = "2024" if month_str == "12" else "2025"
        elif len(year_str) == 2:
            year = "20" + year_str
        else:
            year = year_str

        return f"{day:02d}.{month:02d}.{year}"
    return None


def random_gap() -> timedelta:
    return timedelta(seconds=randint(MIN_GAP_SEC, MAX_GAP_SEC))


def find_images(folder: Path) -> List[Path]:
    return [p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]


def get_base_name(name: str) -> str:
    stem = Path(name).stem
    while True:
        new_stem = SUFFIX_RE.sub("", stem)
        if new_stem == stem:
            return stem
        stem = new_stem


def extract_construction_numbers(path: Path) -> List[int]:
    parts = split_parts(path)
    dir_parts = tuple(parts[:-1])

    date_str = parse_date(dir_parts)
    if not date_str:
        return []

    try:
        i = next(idx for idx, part in enumerate(dir_parts) if date_str in str(part))
    except StopIteration:
        return []

    if i + 1 >= len(dir_parts):
        return []

    seg = (
        dir_parts[i + 1]
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
    )
    nums: list[int] = []
    for token in re.findall(r"\d+(?:\s*-\s*\d+)?", seg):
        if "-" in token:
            a, b = map(int, token.split("-", 1))
            lo, hi = sorted((a, b))
            nums.extend(range(lo, hi + 1))
        else:
            nums.append(int(token))
    return sorted(set(nums))


def detect_stage(path: Path) -> str | None:
    dirs = stage_dirs(path.parent.parent)
    if len(dirs) < 2:
        return None
    folder = path.parent.name
    if folder == dirs[0]:
        return "detected"
    if folder == dirs[-1]:
        return "fixed"
    if folder in dirs:
        return "in_progress"
    return None


def ensure_pptx_compatible(path: str) -> str:
    try:
        with Image.open(path) as im:
            fmt = (im.format or "").upper()
        if fmt == "WEBP":
            out = str(Path(path).with_suffix(".png"))
            with Image.open(path) as im2:
                im2.convert("RGB").save(out, format="PNG")
            return out
    except Exception:
        pass
    return path
