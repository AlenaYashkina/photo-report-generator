"""Shared regular expressions and localisation data used across the project."""
from __future__ import annotations

from datetime import timedelta
import re
from typing import Dict, List

from configs.env_utils import get_env_json
from configs.work_profiles import WORK_STAGE_TITLES

BEFORE_JITTER = timedelta(minutes=15)
WORKS_JITTER = timedelta(minutes=15)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
DATE_RE = re.compile(r'^(\d{1,2})[.\-](\d{1,2})(?:[.\-](\d{2,4}))?$')
SUFFIX_RE = re.compile(
    r'(?i)(?:'
    r'[_\- ](?:filtered|stamped|edited|cropped|retouch(?:ed)?|denoise|noise|upscaled?|rotated?|fix(?:ed)?|v\d+|ver\d+)'
    r'|\s*\(\d+\)'
    r')+$'
)
STAGE_FOLDER_RE = re.compile(r"^\d{1,2}\s+\D")

STAGE_ORDER = {None: 0, "detected": 1, "in_progress": 2, "fixed": 3}

DEFAULT_MONTH_NAMES: Dict[int, str] = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}

RU_MONTH: Dict[int, str] = {
    int(k): str(v)
    for k, v in get_env_json("MONTH_LABELS", DEFAULT_MONTH_NAMES).items()
}

SEASON_ORDER: List[int] = get_env_json(
    "SEASON_ORDER",
    [12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
)

STAGES: Dict[str, List[str]] = {
    code: [str(item) for item in titles] for code, titles in WORK_STAGE_TITLES.items()
}

__all__ = [
    "IMAGE_EXTENSIONS",
    "DATE_RE",
    "SUFFIX_RE",
    "STAGE_FOLDER_RE",
    "STAGE_ORDER",
    "RU_MONTH",
    "SEASON_ORDER",
    "STAGES",
]
