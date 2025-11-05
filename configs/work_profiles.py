"""Definitions for work types and their presentation metadata."""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

from configs.env_utils import get_env_json, load_env_file

load_env_file()

DEFAULT_WORK_PROFILES: Dict[str, Dict[str, object]] = {
    "PR": {
        "label": "PR",
        "regex": r"(?<![A-Za-z])pr(?![A-Za-z])|partical(?:[-_\s]?|)repair|particalrepair",
        "title": "Partical repair required",
        "activity_name": "Partical Repair",
        "activity_short": "PR",
        "stage_titles": [
            "1 Issue detected. Partical repair required",
            "2 Partial repair completed",
            "3 Issue resolved",
        ],
    },
    "DI": {
        "label": "DI",
        "regex": r"\bdi\b|deep\s+inspection",
        "title": "Deep inspection required",
        "activity_name": "Deep Inspection",
        "activity_short": "DI",
        "stage_titles": [
            "1 Deep inspection required",
            "2 Electrical components replaced",
            "3 Issue resolved",
        ],
    },
    "Flags": {
        "label": "Flags",
        "regex": r"\bflags?\b",
        "title": "Flag structures require alignment",
        "activity_name": "Flag Alignment",
        "activity_short": "Flags",
        "stage_titles": [
            "1 Flag alignment required",
            "2 Flag structures adjusted",
            "3 Issue resolved",
        ],
    },
    "Polish": {
        "label": "Polish",
        "regex": r"\bpolish\b|\bwipe\b",
        "title": "Surface polishing required",
        "activity_name": "Surface Polishing",
        "activity_short": "Polish",
        "stage_titles": [
            "1 Surface polishing required",
            "2 Surface polished",
            "3 Issue resolved",
        ],
    },
    "SnowClear": {
        "label": "SnowClear",
        "regex": r"\bsnow\b|\bclear\b|\bremoval\b|\bclearance\b|\bcleaning\b",
        "title": "Snow removal required",
        "activity_name": "Snow Removal",
        "activity_short": "Snow",
        "stage_titles": [
            "1 Snow removal required",
            "2 Snow cleared",
            "3 Issue resolved",
        ],
    },
}

RAW_WORK_PROFILES: Dict[str, Dict[str, object]] = get_env_json("WORK_PROFILES", DEFAULT_WORK_PROFILES)

WORK_PROFILES: Dict[str, Dict[str, object]] = {}
WORK_ALIAS_TO_CODE: Dict[str, str] = {}


def _normalized_token(value: str | None) -> str:
    if value is None:
        return ""
    return (
        str(value)
        .strip()
        .lower()
        .replace("\u00a0", "")
        .replace(" ", "")
    )


for raw_code, raw_profile in RAW_WORK_PROFILES.items():
    profile = dict(raw_profile)
    label = str(profile.get("label") or profile.get("code") or raw_code).strip()
    if not label:
        label = raw_code
    profile["label"] = label
    WORK_PROFILES[label] = profile

    aliases = {
        raw_code,
        label,
        profile.get("activity_name"),
        profile.get("activity_short"),
    }
    for alias in aliases:
        key = _normalized_token(alias)
        if not key:
            continue
        WORK_ALIAS_TO_CODE.setdefault(key, label)

WORK_PATTERNS: List[Tuple[str, re.Pattern[str]]] = []
for code, profile in WORK_PROFILES.items():
    regex = str(profile.get("regex", "")).strip()
    if not regex:
        continue
    WORK_PATTERNS.append((code, re.compile(regex, re.I)))

WORK_TITLES: Dict[str, str] = {
    code: str(profile.get("title", code)) for code, profile in WORK_PROFILES.items()
}

WORK_ACTIVITY_MAP: Dict[str, Tuple[str, str]] = {
    code: (
        str(profile.get("activity_name", code)),
        str(profile.get("activity_short", code)),
    )
    for code, profile in WORK_PROFILES.items()
}

WORK_STAGE_TITLES: Dict[str, List[str]] = {
    code: [str(item) for item in profile.get("stage_titles", [])]
    for code, profile in WORK_PROFILES.items()
}

WORK_TYPE_ORDER: List[str] = list(WORK_PROFILES.keys())


def normalize_work_type(name: str | None) -> str | None:
    if name is None:
        return None
    normalized = WORK_ALIAS_TO_CODE.get(_normalized_token(name))
    if normalized:
        return normalized
    cleaned = str(name).strip()
    return cleaned or None

__all__ = [
    "WORK_PROFILES",
    "WORK_PATTERNS",
    "WORK_TITLES",
    "WORK_ACTIVITY_MAP",
    "WORK_STAGE_TITLES",
    "WORK_TYPE_ORDER",
    "WORK_ALIAS_TO_CODE",
    "normalize_work_type",
]

