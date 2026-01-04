"""Core configuration values for the photo reports automation."""
from __future__ import annotations

import collections.abc
import logging
import shutil
from datetime import time, timedelta
from pathlib import Path
from typing import Dict, List

collections.Sequence = collections.abc.Sequence

from configs.env_utils import get_env_json, get_env_path, get_env_str, load_env_file
from configs.work_profiles import WORK_TYPE_ORDER

load_env_file()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

TITLE_CONTENT = get_env_str(
    "TITLE_CONTENT",
    (
        "{activity}\n"
        "Contractor: Example Contractor\n"
        "Client: Example Client\n"
        "Agreement #111-11-11 dated 01.01.2025\n"
        "Structure type F6\n"
        "Construction count: 2 pcs.\n\n"
        "Installation address: Sample District, Example Street 137A, pedestrian area \n"
        "Reporting period: {period}\n"
    ),
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_FOLDER_PATH = PROJECT_ROOT / "examples" / "reference" / "ExampleLocation"
DEFAULT_FOLDER_PATH = PROJECT_ROOT / "examples" / "workspace" / "ExampleLocation"

configured_folder = Path(
    get_env_path("FOLDER_PATH", str(DEFAULT_FOLDER_PATH))
)

if not configured_folder.exists() and REFERENCE_FOLDER_PATH.exists():
    logger.info(
        "Workspace folder %s missing; seeding from reference dataset.",
        configured_folder,
    )
    configured_folder.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(REFERENCE_FOLDER_PATH, configured_folder, dirs_exist_ok=True)

FOLDER_PATH = configured_folder.resolve()

WORKS_PER_MONTH: Dict[int, int] = {12: 0, 1: 0, 2: 2, 3: 1}
NUM_CONSTRUCTIONS = 4
WIPE_DATES: List[str] = [
    "08.12.2024",
    "16.12.2024",
    "24.12.2024",
    "31.12.2024",
    "08.01.2025",
    "16.01.2025",
    "24.01.2025",
    "31.01.2025",
    "07.02.2025",
    "14.02.2025",
    "21.02.2025",
    "28.02.2025",
    "07.03.2025",
]
SNOW_DATES: List[str] = [
    "07.12.2024",
    "09.12.2024",
    "15.12.2024",
    "19.12.2024",
    "25.12.2024",
    "01.01.2025",
    "04.12.2024",
    "15.02.2025",
    "16.02.2025",
    "18.02.2025",
    "20.02.2025",
]

REPORT_MODE = get_env_str("REPORT_MODE", "daily")
ENABLED_WORK_TYPE = get_env_str("ENABLED_WORK_TYPE", WORK_TYPE_ORDER[-1] if WORK_TYPE_ORDER else "SnowClear")
BATCH_WORK_TYPES: List[str] = get_env_json("BATCH_WORK_TYPES", WORK_TYPE_ORDER)

RAW_IMAGE_MODE = get_env_str("RAW_IMAGE_MODE", "false").strip().lower() in {"1", "true", "yes"}
PHOTO_LABELS_ENABLED = get_env_str("PHOTO_LABELS_ENABLED", "true").strip().lower() not in {"0", "false", "no"}
HIDE_HEADER_DATES = get_env_str("HIDE_HEADER_DATES", "false").strip().lower() in {"1", "true", "yes"}
REVERSE_LEAF_PHOTO_ORDER = get_env_str("REVERSE_LEAF_PHOTO_ORDER", "false").strip().lower() in {"1", "true", "yes"}

LOCATION_GROUPS_DEFAULT = {
    "default": [
        "Central Avenue 1\nExample City",
        "Central Avenue 2\nExample City",
        "Central Avenue 3\nExample City",
        "Central Avenue 4\nExample City",
        "Central Avenue 5\nExample City",
    ]
}

LOCATION_GROUPS: Dict[str, List[str]] = get_env_json("LOCATION_GROUPS", LOCATION_GROUPS_DEFAULT)
if not LOCATION_GROUPS:
    LOCATION_GROUPS = LOCATION_GROUPS_DEFAULT

ACTIVE_LOCATION_GROUP = get_env_str("ACTIVE_LOCATION_GROUP", next(iter(LOCATION_GROUPS)))
LOCATIONS = LOCATION_GROUPS.get(ACTIVE_LOCATION_GROUP) or next(iter(LOCATION_GROUPS.values()))

START_TIME = time(17, 40)
MIN_GAP_SEC = 20
MAX_GAP_SEC = 200
DURATION_FOR_DAYS = timedelta(hours=2)
DURATION_BEFORE_WORKS = timedelta(hours=2)
DURATION_FOR_WORKS = timedelta(hours=3)

START_DATE = get_env_str("START_DATE", "01.12.2024")
END_DATE = get_env_str("END_DATE", "10.03.2025")

GENERATE_ALL_REPORTS = get_env_str("GENERATE_ALL_REPORTS", "true").lower() in {"1", "true", "yes"}

__all__ = [
    "TITLE_CONTENT",
    "FOLDER_PATH",
    "WORKS_PER_MONTH",
    "NUM_CONSTRUCTIONS",
    "WIPE_DATES",
    "SNOW_DATES",
    "REPORT_MODE",
    "ENABLED_WORK_TYPE",
    "GENERATE_ALL_REPORTS",
    "BATCH_WORK_TYPES",
    "LOCATIONS",
    "START_TIME",
    "MIN_GAP_SEC",
    "MAX_GAP_SEC",
    "DURATION_FOR_DAYS",
    "DURATION_BEFORE_WORKS",
    "DURATION_FOR_WORKS",
    "START_DATE",
    "END_DATE",
    "logger",
    "RAW_IMAGE_MODE",
    "PHOTO_LABELS_ENABLED",
    "HIDE_HEADER_DATES",
    "REVERSE_LEAF_PHOTO_ORDER",
]
