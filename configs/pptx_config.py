"""Presentation rendering configuration and work-type mappings."""
from __future__ import annotations

from typing import Dict

from pptx.util import Cm, Inches

from configs.env_utils import get_env_json, get_env_str
from configs.utils_config import RU_MONTH
from configs.work_profiles import WORK_PATTERNS, WORK_TITLES

MONTH_NAMES = [""] + [RU_MONTH.get(i, f"Month {i}") for i in range(1, 13)]

ADDRESS_PREFIX = get_env_str("TITLE_ADDRESS_PREFIX", "Installation address:")
REPORT_FALLBACK_NAME = get_env_str("REPORT_FALLBACK_NAME", "Report")
CONSTRUCTION_SINGLE_TEMPLATE = get_env_str("CONSTRUCTION_SINGLE_TEMPLATE", "Construction #{num}")
CONSTRUCTION_MULTI_PREFIX = get_env_str("CONSTRUCTION_MULTI_PREFIX", "Constructions ")
CONSTRUCTION_NUMBER_TEMPLATE = get_env_str("CONSTRUCTION_NUMBER_TEMPLATE", "#{num}")
DATE_RANGE_SEPARATOR = get_env_str("DATE_RANGE_SEPARATOR", " - ", multiline=False)

DEFAULT_STAGE_FALLBACK_TITLES = {
    "detected": "Issue detected",
    "in_progress": "Work in progress",
    "fixed": "Resolved",
}
WORK_STAGE_FALLBACK_TITLES: Dict[str, str] = {
    key: str(value)
    for key, value in get_env_json("WORK_STAGE_FALLBACK_TITLES", DEFAULT_STAGE_FALLBACK_TITLES).items()
}

PHOTO_HEIGHT = Inches(13.06 * 1 / 2.54)
LABEL_HEIGHT = Inches(2.00 * 1 / 2.54)
LABEL_VERTICAL_SHIFT = Cm(0.2)
LABEL_PHOTO_GAP = Cm(0.3)
PHOTO_TOP = Cm(4.0)
LABEL_FONT_PATH = r"C:\Windows\Fonts\times.ttf"
LABEL_FONT_FAMILY = "Times New Roman"
LABEL_FONT_SIZE = 24
SIDE_MARGIN = Cm(1.0)
GAP = Cm(0.9)
LABEL_BOX = (9.8, 2)
CONTENT_WIDTH_RATIO = 0.92
MAX_GAP_MULTIPLIER = 2.6
WORK_SLIDE_SIDE_MARGIN = Cm(1.0)
WORK_SLIDE_TOP_MARGIN = Cm(3.8)
WORK_SLIDE_BOTTOM_MARGIN = Cm(1.0)
WORK_SLIDE_MIN_GAP = Cm(0.6)

__all__ = [
    "WORK_PATTERNS",
    "WORK_TITLES",
    "PHOTO_HEIGHT",
    "LABEL_HEIGHT",
    "LABEL_VERTICAL_SHIFT",
    "LABEL_PHOTO_GAP",
    "PHOTO_TOP",
    "LABEL_FONT_PATH",
    "LABEL_FONT_FAMILY",
    "LABEL_FONT_SIZE",
    "SIDE_MARGIN",
    "GAP",
    "LABEL_BOX",
    "CONTENT_WIDTH_RATIO",
    "MAX_GAP_MULTIPLIER",
    "WORK_SLIDE_SIDE_MARGIN",
    "WORK_SLIDE_TOP_MARGIN",
    "WORK_SLIDE_BOTTOM_MARGIN",
    "WORK_SLIDE_MIN_GAP",
]
