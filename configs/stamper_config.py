"""Formatting helpers for the photo stamper."""
from __future__ import annotations

from configs.config import FOLDER_PATH
from configs.env_utils import get_env_list, get_env_str

STAMP_METADATA_PATH = FOLDER_PATH / ".stamps_metadata.json"
MONTHS = get_env_list(
    "STAMP_MONTHS",
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
)
YEAR_SUFFIX = get_env_str("STAMP_YEAR_SUFFIX", "")

__all__ = ["MONTHS", "YEAR_SUFFIX"]
