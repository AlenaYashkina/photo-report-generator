from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List


@dataclass(slots=True)
class ImageRecord:
    """In-memory representation of a parsed image record."""

    date: str
    construction_number: List[int]
    path: str
    stage: str | None = None
    work_type: str = ""
    session: str = "single"
    time: str = ""
    location: str = ""
    base_date: str = field(init=False)

    def __post_init__(self) -> None:
        self.base_date = self.date

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable representation of the record."""
        return asdict(self)


@dataclass(slots=True)
class StampMetadata:
    """Information captured when stamping an image."""

    date: str
    time: str
    location: str
    source_path: str
    stamped_path: str
    stage: str
    ext: str
    updated_at: str
    locale_key: str = ""

    def to_dict(self) -> Dict[str, str]:
        """Convert metadata to a regular dict for JSON serialisation."""
        return {
            "date": self.date,
            "time": self.time,
            "location": self.location,
            "source_path": self.source_path,
            "stamped_path": self.stamped_path,
            "stage": self.stage,
            "ext": self.ext,
            "updated_at": self.updated_at,
            "locale_key": self.locale_key,
        }


def build_stamp_key(folder_norm: str, base_norm: str, date: str, stage: str, ext: str) -> str:
    """Generate a consistent lookup key for stamp metadata storage."""
    return "|".join((folder_norm, base_norm, date, stage, ext))


__all__ = ["ImageRecord", "StampMetadata", "build_stamp_key"]
