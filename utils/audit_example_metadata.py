"""Inventory privacy-sensitive metadata in example assets without printing values.

This is a triage check, not a substitute for reviewing image pixels or
verifying rights to publish the underlying photographs.
"""
from __future__ import annotations

import json
import re
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image, ExifTags

EXAMPLE_ROOT = Path("examples")
EMAIL = re.compile(rb"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", re.I)
PERSON_FIELDS = {"Artist", "Copyright", "XPAuthor", "CameraOwnerName", "OwnerName", "BodySerialNumber", "SerialNumber"}
PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
OFFICE_EXTS = {".pptx", ".xlsx", ".docx"}


def inventory(root: Path = EXAMPLE_ROOT) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in PHOTO_EXTS:
            counts["images"] += 1
            try:
                with Image.open(path) as img:
                    exif = img.getexif()
                    if not exif:
                        continue
                    named = {ExifTags.TAGS.get(key, str(key)): value for key, value in exif.items()}
                    if 34853 in exif or "GPSInfo" in named:
                        counts["images_with_gps"] += 1
                    if any(named.get(name) for name in PERSON_FIELDS):
                        counts["images_with_owner_or_serial_metadata"] += 1
            except (OSError, ValueError, TypeError):
                counts["unreadable_images"] += 1
        elif suffix in OFFICE_EXTS:
            counts["office_documents"] += 1
            try:
                with zipfile.ZipFile(path) as archive:
                    xml_parts = [
                        entry for entry in archive.namelist()
                        if entry.endswith(".xml") and not entry.startswith("_rels/")
                    ]
                    if any(EMAIL.search(archive.read(entry)) for entry in xml_parts):
                        counts["office_documents_with_email_like_text"] += 1
            except (OSError, zipfile.BadZipFile, RuntimeError):
                counts["unreadable_office_documents"] += 1
    return {key: counts[key] for key in (
        "images", "images_with_gps", "images_with_owner_or_serial_metadata",
        "unreadable_images", "office_documents",
        "office_documents_with_email_like_text", "unreadable_office_documents",
    )}


if __name__ == "__main__":
    print(json.dumps(inventory(), indent=2))
    print("Scope: EXIF ownership/GPS and email-shaped XML only; no face or license assessment.")
