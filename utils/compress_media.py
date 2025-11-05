"""Utilities to downsize example media without noticeable quality loss."""
from __future__ import annotations

import io
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

from PIL import Image, ImageFile, ImageOps

ImageFile.LOAD_TRUNCATED_IMAGES = True

BASE_DIR = Path(__file__).resolve().parents[1] / "examples"
LONG_SIDE_MAX = 1024
IMAGE_JPEG_QUALITY = 68
PPTX_JPEG_QUALITY = 66
IMAGE_SIZE_THRESHOLD = 0.999
PPTX_MEDIA_THRESHOLD = 0.999
PPTX_FILE_THRESHOLD = 0.999


try:
    RESAMPLE = Image.Resampling.LANCZOS  # Pillow >= 10
except AttributeError:  # pragma: no cover - fallback for older Pillow
    RESAMPLE = Image.LANCZOS  # type: ignore[attr-defined]


@dataclass
class CompressionStats:
    processed: int = 0
    updated: int = 0
    bytes_before: int = 0
    bytes_after: int = 0

    def add(self, original: int, new: int, updated: bool) -> None:
        self.processed += 1
        self.bytes_before += original
        if updated:
            self.updated += 1
            self.bytes_after += new
        else:
            self.bytes_after += original

    @property
    def bytes_saved(self) -> int:
        return self.bytes_before - self.bytes_after


def _save_jpeg(image: Image.Image, quality: int) -> bytes:
    buf = io.BytesIO()
    params = {
        "format": "JPEG",
        "quality": quality,
        "optimize": True,
        "progressive": True,
    }
    icc_profile = image.info.get("icc_profile")
    if icc_profile:
        params["icc_profile"] = icc_profile  # Preserve embedded profile when present.
    image.save(buf, **params)
    return buf.getvalue()


def _prepare_image(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    width, height = image.size
    longest = max(width, height)
    if longest > LONG_SIDE_MAX:
        ratio = LONG_SIDE_MAX / float(longest)
        new_size = (
            max(1, int(round(width * ratio))),
            max(1, int(round(height * ratio))),
        )
        image = image.resize(new_size, RESAMPLE)
    else:
        image.load()
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    return image


def compress_jpeg_file(path: Path, threshold: float, quality: int) -> Tuple[bool, int, int]:
    original_bytes = path.read_bytes()
    try:
        with Image.open(io.BytesIO(original_bytes)) as img:
            original_size = img.size
            prepared = _prepare_image(img)
            new_bytes = _save_jpeg(prepared, quality)
    except OSError:
        return False, len(original_bytes), len(original_bytes)

    should_replace = len(new_bytes) < len(original_bytes) * threshold or prepared.size != original_size
    if not should_replace:
        return False, len(original_bytes), len(original_bytes)

    path.write_bytes(new_bytes)
    return True, len(original_bytes), len(new_bytes)


def compress_jpeg_bytes(data: bytes, threshold: float, quality: int) -> Tuple[bool, bytes]:
    try:
        with Image.open(io.BytesIO(data)) as img:
            original_size = img.size
            prepared = _prepare_image(img)
            new_bytes = _save_jpeg(prepared, quality)
    except OSError:
        return False, data

    should_replace = len(new_bytes) < len(data) * threshold or prepared.size != original_size
    if not should_replace:
        return False, data
    return True, new_bytes


def iter_image_files(base_dir: Path) -> Iterable[Path]:
    for path in base_dir.rglob("*.jpg"):
        if path.is_file():
            yield path


def compress_images(base_dir: Path = BASE_DIR) -> CompressionStats:
    stats = CompressionStats()
    for path in iter_image_files(base_dir):
        updated, original, new = compress_jpeg_file(path, IMAGE_SIZE_THRESHOLD, IMAGE_JPEG_QUALITY)
        stats.add(original, new if updated else original, updated)
    return stats


def compress_media_entries(zip_path: Path) -> Dict[str, bytes]:
    replacements: Dict[str, bytes] = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            name = info.filename
            suffix = Path(name).suffix.lower()
            if not name.lower().startswith("ppt/media/"):
                continue
            if suffix not in {".jpg", ".jpeg"}:
                continue
            data = zf.read(name)
            updated, new_bytes = compress_jpeg_bytes(data, PPTX_MEDIA_THRESHOLD, PPTX_JPEG_QUALITY)
            if updated:
                replacements[name] = new_bytes
    return replacements


def compress_pptx_file(path: Path) -> Tuple[bool, int, int]:
    original_size = path.stat().st_size
    replacements = compress_media_entries(path)
    if not replacements:
        return False, original_size, original_size

    with zipfile.ZipFile(path, "r") as src:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "compressed.pptx"
            with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as dst:
                for info in src.infolist():
                    data = replacements.get(info.filename)
                    if data is None:
                        data = src.read(info.filename)
                    dst.writestr(info, data)
            new_size = tmp_path.stat().st_size
            if new_size >= original_size * PPTX_FILE_THRESHOLD:
                return False, original_size, original_size
            shutil.move(str(tmp_path), path)
    return True, original_size, new_size


def compress_presentations(base_dir: Path = BASE_DIR) -> CompressionStats:
    stats = CompressionStats()
    for path in base_dir.rglob("*.pptx"):
        if not path.is_file():
            continue
        updated, original, new = compress_pptx_file(path)
        stats.add(original, new if updated else original, updated)
    return stats


def main() -> None:
    image_stats = compress_images()
    pptx_stats = compress_presentations()

    def summary(label: str, stats: CompressionStats) -> str:
        saved_mb = stats.bytes_saved / (1024 * 1024)
        before_mb = stats.bytes_before / (1024 * 1024) if stats.bytes_before else 0
        after_mb = stats.bytes_after / (1024 * 1024) if stats.bytes_after else 0
        return (
            f"{label}: processed={stats.processed}, updated={stats.updated}, "
            f"before={before_mb:.2f} MB, after={after_mb:.2f} MB, saved={saved_mb:.2f} MB"
        )

    print(summary("Images", image_stats))
    print(summary("Presentations", pptx_stats))


if __name__ == "__main__":
    main()
