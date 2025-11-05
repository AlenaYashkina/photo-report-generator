"""Stamping utilities that annotate images with metadata overlays."""

import io
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import cairo
import pandas as pd
from PIL import UnidentifiedImageError, Image, ImageOps

from configs.config import logger, FOLDER_PATH, RAW_IMAGE_MODE
from configs.stamper_config import MONTHS, STAMP_METADATA_PATH, YEAR_SUFFIX
from photo_reports.models import StampMetadata, build_stamp_key
from utils.utils import get_base_name


def stamp_image(
    image_path: str,
    date: str,
    time: str,
    location: str
) -> str | None:
    try:
        dt = datetime.strptime(date, "%d.%m.%Y")
        date_str = f"{dt.day:02d} {MONTHS[dt.month - 1]}. {dt.year}{YEAR_SUFFIX}"
    except ValueError:
        date_str = date

    try:
        with Image.open(image_path) as raw_img:
            pil_img = ImageOps.exif_transpose(raw_img).convert("RGBA")

            width, height = pil_img.size
            padding_x = width * 0.004
            padding_y = height * 0.004
            font_size = int(height * 0.031)

            with io.BytesIO() as buf:
                pil_img.save(buf, format="PNG")
                buf.seek(0)
                src_surface = cairo.ImageSurface.create_from_png(buf)

            surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
            ctx = cairo.Context(surface)
            ctx.set_source_surface(src_surface, 0, 0)
            ctx.paint()

            ctx.select_font_face("Noto Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
            ctx.set_source_rgb(1, 1, 1)

            max_width = width - 2 * padding_x
            line_h = int(font_size * 1.3)

            def fits(text: str, size: int) -> bool:
                ctx.set_font_size(size)
                return ctx.text_extents(text).width <= max_width

            hdr = f"{date_str} {time}".strip()
            hdr_size = font_size
            while hdr_size > int(font_size * 0.7) and not fits(hdr, hdr_size):
                hdr_size -= 1
            ctx.set_font_size(hdr_size)
            hdr_ext = ctx.text_extents(hdr)
            y = padding_y + hdr_size
            x = width - padding_x - hdr_ext.width
            ctx.move_to(x, y)
            ctx.show_text(hdr)
            y += line_h

            def wrap_line(text: str) -> list[str]:
                words = text.split()
                lines, cur = [], ""
                ctx.set_font_size(font_size)
                for w in words:
                    test = (cur + " " + w).strip()
                    if ctx.text_extents(test).width <= max_width:
                        cur = test
                    else:
                        if cur:
                            lines.append(cur)
                        cur = w
                if cur:
                    lines.append(cur)
                return lines

            for raw in location.split("\n"):
                for line in wrap_line(raw.strip()):
                    ctx.set_font_size(font_size)
                    ext = ctx.text_extents(line)
                    x = width - padding_x - ext.width
                    ctx.move_to(x, y)
                    ctx.show_text(line)
                    y += line_h

            src_path = Path(image_path)
            suffix = src_path.suffix.lower()
            if suffix in (".jpg", ".jpeg"):
                output_path = src_path.parent / f"{src_path.stem}_stamped.jpg"
                with io.BytesIO() as out_buf:
                    surface.write_to_png(out_buf)
                    out_buf.seek(0)
                    with Image.open(out_buf) as out_img:
                        out_img.convert("RGB").save(str(output_path), format="JPEG")
            else:
                output_path = src_path.parent / f"{src_path.stem}_stamped.png"
                surface.write_to_png(str(output_path))

            surface.flush()
            surface.finish()
            src_surface.finish()
            return str(output_path)

    except (UnidentifiedImageError, IOError, cairo.Error) as e:
        logger.error("Stamp error %s: %s", image_path, e)
        return None



def _load_stamp_metadata() -> Dict[str, StampMetadata]:
    """Load persisted stamp metadata if available."""
    if not STAMP_METADATA_PATH.exists():
        return {}
    try:
        with STAMP_METADATA_PATH.open("r", encoding="utf-8-sig") as fh:
            raw = fh.read().strip()
            if not raw:
                return {}
            data = json.loads(raw)
    except Exception as exc:
        logger.warning("Failed to read stamp metadata: %s", exc)
        return {}

    if not isinstance(data, dict):
        return {}

    meta: Dict[str, StampMetadata] = {}
    for key, payload in data.items():
        if not isinstance(payload, dict):
            continue
        try:
            meta[str(key)] = StampMetadata(
                date=str(payload.get("date", "")),
                time=str(payload.get("time", "")),
                location=str(payload.get("location", "")),
                source_path=str(payload.get("source_path", "")),
                stamped_path=str(payload.get("stamped_path", "")),
                stage=str(payload.get("stage", "")),
                ext=str(payload.get("ext", "")),
                updated_at=str(payload.get("updated_at", "")),
                locale_key=str(payload.get("locale_key", "")),
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Skip invalid metadata entry %s: %s", key, exc)
    return meta


def _save_stamp_metadata(meta: Dict[str, StampMetadata]) -> None:
    """Persist stamp metadata to disk."""
    try:
        STAMP_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        serialised = {key: value.to_dict() for key, value in meta.items()}
        with STAMP_METADATA_PATH.open("w", encoding="utf-8") as fh:
            json.dump(serialised, fh, ensure_ascii=False, indent=2)
        logger.debug("Persisted stamp metadata to %s (%d entries)", STAMP_METADATA_PATH, len(serialised))
    except Exception as exc:
        logger.error("Failed to write stamp metadata: %s", exc)


def stamp_all_images(_: List[Dict[str, Any]] | None = None) -> None:
    """Stamp images according to the current state of parsed_records.xlsx."""
    if RAW_IMAGE_MODE:
        logger.info("RAW_IMAGE_MODE enabled; skipping stamping.")
        return
    excel_path = FOLDER_PATH / "parsed_records.xlsx"
    if not excel_path.exists():
        logger.warning("Excel not found: %s", excel_path)
        return

    df = pd.read_excel(excel_path, dtype=str)
    metadata = _load_stamp_metadata()

    def _folder_norm(p: str) -> str:
        return str(Path(p).parent).replace("\\", "/").lower()

    def _base_norm(p: str) -> str:
        return get_base_name(p).lower()

    def _stage_norm(s: Any) -> str:
        if pd.isna(s):
            return "none"
        s = str(s).strip().lower()
        return s if s in {"detected", "in_progress", "fixed", "none"} else "none"

    df["path"] = df["path"].astype(str)
    df["date"] = df["date"].astype(str).str.strip()
    df["time"] = df["time"].astype(str).str.strip()
    df["location"] = df["location"].astype(str)

    df["folder_norm"] = df["path"].apply(_folder_norm)
    df["base_norm"] = df["path"].apply(_base_norm)
    df["stage_norm"] = df["stage"].apply(_stage_norm)

    # Deduplicate by (folder, base, date, stage, extension) keeping the latest Excel entry
    df["ext"] = df["path"].apply(lambda p: Path(p).suffix.lower())
    dfx = df.drop_duplicates(
        subset=["folder_norm", "base_norm", "date", "stage_norm", "ext"],
        keep="last"
    )

    seen: set[tuple[str, str, str, str, str]] = set()

    current_locale = "|".join(MONTHS) + "|" + (YEAR_SUFFIX or "")

    folder_cache: Dict[Path, List[Path]] = {}
    metadata_changed = False

    restamp_tasks: list[dict[str, Any]] = []
    fresh_tasks: list[dict[str, Any]] = []

    for _, row in dfx.iterrows():
        folder_norm = row["folder_norm"]
        base_norm = row["base_norm"]
        stage_val = row["stage_norm"]
        date_val = row["date"]
        time_val = row["time"]
        loc_val = row["location"]
        ext = row["ext"]

        # Skip records with empty timestamps
        if not time_val or time_val.lower() in {"nan", "none"}:
            logger.warning(
                "Skip: empty time for (%s, %s, %s, %s)",
                folder_norm,
                base_norm,
                date_val,
                stage_val,
            )
            continue

        seen_key = (folder_norm, base_norm, date_val, stage_val, ext)
        if seen_key in seen:
            continue
        seen.add(seen_key)

        meta_key = build_stamp_key(*seen_key)
        prev_meta = metadata.get(meta_key)
        force_stamp = False
        metadata_matches = False
        prev_stamp_path = None
        restamp_reasons: list[str] = []
        if prev_meta:
            mismatches: list[str] = []
            if prev_meta.date != date_val:
                mismatches.append(f"date {prev_meta.date} -> {date_val}")
            if prev_meta.time != time_val:
                mismatches.append(f"time {prev_meta.time} -> {time_val}")
            if prev_meta.location != loc_val:
                mismatches.append("location changed")
            if getattr(prev_meta, "locale_key", "") != current_locale:
                mismatches.append("locale changed")
            if mismatches:
                logger.warning(
                    "Stamped metadata mismatch for %s (%s): %s",
                    row["path"],
                    stage_val,
                    "; ".join(mismatches),
                )
                force_stamp = True
                restamp_reasons.extend(mismatches)
            else:
                metadata_matches = True
                prev_stamp_path = Path(prev_meta.stamped_path) if prev_meta.stamped_path else None

        src_path = Path(row["path"])
        base_name = get_base_name(row["path"])

        parent = src_path.parent
        if not parent.exists():
            logger.warning("Folder not found: %s", parent)
            continue

        files = folder_cache.get(parent)
        if files is None:
            try:
                files = list(parent.iterdir())
            except FileNotFoundError:
                logger.warning("Folder disappeared before stamping: %s", parent)
                continue
            folder_cache[parent] = files

        # Fast skip if a stamped version already exists for this base/ext
        # Find source candidate (exclude already stamped)
        candidates = [
            f for f in files
            if f.is_file()
               and f.suffix.lower() == ext
               and f.stem.startswith(base_name)
               and "_stamped" not in f.stem.lower()
        ]
        if not candidates:
            logger.warning("No candidates to stamp for %s in %s", base_name, parent)
            continue

        # Prefer the most specific filename
        src = max(candidates, key=lambda p: len(p.stem))

        try:
            src_mtime = src.stat().st_mtime
        except FileNotFoundError:
            logger.warning("Source disappeared before stamping: %s", src)
            continue

        stamped_targets: list[Path] = []
        if src.suffix.lower() in (".jpg", ".jpeg"):
            stamped_targets.append(src.with_name(f"{src.stem}_stamped.jpg"))
        else:
            stamped_targets.append(src.with_name(f"{src.stem}_stamped.png"))

        if metadata_matches:
            targets_exist = all(target.exists() for target in stamped_targets)
            prev_exists = prev_stamp_path.exists() if prev_stamp_path else False
            if targets_exist and (prev_stamp_path is None or prev_exists):
                continue

        if not prev_meta:
            existing_stamped = [target for target in stamped_targets if target.exists()]
            if existing_stamped:
                msg = "metadata missing for existing stamp"
                logger.warning(
                    "Stamped metadata mismatch for %s (%s): %s",
                    row["path"],
                    stage_val,
                    msg,
                )
                restamp_reasons.append(msg)
                force_stamp = True

        up_to_date = True
        for target in stamped_targets:
            if not target.exists():
                up_to_date = False
                break
            try:
                if target.stat().st_mtime < src_mtime:
                    up_to_date = False
                    break
            except FileNotFoundError:
                up_to_date = False
                break
        if up_to_date and not force_stamp:
            continue

        task_payload = {
            "src": src,
            "date": date_val,
            "time": time_val,
            "location": loc_val,
            "stage": stage_val,
            "ext": ext,
            "meta_key": meta_key,
            "base_name": base_name,
            "parent": parent,
            "targets": stamped_targets,
            "force": force_stamp,
        }
        if force_stamp or restamp_reasons:
            restamp_tasks.append(task_payload)
        else:
            fresh_tasks.append(task_payload)

    tasks = restamp_tasks + fresh_tasks

    if not tasks:
        if metadata_changed:
            _save_stamp_metadata(metadata)
        return

    max_workers_env = os.environ.get("STAMP_WORKERS")
    if max_workers_env and max_workers_env.isdigit():
        max_workers = max(1, int(max_workers_env))
    else:
        max_workers = max(1, min(6, (os.cpu_count() or 1)))

    def _stamp_task(payload: dict[str, Any]) -> tuple[dict[str, Any], str | None, Exception | None]:
        try:
            result = stamp_image(
                str(payload["src"]),
                payload["date"],
                payload["time"],
                payload["location"],
            )
            return payload, result, None
        except Exception as exc:  # pragma: no cover - defensive
            return payload, None, exc

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_stamp_task, task) for task in tasks]
        for future in as_completed(futures):
            payload, out_path, err = future.result()
            if err is not None:
                logger.error("Stamp worker failed for %s: %s", payload["src"], err)
                continue
            if not out_path:
                continue
            logger.info("Saved stamped image: %s", out_path)
            metadata[payload["meta_key"]] = StampMetadata(
                date=payload["date"],
                time=payload["time"],
                location=payload["location"],
                source_path=str(payload["src"]),
                stamped_path=out_path,
                stage=payload["stage"],
                ext=payload["ext"],
                updated_at=datetime.now().isoformat(timespec="seconds"),
                locale_key=current_locale,
            )
            metadata_changed = True
            _save_stamp_metadata(metadata)
            parent = payload["parent"]
            refreshed = folder_cache.get(parent)
            if refreshed is not None:
                for target in payload["targets"]:
                    if target not in refreshed:
                        refreshed.append(target)
                folder_cache[parent] = refreshed

    if metadata_changed:
        _save_stamp_metadata(metadata)
