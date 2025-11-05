import hashlib
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from tqdm import tqdm

from configs.config import logger, FOLDER_PATH, RAW_IMAGE_MODE
from configs.filters_config import PHONE_PROFILES, LOOK_STRENGTH

EXT_RANK: dict[str, int] = {
    ".jpg": 5,
    ".jpeg": 4,
    ".png": 3,
    ".webp": 2,
    ".bmp": 1,
    ".tif": 1,
    ".tiff": 1,
}

def _log_bar(msg: str, *args) -> None:
    line = msg % args if args else msg
    writer = getattr(tqdm, "write", None) if "tqdm" in globals() else None
    if callable(writer):
        writer(line)
    else:
        logger.info(line)


def _progress(iterable, total, desc: str):
    if tqdm is not None:
        return tqdm(
            iterable, total=total, desc=desc,
            unit="img", leave=False, dynamic_ncols=True,
            mininterval=0.3, file=sys.stdout
        )
    return iterable


def _get_filter_seed(date_str: str) -> int:
    d = hashlib.md5(date_str.encode("utf-8")).digest()
    return int.from_bytes(d[:4], "big", signed=False)


def _stage_norm(value: Any) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip().lower()
    if s in {"", "none", "nan"}:
        return ""
    return s


def _is_allowed_stage(stage_value: Any) -> bool:
    return _stage_norm(stage_value) in {"", "detected"}


def _adjust_white_balance(img: Image.Image, red_factor: float, blue_factor: float) -> Image.Image:
    mode = img.mode
    if mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
        mode = "RGB"

    if mode == "RGBA":
        r, g, b, a = img.split()
    else:
        r, g, b = img.split()
        a = None

    red_factor = max(0.0, float(red_factor))
    blue_factor = max(0.0, float(blue_factor))
    lut_r = [min(255, int(round(i * red_factor))) for i in range(256)]
    lut_b = [min(255, int(round(i * blue_factor))) for i in range(256)]

    r = r.point(lut_r)
    b = b.point(lut_b)

    if a is not None:
        return Image.merge("RGBA", (r, g, b, a))
    return Image.merge("RGB", (r, g, b))


def _adjust_hue(img: Image.Image, hue_factor: float) -> Image.Image:
    alpha = img.getchannel("A") if img.mode == "RGBA" else None
    base = img.convert("RGB") if img.mode not in ("RGB", "RGBA") else img.convert("RGB")
    h, s, v = base.convert("HSV").split()
    offset = int(round(hue_factor * 255)) % 256
    lut = [(i + offset) & 0xFF for i in range(256)]
    h2 = h.point(lut)

    out = Image.merge("HSV", (h2, s, v)).convert("RGB")

    if alpha is not None:
        out = out.convert("RGBA")
        out.putalpha(alpha)
    return out


def _apply_filters(img: Image.Image, seed: int) -> Image.Image:
    rng = random.Random(seed)

    profile = PHONE_PROFILES[seed % len(PHONE_PROFILES)]
    _, r_fac, b_fac, hue, br, ct, col, sh, blur = profile

    def j(x: float, rel: float) -> float:
        return x * (1.0 + rng.uniform(-rel, rel))

    r_fac = j(r_fac, 0.01)
    b_fac = j(b_fac, 0.01)
    hue = hue + rng.uniform(-0.004, 0.004)
    br = j(br, 0.02)
    ct = j(ct, 0.03)
    col = j(col, 0.03)
    sh = j(sh, 0.05)
    blur = max(0.0, blur + rng.uniform(-0.05, 0.05))
    s = float(LOOK_STRENGTH)

    def amplify(v: float) -> float:
        return 1.0 + (v - 1.0) * s

    r_fac = amplify(r_fac)
    b_fac = amplify(b_fac)
    br = amplify(br)
    ct = amplify(ct)
    col = amplify(col)
    sh = amplify(sh)
    hue = hue * s
    blur = max(0.0, blur * (0.5 + 0.5 * s))

    alpha = img.getchannel("A") if img.mode == "RGBA" else None
    base = img.convert("RGB")

    base = _adjust_white_balance(base, r_fac, b_fac)
    base = _adjust_hue(base, hue)
    base = ImageEnhance.Brightness(base).enhance(br)
    base = ImageEnhance.Contrast(base).enhance(ct)
    base = ImageEnhance.Color(base).enhance(col)
    base = ImageEnhance.Sharpness(base).enhance(sh)
    if blur > 0:
        base = base.filter(ImageFilter.GaussianBlur(radius=blur))

    if alpha is not None:
        base = base.convert("RGBA")
        base.putalpha(alpha)
    return base


def filter_images(records: List[Dict[str, str]]) -> None:
    t0 = time.time()

    if RAW_IMAGE_MODE:
        logger.info("RAW_IMAGE_MODE enabled; skipping image filtering.")
        return

    excel_path = FOLDER_PATH / "parsed_records.xlsx"
    excel_exists = excel_path.exists()
    keys: set[tuple[Path, str, str, str]] = set()  # (folder, base_name, ext, base_date)

    if excel_exists:
        df = pd.read_excel(excel_path, dtype=str)
        cols = ["path", "base_name", "base_date"]
        if "stage" in df.columns:
            cols.append("stage")
        df2 = (
            df.loc[:, cols]
              .dropna(subset=["path", "base_name", "base_date"])
              .drop_duplicates(subset=["path"])
        )
        if "stage" in df2.columns:
            df2 = df2[df2["stage"].apply(_is_allowed_stage)]
        for _, row in df2.iterrows():
            p = Path(row["path"])
            keys.add((p.parent, str(row["base_name"]), p.suffix.lower(), str(row["base_date"])))
    else:
        for rec in records:
            path = rec.get("path")
            base_name = rec.get("base_name")
            base_date = rec.get("base_date")
            stage = rec.get("stage")
            if not path or not base_name or not base_date:
                continue
            if not _is_allowed_stage(stage):
                continue
            p = Path(path)
            keys.add((p.parent, str(base_name), p.suffix.lower(), str(base_date)))

    total = len(keys)
    logger.info("Planned: %d to process (Excel-driven%s).",
                total, "" if excel_exists else " fallback")

    processed = 0
    errors = 0
    logged_dates: set[str] = set()

    date_cache: Dict[str, datetime] = {}

    def _date_key(value: str) -> datetime:
        dt = date_cache.get(value)
        if dt is None:
            dt = datetime.strptime(value, "%d.%m.%Y")
            date_cache[value] = dt
        return dt

    sorted_keys = sorted(
        keys,
        key=lambda item: (_date_key(item[3]), str(item[0]), item[1], item[2]),
    )
    folder_cache: Dict[Path, List[Path]] = {}
    tasks: List[Tuple[Path, str, str, str]] = []
    folder_locks: Dict[Path, threading.Lock] = {}
    for folder, base, ext, base_date in sorted_keys:
        tasks.append((folder, base, ext, base_date))
        folder_locks.setdefault(folder, threading.Lock())

    max_workers_env = os.environ.get("FILTER_WORKERS")
    if max_workers_env and max_workers_env.isdigit():
        max_workers = max(1, int(max_workers_env))
    else:
        max_workers = max(1, min(6, (os.cpu_count() or 1)))

    progress_monitor = _progress(range(total), total=total, desc="Filtering images")

    def _process_entry(payload: Tuple[Path, str, str, str]) -> Tuple[int, int]:
        nonlocal folder_cache
        folder, base, ext, base_date = payload
        errors_local = 0
        processed_local = 0

        if not folder.exists():
            logger.debug("Skip missing folder for %s in %s", base, folder)
            return processed_local, errors_local

        try:
            lock = folder_locks[folder]
        except KeyError:
            lock = folder_locks.setdefault(folder, threading.Lock())

        with lock:
            files = folder_cache.get(folder)
            if files is None:
                try:
                    files = list(folder.iterdir())
                except FileNotFoundError:
                    logger.debug("Folder disappeared while processing: %s", folder)
                    return processed_local, errors_local
                folder_cache[folder] = files
            filtered_existing = [
                f for f in files
                if f.is_file()
                and f.stem.startswith(base)
                and "_filtered" in f.stem.lower()
            ]

            candidates = [
                f for f in files
                if f.is_file()
                and f.stem.startswith(base)
                and "_stamped" not in f.stem.lower()
                and "_filtered" not in f.stem.lower()
            ]

        if not candidates:
            logger.debug("No candidates for %s in %s", base, folder)
            return processed_local, errors_local

        def score(path: Path) -> tuple[int, int, int, str]:
            tail = len(path.stem) - len(base)
            er = EXT_RANK.get(path.suffix.lower(), 0)
            return tail, er, len(path.stem), str(path)

        src = max(candidates, key=score)

        seed = _get_filter_seed(base_date)
        profile_name = PHONE_PROFILES[seed % len(PHONE_PROFILES)][0]
        if base_date not in logged_dates and not filtered_existing:
            _log_bar("Date %s -> profile: %s", base_date, profile_name)
            logged_dates.add(base_date)

        dst = src.with_stem(f"{src.stem}_filtered")
        try:
            src_mtime = src.stat().st_mtime
        except FileNotFoundError:
            logger.debug("Source disappeared before processing: %s", src)
            return processed_local, errors_local

        if filtered_existing:
            has_up_to_date = False
            rewrite_target: Path | None = None
            rewrite_mtime = float("-inf")
            for existing in filtered_existing:
                try:
                    existing_mtime = existing.stat().st_mtime
                except FileNotFoundError:
                    continue
                if existing_mtime >= src_mtime:
                    has_up_to_date = True
                    break
                if existing_mtime > rewrite_mtime:
                    rewrite_target = existing
                    rewrite_mtime = existing_mtime
            if has_up_to_date:
                return processed_local, errors_local
            if rewrite_target is not None:
                dst = rewrite_target
        else:
            if dst.exists():
                try:
                    if dst.stat().st_mtime >= src_mtime:
                        return processed_local, errors_local
                except FileNotFoundError:
                    pass

        try:
            with Image.open(src) as img:
                prepared = ImageOps.exif_transpose(img)
                out = _apply_filters(prepared, seed)
                out.save(dst, quality=90)
            processed_local += 1
            logger.debug("Saved filtered image: %s", dst)
        except Exception as e:  # pragma: no cover - defensive
            errors_local += 1
            logger.error("Failed on %s: %s", src, e)
            return processed_local, errors_local

        with lock:
            files = folder_cache.get(folder)
            if files is not None and dst not in files:
                files.append(dst)
        return processed_local, errors_local

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_process_entry, payload) for payload in tasks]
            for future, _ in zip(as_completed(futures), progress_monitor):
                done, err = future.result()
                processed += done
                errors += err
    finally:
        closer = getattr(progress_monitor, "close", None)
        if callable(closer):
            closer()

    dt = time.time() - t0
    logger.info("Done: %d processed, %d errors in %.2fs.", processed, errors, dt)
