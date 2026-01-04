"""Utilities for parsing raw photo reports into structured records."""

import ast
import hashlib
import random
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Sequence

import pandas as pd

from configs.config import (
    START_TIME, DURATION_FOR_DAYS, LOCATIONS,
    DURATION_BEFORE_WORKS, DURATION_FOR_WORKS, FOLDER_PATH,
    REVERSE_LEAF_PHOTO_ORDER,
)
from configs.pptx_config import WORK_PATTERNS
from configs.work_profiles import WORK_PROFILES, WORK_STAGE_TITLES, normalize_work_type
from configs.utils_config import BEFORE_JITTER, STAGE_ORDER, SEASON_ORDER, WORKS_JITTER
from photo_reports.models import ImageRecord
from utils.utils import (
    stage_dirs, random_gap, find_images, detect_stage,
    extract_construction_numbers, get_base_name, parse_date,
)

def _excel_path() -> Path:
    """Return the canonical path to the cached Excel file."""
    return FOLDER_PATH / "parsed_records.xlsx"


def _cn_sort_value(value: Any) -> int:
    seq: List[Any] = []
    if isinstance(value, (list, tuple)):
        seq = list(value)
    elif isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except Exception:
            parsed = []
        if isinstance(parsed, (list, tuple)):
            seq = list(parsed)
        elif parsed not in (None, ""):
            seq = [parsed]
    if not seq:
        return -1
    nums: List[int] = []
    for item in seq:
        try:
            nums.append(int(item))
        except (TypeError, ValueError):
            continue
    return min(nums) if nums else -1


def _stage_sort_value(stage: Any) -> int:
    if stage is None:
        return STAGE_ORDER.get(None, 0)
    stage_str = str(stage).strip().lower()
    if stage_str in ("", "none", "nan"):
        return STAGE_ORDER.get(None, 0)
    return STAGE_ORDER.get(stage_str, 99)


_MONTH_ORDER = {month: idx for idx, month in enumerate(SEASON_ORDER)}

# Ensure deterministic ordering: Day sessions first, then Night, then sessions
# with no explicit suffix ("single").
SESSION_ORDER = ("day", "night", "single")
SESSION_OFFSETS = {
    "day": timedelta(hours=-12),
    "single": timedelta(hours=0),
    "night": timedelta(hours=0),
}

SESSION_ORDER_MAP = {name: idx for idx, name in enumerate(SESSION_ORDER)}


def _detect_session(parts: Sequence[str]) -> str:
    """Derive shooting session (day/night) from folder names."""
    for part in parts:
        lower = part.lower()
        if lower.endswith(" night"):
            return "night"
        if lower.endswith(" day"):
            return "day"
    return "single"


def _month_sort_value(date_str: Any) -> int:
    if not date_str or not isinstance(date_str, str):
        return len(_MONTH_ORDER)
    try:
        month = int(date_str.split(".")[1])
    except (ValueError, IndexError):
        return len(_MONTH_ORDER)
    return _MONTH_ORDER.get(month, len(_MONTH_ORDER))


def _sort_records_df(df: pd.DataFrame) -> pd.DataFrame:
    df_sorted = df.copy()
    required_columns = ["base_date", "construction_number", "stage", "date", "time", "path", "work_type", "session"]
    for column in required_columns:
        if column not in df_sorted.columns:
            df_sorted[column] = pd.Series(dtype="object")

    df_sorted["__month_order"] = df_sorted["base_date"].apply(_month_sort_value)
    df_sorted["__base_dt"] = pd.to_datetime(df_sorted["base_date"], format="%d.%m.%Y", errors="coerce")
    df_sorted["__cn_key"] = df_sorted["construction_number"].apply(_cn_sort_value)
    df_sorted["__stage_key"] = df_sorted["stage"].apply(_stage_sort_value)
    date_series = df_sorted["date"].fillna("")
    time_series = df_sorted["time"].fillna("")
    df_sorted["__dt_key"] = pd.to_datetime(
        date_series + " " + time_series,
        format="%d.%m.%Y %H:%M:%S",
        errors="coerce",
    )
    session_series = df_sorted["session"].fillna("single")
    df_sorted["__session_key"] = session_series.apply(
        lambda value: SESSION_ORDER_MAP.get(str(value).strip().lower(), len(SESSION_ORDER))
    )
    df_sorted.sort_values(
        ["__month_order", "__base_dt", "__session_key", "__cn_key", "__stage_key", "__dt_key", "path"],
        inplace=True,
        kind="mergesort",
    )
    df_sorted.drop(columns=["__month_order", "__base_dt", "__cn_key", "__stage_key", "__dt_key", "__session_key"], inplace=True)
    df_sorted.reset_index(drop=True, inplace=True)
    return df_sorted


def _stable_location_choice(base: str, date: str, stage: str) -> str:
    """Pick a deterministic location for a record using hashed metadata."""
    if not LOCATIONS:
        return ""
    key = f"{base}|{date}|{stage}".encode("utf-8", errors="ignore")
    digest = hashlib.blake2s(key, digest_size=8).digest()
    idx = int.from_bytes(digest, "big") % len(LOCATIONS)
    return LOCATIONS[idx]


def _normalise_locations(df: pd.DataFrame) -> None:
    """Ensure cached records use the current location pool from settings."""
    if "location" not in df.columns or not LOCATIONS:
        return

    def needs_update(value: object) -> bool:
        if not isinstance(value, str) or not value.strip():
            return True
        return value not in LOCATIONS

    mask = df["location"].apply(needs_update)
    if not mask.any():
        return

    def resolve(row: pd.Series) -> str:
        base = str(row.get("base_name") or get_base_name(str(row.get("path", ""))))
        date = str(row.get("date") or "")
        stage = str(row.get("stage") or "")
        choice = _stable_location_choice(base, date, stage)
        return choice or str(row.get("location") or "")

    df.loc[mask, "location"] = df.loc[mask].apply(resolve, axis=1)


def _split_by_session(
    bucket: Dict[int, Dict[str, List[ImageRecord]]],
) -> Dict[str, Dict[int, Dict[str, List[ImageRecord]]]]:
    sessions: Dict[str, Dict[int, Dict[str, List[ImageRecord]]]] = {
        key: defaultdict(lambda: defaultdict(list)) for key in SESSION_ORDER
    }
    for cn, stage_map in bucket.items():
        for stage_name, records in stage_map.items():
            for rec in records:
                session = getattr(rec, "session", "single") or "single"
                if session not in sessions:
                    session = "single"
                sessions[session][cn][stage_name].append(rec)
    return sessions


def _session_has_records(session_bucket: Dict[int, Dict[str, List[ImageRecord]]]) -> bool:
    for cn_map in session_bucket.values():
        for records in cn_map.values():
            if records:
                return True
    return False


def _order_records_in_bucket(records: Sequence[ImageRecord]) -> List[ImageRecord]:
    """Order photos by path, optionally reversing within each leaf folder."""
    if not records:
        return []
    if not REVERSE_LEAF_PHOTO_ORDER:
        return sorted(records, key=lambda rec: rec.path)

    grouped: Dict[Path, List[ImageRecord]] = {}
    parent_order: List[Path] = []
    for rec in records:
        parent = Path(rec.path).parent
        if parent not in grouped:
            grouped[parent] = []
            parent_order.append(parent)
        grouped[parent].append(rec)

    ordered: List[ImageRecord] = []
    for parent in parent_order:
        group = grouped[parent]
        group.sort(key=lambda rec: Path(rec.path).name, reverse=True)
        ordered.extend(group)
    return ordered


def _process_session_bucket(
    bucket: Dict[int, Dict[str, List[ImageRecord]]],
    rng: random.Random,
    session_start: datetime,
) -> List[ImageRecord]:
    if not bucket:
        return []

    out: List[ImageRecord] = []
    current = session_start + timedelta(seconds=rng.randint(0, int(DURATION_FOR_DAYS.total_seconds())))

    for cn in sorted(bucket):
        for r in _order_records_in_bucket(bucket[cn]["none"]):
            r.date, r.time = _stamp_dt_loc(r, current, rng)
            out.append(r)
            current += random_gap()

        for r in _order_records_in_bucket(bucket[cn]["detected"]):
            r.date, r.time = _stamp_dt_loc(r, current, rng)
            out.append(r)
            current += random_gap()

    last_evening_time = current

    worked_cns = [cn for cn in sorted(bucket) if bucket[cn]["in_progress"]]
    if worked_cns:
        jitter_b = rng.randint(-int(BEFORE_JITTER.total_seconds()), int(BEFORE_JITTER.total_seconds()))
        works_start = last_evening_time + DURATION_BEFORE_WORKS + timedelta(seconds=jitter_b)

        jitter_w = rng.randint(-int(WORKS_JITTER.total_seconds()), int(WORKS_JITTER.total_seconds()))
        works_total = max(DURATION_FOR_WORKS + timedelta(seconds=jitter_w), timedelta(minutes=10))
        works_end = works_start + works_total
        window_secs = int((works_end - works_start).total_seconds())

        total_photos = sum(len(bucket[cn]["in_progress"]) for cn in worked_cns)
        cursor = works_start
        last_inprogress_time = works_start

        for i, cn in enumerate(worked_cns):
            photos = _order_records_in_bucket(bucket[cn]["in_progress"])
            if not photos:
                continue

            if i == len(worked_cns) - 1:
                cn_end = works_end
            else:
                share = len(photos) / max(total_photos, 1)
                span_secs = max(60, int(window_secs * share))
                cn_end = min(cursor + timedelta(seconds=span_secs), works_end)

            span = max((cn_end - cursor).total_seconds(), 60.0)
            offsets = sorted(rng.uniform(0, span) for _ in range(len(photos)))
            times = [cursor + timedelta(seconds=o) for o in offsets]

            for r, t in zip(photos, times):
                r.date, r.time = _stamp_dt_loc(r, t, rng)
                out.append(r)
            last_inprogress_time = times[-1]
            cursor = cn_end

    fixed_cns = [cn for cn in sorted(worked_cns) if bucket[cn]["fixed"]]
    if fixed_cns:
        fixed_cursor = last_evening_time + random_gap()
        for cn in fixed_cns:
            for r in _order_records_in_bucket(bucket[cn]["fixed"]):
                r.date, r.time = _stamp_dt_loc(r, fixed_cursor, rng)
                out.append(r)
                fixed_cursor += random_gap()

    return out


def _build_group_index(records: List[ImageRecord]) -> Dict[str, Dict[int, Dict[str, List[ImageRecord]]]]:
    """Group records by base date, construction number and stage for scheduling."""
    grouped: Dict[str, Dict[int, Dict[str, List[ImageRecord]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for r in records:
        cn = min(r.construction_number) if r.construction_number else -1
        grouped[r.base_date][cn][r.stage or "none"].append(r)
    return grouped


def _process_day(
    bd: str,
    bucket: Dict[int, Dict[str, List[ImageRecord]]],
    rng: random.Random,
) -> List[ImageRecord]:
    """Distribute timestamps and locations for a single base date."""
    out: List[ImageRecord] = []

    session_buckets = _split_by_session(bucket)
    base_dt = datetime.combine(datetime.strptime(bd, "%d.%m.%Y"), START_TIME)

    for session in SESSION_ORDER:
        session_bucket = session_buckets.get(session)
        if not session_bucket or not _session_has_records(session_bucket):
            continue
        session_start = base_dt + SESSION_OFFSETS.get(session, timedelta())
        out.extend(_process_session_bucket(session_bucket, rng, session_start))

    return out


def collect_all_images(base_folder: Path) -> List[ImageRecord]:
    """Collect raw images from the filesystem and prepare lightweight records."""
    files = find_images(base_folder)

    by_date: Dict[str, List[Path]] = defaultdict(list)
    dt_map: Dict[Path, str | None] = {}
    for p in files:
        dt = parse_date(tuple(p.parts[:-1]))
        dt_map[p] = dt
        if dt:
            by_date[dt].append(p)

    temp: List[ImageRecord] = []
    ordered_dates = sorted(
        by_date.keys(),
        key=lambda date: datetime.strptime(date, "%d.%m.%Y"),
    )
    for dt in ordered_dates:
        paths = by_date[dt]
        roots = {p.parent.parent for p in paths}
        stage_present = any(stage_dirs(root) for root in roots)

        ext_rank = {".jpg": 5, ".jpeg": 4, ".png": 3, ".webp": 2, ".bmp": 1, ".tif": 1, ".tiff": 1}
        best: Dict[tuple[Path, str], Path] = {}

        for p in paths:
            stem_l = p.stem.lower()
            if "_filtered" in stem_l or "_stamped" in stem_l:
                continue
            base = get_base_name(str(p))
            key = (p.parent, base)

            def score(path: Path) -> tuple[int, int, int, str]:
                tail = len(path.stem) - len(base)
                er = ext_rank.get(path.suffix.lower(), 0)
                return tail, er, len(path.stem), str(path)

            prev = best.get(key)
            if prev is None or score(p) > score(prev):
                best[key] = p

        for p in sorted(best.values(), key=lambda x: str(x)):
            rec = ImageRecord(
                date=dt,
                construction_number=extract_construction_numbers(p),
                path=str(p),
                stage=detect_stage(p) if stage_present else None,
                session=_detect_session(p.parts[:-1]),
            )
            # Limit work-type detection to meaningful parts of the path:
            # file name + two nearest folders and the detected stage.
            pth = Path(rec.path)
            parent1 = pth.parent.name
            parent2 = getattr(pth.parent, 'parent', pth.parent).name
            # Only look at file name + two nearest folders. Do NOT include stage.
            haystack = " ".join([pth.stem, parent1, parent2]).strip()
            for code, patt in WORK_PATTERNS:
                if patt.search(haystack):
                    rec.work_type = code
                    break
            if not rec.work_type:
                stage_name = pth.parent.name.strip()
                for code, titles in WORK_STAGE_TITLES.items():
                    if stage_name in titles:
                        rec.work_type = code
                        break
            if not rec.work_type:
                for token in (parent1, parent2):
                    candidate = token.strip()
                    if not candidate:
                        continue
                    pieces = candidate.split()
                    label_guess = pieces[-1] if pieces else candidate
                    if not any(ch.isalpha() for ch in label_guess):
                        continue
                    code = normalize_work_type(label_guess)
                    if code and code in WORK_PROFILES:
                        rec.work_type = code
                        break
            if not rec.work_type and not stage_present:
                rec.work_type = "CI"
            temp.append(rec)

    return temp


def enrich_all_images(records: List[ImageRecord], *, refresh_cache: bool = True) -> List[Dict[str, Any]]:
    """Persist enriched data to Excel and return row dictionaries.

    When ``refresh_cache`` is False and only cached data is needed, the Excel
    file is left untouched.
    """
    excel_path = _excel_path()

    if records:
        grouped = _build_group_index(records)

        out: List[ImageRecord] = []
        rng = random.Random()
        for bd in sorted(grouped, key=lambda date: datetime.strptime(date, "%d.%m.%Y")):
            out.extend(_process_day(bd, grouped[bd], rng))

        session_order = {"day": 0, "night": 1, "single": 2}

        out.sort(
            key=lambda rec: (
                datetime.strptime(rec.base_date, "%d.%m.%Y"),
                session_order.get((rec.session or "single").lower(), len(session_order)),
                (min(rec.construction_number) if rec.construction_number else -1),
                STAGE_ORDER.get(rec.stage, 0),
                datetime.strptime(f"{rec.date} {rec.time}", "%d.%m.%Y %H:%M:%S"),
            )
        )

        recs: List[Dict[str, Any]] = []
        for r in out:
            payload = r.to_dict()
            payload["base_name"] = get_base_name(r.path)
            recs.append(payload)

        df_new = pd.DataFrame(recs)
        df_sorted = _sort_records_df(df_new)
        if "base_name" not in df_sorted.columns:
            df_sorted["base_name"] = pd.Series(dtype="object")
        _normalise_locations(df_sorted)

        excel_path.parent.mkdir(parents=True, exist_ok=True)
        if refresh_cache:
            df_sorted.to_excel(excel_path, index=False)
        return df_sorted.to_dict(orient="records")

    if excel_path.exists():
        df = pd.read_excel(excel_path, dtype=str)
        df_sorted = _sort_records_df(df)
        _normalise_locations(df_sorted)
        if refresh_cache:
            df_sorted.to_excel(excel_path, index=False)
        return df_sorted.to_dict(orient="records")

    return []


def _stamp_dt_loc(rec: ImageRecord, dt: datetime, rng: random.Random):
    """Assign a timestamp and randomised location to a record."""
    rec.time = dt.strftime("%H:%M:%S")
    rec.location = rng.choice(LOCATIONS)
    rec.date = dt.strftime("%d.%m.%Y")
    return rec.date, rec.time
