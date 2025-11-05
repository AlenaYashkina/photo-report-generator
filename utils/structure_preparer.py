"""Prepare the filesystem layout expected by the photo parser.

The script reads the monthly work plan and creates a scaffold under ``FOLDER_PATH``
using the structure ``<ordinal month>/<dd.mm.yyyy>/<CN [label]>/<stage>``.
Work and stage names are pulled from the active ``WORK_PROFILES`` definition.
"""
from __future__ import annotations

import argparse
import calendar
import json
import random
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set

if __package__ is None:  # pragma: no cover - allows running as `python utils/structure_preparer.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.config import FOLDER_PATH
from configs.env_utils import get_env_json, get_env_str
from configs.work_profiles import WORK_PROFILES, WORK_STAGE_TITLES, normalize_work_type
from configs.utils_config import RU_MONTH

DATE_FMT = "%d.%m.%Y"
DEFAULT_STAGE_FOLDERS: Sequence[str] = (
    "1 Detected",
    "2 In progress",
    "3 Fixed",
)

_HYPHEN_CHARS = r"\-\u2010\u2011\u2012\u2013\u2014\u2015\u2212\uFF0D"
PLAN_TOKEN = re.compile(
    rf"^\s*(.+?)\s*[{_HYPHEN_CHARS}]\s*(\d+)\s*\*\s*([A-Za-z0-9_]+)\s*$"
)


@dataclass(frozen=True)
class WorkRequirement:
    work_type: str
    sessions: int
    cover: int
    raw: str


@dataclass(frozen=True)
class MonthPlan:
    month: str
    num_constructions: int | None
    requirements: List[WorkRequirement]


def _ensure_list(value: object) -> List[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _canonical_type(value: str) -> str:
    token = (value or "").strip()
    if not token:
        return ""
    normalised = normalize_work_type(token)
    if normalised:
        return normalised
    return token.upper()


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


GLOBAL_NUM_CONSTRUCTIONS = _parse_int(get_env_str("NUM_CONSTRUCTIONS", ""))


def _resolve_env_reference(token: str | None) -> int | None:
    if token is None:
        return None
    cleaned = token.strip()
    if not cleaned:
        return None
    if cleaned.upper() == "NUM_CONSTRUCTIONS":
        return GLOBAL_NUM_CONSTRUCTIONS
    return _parse_int(cleaned)


def _clean_token(value: str | None) -> str:
    if value is None:
        return ""
    return re.sub(r"[^A-Za-z0-9\u0400-\u04FF]+", "", str(value)).lower()


def _parse_plan_entry(raw: object) -> WorkRequirement:
    text = str(raw)
    match = PLAN_TOKEN.match(text)
    if not match:
        raise ValueError(f"Malformed plan token '{text}' (expected TYPE-count*cover)")
    work_type = _canonical_type(match.group(1))
    sessions = int(match.group(2))
    cover_token = match.group(3)
    cover = _resolve_env_reference(cover_token)
    if cover is None:
        raise ValueError(f"Invalid cover value '{cover_token}' for '{text}'")
    return WorkRequirement(work_type=work_type, sessions=sessions, cover=cover, raw=text.strip())


def _parse_month_plan(month: str, payload: object) -> MonthPlan:
    num_constructions: int | None = None
    entries: Sequence[object]

    if isinstance(payload, dict):
        raw_num = payload.get("num_constructions")
        if isinstance(raw_num, int):
            num_constructions = raw_num
        else:
            num_constructions = _resolve_env_reference(str(raw_num) if raw_num is not None else None)
        entries = _ensure_list(payload.get("items") or payload.get("works") or [])
    else:
        entries = _ensure_list(payload)

    requirements: List[WorkRequirement] = []
    for entry in entries:
        try:
            requirements.append(_parse_plan_entry(entry))
        except ValueError as exc:
            print(f"[WARN] {exc}")
    return MonthPlan(month=month, num_constructions=num_constructions, requirements=requirements)


def _parse_monthly_plan(raw_plan: object) -> Dict[str, MonthPlan]:
    plan: Dict[str, MonthPlan] = {}
    if not isinstance(raw_plan, dict):
        return plan
    for month, payload in raw_plan.items():
        month_key = str(month).strip()
        if not month_key:
            continue
        plan[month_key] = _parse_month_plan(month_key, payload)
    return plan


def _parse_allowed_dates(raw: object) -> Dict[str, Dict[str, Set[str]]]:
    result: Dict[str, Dict[str, Set[str]]] = {}
    if not isinstance(raw, dict):
        return result
    for work_type, payload in raw.items():
        canonical = _canonical_type(work_type)
        if not canonical:
            continue
        month_map: Dict[str, Set[str]] = {}
        if isinstance(payload, dict):
            for month, dates in payload.items():
                month_map[str(month)] = {str(d).strip() for d in _ensure_list(dates) if str(d).strip()}
        else:
            month_map["*"] = {str(d).strip() for d in _ensure_list(payload) if str(d).strip()}
        result[canonical] = month_map
    return result


RAW_PLAN = get_env_json("MONTHLY_WORK_PLAN", {})
MONTHLY_PLAN = _parse_monthly_plan(RAW_PLAN)
RAW_ALLOWED_DATES = get_env_json("WORK_ALLOWED_DATES", None)
if RAW_ALLOWED_DATES is None:
    fallback_data: Dict[str, object] = {}
    for candidate in (Path(".env"), Path(".env.example")):
        try:
            text = candidate.read_text(encoding="utf-8")
        except OSError:
            continue
        match = re.search(r"WORK_ALLOWED_DATES\s*=\s*['`](.*?)['`]", text, re.S)
        if match:
            try:
                fallback_data = json.loads(match.group(1))
            except json.JSONDecodeError:
                fallback_data = {}
            if fallback_data:
                break
    RAW_ALLOWED_DATES = fallback_data or {}
ALLOWED_DATES = _parse_allowed_dates(RAW_ALLOWED_DATES or {})
RAW_START_DATE = get_env_str("START_DATE", "").strip()
RAW_END_DATE = get_env_str("END_DATE", "").strip()


def _parse_date(value: str) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, DATE_FMT).date()
    except ValueError:
        return None


START_DATE_BOUND = _parse_date(RAW_START_DATE)
END_DATE_BOUND = _parse_date(RAW_END_DATE)
if START_DATE_BOUND and END_DATE_BOUND and END_DATE_BOUND < START_DATE_BOUND:
    START_DATE_BOUND, END_DATE_BOUND = END_DATE_BOUND, START_DATE_BOUND


def _calendar_dates(month_key: str) -> List[str]:
    try:
        year_str, month_str = month_key.split("-", 1)
        year = int(year_str)
        month = int(month_str)
        _, last_day = calendar.monthrange(year, month)
    except ValueError:
        return []
    return [f"{day:02d}.{month:02d}.{year}" for day in range(1, last_day + 1)]


def _month_bounds(month_key: str) -> tuple[date | None, date | None]:
    try:
        year_str, month_str = month_key.split("-", 1)
        year = int(year_str)
        month = int(month_str)
        last_day = calendar.monthrange(year, month)[1]
        first = date(year, month, 1)
        last = date(year, month, last_day)
        return first, last
    except ValueError:
        return None, None


def _within_window(date_str: str, start: date | None, end: date | None) -> bool:
    dt = _parse_date(date_str)
    if dt is None:
        return False
    if start and dt < start:
        return False
    if end and dt > end:
        return False
    return True


def _month_in_window(month_key: str, start: date | None, end: date | None) -> bool:
    first, last = _month_bounds(month_key)
    if first is None or last is None:
        return True
    if start and last < start:
        return False
    if end and first > end:
        return False
    return True


def _month_folder_name(month_key: str, ordinal: int) -> str:
    try:
        _, month_str = month_key.split("-", 1)
        month_idx = int(month_str)
    except ValueError:
        return f"{ordinal} {month_key}"
    label = RU_MONTH.get(month_idx)
    if label:
        return f"{ordinal} {label}"
    return f"{ordinal} {month_str}"


def _resolve_month_total(plan: MonthPlan, override: int | None) -> int:
    candidates = [
        override,
        plan.num_constructions,
        GLOBAL_NUM_CONSTRUCTIONS,
    ]
    max_cover = max((req.cover for req in plan.requirements), default=0)
    if max_cover:
        candidates.append(max_cover)
    for candidate in candidates:
        if isinstance(candidate, int) and candidate > 0:
            return candidate
    return 1


def _build_profiles() -> Dict[str, Dict[str, Sequence[str] | str]]:
    profiles: Dict[str, Dict[str, Sequence[str] | str]] = {}
    for code, payload in WORK_PROFILES.items():
        label = str(payload.get("label") or code).strip() or code
        raw_titles = WORK_STAGE_TITLES.get(code) or payload.get("stage_titles") or ()
        titles = [
            str(item).strip()
            for item in raw_titles
            if str(item).strip()
        ]
        if not titles:
            titles = list(DEFAULT_STAGE_FOLDERS)
        profiles[code] = {"label": label, "stages": tuple(titles)}
    return profiles


PROFILES = _build_profiles()


def _stage_titles_for(work_type: str) -> Sequence[str]:
    payload = PROFILES.get(work_type)
    if payload:
        stages = payload.get("stages")
        if stages:
            return stages  # type: ignore[return-value]
    return DEFAULT_STAGE_FOLDERS


def _label_for(work_type: str) -> str:
    payload = PROFILES.get(work_type)
    if payload:
        return str(payload.get("label") or work_type)
    return work_type


HEAVY_HINTS: Set[str] = {
    "pr",
    "partialrepair",
    "di",
    "deepinspection",
    "flag",
    "flags",
    "flagalignment",
    "polish",
    "wipe",
    "snow",
    "snowclear",
}

def _derive_single_cn_codes() -> Set[str]:
    derived: Set[str] = set()
    for code, payload in WORK_PROFILES.items():
        tokens = {
            code,
            payload.get("label"),
            payload.get("activity_name"),
            payload.get("activity_short"),
        }
        for token in tokens:
            cleaned = _clean_token(token)
            if cleaned in HEAVY_HINTS:
                derived.add(code)
                break
    return derived


DERIVED_SINGLE_CN = _derive_single_cn_codes()
RAW_SINGLE_CN = get_env_json("SINGLE_CN_WORK_TYPES", [])
SINGLE_CN_WORK_TYPES: Set[str] = set(DERIVED_SINGLE_CN)
for item in _ensure_list(RAW_SINGLE_CN):
    canonical = _canonical_type(str(item))
    if canonical:
        SINGLE_CN_WORK_TYPES.add(canonical)


EXCLUSIVE_WORK_GROUPS_RAW: Sequence[Sequence[str]] = [
    ("Polish", "SnowClear"),
    ("polish", "snow"),
]
EXCLUSIVE_GROUPS: List[Set[str]] = [
    {_canonical_type(token) for token in group if _canonical_type(token)}
    for group in EXCLUSIVE_WORK_GROUPS_RAW
]
_BLOCKING_TOKENS: Sequence[str] = ("Polish", "SnowClear", "polish", "snow")
BLOCKING_WORK_TYPES: Set[str] = {
    token for token in (_canonical_type(item) for item in _BLOCKING_TOKENS) if token
}


def _has_conflict(work_type: str, existing: Set[str]) -> bool:
    if not existing:
        return False
    if work_type in BLOCKING_WORK_TYPES:
        return True
    if BLOCKING_WORK_TYPES & existing:
        return True
    for group in EXCLUSIVE_GROUPS:
        if work_type in group and group & existing:
            return True
    return True


def _select_dates(
    work_type: str,
    month_key: str,
    count: int,
    start: date | None,
    end: date | None,
    unique_only: bool,
    reserved: Set[str] | None = None,
    blocked: Set[str] | None = None,
    per_day_works: Dict[str, Set[str]] | None = None,
) -> List[str]:
    if count <= 0:
        return []

    allowed_map = ALLOWED_DATES.get(work_type, {})
    candidate: List[str] = []
    has_specific = False
    if isinstance(allowed_map, Mapping):
        preferred = allowed_map.get(month_key) or allowed_map.get("*")
        if preferred:
            candidate = [str(item).strip() for item in preferred if str(item).strip()]
            has_specific = bool(candidate)
    if not candidate:
        candidate = _calendar_dates(month_key)
    rng = random.Random()
    rng.shuffle(candidate)
    blocked_set = set(blocked or ())
    candidate = [
        d for d in candidate
        if _within_window(d, start, end) and d not in blocked_set
    ]

    supplemental: List[str] = []
    if not has_specific:
        supplemental = [
            d for d in _calendar_dates(month_key)
            if d not in candidate and _within_window(d, start, end) and d not in blocked_set
        ]
    ordered = list(dict.fromkeys(candidate + supplemental))
    if not ordered:
        ordered = [f"{month_key}_slot"]

    result: List[str] = []
    if unique_only:
        seen = set(blocked_set)
        if reserved:
            seen.update(reserved)
        for d in ordered:
            if d in seen:
                continue
            if per_day_works and _has_conflict(work_type, per_day_works.get(d, set())):
                continue
            seen.add(d)
            result.append(d)
            if len(result) >= count:
                break
        first_day, last_day = _month_bounds(month_key)
        offset = 0
        while len(result) < count:
            fallback: str
            if first_day and last_day:
                attempt = first_day + timedelta(days=offset)
                offset += 1
                if attempt > last_day + timedelta(days=60):
                    fallback = f"{month_key}_extra{offset:02d}"
                else:
                    fallback = attempt.strftime(DATE_FMT)
            else:
                fallback = f"{month_key}_extra{offset:02d}"
                offset += 1
            if fallback in seen:
                continue
            fallback_dt = _parse_date(fallback)
            if fallback_dt is not None:
                if start and fallback_dt < start:
                    continue
                if end and fallback_dt > end:
                    continue
            seen.add(fallback)
            result.append(fallback)
    else:
        idx = 0
        while len(result) < count:
            candidate_date = ordered[idx % len(ordered)]
            if candidate_date in blocked_set:
                idx += 1
                if idx > len(ordered) * 3:
                    break
                continue
            if per_day_works and _has_conflict(work_type, per_day_works.get(candidate_date, set())):
                idx += 1
                if idx > len(ordered) * 3:
                    break
                continue
            result.append(candidate_date)
            idx += 1

    return result[:count]


def _allocate_cn_groups(
    month_key: str,
    work_type: str,
    total: int,
    per_slot: int,
    slots: int,
) -> List[List[int]]:
    total = max(1, total)
    per_slot = max(1, min(per_slot, total))
    base = list(range(1, total + 1))
    if per_slot >= total:
        return [base[:] for _ in range(slots)]
    if per_slot == 1:
        return [[base[i % total]] for i in range(slots)]
    rng = random.Random()
    allocations: List[List[int]] = []
    for _ in range(slots):
        group = sorted(rng.sample(base, per_slot))
        allocations.append(group)
    return allocations


def _cn_dir_name(cn: int, work_types: Sequence[str]) -> str:
    if not work_types:
        return str(cn)
    labels: List[str] = []
    for work_type in work_types:
        label = _label_for(work_type).strip()
        if not label:
            continue
        if label not in labels:
            labels.append(label)
    suffix = " + ".join(labels) if labels else ""
    return f"{cn} {suffix}".strip()


def _ensure_hierarchy(paths: Iterable[Path], dry_run: bool) -> int:
    created = 0
    for path in paths:
        if dry_run:
            print(f"[DRY] {path}")
            continue
        if path.exists():
            continue
        path.mkdir(parents=True, exist_ok=True)
        created += 1
    return created


def _build_structure(
    base_path: Path,
    plan: Mapping[str, MonthPlan],
    work_filter: Sequence[str] | None = None,
    dry_run: bool = False,
    cn_override: int | None = None,
    start: date | None = None,
    end: date | None = None,
) -> int:
    created_dirs = 0
    work_filter_set: Set[str] | None = None
    if work_filter:
        resolved: Set[str] = set()
        for token in work_filter:
            canonical = _canonical_type(token)
            if canonical:
                resolved.add(canonical)
        work_filter_set = resolved

    month_items = sorted(
        plan.items(),
        key=lambda item: datetime.strptime(item[0], "%Y-%m"),
    )

    for ordinal, (month_key, month_plan) in enumerate(month_items, start=1):
        if not _month_in_window(month_key, start, end):
            continue

        month_total = _resolve_month_total(month_plan, cn_override)
        if month_total <= 0:
            continue

        month_root = base_path / _month_folder_name(month_key, ordinal)
        day_work_map: Dict[str, Dict[int, List[str]]] = {}
        base_dates: Set[str] = set()
        single_reserved: Set[str] = set()
        single_blocked_dates: Set[str] = set()
        per_day_works: Dict[str, Set[str]] = {}
        per_day_cn_assignments: Dict[str, Dict[int, str]] = {}

        ordered_requirements = sorted(
            enumerate(month_plan.requirements),
            key=lambda item: (0 if item[1].work_type in BLOCKING_WORK_TYPES else 1, item[0]),
        )

        for _, requirement in ordered_requirements:
            work_type = requirement.work_type
            if work_filter_set and work_type not in work_filter_set:
                continue
            if requirement.sessions <= 0:
                continue

            if work_type.upper() == "CI":
                dates = _select_dates(
                    work_type,
                    month_key,
                    requirement.sessions,
                    start,
                    end,
                    unique_only=False,
                )
                base_dates.update(dates)
                continue

            single_cn = work_type in SINGLE_CN_WORK_TYPES
            cover = max(1, requirement.cover)
            per_slot = 1 if single_cn else min(cover, month_total)
            slot_count = requirement.sessions * cover if single_cn else requirement.sessions
            dates = _select_dates(
                work_type,
                month_key,
                slot_count,
                start,
                end,
                unique_only=single_cn,
                reserved=single_reserved if single_cn else None,
                blocked=single_blocked_dates if single_cn else None,
                per_day_works=per_day_works,
            )
            if not dates:
                continue

            allocations = _allocate_cn_groups(
                month_key=month_key,
                work_type=work_type,
                total=month_total,
                per_slot=per_slot,
                slots=len(dates),
            )
            assigned_dates: List[str] = []
            for date_str, cn_group in zip(dates, allocations):
                day_assignments = per_day_cn_assignments.setdefault(date_str, {})
                if day_assignments:
                    continue
                entry = day_work_map.setdefault(date_str, {})
                assigned = False
                if work_type in BLOCKING_WORK_TYPES:
                    # Snow and Polish occupy every construction for the day.
                    full_range = list(range(1, month_total + 1))
                    if all(not entry.get(cn) for cn in full_range):
                        for cn in full_range:
                            entry.setdefault(cn, []).append(work_type)
                            day_assignments[cn] = work_type
                        assigned = True
                else:
                    assignable: List[int] = []
                    for cn in cn_group:
                        if entry.get(cn):
                            assignable = []
                            break
                        assignable.append(cn)
                    if assignable:
                        for cn in assignable:
                            entry.setdefault(cn, []).append(work_type)
                            day_assignments[cn] = work_type
                        assigned = True
                if not assigned:
                    continue
                per_day_works.setdefault(date_str, set()).add(work_type)
                base_dates.add(date_str)
                assigned_dates.append(date_str)

            if single_cn and assigned_dates:
                single_reserved.update(assigned_dates)
                single_blocked_dates.update(assigned_dates)

        if not base_dates and not day_work_map:
            continue

        all_dates = sorted(base_dates | set(day_work_map.keys()), key=lambda d: datetime.strptime(d, DATE_FMT))
        for date_str in all_dates:
            day_root = month_root / date_str
            created_dirs += _ensure_hierarchy([month_root, day_root], dry_run)

            works_for_day = day_work_map.get(date_str, {})
            for cn in range(1, month_total + 1):
                work_types = works_for_day.get(cn, [])
                cn_dir = day_root / _cn_dir_name(cn, work_types)
                created_dirs += _ensure_hierarchy([cn_dir], dry_run)
                if not work_types:
                    continue
                if len(work_types) == 1:
                    stage_titles = _stage_titles_for(work_types[0])
                    paths = [cn_dir / stage for stage in stage_titles]
                    created_dirs += _ensure_hierarchy(paths, dry_run)
                else:
                    for work_type in work_types:
                        label = _label_for(work_type)
                        work_root = cn_dir / label
                        stage_titles = _stage_titles_for(work_type)
                        paths = [work_root] + [work_root / stage for stage in stage_titles]
                        created_dirs += _ensure_hierarchy(paths, dry_run)

    return created_dirs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-create folder scaffolding for planned work sessions.",
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        default=FOLDER_PATH,
        help="Root directory for generated folders (defaults to FOLDER_PATH).",
    )
    parser.add_argument(
        "--work-types",
        nargs="+",
        help="Limit generation to selected work codes (e.g. PR DI SnowClear).",
    )
    parser.add_argument(
        "--cn-total",
        type=int,
        help="Override number of constructions per session.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned directories without creating them.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not MONTHLY_PLAN:
        print("[WARN] MONTHLY_WORK_PLAN is empty; nothing to prepare.")
        if args.dry_run:
            print("[INFO] Dry run complete. No directories were created.")
        else:
            print(f"[INFO] Prepared 0 directories under {args.base_path}")
        return

    created = _build_structure(
        base_path=args.base_path,
        plan=MONTHLY_PLAN,
        work_filter=args.work_types,
        dry_run=args.dry_run,
        cn_override=args.cn_total,
        start=START_DATE_BOUND,
        end=END_DATE_BOUND,
    )
    if args.dry_run:
        if created:
            print(f"[INFO] Dry run complete. {created} directories would be created.")
        else:
            print("[INFO] Dry run complete. No directories were created.")
    else:
        print(f"[INFO] Prepared {created} directories under {args.base_path}")


if __name__ == "__main__":
    main()
