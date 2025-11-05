"""Consistency checker for parsed photo report data based on monthly work plans."""
from __future__ import annotations

import ast
import calendar
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

from configs.config import FOLDER_PATH
from configs.env_utils import get_env_json, get_env_str
from configs.work_profiles import normalize_work_type

DATE_FMT = "%d.%m.%Y"
PLAN_TOKEN = re.compile(r"^\s*(.+?)\s*-\s*(\d+)\s*\*\s*([A-Za-z0-9_]+)\s*$")
REQUIRED_STAGES: Set[str] = {"detected", "in_progress", "fixed"}

EXCEL_PATH = FOLDER_PATH / "parsed_records.xlsx"


# ---------------------------------------------------------------------------
# Plan parsing
# ---------------------------------------------------------------------------


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


def _parse_int(value: str | None) -> Optional[int]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


@dataclass(frozen=True)
class WorkRequirement:
    work_type: str
    sessions: int
    cover: int
    raw: str


@dataclass(frozen=True)
class MonthPlan:
    month: str
    num_constructions: Optional[int]
    requirements: List[WorkRequirement]


GLOBAL_NUM_CONSTRUCTIONS = _parse_int(get_env_str("NUM_CONSTRUCTIONS", ""))


def _resolve_cover_token(token: str) -> Optional[int]:
    cleaned = (token or "").strip()
    if not cleaned:
        return None
    if cleaned.upper() == "NUM_CONSTRUCTIONS":
        return GLOBAL_NUM_CONSTRUCTIONS
    return _parse_int(cleaned)


def _parse_plan_entry(raw: object) -> WorkRequirement:
    text = str(raw)
    match = PLAN_TOKEN.match(text)
    if not match:
        raise ValueError(f"Malformed plan token '{text}' (expected TYPE-count*cover)")
    work_type = _canonical_type(match.group(1))
    sessions = int(match.group(2))
    cover_token = match.group(3)
    cover = _resolve_cover_token(cover_token)
    if cover is None:
        raise ValueError(f"Invalid cover value '{cover_token}' for '{text}'")
    return WorkRequirement(work_type=work_type, sessions=sessions, cover=cover, raw=text.strip())


def _parse_month_plan(month: str, payload: object) -> MonthPlan:
    num_constructions: Optional[int] = None
    entries: Sequence[object]

    if isinstance(payload, dict):
        raw_num = payload.get("num_constructions")
        if isinstance(raw_num, int):
            num_constructions = raw_num
        else:
            num_constructions = _resolve_cover_token(str(raw_num) if raw_num is not None else None)
        entries = _ensure_list(payload.get("items") or payload.get("works") or [])
    else:
        entries = _ensure_list(payload)

    requirements = [ _parse_plan_entry(entry) for entry in entries ]
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

RAW_ALLOWED_DATES = get_env_json("WORK_ALLOWED_DATES", {})
ALLOWED_DATES = _parse_allowed_dates(RAW_ALLOWED_DATES)

GLOBAL_NUM_CONSTRUCTIONS = _parse_int(get_env_str("NUM_CONSTRUCTIONS", ""))


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _month_key(date_str: str) -> str:
    return datetime.strptime(date_str, DATE_FMT).strftime("%Y-%m")


def _load_df_from_excel(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]

    df["stage"] = df["stage"].apply(
        lambda x: None if (pd.isna(x) or str(x).strip().lower() in {"", "nan", "none"})
        else str(x).strip().lower()
    )

    df["construction_number"] = df["construction_number"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else (x if isinstance(x, list) else [])
    )
    df = df.explode("construction_number", ignore_index=True)
    df["construction_number"] = df["construction_number"].apply(
        lambda value: int(value) if str(value).isdigit() else None
    )
    df = df.dropna(subset=["construction_number"]).rename(columns={"construction_number": "cn"})
    df["cn"] = df["cn"].astype(int)

    df["base_date"] = df["base_date"].astype(str).str.strip()

    def norm_work_type(value: object) -> Optional[str]:
        if pd.isna(value):
            return None
        token = str(value).strip()
        if not token:
            return None
        resolved = normalize_work_type(token)
        if resolved:
            return resolved
        return token.upper()

    df["work_type"] = df["work_type"].apply(norm_work_type)
    df["month_key"] = df["base_date"].apply(_month_key)
    return df


def _allowed_dates_for(work_type: str, month: str) -> Set[str]:
    mapping = ALLOWED_DATES.get(work_type)
    if not mapping:
        return set()
    dates: Set[str] = set()
    if "*" in mapping:
        dates |= mapping["*"]
    if month in mapping:
        dates |= mapping[month]
    return dates


def _calendar_dates_for_month(month_key: str) -> Set[str]:
    """Return all calendar dates for the given YYYY-MM month in dd.mm.yyyy format."""
    try:
        year_str, month_str = month_key.split("-", 1)
        year = int(year_str)
        month = int(month_str)
        _, last_day = calendar.monthrange(year, month)
    except ValueError:
        return set()

    formatted: Set[str] = set()
    for day in range(1, last_day + 1):
        formatted.add(f"{day:02d}.{month:02d}.{year}")
    return formatted


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _collect_sessions(
    df_month: pd.DataFrame,
    work_type: str,
) -> Tuple[List[Tuple[str, Set[int]]], Dict[str, List[str]], pd.DataFrame]:
    """Return (sessions, missing_details, subset_df)."""
    missing: Dict[str, List[str]] = {}

    if work_type == "CI":
        detection_subset = df_month[
            df_month["stage"].isna() | (df_month["stage"] == "detected")
        ]

        detection_map: Dict[str, Set[int]] = {}
        for date, group in detection_subset.groupby("base_date"):
            coverage = {
                int(cn)
                for cn in group["cn"]
                if pd.notna(cn)
            }
            detection_map[date] = coverage

        all_cns: List[int] = sorted(
            {int(cn) for cn in df_month["cn"] if pd.notna(cn)}
        )

        sessions: List[Tuple[str, Set[int]]] = []
        base_dates = sorted(
            (str(date) for date in df_month["base_date"].unique()),
            key=lambda value: datetime.strptime(value, DATE_FMT),
        )
        for date in base_dates:
            coverage = detection_map.get(date, set())
            sessions.append((date, coverage))
            if all_cns and len(coverage) < len(all_cns):
                details = [
                    f"CN{cn}: missing detected/none stage photo"
                    for cn in all_cns
                    if cn not in coverage
                ]
                if details:
                    missing[date] = details

        return sessions, missing, detection_subset

    subset = df_month[df_month["work_type"] == work_type]
    sessions = []
    for date, group in subset.groupby("base_date"):
        good: Set[int] = set()
        issues: List[str] = []
        for cn, cn_group in group.groupby("cn"):
            stages = {stage for stage in cn_group["stage"] if stage}
            missing_stages = sorted(REQUIRED_STAGES - stages)
            if missing_stages:
                issues.append(f"CN{cn}: missing {', '.join(missing_stages)}")
                continue
            good.add(int(cn))
        sessions.append((date, good))
        if issues:
            missing[date] = issues
    sessions.sort(key=lambda item: datetime.strptime(item[0], DATE_FMT))
    return sessions, missing, subset


def _format_details(details: Iterable[str], limit: int = 4) -> str:
    items = list(details)
    if not items:
        return ""
    if len(items) > limit:
        preview = "; ".join(items[:limit])
        return f"{preview}; ..."
    return "; ".join(items)


def _preview(items: Iterable[str], limit: int = 4) -> str:
    seq = [str(item) for item in items]
    if not seq:
        return ""
    if len(seq) <= limit:
        return ", ".join(seq)
    return ", ".join(seq[:limit]) + ", ..."


def _validate_requirement(
    month: str,
    requirement: WorkRequirement,
    df_month: pd.DataFrame,
    total_constructions: int,
) -> List[str]:
    issues: List[str] = []

    sessions, missing, subset = _collect_sessions(df_month, requirement.work_type)
    allowed_dates = _allowed_dates_for(requirement.work_type, month)
    actual_dates = {str(date) for date, _ in sessions}

    if allowed_dates:
        invalid = sorted(actual_dates - allowed_dates)
        if invalid:
            issues.append(
                f"{month} {requirement.work_type}: data found on non-permitted dates {', '.join(invalid)}"
            )
        sessions = [session for session in sessions if session[0] in allowed_dates]
        valid_dates = {date for date, _ in sessions}
        missing = {date: details for date, details in missing.items() if date in valid_dates}

    coverage_counter: Counter[int] = Counter()
    for _, coverage in sessions:
        for cn in coverage:
            coverage_counter[int(cn)] += 1

    all_cns = sorted({int(cn) for cn in df_month["cn"].dropna().unique()})
    if total_constructions:
        all_cns = sorted(set(all_cns) | set(range(1, total_constructions + 1)))
    if not all_cns:
        all_cns = sorted(coverage_counter.keys())

    if requirement.sessions == 0:
        if subset.empty:
            return issues
        active = [f"CN{cn}: {count}" for cn, count in sorted(coverage_counter.items()) if count]
        if active:
            issues.append(
                f"{month} {requirement.work_type}: expected 0 sessions but found activity ({', '.join(active)})"
            )
        return issues

    if requirement.cover > total_constructions:
        issues.append(
            f"{month} {requirement.work_type}: plan requires {requirement.cover} constructions "
            f"per session but only {total_constructions} are available"
        )

    actual_dates = sorted({date for date, coverage in sessions if coverage})
    recorded_dates = set(actual_dates)
    candidate_dates = allowed_dates if allowed_dates else _calendar_dates_for_month(month)
    missing_dates = sorted(candidate_dates - recorded_dates) if candidate_dates else []

    if len(actual_dates) < requirement.sessions:
        missing_fragment = _preview(missing_dates, limit=6) or "no available calendar dates"
        issues.append(
            f"{month} {requirement.work_type}: sessions recorded {len(actual_dates)}/{requirement.sessions}; "
            f"missing dates: {missing_fragment}"
        )

    cn_date_map: Dict[int, List[str]] = {}
    for date, coverage in sessions:
        for cn in coverage:
            cn_date_map.setdefault(int(cn), []).append(date)

    eligible_cns = [cn for cn in all_cns if len(cn_date_map.get(cn, [])) >= requirement.sessions]
    if len(eligible_cns) < requirement.cover:
        summary = ", ".join(
            f"CN{cn}: {len(cn_date_map.get(cn, []))}"
            for cn in all_cns
        ) or "no data"
        issues.append(
            f"{month} {requirement.work_type}: constructions meeting the target {len(eligible_cns)}/{requirement.cover} "
            f"(sessions per CN: {summary})"
        )
        deficits = []
        for cn in all_cns:
            count = len(cn_date_map.get(cn, []))
            if count < requirement.sessions:
                have_preview = _preview(sorted(cn_date_map.get(cn, [])), limit=3) or "none"
                deficits.append(
                    f"CN{cn}: short by {requirement.sessions - count} (have: {have_preview})"
                )
        if deficits:
            if deficits and allowed_dates:
                issues.append("  Coverage shortfall: " + ", ".join(deficits))

    if missing:
        for date, info in sorted(missing.items()):
            detail = _format_details(info)
            if detail:
                issues.append(
                    f"{month} {requirement.work_type}: {date} has records but missing stages ({detail})"
                )

    return issues


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def check_dataset(
    _base_root: Path,
    excel_path: Path = EXCEL_PATH,
) -> None:
    if not excel_path.exists():
        print(f"[ERROR] Excel not found: {excel_path}")
        return

    df = _load_df_from_excel(excel_path)
    if df.empty:
        print("[ERROR] parsed_records.xlsx is empty; nothing to validate.")
        return

    all_constructions = sorted(df["cn"].unique().tolist())
    global_total = GLOBAL_NUM_CONSTRUCTIONS or len(all_constructions)

    issues: List[str] = []

    for month_key, plan in sorted(MONTHLY_PLAN.items()):
        df_month = df[df["month_key"] == month_key]
        month_total = plan.num_constructions or global_total

        for requirement in plan.requirements:
            issues.extend(_validate_requirement(month_key, requirement, df_month, month_total))

    if not issues:
        print("All checks passed: monthly work requirements satisfied.")
    else:
        print("Checks failed:")
        for line in issues:
            print(f"- {line}")


if __name__ == "__main__":
    check_dataset(FOLDER_PATH)
