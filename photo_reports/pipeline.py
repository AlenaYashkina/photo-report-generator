"""High-level orchestration for generating photo reports."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, List, Dict, Any

from photo_reports.filters import filter_images
from photo_reports.parser import collect_all_images, enrich_all_images
from photo_reports.pptx_creator import create_daily_presentation, create_work_presentation
from photo_reports.stamper import stamp_all_images
from utils.completeness_checker import check_dataset
from configs.work_profiles import WORK_TYPE_ORDER


@dataclass(frozen=True, slots=True)
class PipelineSettings:
    """Configuration container for the report-generation pipeline."""

    folder_path: Path
    report_mode: str
    enabled_work_type: str
    generate_all_reports: bool
    batch_work_types: Sequence[str]


class ReportPipeline:
    """Encapsulates the high-level orchestration of the reporting workflow."""

    def __init__(self, settings: PipelineSettings) -> None:
        self._settings = settings

    def run(self) -> None:
        records = self._build_records()
        check_dataset(self._settings.folder_path)

        if self._settings.generate_all_reports:
            self._run_full_batch(records)
            return

        if self._settings.report_mode == "daily":
            self._produce_daily_reports(records)
            return

        self._produce_work_report(records, self._settings.enabled_work_type)

    def _build_records(self) -> List[Dict[str, Any]]:
        excel_path = self._settings.folder_path / "parsed_records.xlsx"
        if excel_path.exists():
            try:
                excel_mtime = excel_path.stat().st_mtime
            except OSError:
                excel_mtime = 0.0

            latest_source_mtime = 0.0
            for path in self._settings.folder_path.rglob("*"):
                if not path.is_file():
                    continue
                suffix = path.suffix.lower()
                if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
                    continue
                stem_lower = path.stem.lower()
                if "_filtered" in stem_lower or "_stamped" in stem_lower:
                    continue
                try:
                    mtime = path.stat().st_mtime
                except OSError:
                    continue
                if mtime > latest_source_mtime:
                    latest_source_mtime = mtime
                    if latest_source_mtime > excel_mtime:
                        break

            if excel_mtime >= latest_source_mtime:
                return enrich_all_images([], refresh_cache=False)

        raw = collect_all_images(self._settings.folder_path)
        return enrich_all_images(raw)

    @staticmethod
    def _prepare_work_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prepared: List[Dict[str, Any]] = []
        for rec in records:
            original = Path(rec["path"])
            stamped = original.with_name(original.stem + "_stamped" + original.suffix)
            selected = stamped if stamped.exists() else original
            rec_copy = rec.copy()
            rec_copy["path"] = str(selected)
            prepared.append(rec_copy)
        return prepared

    def _run_full_batch(self, records: List[Dict[str, Any]]) -> None:
        filter_images(records)
        stamp_all_images(records)
        create_daily_presentation(report_mode="daily")

        if isinstance(self._settings.batch_work_types, str):
            configured = [self._settings.batch_work_types]
        else:
            configured = list(self._settings.batch_work_types)

        normalized: List[str] = []
        for item in configured:
            token = (item or "").strip()
            if not token:
                continue
            if token not in normalized:
                normalized.append(token)

        if not normalized:
            normalized = list(WORK_TYPE_ORDER)

        for work_type in normalized:
            self._produce_work_report(records, work_type)

    def _produce_daily_reports(self, records: List[Dict[str, Any]]) -> None:
        filter_images(records)
        stamp_all_images(records)
        create_daily_presentation()

    def _produce_work_report(self, records: List[Dict[str, Any]], work_type: str) -> None:
        work_records = self._prepare_work_records(records)
        create_work_presentation(work_records, work_type=work_type)


__all__ = ["PipelineSettings", "ReportPipeline"]
