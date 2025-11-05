
"""Entry point for automated photo report generation."""

from configs.config import (
    REPORT_MODE,
    FOLDER_PATH,
    ENABLED_WORK_TYPE,
    GENERATE_ALL_REPORTS,
    BATCH_WORK_TYPES,
)
from photo_reports.pipeline import PipelineSettings, ReportPipeline


def main() -> None:
    settings = PipelineSettings(
        folder_path=FOLDER_PATH,
        report_mode=REPORT_MODE,
        enabled_work_type=ENABLED_WORK_TYPE,
        generate_all_reports=GENERATE_ALL_REPORTS,
        batch_work_types=BATCH_WORK_TYPES,
    )
    ReportPipeline(settings).run()


if __name__ == "__main__":
    main()
