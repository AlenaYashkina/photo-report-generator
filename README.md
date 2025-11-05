# Photo Reports Automation

End-to-end pipeline that turns raw construction or maintenance photo dumps into
ready-to-share PowerPoint reports. Everything is configuration-driven so a new
site or work type can be onboarded by editing `.env`, not by touching code.

- Input: dated photo folders exported from field teams.
- Output: filtered and stamped images plus daily and work-type presentations.
- Stack: pure Python 3.10+, concurrency for heavy image passes, PPTX generation via `python-pptx`.

## Why this project exists

As a lighting engineer I was the only person responsible for hundreds of photo
reports each month. The work was repetitive: rename images, pick the best shot,
stamp metadata, build a slideshow with the correct stage titles. This project
codifies that routine so I can deliver accurate reports in minutes and spend the
saved time on analysis, storytelling, or experimenting with AI-assisted tooling.

## How the pipeline works

1. **Parse** folder trees (`photo_reports.parser`) and extract structured
   records: date, construction numbers, detected stage, inferred work type.
2. **Filter** photos (`photo_reports.filters`) to pick the best variant per
   scene, apply gentle color correction, and skip already processed files.
3. **Stamp** images (`photo_reports.stamper`) with date, time, and location
   overlays using Cairo so every slide is audit-ready.
4. **Compile PowerPoint** decks (`photo_reports.pptx_creator`) for daily
   progress and for each enabled work profile, keeping layouts consistent.
5. **Validate coverage** (`utils.completeness_checker`) against the monthly
   work plan to ensure contractual scope is met.

The orchestration lives in `photo_reports.pipeline.ReportPipeline`. Running
`python main.py` executes the full chain with concurrency and caching so
re-runs only touch deltas.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env  # adjust paths, dates, work profiles
python main.py
```

The default `.env` already points `FOLDER_PATH` to
`examples/workspace/ExampleLocation`, a writable copy of the anonymised demo
dataset. If the workspace folder is missing, it is seeded automatically from
`examples/reference/ExampleLocation` on first import. Output `.pptx` files are
**not** versioned to keep the repository lightweight-running the pipeline will
recreate them in-place.

## Key configuration (see `configs/config.py`)

- `FOLDER_PATH` - root directory with source photo archives.
- `REPORT_MODE` - `daily` or `batch`.
- `ENABLED_WORK_TYPE` / `BATCH_WORK_TYPES` - select activities for reports.
- `START_DATE` / `END_DATE` - reporting window.
- `RAW_IMAGE_MODE`, `PHOTO_LABELS_ENABLED` - toggle filtering and captions.
- `TITLE_CONTENT`, `MONTH_LABELS`, `WORK_PROFILES` - presentation metadata.

`.env.example` ships with a fully anonymised template that mirrors a real-world
lighting maintenance contract.

## Sample dataset

```
examples/
|-- reference/
|   `-- ExampleLocation/    # pristine snapshot (read-only)
|-- workspace/
|   `-- ExampleLocation/    # default working copy used by the pipeline
`-- source_photos/
    |-- Days/<CN>/...       # sample daytime shots for populators
    `-- Works/...           # stage-2 shots for in-progress populators
```

- Folder tree mirrors the real-world naming convention (month -> day ->
  construction -> stage).
- Excel cache and stamp metadata are pre-generated so the pipeline can verify
  coverage and metadata on the first run.
- Generated PowerPoint reports are intentionally excluded (they exceed Git
  hosting limits). Execute `python main.py` to reproduce them locally.
- Utility scripts `utils/random_photo_populator.py` and
  `utils/random_inprogress_photo_populator.py` pull sample images from
  `examples/source_photos`.

## Repository layout

- `configs/` - env parsing, default work profiles, layout constants.
- `photo_reports/filters.py` - image selection and enhancement tuned per device.
- `photo_reports/parser.py` - deterministic heuristics for stage detection and
  address assignment.
- `photo_reports/stamper.py` - Cairo-based overlay engine.
- `photo_reports/pptx_creator.py` - slide authoring and adaptive layouts.
- `utils/` - data quality checks, directory scaffolders, sample data populators.

## What this demonstrates for hiring managers

- **Automation mindset**: replaces manual, error-prone reporting with an
  idempotent pipeline that can be scheduled or wrapped in a UI.
- **Data wrangling**: cleans messy folder structures, infers missing metadata,
  keeps an auditable Excel cache, and validates work plans.
- **Presentation tooling**: programmatic PPT generation with adaptive layouts
  and branded headers.
- **Extensibility**: configuration-first approach makes it easy to plug into
  other systems (e.g., future LLM assistants for anomaly narratives or
  automatic photo tagging).

## Roadmap ideas

1. Sample dataset and before/after gallery for recruiters.
2. Optional Streamlit or FastAPI front-end to trigger runs without the CLI.
3. Hooks for AI-driven captions or anomaly summaries using the cached records.
4. Packaging as a `pipx`-friendly CLI with typed configs.

## About the author

Alena Yashkina - lighting engineer/designer (9 years) who loves automating field
operations with Python and AI tools. Looking for remote roles at the intersection
of workflow automation, computer vision, and creative tooling.


