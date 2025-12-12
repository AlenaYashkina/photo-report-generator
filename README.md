# Photo Reports Automation (archived)

![status](https://img.shields.io/badge/status-archived-lightgrey)
![python](https://img.shields.io/badge/Python-3.10%2B-blue)

Windows-first Python CLI that turns dated photo folders from field crews into stamped images and PowerPoint decks (daily and per work type). Built for a production maintenance contract; kept as a reproducible work sample.

> Archived snapshot: stable and used in production; no new features planned.

## Why it exists

- Field teams shoot proof photos for maintenance/installation works; every shot must carry date/time/location stamps.
- Deliverables are daily control decks plus work-type decks (snow clearing, polishing, flags, etc.).
- Manual Photoshop + PPTX layout across thousands of photos was slow and error-prone; this pipeline standardises the loop (parse -> enrich -> stamp -> export PPTX).

## Highlights

- Config-first CLI for Windows: `.env` and `configs/` define work types, date ranges, addresses, titles, and layout constants.
- End-to-end automation: parses folder trees into `parsed_records.xlsx`, filters images, stamps date/time/location overlays, and exports `.pptx` with a consistent layout.
- Sample dataset and auto-seeding: if your workspace folder is missing, it is copied from `examples/reference/ExampleLocation` into `examples/workspace/ExampleLocation` so you can run immediately.
- Deterministic caching: reuses `parsed_records.xlsx` and stamp metadata to skip already processed shots.
- Helper utilities: scaffold folder structures, populate sample photos, check completeness against a monthly plan, and compress media for sharing.

## Pipeline flow

1. Parse the folder tree under `FOLDER_PATH`, infer base date/stage/session and construction numbers, and cache to `parsed_records.xlsx`.
2. Filter images to align look across devices (skipped when `RAW_IMAGE_MODE=true`).
3. Stamp date/time/location overlays on selected images (skipped when `RAW_IMAGE_MODE=true`; reuses existing `_stamped` files).
4. Build PowerPoint decks:
   - daily report when `REPORT_MODE=daily`;
   - work-type reports (SnowClear, Polish, Flags, etc.) or all of them when `GENERATE_ALL_REPORTS=true`.
5. Decks are saved next to `FOLDER_PATH`; stamped images stay alongside their sources.

## Setup (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install pandas python-pptx pycairo  # dependencies used by the pipeline but not pinned above
copy .env.example .env  # set FOLDER_PATH, START_DATE, END_DATE, and presentation fields
python main.py
```

- Input: dated photo folders under `FOLDER_PATH`.
- Output: `parsed_records.xlsx`, stamped JPG/PNG files, and `.pptx` decks in the same workspace.

## Configuration cheatsheet (`.env` or `configs/config.py`)

- `FOLDER_PATH`: workspace with photo folders; auto-copied from `examples/reference/ExampleLocation` if missing.
- `START_DATE` / `END_DATE`: reporting window.
- `REPORT_MODE`: `daily` or `batch`.
- `GENERATE_ALL_REPORTS`: run daily plus all work-type decks in one go.
- `ENABLED_WORK_TYPE` / `BATCH_WORK_TYPES`: target work types when not running the full batch.
- `RAW_IMAGE_MODE`: skip filtering and stamping; use raw photos in decks.
- `PHOTO_LABELS_ENABLED`: toggle per-photo captions in PPTX.
- Presentation metadata: `TITLE_CONTENT`, `TITLE_ADDRESS_PREFIX`, `MONTH_LABELS`, `STAMP_MONTHS`, `STAMP_YEAR_SUFFIX`.
- Work metadata: `WORK_PROFILES`, `WORK_STAGE_FALLBACK_TITLES`, `WORK_ALLOWED_DATES`, `MONTHLY_WORK_PLAN`, `NUM_CONSTRUCTIONS`.
- Location pools: `LOCATION_GROUPS`, `ACTIVE_LOCATION_GROUP` (deterministic address assignment in stamps).

`.env.example` ships with an anonymised contract-style template so you can adjust settings without code changes.

## Sample data

```
examples/
  reference/ExampleLocation/   # pristine snapshot (read-only)
  workspace/ExampleLocation/   # working copy; auto-created from reference if missing
  source_photos/               # pool for populators
    Days/<CN>/...
    Works/...
```

Run the pipeline with defaults to regenerate the bundled Excel cache and PPTX decks locally.

## Helper scripts

- `utils/structure_preparer.py`: build the expected folder scaffold from `MONTHLY_WORK_PLAN` and allowed dates.
- `utils/random_photo_populator.py`: fill the workspace with random day-session images from `examples/source_photos`.
- `utils/random_inprogress_photo_populator.py`: populate stage-2 folders with in-progress shots.
- `utils/completeness_checker.py`: compare `parsed_records.xlsx` against the monthly plan (counts, stages, dates).
- `utils/compress_media.py`: downsize media when sharing results.

## Repository layout

- `configs/`: env parsing, work profiles, PPTX/stamp/layout constants.
- `photo_reports/filters.py`: image filtering and device-normalisation pipeline.
- `photo_reports/parser.py`: heuristics for date/stage/session detection and Excel cache writer.
- `photo_reports/stamper.py`: overlay engine for date/time/location stamps.
- `photo_reports/pptx_creator.py`: PowerPoint rendering (daily + work-type decks).
- `photo_reports/pipeline.py`: CLI orchestration used by `main.py`.
- `utils/`: data validation, folder scaffolding, sample data loaders, compression helpers.

## Status

Production-ready snapshot kept as an archived work sample; no new features planned. Local config tweaks are still possible if you reuse the pipeline.

## Author

Alena Yashkina - Python-first automation engineer (image/report pipelines, Windows CLI tools). Open to remote roles in process automation and AI-assisted ops tooling.
