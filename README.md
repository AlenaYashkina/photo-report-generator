# Photo Report Generator — Production Automation (Archived)

![status](https://img.shields.io/badge/status-archived-lightgrey)
![python](https://img.shields.io/badge/Python-3.10%2B-blue)

A finished **Python CLI pipeline** that turns dated photo folders into **stamped images** and **ready-to-share PowerPoint reports**.
Focus: practical process automation & batch image handling for field ops.

> Repository is frozen. No new features planned; used in production and kept as a work sample.

---

## Impact (measurable)

**Per‑photo effort (conservative):**

| Step | Manual time / photo | Automated time / photo |
|---|---:|---:|
| Read/select + find metadata | 20–30 s | ~0.5–1.0 s |
| Stamp date/time/location overlay | 25–40 s | ~0.5–1.0 s |
| Insert into PPTX with correct layout | 15–20 s | ~0.5–1.0 s |
| **Total per photo** | **60–90 s** | **~1.5–3.0 s** |

**At 9,000 photos / month (typical peak):**

- Manual: **~150–225 hours / month**  
- Automated: **~3.8–7.5 hours / month**  
- **Time saved:** **~146–221 hours / month** (≈ **3.6–5.5 work weeks**).  
- Plus: fewer typos in overlays, consistent slide layouts, predictable output.

**Development timeline:** initial build ≈ **2 months**; afterward only small fixes and adjustments discovered in use.

<sub>*Assumptions used: office PC with SSD; manual flow = Photoshop stamp + PPTX placement per photo; automated flow runs end‑to‑end in one command. Values are conservative to avoid overstating speed‑up.*</sub>

---

## Target roles (remote)

- **AI Automation / Prompt Engineer** — LLM-assisted document & photo workflows, API orchestration, structured extraction.
- **Python Automation Engineer** — batch image processing, metadata pipelines, reproducible CLI tools (Windows).
- **LLM Integrations Engineer** — OpenAI/HF APIs, prompt chains, lightweight retrieval for reports; productionizing prototypes.
- **CV / Content Ops (Entry)** — pre/post-processing for computer-vision tasks, dataset prep, stamping/annotation, export to PPTX.
- **Creative Technologist (GenAI: imaging/audio)** — rapid prototyping with SD/inpainting/TTS for content pipelines.

---

## What this project demonstrates

- Ability to **design and deliver** a Windows‑friendly automation pipeline.
- **Config‑first** design: new sites/locations are added via `.env` (no code edits).
- File‑tree parsing and **metadata extraction** (dates, stages, locations).
- **Programmatic PPTX** assembly (`python-pptx`) with consistent slide layouts.
- A small sample dataset and docs for **local reproducibility**.

---

## How it works (short)

1. **Parse** dated folder trees → extract date / stage / location.  
2. **Select / skip**: avoid reprocessing already stamped images.  
3. **Stamp** images with date/time/location overlays.  
4. **Compile** `.pptx` decks (daily and by work type) with stable layouts.

Orchestration entrypoint: `python main.py` (see `photo_reports/pipeline.py` for composition).

---

## Quick start (Windows)

```powershell
python -m venv .venv
.\.venv\Scriptsctivate
pip install -r requirements.txt
copy .env.example .env  # set FOLDER_PATH and reporting window
python main.py
```

**Input:** dated photo folders exported from field teams.  
**Output:** stamped JPGs + `.pptx` reports in your workspace.

---

## Key configuration (see `configs/config.py`)

- `FOLDER_PATH` — root directory with source photo archives.
- `REPORT_MODE` — `daily` or `batch`.
- `ENABLED_WORK_TYPE` / `BATCH_WORK_TYPES` — which activities go to reports.
- `START_DATE` / `END_DATE` — reporting window.
- `RAW_IMAGE_MODE`, `PHOTO_LABELS_ENABLED` — filtering and caption toggles.
- `TITLE_CONTENT`, `MONTH_LABELS`, `WORK_PROFILES` — presentation metadata.

`.env.example` ships with an anonymised template mirroring a real maintenance contract.

---

## Sample dataset

```
examples/
|-- reference/
|   `-- ExampleLocation/    # pristine snapshot (read-only)
|-- workspace/
|   `-- ExampleLocation/    # working copy used by the pipeline
`-- source_photos/
    |-- Days/<CN>/...       # daytime shots for populators
    `-- Works/...           # stage-2 shots for in-progress populators
```

- Folder tree mirrors a real naming convention (month → day → construction → stage).
- Excel cache and stamp metadata are pre-generated so the pipeline can verify coverage on the first run.
- Generated `.pptx` files are intentionally excluded from version control (size). Run the pipeline locally to recreate them.
- Utility scripts `utils/random_photo_populator.py` and
  `utils/random_inprogress_photo_populator.py` pull sample images from `examples/source_photos`.

---

## Repository layout

- `configs/` — env parsing, default work profiles, layout constants.
- `photo_reports/filters.py` — image selection/enhancement tuned per device.
- `photo_reports/parser.py` — heuristics for stage detection and address assignment.
- `photo_reports/stamper.py` — overlay engine.
- `photo_reports/pptx_creator.py` — slide authoring and stable layouts.
- `photo_reports/pipeline.py` — orchestration glue for the end‑to‑end run.
- `utils/` — data checks, directory scaffolders, sample data populators.

---

## Scope & status

- Purpose‑built **process automation** around images and reports (CLI).
- **Archived**: kept as a finished, reproducible work sample used in production.

---

## About the author

**Alena Yashkina** — automation‑minded engineer with 9 years in architectural lighting.
Looking for **remote** roles in Python‑based automation and ops tooling. Languages: RU / EN (B2).
