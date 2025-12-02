# Photo Report Generator — production photo-report automation (archived)

A Python CLI pipeline that turns dated photo folders into **stamped images** and **ready PowerPoint reports**.  
Built for architectural / festive lighting maintenance, where photographers bring back thousands of photos every time the city installs decorative lighting for public holidays.

> Repository is frozen. No new features planned; **actively used in production** and kept as a work sample.

---

## Problem it solves

Every festive lighting season we get **thousands of raw photos** from daily walk-throughs and separate repair jobs.  
Before this tool, the office had to:

- manually list all photos in Excel
- figure out **day, construction number, address, work type, issue stage** from folder names
- invent realistic **date & time** for each shot
- add watermarks in Photoshop
- build PowerPoint reports for:
  - daily walk-throughs (all constructions “as of today”)
  - repair chains: **detected → in_progress → fixed** for each issue and work type

This easily turned into **dozens of hours of routine** on every big project.

---

## What the tool does

Given a root folder with photo drops, the tool:

- walks all subfolders and collects photos from the input tree  
- parses metadata from the path (day, construction number, work type, issue stage)  
- writes everything into an **Excel table** (main control surface: can be edited by a non-technical user)
- generates realistic **timestamps and locations** from a config and fills missing date/time
- automatically rolls over to the **next day** if the time range crosses midnight
- stamps each photo with a watermark: **date, time, location / address**
- builds **PowerPoint decks**:
  - daily reports for walk-throughs
  - work-done reports with chains like *detected → in_progress → fixed* per construction / work type

In practice the user only selects the photos and, if needed, tweaks the Excel file —  
**all stamping and PPTX assembly is automated.**

---

## Impact

On a real festive-lighting contract (~9,000 photos per peak month):

- manual reporting: **150–225 hours/month**
- automated with this tool: **4–8 hours/month** (quality check only)

This saves **3–5 full work weeks every month** and keeps reports consistent and client-ready.

---

## Usage

```bash
pip install -r requirements.txt
python main.py --config config.json
```

---

## Repository layout

```
photo-report-generator/
├── config/
├── input/
├── output/
├── src/
│   ├── watermark.py
│   ├── grouping.py
│   ├── ppt_builder.py
│   └── utils/
└── README.md
```

---

## About the author

Built by **Alena Yashkina** — lighting engineer turned AI‑automation developer.  
Portfolio and contact links:

- GitHub: https://github.com/AlenaYashkina
- LinkedIn: https://www.linkedin.com/in/alena-yashkina-a9994a35a/
