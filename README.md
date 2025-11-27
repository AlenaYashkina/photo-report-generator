# Photo Report Generator — production photo-report automation (archived)

A Python CLI pipeline that turns dated photo folders into **stamped images** and **ready PowerPoint reports**.  
Built for architectural / festive lighting maintenance, where photographers bring back thousands of photos every time the city installs decorative lighting for public holidays.

> Repository is frozen. No new features planned; **actively used in production** and kept as a work sample.

---

## Problem it solves

During festive lighting periods, photographers walk through **all addresses in the city** every day and take photos of each decorative construction.

If they notice an issue — **deep inspection**, **partial repair**, **slow blinking**, **flags**, **cleaning**, etc. — that photo is marked as **detected**.

A few hours later a maintenance crew goes to the same location and fixes the issue.  
This produces two report types:

- **Daily status reports** — “everything works today”, grouped by address  
- **Work-done reports** — full chain: **detected → in_progress → fixed**

Field photos come with **no EXIF time/location**, so the office previously had to:

1. Open every photo manually  
2. Add a watermark with **date, time, location** (derived from folder structure/logs)  
3. Sort photos by **object**, **issue type**, **fix stage**  
4. Insert them into PowerPoint in the correct order (**detected → in_progress → fixed**)  
5. Build daily decks and work-done decks for the client  

Manual Photoshop + PowerPoint took **hundreds of hours per large address** and constantly produced timestamp/ordering mistakes.

This tool fully automated the workflow.

---

## What the tool does

Given a root folder with daily photo drops, the tool:

- scans all dated folders within a reporting window  
- uses folder naming + a small metadata table to match each photo to  
  **object**, **issue type**, **fix stage** (`detected`, `in_progress`, `fixed`)  
- generates **stamped JPGs** with date, time and location (as watermarks)  
- builds a **normalized folder structure** grouped by object / issue / stage  
- assembles **PowerPoint decks**:  
  - **daily reports**  
  - **work-done reports** showing the full chain: detected → process → fixed  

---

## Impact

On a real festive-lighting contract (~9,000 photos per peak month):

- Manual reporting: **150–225 hours/month**  
- Automated: **4–8 hours/month** (mostly reviewing)

→ Saves **3–5 full work weeks every month**  
→ Eliminates timestamp and ordering mistakes  
→ Produces consistent, client-ready PPTX decks

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

Built by **Alena Yashkina** — lighting engineer turned AI-automation developer.  
More info: GitHub profile / LinkedIn.
