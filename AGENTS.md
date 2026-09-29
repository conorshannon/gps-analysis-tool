# AGENTS.md

## Project Overview
Single-file Streamlit app for sport GPS data analysis. The entire application
lives in `Qwen_python_20260916_959e2d74d.py`. Dependencies are in `requirements.txt`.

## Stack
- **Framework**: Streamlit (single-file app)
- **Data**: SQLite (local file at `data/sport_gps.db`, set via `SPORT_GPS_DB` env var)
- **Libraries**: pandas, numpy, plotly

## Running
```
docker compose -f docker-compose.base44.yml up -d --build
```
The app serves on port 3000. Streamlit runs with `runOnSave=true` so edits to the
Python file auto-rerun in the browser.

## No External Credentials
All data is local SQLite. No external services, API keys, or credentials needed.

## Verifying
- Health check: `curl http://localhost:3000/_stcore/health` → `ok`
- The app has a built-in "Load demo data" button (Import tab) that generates 6 weeks
  of synthetic GPS data for 11 players, useful for testing all features.
