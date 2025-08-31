# Async Site Scraper

A production-grade, polite, **async** Python 3.11+ scraper that crawls a list of websites (BFS), extracts readable page text, downloads PDFs/Word docs, records external URLs, and logs everything.

## Features

- ✅ BFS crawl with robust URL normalization & de-duplication  
- ✅ Scope control by **registrable domain** (with optional subdomains)  
- ✅ Seeds from **robots.txt** and discovered **sitemaps**  
- ✅ Extracts readable text (removes script/style/nav/footer)  
- ✅ Downloads **PDFs** and **Word docs** (`.doc`, `.docx`)  
- ✅ Collects **external URLs** (unique & sorted)  
- ✅ Respects **robots.txt** (configurable) with retries & backoff  
- ✅ Site-specific results folder with logs, pages, files, and visited set  
- ✅ Minimal, pinned dependencies

## Installation

> Requires **Python 3.11+**.

```bash
# 1) Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # On Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run application
python scraper.py --config config.yaml
```