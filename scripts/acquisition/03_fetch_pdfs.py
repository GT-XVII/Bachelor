#!/usr/bin/env python3
"""Download CNCF/LF PDF whitepapers and extract text sections."""

import json
import re
import requests
import pdfplumber
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "data/raw/pdfs"
CORPUS_DIR  = BASE_DIR / "data/corpus"

BASE_TAG_SECURITY   = "https://github.com/cncf/tag-security/raw/main"
BASE_TAG_APP        = "https://github.com/cncf/tag-app-delivery/raw/main"
BASE_TAG_STORAGE    = "https://raw.githubusercontent.com/cncf/tag-storage/master"
BASE_FOUNDATION     = "https://github.com/cncf/foundation/raw/main"
BASE_WG_SERVERLESS  = "https://github.com/cncf/wg-serverless/raw/master"
BASE_TOC            = "https://github.com/cncf/toc/raw/main"

PDFS = [
    # ── tag-security ──────────────────────────────────────────────────────────
    {
        "name": "cncf-cloud-native-security-whitepaper-v2",
        "title": "CNCF Cloud Native Security Whitepaper v2",
        "url": f"{BASE_TAG_SECURITY}/community/resources/security-whitepaper/v2/CNCF_cloud-native-security-whitepaper-May2022-v2.pdf",
        "published": "2022-05",
    },
    {
        "name": "cncf-supply-chain-security-whitepaper-v2",
        "title": "CNCF Software Supply Chain Best Practices Whitepaper v2",
        "url": f"{BASE_TAG_SECURITY}/community/working-groups/supply-chain-security/supply-chain-security-paper-v2/Software_Supply_Chain_Practices_whitepaper_v2.pdf",
        "published": "2022",
    },
    {
        "name": "cncf-supply-chain-security-whitepaper-v1",
        "title": "CNCF Software Supply Chain Best Practices Whitepaper v1",
        "url": f"{BASE_TAG_SECURITY}/community/working-groups/supply-chain-security/supply-chain-security-paper/CNCF_SSCP_v1.pdf",
        "published": "2021",
    },
    {
        "name": "cncf-security-fuzzing-handbook",
        "title": "CNCF Security Fuzzing Handbook",
        "url": f"{BASE_TAG_SECURITY}/community/resources/security-fuzzing-handbook/handbook-fuzzing.pdf",
        "published": "2022",
    },
    {
        "name": "cncf-open-and-secure-assessment",
        "title": "CNCF Open and Secure: A Manual for Practicing Threat Modeling",
        "url": f"{BASE_TAG_SECURITY}/community/assessments/Open_and_Secure.pdf",
        "published": "2021",
    },
    # ── tag-storage ───────────────────────────────────────────────────────────
    {
        "name": "cncf-storage-whitepaper-v2",
        "title": "CNCF Cloud Native Storage Whitepaper V2",
        "url": f"{BASE_TAG_STORAGE}/CNCF%20Storage%20Whitepaper%20V2.pdf",
        "published": "2023",
    },
    {
        "name": "cncf-cloud-native-disaster-recovery-v2",
        "title": "CNCF Cloud Native Disaster Recovery Whitepaper v2",
        "url": f"{BASE_TAG_STORAGE}/cloud-native-disaster-recovery-whitepaper/Cloud%20Native%20Disaster%20Recovery%20v2.pdf",
        "published": "2023",
    },
    # ── tag-app-delivery ──────────────────────────────────────────────────────
    {
        "name": "cncf-operator-whitepaper-v1",
        "title": "CNCF Operator Whitepaper v1",
        "url": f"{BASE_TAG_APP}/operator-whitepaper/v1/CNCF_Operator_WhitePaper_v1-0_20210715.pdf",
        "published": "2021-07",
    },
    {
        "name": "cncf-platforms-whitepaper-latest",
        "title": "CNCF Platforms Whitepaper",
        "url": f"{BASE_TAG_APP}/platforms-whitepaper/latest/assets/platforms-def-latest.pdf",
        "published": "2023",
    },
    {
        "name": "cncf-platform-engineering-maturity-model",
        "title": "CNCF Platform Engineering Maturity Model v1",
        "url": f"{BASE_TAG_APP}/platforms-maturity-model/v1/assets/platform-eng-maturity-model-v1.0.pdf",
        "published": "2023",
    },
    # ── foundation ────────────────────────────────────────────────────────────
    {
        "name": "cncf-strategy-2024",
        "title": "CNCF Strategy 2024",
        "url": f"{BASE_FOUNDATION}/documents/2024%20CNCF%20Strategy.pdf",
        "published": "2024",
    },
    {
        "name": "cncf-strategy-2023",
        "title": "CNCF Strategy 2023",
        "url": f"{BASE_FOUNDATION}/documents/2023%20CNCF%20Strategy.pdf",
        "published": "2023",
    },
    # ── wg-serverless ─────────────────────────────────────────────────────────
    {
        "name": "cncf-serverless-whitepaper-v1",
        "title": "CNCF Serverless Whitepaper v1",
        "url": f"{BASE_WG_SERVERLESS}/whitepapers/serverless-overview/cncf_serverless_whitepaper_v1.0.pdf",
        "published": "2018",
    },
    # ── toc ───────────────────────────────────────────────────────────────────
    {
        "name": "cncf-ai-assisted-development-survey-2024",
        "title": "CNCF State of AI-Assisted Development Survey 2024",
        "url": f"{BASE_TOC}/tags/tag-developer-experience/initiatives/state-of-ai-assisted-development/assets/survey-results/survey%20results%20report.pdf",
        "published": "2024",
    },
]

MIN_PAGE_CHARS = 100
MAX_CONTENT    = 8000
PAGES_PER_SECTION = 4   # sliding window — keeps sections coherent and retrievable


def is_junk_page(text):
    """Discard cover pages, ToC pages, and near-blank pages."""
    if not text or len(text) < MIN_PAGE_CHARS:
        return True
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return True
    # ToC heuristic: most lines are very short (page references, dot leaders)
    short = sum(1 for l in lines if len(l) < 50)
    return len(lines) > 4 and (short / len(lines)) > 0.70


def extract_sections(pages):
    """
    Split a document into overlapping page-window sections.
    More reliable than heading detection across diverse PDF styles.
    """
    good = [(num, txt) for num, txt in pages if not is_junk_page(txt)]
    if not good:
        return []

    sections = []
    step = max(1, PAGES_PER_SECTION - 1)   # 1-page overlap between windows
    for i in range(0, len(good), step):
        window = good[i : i + PAGES_PER_SECTION]
        content = "\n\n".join(txt for _, txt in window)
        page_nums = [num for num, _ in window]
        # Use first non-trivial line of the window as section title
        first_lines = [l.strip() for l in window[0][1].splitlines() if l.strip()]
        title = first_lines[0][:80] if first_lines else f"Section {len(sections)+1}"
        sections.append((title, content, page_nums))

    return sections


def download_pdf(pdf_info):
    dest = OUTPUT_DIR / f"{pdf_info['name']}.pdf"
    if dest.exists():
        print(f"  Already downloaded: {dest.name}")
        return dest

    print(f"  Downloading {pdf_info['name']}...")
    try:
        resp = requests.get(
            pdf_info["url"], timeout=90,
            headers={"User-Agent": "Mozilla/5.0 (academic research)"},
            allow_redirects=True,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"  FAILED: {e}")
        return None

    dest.write_bytes(resp.content)
    print(f"  {len(resp.content)//1024} KB → {dest.name}")
    return dest


def process_pdf(pdf_info):
    dest = download_pdf(pdf_info)
    if dest is None:
        return []

    try:
        with pdfplumber.open(dest) as pdf:
            pages = [(i + 1, page.extract_text() or "") for i, page in enumerate(pdf.pages)]
    except Exception as e:
        print(f"  Text extraction failed: {e}")
        return []

    sections = extract_sections(pages)
    records = []

    for idx, (title, content, page_nums) in enumerate(sections):
        if len(content) < MIN_PAGE_CHARS:
            continue
        page_range = f"{min(page_nums)}–{max(page_nums)}" if page_nums else "?"
        records.append({
            "source_type": "pdf_whitepaper",
            "id": f"{pdf_info['name']}/section-{idx+1}",
            "title": f"{pdf_info['title']} — {title}",
            "content": content[:MAX_CONTENT],
            "metadata": {
                "document": pdf_info["title"],
                "url": pdf_info["url"],
                "page_range": page_range,
                "published": pdf_info["published"],
            },
        })

    return records


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    all_records = []

    for pdf_info in PDFS:
        print(f"\n{pdf_info['title']}")
        records = process_pdf(pdf_info)
        print(f"  → {len(records)} sections extracted")
        all_records.extend(records)

    corpus = CORPUS_DIR / "pdf_whitepapers.jsonl"
    with open(corpus, "w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")
    print(f"\nDone. {len(all_records)} total sections → {corpus}")


if __name__ == "__main__":
    main()
