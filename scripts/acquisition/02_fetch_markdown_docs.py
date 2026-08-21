#!/usr/bin/env python3
"""Sparse-clone and process Markdown/RST documentation repos."""

import json
import re
import subprocess
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "data/raw/markdown_docs"
CORPUS_DIR  = BASE_DIR / "data/corpus"

REPOS = [
    {
        "name": "kubernetes-website",
        "url": "https://github.com/kubernetes/website.git",
        "sparse_dirs": ["content/en/docs"],
        "extensions": [".md"],
        "base_url": "https://kubernetes.io/docs/",
    },
    {
        "name": "ansible-documentation",
        "url": "https://github.com/ansible/ansible-documentation.git",
        "sparse_dirs": ["docs"],
        "extensions": [".rst", ".md"],
        "base_url": "https://docs.ansible.com/ansible/latest/",
    },
    {
        "name": "cncf-tag-observability",
        "url": "https://github.com/cncf/tag-observability.git",
        "sparse_dirs": [],          # full clone — small repo, content at root and working-groups/
        "extensions": [".md"],
        "base_url": "https://github.com/cncf/tag-observability/blob/main/",
        "skip_paths_extra": {"themes", "layouts", "i18n", "static", "assets", "website"},
    },
]

MIN_CONTENT  = 500   # characters after stripping
MAX_CONTENT  = 8000  # cap to keep records manageable
SKIP_PATHS   = {"_includes", "_layouts", "assets", "node_modules", ".git",
                "_vendor", "static", "images", "img", "js", "css"}


def strip_markup(text):
    """Strip Markdown and RST markup to readable plain text."""
    # YAML front matter
    text = re.sub(r'^---\n.*?\n---\n', '', text, flags=re.DOTALL)
    # Fenced code blocks
    text = re.sub(r'```[^\n]*\n.*?```', ' ', text, flags=re.DOTALL)
    text = re.sub(r'`[^`\n]+`', ' ', text)
    # HTML tags and shortcodes
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\{\{[^}]+\}\}', '', text)
    # Markdown links → keep label
    text = re.sub(r'!\[[^\]]*\]\([^\)]+\)', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Markdown headings
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # RST directives and roles
    text = re.sub(r'\.\.\s+\w[^:]*::[^\n]*', '', text)
    text = re.sub(r'^\s+:[\w-]+:.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r':\w+:`([^`]+)`', r'\1', text)
    # RST section underlines
    text = re.sub(r'^[=\-~\^#\*\+\.]{3,}\s*$', '', text, flags=re.MULTILINE)
    # Collapse blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_title(raw_text, filepath):
    m = re.search(r'^#{1,2}\s+(.+)$', raw_text, re.MULTILINE)
    if m:
        return m.group(1).strip()
    m = re.search(r'^(.+)\n[=\-]{3,}$', raw_text, re.MULTILINE)
    if m:
        return m.group(1).strip()
    return Path(filepath).stem.replace('-', ' ').replace('_', ' ').title()


def clone_repo(repo_info, clone_base):
    repo_dir = clone_base / repo_info["name"]
    if repo_dir.exists() and any(repo_dir.iterdir()):
        print(f"  {repo_info['name']} already cloned — skipping clone step.")
        return repo_dir

    sparse_dirs = repo_info.get("sparse_dirs", [])
    if sparse_dirs:
        print(f"  Cloning {repo_info['url']} (sparse, depth=1)...")
        subprocess.run([
            "git", "clone", "--depth=1", "--filter=blob:none", "--sparse",
            repo_info["url"], str(repo_dir)
        ], check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", str(repo_dir), "sparse-checkout", "set"] + sparse_dirs,
            check=True, capture_output=True,
        )
    else:
        print(f"  Cloning {repo_info['url']} (full, depth=1)...")
        subprocess.run([
            "git", "clone", "--depth=1", repo_info["url"], str(repo_dir)
        ], check=True, capture_output=True)

    print(f"  Done.")
    return repo_dir


def process_repo(repo_info, clone_base):
    repo_dir = clone_repo(repo_info, clone_base)
    all_files = []
    for ext in repo_info["extensions"]:
        all_files.extend(repo_dir.rglob(f"*{ext}"))

    print(f"  {len(all_files)} files found — filtering...")
    skip = SKIP_PATHS | set(repo_info.get("skip_paths_extra", set()))
    records = []

    for filepath in sorted(all_files):
        parts = set(filepath.parts)
        if parts & skip:
            continue
        if filepath.stat().st_size < 1024:
            continue

        try:
            raw = filepath.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        title   = extract_title(raw, filepath)
        content = strip_markup(raw)

        if len(content) < MIN_CONTENT:
            continue

        rel  = filepath.relative_to(repo_dir)
        record = {
            "source_type": "markdown_doc",
            "id": f"{repo_info['name']}/{rel.with_suffix('')}",
            "title": title,
            "content": content[:MAX_CONTENT],
            "metadata": {
                "repo": repo_info["url"].removeprefix("https://github.com/").removesuffix(".git"),
                "file_path": str(rel),
                "base_url": repo_info["base_url"],
            },
        }
        records.append(record)

    return records


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    clone_base = OUTPUT_DIR / "clones"
    clone_base.mkdir(exist_ok=True)

    all_records = []

    for repo_info in REPOS:
        print(f"\nProcessing {repo_info['name']}...")
        records = process_repo(repo_info, clone_base)
        print(f"  {len(records)} qualifying documents")

        out = OUTPUT_DIR / f"{repo_info['name']}_docs.jsonl"
        with open(out, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"  Saved → {out.name}")
        all_records.extend(records)

    corpus = CORPUS_DIR / "markdown_docs.jsonl"
    with open(corpus, "w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")
    print(f"\nDone. {len(all_records)} total docs → {corpus}")


if __name__ == "__main__":
    main()
