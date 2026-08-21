#!/usr/bin/env python3
"""Fetch closed GitHub issues from kubernetes/kubernetes and microsoft/vscode."""

import json
import time
import sys
import requests
from pathlib import Path

GITHUB_TOKEN = "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"
BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "data/raw/github_issues"
CORPUS_DIR  = BASE_DIR / "data/corpus"

REPOS = [
    ("kubernetes", "kubernetes"),
    ("microsoft",  "vscode"),
]

TARGET_PER_REPO = 100
MIN_BODY_LEN    = 200
EXCLUDE_LABELS  = {"duplicate", "invalid", "wontfix"}

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}


def check_token():
    resp = requests.get("https://api.github.com/rate_limit", headers=HEADERS, timeout=10)
    if resp.status_code != 200:
        print("ERROR: GitHub token invalid or unreachable.", file=sys.stderr)
        sys.exit(1)
    remaining = resp.json()["resources"]["core"]["remaining"]
    print(f"GitHub API: {remaining} requests remaining this hour.")


def api_get(url, params=None):
    while True:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if resp.status_code == 403:
            reset = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait = max(reset - time.time(), 0) + 5
            print(f"  Rate limited — waiting {wait:.0f}s...")
            time.sleep(wait)
            continue
        return resp


def fetch_issues(owner, repo):
    issues = []
    page = 1
    print(f"\nFetching issues from {owner}/{repo}...")
    while len(issues) < TARGET_PER_REPO:
        resp = api_get(
            f"https://api.github.com/repos/{owner}/{repo}/issues",
            params={"state": "closed", "per_page": 100, "page": page,
                    "sort": "updated", "direction": "desc"},
        )
        if resp.status_code != 200:
            print(f"  HTTP {resp.status_code} — stopping.")
            break
        batch = resp.json()
        if not batch:
            break
        for issue in batch:
            if "pull_request" in issue:
                continue
            body = issue.get("body") or ""
            if len(body) < MIN_BODY_LEN:
                continue
            if issue.get("comments", 0) < 1:
                continue
            labels = {l["name"].lower() for l in issue.get("labels", [])}
            if labels & EXCLUDE_LABELS:
                continue
            issues.append(issue)
            if len(issues) >= TARGET_PER_REPO:
                break
        page += 1
        time.sleep(0.3)
    return issues[:TARGET_PER_REPO]


def fetch_comments(owner, repo, number):
    resp = api_get(
        f"https://api.github.com/repos/{owner}/{repo}/issues/{number}/comments",
        params={"per_page": 50},
    )
    if resp.status_code == 200:
        return resp.json()
    return []


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    check_token()

    all_records = []

    for owner, repo in REPOS:
        issues = fetch_issues(owner, repo)
        print(f"  {len(issues)} qualifying issues")

        records = []
        for i, issue in enumerate(issues):
            number = issue["number"]
            print(f"  [{i+1:3}/{len(issues)}] #{number} — fetching comments...")
            comments = fetch_comments(owner, repo, number)
            comment_texts = [c["body"] for c in comments if c.get("body")]

            parts = [issue.get("body", "")]
            if comment_texts:
                parts.append("---")
                parts.extend(comment_texts)

            record = {
                "source_type": "github_issue",
                "id": f"{owner}-{repo}#{number}",
                "title": issue["title"],
                "content": "\n\n".join(p for p in parts if p),
                "metadata": {
                    "repo": f"{owner}/{repo}",
                    "url": issue["html_url"],
                    "labels": [l["name"] for l in issue.get("labels", [])],
                    "state": issue["state"],
                    "created_at": issue["created_at"],
                    "comment_count": issue.get("comments", 0),
                },
            }
            records.append(record)
            all_records.append(record)
            time.sleep(0.2)

        out = OUTPUT_DIR / f"{owner}_{repo}_issues.jsonl"
        with open(out, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"  Saved {len(records)} records → {out.name}")

    corpus = CORPUS_DIR / "github_issues.jsonl"
    with open(corpus, "w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")
    print(f"\nDone. {len(all_records)} total issues → {corpus}")


if __name__ == "__main__":
    main()
