"""
Validate queries.jsonl schema and print a count breakdown.

Run from repo root:
    python -m scripts.evaluation.check_queries

Exits 0 if all records are valid, non-zero otherwise.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUERIES_PATH = ROOT / "data" / "eval" / "queries.jsonl"

REQUIRED_FIELDS   = {"id", "question", "ground_truth", "query_type", "source_type", "source_ids", "notes"}
VALID_QUERY_TYPES  = {"factual", "synthesis", "ambiguous", "unanswerable"}
VALID_SOURCE_TYPES = {"github_issue", "markdown_doc", "pdf_whitepaper", "cross_source", "none"}


def main() -> None:
    if not QUERIES_PATH.exists():
        print(f"Error: {QUERIES_PATH} not found.", file=sys.stderr)
        print("Complete the manual review step and save queries.jsonl first.", file=sys.stderr)
        sys.exit(1)

    records: list[dict] = []
    errors:  list[str]  = []

    with open(QUERIES_PATH) as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as exc:
                errors.append(f"Line {lineno}: JSON parse error — {exc}")
                continue

            rec_id = rec.get("id", f"<line {lineno}>")

            missing = REQUIRED_FIELDS - set(rec.keys())
            if missing:
                errors.append(f"{rec_id}: missing fields: {sorted(missing)}")

            qt = rec.get("query_type")
            if qt not in VALID_QUERY_TYPES:
                errors.append(f"{rec_id}: invalid query_type '{qt}'")

            st = rec.get("source_type")
            if st not in VALID_SOURCE_TYPES:
                errors.append(f"{rec_id}: invalid source_type '{st}'")

            if not isinstance(rec.get("source_ids"), list):
                errors.append(f"{rec_id}: source_ids must be a list")

            if not str(rec.get("question", "")).strip():
                errors.append(f"{rec_id}: question is empty")

            if qt != "ambiguous" and qt != "unanswerable" and not str(rec.get("ground_truth", "")).strip():
                errors.append(f"{rec_id}: ground_truth is empty for {qt} query")

            records.append(rec)

    # ── Count breakdown ───────────────────────────────────────────────────────
    by_type:   dict[str, int] = {}
    by_source: dict[str, int] = {}
    for rec in records:
        qt = rec.get("query_type", "unknown")
        st = rec.get("source_type", "unknown")
        by_type[qt]   = by_type.get(qt, 0)   + 1
        by_source[st] = by_source.get(st, 0) + 1

    print(f"Total records: {len(records)}")
    print()
    print("By query_type:")
    for qt in ("factual", "synthesis", "ambiguous", "unanswerable"):
        n = by_type.get(qt, 0)
        flag = "" if n > 0 else "  ← MISSING"
        print(f"  {qt:<20s}: {n:3d}{flag}")

    print()
    print("By source_type:")
    for st in ("github_issue", "markdown_doc", "pdf_whitepaper", "cross_source", "none"):
        n = by_source.get(st, 0)
        if n:
            print(f"  {st:<20s}: {n:3d}")

    # ── Errors ────────────────────────────────────────────────────────────────
    if errors:
        print(f"\n{len(errors)} error(s) found:", file=sys.stderr)
        for err in errors:
            print(f"  {err}", file=sys.stderr)
        sys.exit(1)

    print()
    target = 90
    if len(records) == target:
        print(f"All {len(records)} records valid. Ready for evaluation run.")
    else:
        delta = target - len(records)
        direction = "need" if delta > 0 else "have"
        print(f"All records valid. {len(records)}/{target} — {direction} {abs(delta)} more.")
    sys.exit(0)


if __name__ == "__main__":
    main()
