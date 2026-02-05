#!/usr/bin/env python3
# pylint: disable=direct-environment-variable-reference
"""Collect "prompt is too long" errors from LangSmith.

Queries LangSmith for error runs and extracts trace information
for downstream benchmark dataset building.

Usage:
    poetry run python scripts/collect_prompt_too_long_errors_from_langsmith.py \
--project-id <langsmith-project-id> \
--start-date 2025-11-01 \
--out scripts/prompt_too_long_production_errors.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

PROMPT_TOO_LONG_RE = re.compile(
    r"prompt is too long:\s*(\d+)\s*tokens?\s*>\s*(\d+)", re.IGNORECASE
)


class LangSmithClient:
    def __init__(self, api_key: str, api_url: str = "https://api.smith.langchain.com"):
        self.api_key = api_key
        self.api_url = api_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "X-API-Key": self.api_key,
                "Content-Type": "application/json",
            }
        )

    def fetch_error_runs(
        self,
        project_id: str,
        start_time: datetime,
        limit: int = 100,
        cursor: str | None = None,
    ):
        """Fetch runs with errors from the project."""
        url = f"{self.api_url}/runs/query"
        start_iso = start_time.isoformat().replace("+00:00", "Z")

        payload = {
            "session": [project_id],
            "error": True,
            "filter": f'gte(start_time, "{start_iso}")',
            "limit": limit,
            "select": ["id", "trace_id", "name", "error", "extra"],
        }
        if cursor:
            payload["cursor"] = cursor

        try:
            resp = self.session.post(url, json=payload, timeout=60)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                print("Rate limited - waiting 30s and retrying...", file=sys.stderr)
                time.sleep(30)
                return self.fetch_error_runs(
                    project_id, start_time, limit, cursor
                )  # Retry

            print(f"Error {resp.status_code}: {resp.text[:200]}", file=sys.stderr)
            return {"runs": [], "cursors": {}}
        except Exception as e:
            print(f"Exception: {e}", file=sys.stderr)
            return {"runs": [], "cursors": {}}


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(
        description="Collect 'prompt is too long' errors from LangSmith"
    )
    ap.add_argument(
        "--project-id",
        required=True,
        help="LangSmith project ID (UUID)",
    )
    ap.add_argument(
        "--start-date", required=True, help='Start date (e.g. "2025-01-01")'
    )
    ap.add_argument("--out", default="prompt_too_long_errors.jsonl")
    ap.add_argument("--limit-per-page", type=int, default=100)
    ap.add_argument(
        "--max-pages",
        type=int,
        default=100,
        help="Max pages to fetch (0 for unlimited)",
    )
    args = ap.parse_args()

    api_key = os.environ.get("LANGCHAIN_API_KEY")
    if not api_key:
        print("Missing LANGCHAIN_API_KEY environment variable", file=sys.stderr)
        return 1

    # Parse start date
    s = args.start_date.strip()
    start_dt = datetime.fromisoformat(s if "T" in s else s + "T00:00:00+00:00")
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)

    client = LangSmithClient(api_key=api_key)

    total_runs = 0
    matching_runs = 0
    page = 0
    cursor = None

    print(f"Fetching error runs from {start_dt.date()}", file=sys.stderr)

    with open(args.out, "w", encoding="utf-8") as f:
        while True:
            page += 1
            if args.max_pages and page > args.max_pages:
                print(f"Reached max pages ({args.max_pages})", file=sys.stderr)
                break

            data = client.fetch_error_runs(
                args.project_id, start_dt, limit=args.limit_per_page, cursor=cursor
            )
            runs = data.get("runs", [])

            if not runs:
                print("No more runs", file=sys.stderr)
                break

            total_runs += len(runs)
            print(
                f"Page {page}: {len(runs)} runs (total: {total_runs})", file=sys.stderr
            )

            for run in runs:
                error_text = run.get("error") or ""
                match = PROMPT_TOO_LONG_RE.search(error_text)

                if match:
                    matching_runs += 1
                    extra = run.get("extra", {}) or {}
                    invocation_params = extra.get("invocation_params", {}) or {}

                    record = {
                        "trace_id": run.get("trace_id"),
                        "run_id": run.get("id"),
                        "name": run.get("name"),
                        "tokens_reported": int(match.group(1)),
                        "max_tokens": int(match.group(2)),
                        "model": invocation_params.get("model"),
                        "error": error_text[:500],  # Truncated for reviewability
                    }
                    f.write(json.dumps(record) + "\n")

            cursor = data.get("cursors", {}).get("next")
            if not cursor:
                print("No more pages", file=sys.stderr)
                break

            time.sleep(1.0)  # Rate limiting

    print(
        json.dumps(
            {
                "output_file": args.out,
                "total_runs_scanned": total_runs,
                "prompt_too_long_matches": matching_runs,
            },
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
