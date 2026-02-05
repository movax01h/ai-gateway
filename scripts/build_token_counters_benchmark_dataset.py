#!/usr/bin/env python3
# pylint: disable=direct-environment-variable-reference
"""Create benchmark dataset from LangSmith error traces.

Takes the JSONL output from collect_prompt_too_long_errors.py and enriches it
with the full message payloads needed for token counter benchmarking.

Outputs in LangSmith Dataset format ({inputs: {...}, outputs: {...}}) which can be:
- Uploaded to LangSmith Datasets UI for evaluation
- Used directly by the benchmark evaluation script

Usage:
    python build_token_counters_benchmark_dataset.py \
--input prompt_too_long_production_errors.jsonl \
--output benchmark_dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import requests
from dotenv import load_dotenv


@dataclass
class BenchmarkRecord:
    trace_id: str
    run_id: str
    model: str | None
    anthropic_reported_tokens: int
    max_context_tokens: int
    messages_lc: list[Any]  # Raw LangChain serialization format
    tools: list[Any]  # Tool definitions from invocation_params
    content_stats: dict[str, Any]
    fetch_error: str | None = None


class LangSmithClient:
    def __init__(self, api_key: str, api_url: str = "https://api.smith.langchain.com"):
        self.api_key = api_key
        self.api_url = api_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"X-API-Key": self.api_key})

    def fetch_run(self, run_id: str) -> dict[str, Any] | None:
        """Fetch a single run by ID."""
        url = f"{self.api_url}/runs/{run_id}"
        try:
            resp = self.session.get(url, timeout=60)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                print("  Rate limited, waiting 30s...", file=sys.stderr)
                time.sleep(30)
                return self.fetch_run(run_id)  # Retry
            print(
                f"  Error fetching run {run_id}: {resp.status_code}",
                file=sys.stderr,
            )
            return None
        except Exception as e:
            print(f"  Exception fetching run {run_id}: {e}", file=sys.stderr)
            return None

    def fetch_run_with_extras(
        self, trace_id: str, run_name: str = "ChatAnthropic", status: str = "error"
    ) -> dict[str, Any] | None:
        """Fetch a run by trace with extra fields including tools."""
        url = f"{self.api_url}/runs/query"
        payload = {
            "trace": trace_id,
            "filter": f'and(eq(name, "{run_name}"), eq(status, "{status}"))',
            "select": ["id", "inputs", "extra"],
            "limit": 1,
        }
        try:
            resp = self.session.post(url, json=payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                runs = data.get("runs", [])
                return runs[0] if runs else None
            if resp.status_code == 429:
                print("  Rate limited, waiting 30s...", file=sys.stderr)
                time.sleep(30)
                return self.fetch_run_with_extras(trace_id, run_name, status)  # Retry

            print(
                f"  Error querying runs for trace {trace_id}: {resp.status_code}",
                file=sys.stderr,
            )
            return None
        except Exception as e:
            print(
                f"  Exception querying runs for trace {trace_id}: {e}", file=sys.stderr
            )
            return None


def compute_content_stats(messages_lc: list[Any]) -> dict[str, Any]:
    """Compute statistics about the message content."""
    total_chars = 0
    by_role: dict[str, dict[str, int]] = {}

    for msg in messages_lc:
        msg_id = msg.get("id", [])
        role = msg_id[-1] if msg_id else "Unknown"
        kwargs = msg.get("kwargs", {})
        content = kwargs.get("content", "")
        if isinstance(content, str):
            char_count = len(content)
        elif isinstance(content, list):
            char_count = len(json.dumps(content))
        else:
            char_count = 0
        total_chars += char_count
        if role not in by_role:
            by_role[role] = {"count": 0, "chars": 0}
        by_role[role]["count"] += 1
        by_role[role]["chars"] += char_count

    return {
        "total_chars": total_chars,
        "num_messages": len(messages_lc),
        "by_role": by_role,
    }


def find_failing_llm_run_id(error_records: list[dict]) -> str | None:
    """Find the ChatAnthropic run ID from a group of error records for same trace."""
    for record in error_records:
        if record.get("name") == "ChatAnthropic":
            return record.get("run_id")
    # Fallback to first record if no ChatAnthropic found
    return error_records[0].get("run_id") if error_records else None


def process_trace(
    client: LangSmithClient,
    trace_id: str,
    error_records: list[dict],
) -> BenchmarkRecord | None:
    """Process a single trace and extract benchmark data."""

    # Find the failing ChatAnthropic run
    run_id = find_failing_llm_run_id(error_records)
    if not run_id:
        print(f"  No run_id found for trace {trace_id}", file=sys.stderr)
        return None

    # Get metadata from the first error record
    first_record = error_records[0]
    tokens_reported = first_record.get("tokens_reported", 0)
    max_tokens = first_record.get("max_tokens", 0)
    model = first_record.get("model")

    # Try to find model from any record if first doesn't have it
    if not model:
        for r in error_records:
            if r.get("model"):
                model = r.get("model")
                break

    # Fetch the full run data with extras (includes tools)
    run_data = client.fetch_run_with_extras(trace_id)
    if not run_data:
        # Fallback to simple fetch
        run_data = client.fetch_run(run_id)

    if not run_data:
        return BenchmarkRecord(
            trace_id=trace_id,
            run_id=run_id,
            model=model,
            anthropic_reported_tokens=tokens_reported,
            max_context_tokens=max_tokens,
            messages_lc=[],
            tools=[],
            content_stats={},
            fetch_error="Failed to fetch run data",
        )

    # Extract tools from extra.invocation_params
    extra = run_data.get("extra", {}) or {}
    invocation_params = extra.get("invocation_params", {}) or {}
    tools = invocation_params.get("tools", []) or []

    # Extract messages from inputs
    inputs = run_data.get("inputs", {})
    if not inputs:
        return BenchmarkRecord(
            trace_id=trace_id,
            run_id=run_id,
            model=model,
            anthropic_reported_tokens=tokens_reported,
            max_context_tokens=max_tokens,
            messages_lc=[],
            tools=tools,
            content_stats={},
            fetch_error="No inputs in run data",
        )

    # Messages are at inputs.messages[0] in LangChain format
    messages_wrapper = inputs.get("messages", [])
    if not messages_wrapper:
        return BenchmarkRecord(
            trace_id=trace_id,
            run_id=run_id,
            model=model,
            anthropic_reported_tokens=tokens_reported,
            max_context_tokens=max_tokens,
            messages_lc=[],
            tools=tools,
            content_stats={},
            fetch_error="No messages in inputs",
        )

    # The actual messages list is nested inside
    messages_lc = messages_wrapper[0] if messages_wrapper else []

    if not isinstance(messages_lc, list):
        return BenchmarkRecord(
            trace_id=trace_id,
            run_id=run_id,
            model=model,
            anthropic_reported_tokens=tokens_reported,
            max_context_tokens=max_tokens,
            messages_lc=[],
            tools=tools,
            content_stats={},
            fetch_error=f"Unexpected messages format: {type(messages_lc)}",
        )

    # Compute content statistics
    content_stats = compute_content_stats(messages_lc)

    return BenchmarkRecord(
        trace_id=trace_id,
        run_id=run_id,
        model=model,
        anthropic_reported_tokens=tokens_reported,
        max_context_tokens=max_tokens,
        messages_lc=messages_lc,
        tools=tools,
        content_stats=content_stats,
    )


def to_langsmith_format(record: BenchmarkRecord) -> dict:
    """Convert a BenchmarkRecord to LangSmith Dataset format.

    LangSmith expects: {"inputs": {...}, "outputs": {...}}
    - inputs: the data your evaluation will process
    - outputs: the ground truth / expected values
    """
    return {
        "inputs": {
            "trace_id": record.trace_id,
            "run_id": record.run_id,
            "model": record.model,
            "messages_lc": record.messages_lc,
            "tools": record.tools,
            "content_stats": record.content_stats,
        },
        "outputs": {
            "anthropic_reported_tokens": record.anthropic_reported_tokens,
            "max_context_tokens": record.max_context_tokens,
        },
    }


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(
        description="Build benchmark dataset from LangSmith error traces"
    )
    ap.add_argument(
        "--input",
        default="prompt_too_long_errors.jsonl",
        help="Input JSONL from error collection script",
    )
    ap.add_argument(
        "--output",
        default="benchmark_dataset.jsonl",
        help="Output JSONL file in LangSmith Dataset format",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of traces to process (0 for all)",
    )
    ap.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between API calls in seconds",
    )
    args = ap.parse_args()

    api_key = os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        print("Missing LANGCHAIN_API_KEY environment variable", file=sys.stderr)
        return 1

    # Read input file and group by trace_id
    print(f"Reading {args.input}...", file=sys.stderr)
    traces: dict[str, list[dict]] = {}

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            trace_id = record.get("trace_id")
            if trace_id:
                if trace_id not in traces:
                    traces[trace_id] = []
                traces[trace_id].append(record)

    print(f"Found {len(traces)} unique traces", file=sys.stderr)

    # Process each trace
    client = LangSmithClient(api_key=api_key)
    processed = 0
    errors = 0

    trace_ids = list(traces.keys())
    if args.limit > 0:
        trace_ids = trace_ids[: args.limit]

    with open(args.output, "w", encoding="utf-8") as f:
        for i, trace_id in enumerate(trace_ids):
            print(
                f"Processing {i + 1}/{len(trace_ids)}: {trace_id[:8]}...",
                file=sys.stderr,
            )

            record = process_trace(client, trace_id, traces[trace_id])

            if record:
                if record.fetch_error:
                    errors += 1
                    print(f"  Error: {record.fetch_error}", file=sys.stderr)
                else:
                    output = to_langsmith_format(record)
                    f.write(json.dumps(output, ensure_ascii=False) + "\n")
                    processed += 1
                    print(
                        f"  OK: {record.content_stats.get('num_messages', 0)} messages, "
                        f"{record.content_stats.get('total_chars', 0)} chars, "
                        f"{len(record.tools)} tools, "
                        f"{record.anthropic_reported_tokens} tokens reported",
                        file=sys.stderr,
                    )

            if args.delay > 0 and i < len(trace_ids) - 1:
                time.sleep(args.delay)

    # Summary
    print(
        json.dumps(
            {
                "input_file": args.input,
                "output_file": args.output,
                "output_format": "langsmith",
                "total_traces": len(traces),
                "processed": processed,
                "errors": errors,
            },
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
