#!/usr/bin/env python3
"""Evaluate token counters against production ground truth.

Uses the benchmark dataset (in LangSmith Dataset format) to compare
TikTokenCounter estimates against Anthropic's reported token counts.

Usage:
    python benchmark_token_counters_production.py --input benchmark_dataset.jsonl
    python benchmark_token_counters_production.py --input benchmark_dataset.jsonl --use-anthropic-api --use-tiktoken
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import anthropic
from anthropic.types import MessageParam
from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from duo_workflow_service.token_counter.tiktoken_counter import TikTokenCounter


@dataclass
class EvalResult:
    trace_id: str
    model: str
    anthropic_reported: int  # From error message (ground truth)
    anthropic_api: int | None  # From count_tokens API (validation)
    tiktoken_estimate: int | None
    error: str | None = None


def deserialize_lc_message(msg_data: dict) -> BaseMessage | None:
    """Deserialize a LangChain message from its serialized format."""
    msg_id = msg_data.get("id", [])
    msg_type = msg_id[-1] if msg_id else None
    kwargs = msg_data.get("kwargs", {})

    content = kwargs.get("content", "")

    if msg_type == "SystemMessage":
        return SystemMessage(content=content)
    if msg_type == "HumanMessage":
        return HumanMessage(content=content)
    if msg_type == "AIMessage":
        tool_calls = kwargs.get("tool_calls", [])
        if tool_calls:
            return AIMessage(content=content, tool_calls=tool_calls)
        return AIMessage(content=content)
    if msg_type == "ToolMessage":
        tool_call_id = kwargs.get("tool_call_id", "")
        name = kwargs.get("name", "")
        return ToolMessage(content=content, tool_call_id=tool_call_id, name=name)

    print(f"  Unknown message type: {msg_type}", file=sys.stderr)
    return None


def deserialize_messages(messages_lc: list[dict]) -> list[BaseMessage]:
    """Deserialize all messages from LangChain format."""
    result = []
    for msg_data in messages_lc:
        msg = deserialize_lc_message(msg_data)
        if msg:
            result.append(msg)
    return result


def sanitize_content_block(block: dict) -> dict:
    """Remove extra fields from content blocks that Anthropic API doesn't accept."""
    block_type = block.get("type")

    if block_type == "text":
        return {"type": "text", "text": block.get("text", "")}
    if block_type == "tool_use":
        tool_input = block.get("input", {})
        partial_json = block.get("partial_json", "")

        if (not tool_input or tool_input == {}) and partial_json:
            try:
                tool_input = json.loads(partial_json)
            except json.JSONDecodeError:
                tool_input = {"_partial": partial_json}

        return {
            "type": "tool_use",
            "id": block.get("id", ""),
            "name": block.get("name", ""),
            "input": tool_input,
        }
    if block_type == "tool_result":
        result = {
            "type": "tool_result",
            "tool_use_id": block.get("tool_use_id", ""),
            "content": block.get("content", ""),
        }
        if block.get("is_error"):
            result["is_error"] = block["is_error"]
        return result

    return block


def sanitize_content(content: Any) -> Any:
    """Sanitize message content, handling both string and list formats."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return [
            sanitize_content_block(b) if isinstance(b, dict) else b for b in content
        ]
    return content


def _extract_system_content(msg: SystemMessage) -> str:
    """Extract text content from a SystemMessage."""
    if isinstance(msg.content, str):
        return msg.content
    if isinstance(msg.content, list):
        parts = []
        for block in msg.content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n\n".join(parts)
    return str(msg.content)


def _convert_human_message(msg: HumanMessage) -> MessageParam:
    """Convert HumanMessage to Anthropic API format."""
    if isinstance(msg.content, list):
        return {"role": "user", "content": sanitize_content(msg.content)}
    content = msg.content if isinstance(msg.content, str) else str(msg.content)
    return {"role": "user", "content": content}


def _build_tool_call_blocks(tool_calls: list) -> list[dict]:
    """Build tool_use blocks from LangChain tool_calls."""
    return [
        {
            "type": "tool_use",
            "id": tc.get("id", ""),
            "name": tc.get("name", ""),
            "input": tc.get("args", {}),
        }
        for tc in tool_calls
    ]


def _convert_ai_message_with_tools(msg: AIMessage) -> MessageParam:
    """Convert AIMessage with tool_calls to Anthropic API format."""
    content = msg.content

    if isinstance(content, list):
        has_tool_use = any(
            isinstance(b, dict) and b.get("type") == "tool_use" for b in content
        )
        if has_tool_use:
            return {"role": "assistant", "content": sanitize_content(content)}

        content_blocks = [
            sanitize_content_block(b) if isinstance(b, dict) else b for b in content
        ]
        content_blocks.extend(_build_tool_call_blocks(msg.tool_calls))
        return {
            "role": "assistant",
            "content": content_blocks,  # type: ignore[typeddict-item]
        }

    content_blocks = []
    if content:
        content_blocks.append({"type": "text", "text": content})
    content_blocks.extend(_build_tool_call_blocks(msg.tool_calls))
    return {
        "role": "assistant",
        "content": content_blocks,  # type: ignore[typeddict-item]
    }


def _convert_ai_message(msg: AIMessage) -> MessageParam:
    """Convert AIMessage to Anthropic API format."""
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        return _convert_ai_message_with_tools(msg)

    content = msg.content
    if isinstance(content, list):
        return {"role": "assistant", "content": sanitize_content(content)}
    return {"role": "assistant", "content": content if content else ""}


def _convert_tool_message(msg: ToolMessage) -> MessageParam:
    """Convert ToolMessage to Anthropic API format."""
    return {
        "role": "user",
        "content": [
            {  # type: ignore[misc,list-item]
                "type": "tool_result",
                "tool_use_id": msg.tool_call_id,
                "content": msg.content,
            }
        ],
    }


def to_anthropic_format(messages: list[BaseMessage]) -> tuple[str, list[MessageParam]]:
    """Convert LangChain messages to Anthropic API format.

    Returns (system_prompt, messages_list).
    """
    system_parts = []
    api_messages = []

    for msg in messages:
        if isinstance(msg, SystemMessage):
            system_parts.append(_extract_system_content(msg))
        elif isinstance(msg, HumanMessage):
            api_messages.append(_convert_human_message(msg))
        elif isinstance(msg, AIMessage):
            api_messages.append(_convert_ai_message(msg))
        elif isinstance(msg, ToolMessage):
            api_messages.append(_convert_tool_message(msg))

    return "\n\n".join(system_parts), api_messages


def merge_consecutive_roles(messages: list[MessageParam]) -> list[MessageParam]:
    """Merge consecutive messages with the same role (Anthropic requires alternation)."""
    if not messages:
        return []

    merged: list[MessageParam] = []
    for msg in messages:
        if not merged or merged[-1]["role"] != msg["role"]:
            merged.append(msg)
        else:
            prev_content = merged[-1]["content"]
            curr_content = msg["content"]
            merged[-1]["content"] = _merge_content(prev_content, curr_content)

    return merged


def _merge_content(prev: Any, curr: Any) -> Any:
    """Merge two message contents together."""
    if isinstance(prev, str) and isinstance(curr, str):
        return prev + "\n\n" + curr
    if isinstance(prev, list) and isinstance(curr, list):
        return prev + curr
    if isinstance(prev, str) and isinstance(curr, list):
        return [{"type": "text", "text": prev}] + curr
    if isinstance(prev, list) and isinstance(curr, str):
        return prev + [{"type": "text", "text": curr}]
    return curr


def count_with_anthropic_api(
    client: anthropic.Anthropic,
    model: str,
    system: str,
    messages: list[MessageParam],
    tools: list[dict] | None = None,
) -> int:
    """Count tokens using Anthropic's API."""
    merged = merge_consecutive_roles(messages)

    if merged and merged[0]["role"] != "user":
        placeholder: MessageParam = {
            "role": "user",
            "content": "(continuing conversation)",
        }
        merged = [placeholder] + merged

    if tools:
        response = client.messages.count_tokens(
            model=model,
            system=system,
            messages=merged,
            tools=tools,  # type: ignore[arg-type]
        )
    else:
        response = client.messages.count_tokens(
            model=model,
            system=system,
            messages=merged,
        )
    return response.input_tokens


@dataclass
class ConversationStats:
    """Statistics about a conversation's structure."""

    trace_id: str
    num_messages: int
    num_ai_messages: int
    num_tool_messages: int
    num_tool_calls: int
    total_chars: int
    tool_message_chars: int
    avg_chars_per_tool_msg: float
    tokens_per_char: float  # anthropic tokens / total chars


def extract_conversation_stats(inputs: dict, outputs: dict) -> ConversationStats:
    """Extract conversation structure statistics from a benchmark record."""
    content_stats = inputs.get("content_stats", {})
    by_role = content_stats.get("by_role", {})

    num_ai = by_role.get("AIMessage", {}).get("count", 0)
    num_tool = by_role.get("ToolMessage", {}).get("count", 0)
    tool_chars = by_role.get("ToolMessage", {}).get("chars", 0)
    total_chars = content_stats.get("total_chars", 1)  # avoid div by zero

    # Count actual tool_calls from messages
    num_tool_calls = 0
    for msg in inputs.get("messages_lc", []):
        msg_id = msg.get("id", [])
        if msg_id and msg_id[-1] == "AIMessage":
            tool_calls = msg.get("kwargs", {}).get("tool_calls", [])
            num_tool_calls += len(tool_calls)

    reported_tokens = outputs.get("anthropic_reported_tokens", 0)

    return ConversationStats(
        trace_id=inputs.get("trace_id", ""),
        num_messages=content_stats.get("num_messages", 0),
        num_ai_messages=num_ai,
        num_tool_messages=num_tool,
        num_tool_calls=num_tool_calls,
        total_chars=total_chars,
        tool_message_chars=tool_chars,
        avg_chars_per_tool_msg=tool_chars / num_tool if num_tool > 0 else 0,
        tokens_per_char=reported_tokens / total_chars if total_chars > 0 else 0,
    )


def evaluate_record(
    inputs: dict,
    outputs: dict,
    anthropic_client: anthropic.Anthropic | None,
    tiktoken_counter: Any | None,
) -> EvalResult:
    """Evaluate a single benchmark record in LangSmith Dataset format."""
    trace_id = inputs["trace_id"]
    model = inputs.get("model") or "claude-haiku-4-5-20251001"
    anthropic_reported = outputs["anthropic_reported_tokens"]
    tools = inputs.get("tools", [])

    result = EvalResult(
        trace_id=trace_id,
        model=model,
        anthropic_reported=anthropic_reported,
        anthropic_api=None,
        tiktoken_estimate=None,
    )

    # Deserialize messages
    try:
        messages = deserialize_messages(inputs.get("messages_lc", []))
        if not messages:
            result.error = "No messages after deserialization"
            return result
    except Exception as e:
        result.error = f"Deserialization error: {e}"
        return result

    # Count with TikTokenCounter
    if tiktoken_counter:
        try:
            result.tiktoken_estimate = tiktoken_counter.count_tokens(messages)
        except Exception as e:
            result.error = f"TikToken error: {e}"

    # Count with Anthropic API
    if anthropic_client:
        try:
            system, api_messages = to_anthropic_format(messages)
            result.anthropic_api = count_with_anthropic_api(
                anthropic_client, model, system, api_messages, tools
            )
        except Exception as e:
            result.error = f"Anthropic API error: {e}"

    return result


def calc_stats(errors: list[float]) -> dict:
    """Calculate mean, std dev, min, max, and percentiles for a list of percentage errors."""
    n = len(errors)
    if n == 0:
        return {"mean": 0, "std_dev": 0, "min": 0, "max": 0, "percentiles": {}}
    mean = sum(errors) / n
    variance = sum((e - mean) ** 2 for e in errors) / n
    std_dev = variance**0.5
    sorted_errors = sorted(errors)

    def percentile(p: int) -> float:
        idx = int(n * p / 100)
        idx = min(idx, n - 1)
        return sorted_errors[idx]

    return {
        "mean": mean,
        "std_dev": std_dev,
        "min": min(errors),
        "max": max(errors),
        "percentiles": {
            "p10": percentile(10),
            "p25": percentile(25),
            "p50": percentile(50),
            "p75": percentile(75),
            "p90": percentile(90),
            "p95": percentile(95),
        },
    }


def error_to_multiplier(error_pct: float) -> float:
    """Convert an undercount error percentage to the multiplier needed to correct it."""
    # If tiktoken undercounts by X%, actual = tiktoken * (1 / (1 + X/100))
    # So multiplier = 1 / (1 + error/100)
    # e.g., -20% error means actual = tiktoken / 0.8, so multiplier = 1.25
    return 1 / (1 + error_pct / 100) if error_pct != -100 else float("inf")


def _print_evaluation_feedback(result: EvalResult) -> None:
    """Print immediate feedback for a single evaluation result."""
    if result.error:
        print(f"  Error: {result.error}", file=sys.stderr)
        return

    parts = [f"reported={result.anthropic_reported}"]
    if result.anthropic_api:
        diff = result.anthropic_api - result.anthropic_reported
        parts.append(f"api={result.anthropic_api} (diff={diff:+d})")
    if result.tiktoken_estimate:
        err_vs_reported = (
            (result.tiktoken_estimate - result.anthropic_reported)
            / result.anthropic_reported
            * 100
        )
        parts.append(
            f"tiktoken={result.tiktoken_estimate} (vs_reported={err_vs_reported:+.1f}%)"
        )
        if result.anthropic_api:
            err_vs_api = (
                (result.tiktoken_estimate - result.anthropic_api)
                / result.anthropic_api
                * 100
            )
            parts.append(f"(vs_api={err_vs_api:+.1f}%)")
    print(f"  {', '.join(parts)}", file=sys.stderr)


def _write_detailed_results(results: list[EvalResult], output_path: str) -> None:
    """Write detailed results to JSONL file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            tiktoken_vs_reported = None
            tiktoken_vs_api = None
            if r.tiktoken_estimate and r.anthropic_reported:
                tiktoken_vs_reported = (
                    (r.tiktoken_estimate - r.anthropic_reported)
                    / r.anthropic_reported
                    * 100
                )
            if r.tiktoken_estimate and r.anthropic_api:
                tiktoken_vs_api = (
                    (r.tiktoken_estimate - r.anthropic_api) / r.anthropic_api * 100
                )

            f.write(
                json.dumps(
                    {
                        "trace_id": r.trace_id,
                        "model": r.model,
                        "anthropic_reported": r.anthropic_reported,
                        "anthropic_api": r.anthropic_api,
                        "tiktoken_estimate": r.tiktoken_estimate,
                        "tiktoken_vs_reported_pct": (
                            round(tiktoken_vs_reported, 2)
                            if tiktoken_vs_reported
                            else None
                        ),
                        "tiktoken_vs_api_pct": (
                            round(tiktoken_vs_api, 2) if tiktoken_vs_api else None
                        ),
                        "error": r.error,
                    }
                )
                + "\n"
            )
    print(f"\nDetailed results written to {output_path}", file=sys.stderr)


def _build_header_markdown(
    args: argparse.Namespace, valid_count: int, total: int
) -> list[str]:
    """Build the markdown header section."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return [
        "## Token Counter Benchmark Results\n",
        f"**Date:** {timestamp}  ",
        f"**Dataset:** `{args.input}`  ",
        f"**Total evaluated:** {total} | **Valid:** {valid_count}\n",
    ]


def _build_api_validation_markdown(valid_results: list[EvalResult]) -> list[str]:
    """Build markdown section for Anthropic API validation."""
    api_errs = [
        (r.anthropic_api - r.anthropic_reported) / r.anthropic_reported * 100
        for r in valid_results
        if r.anthropic_api
    ]
    if not api_errs:
        return []

    stats = calc_stats(api_errs)
    return [
        "### Anthropic API vs Production Reported\n",
        "This validates that our benchmark data matches production. "
        "Small differences are expected due to serialization.\n",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Sample size | {len(api_errs)} |",
        f"| Mean error | {stats['mean']:+.1f}% |",
        f"| Std dev | {stats['std_dev']:.1f}% |",
        f"| Max undercount | {stats['min']:+.1f}% |",
        f"| Max overcount | {stats['max']:+.1f}% |\n",
    ]


def _build_error_table(label: str, stats: dict) -> list[str]:
    """Build a markdown error table with multipliers."""
    lines = [
        f"| {label} | Error % | Multiplier Needed |",
        "|--------|---------|-------------------|",
        f"| Mean | {stats['mean']:+.1f}% | {error_to_multiplier(stats['mean']):.2f}x |",
        f"| Max undercount | {stats['min']:+.1f}% "
        f"| {error_to_multiplier(stats['min']):.2f}x |",
        f"| Max overcount | {stats['max']:+.1f}% "
        f"| {error_to_multiplier(stats['max']):.2f}x |\n",
    ]
    return lines


def _build_percentile_table(percentiles: dict) -> list[str]:
    """Build a markdown percentile distribution table."""
    p = percentiles
    lines = [
        "#### Error Distribution (Percentiles)\n",
        "| Percentile | Error % | Multiplier Needed |",
        "|------------|---------|-------------------|",
        f"| p10 (worst 10%) | {p['p10']:+.1f}% | {error_to_multiplier(p['p10']):.2f}x |",
        f"| p25 | {p['p25']:+.1f}% | {error_to_multiplier(p['p25']):.2f}x |",
        f"| p50 (median) | {p['p50']:+.1f}% | {error_to_multiplier(p['p50']):.2f}x |",
        f"| p75 | {p['p75']:+.1f}% | {error_to_multiplier(p['p75']):.2f}x |",
        f"| p90 | {p['p90']:+.1f}% | {error_to_multiplier(p['p90']):.2f}x |",
        f"| p95 | {p['p95']:+.1f}% | {error_to_multiplier(p['p95']):.2f}x |\n",
    ]
    return lines


def _build_tiktoken_vs_reported_markdown(valid_results: list[EvalResult]) -> list[str]:
    """Build markdown for TikTokenCounter vs Production Reported."""
    tiktoken_vs_reported = [
        (r.tiktoken_estimate - r.anthropic_reported) / r.anthropic_reported * 100
        for r in valid_results
        if r.tiktoken_estimate
    ]
    if not tiktoken_vs_reported:
        return []

    reported_stats = calc_stats(tiktoken_vs_reported)
    lines = [
        "### TikTokenCounter vs Production Reported\n",
        "Comparison against the token count from production 'prompt is too long' errors.\n",
    ]
    lines.extend(_build_error_table("Metric", reported_stats))
    lines.extend(_build_percentile_table(reported_stats["percentiles"]))
    return lines


def _build_tiktoken_vs_api_markdown(valid_results: list[EvalResult]) -> list[str]:
    """Build markdown for TikTokenCounter vs Anthropic API."""
    tiktoken_vs_api = [
        (r.tiktoken_estimate - r.anthropic_api) / r.anthropic_api * 100
        for r in valid_results
        if r.tiktoken_estimate and r.anthropic_api
    ]
    if not tiktoken_vs_api:
        return []

    tiktoken_stats = calc_stats(tiktoken_vs_api)
    lines = [
        "### TikTokenCounter vs Anthropic API\n",
    ]
    lines.extend(_build_error_table("Metric", tiktoken_stats))
    lines.extend(_build_percentile_table(tiktoken_stats["percentiles"]))
    return lines


def _avg_conv_stats(
    result_list: list[EvalResult],
    conv_stats_map: dict[str, ConversationStats],
) -> dict[str, float]:
    """Calculate average conversation stats for a list of results."""
    stats_list: list[ConversationStats] = [
        conv_stats_map[r.trace_id] for r in result_list if r.trace_id in conv_stats_map
    ]
    if not stats_list:
        return {}
    n = len(stats_list)
    return {
        "num_messages": sum(s.num_messages for s in stats_list) / n,
        "num_tool_calls": sum(s.num_tool_calls for s in stats_list) / n,
        "num_tool_messages": sum(s.num_tool_messages for s in stats_list) / n,
        "avg_chars_per_tool_msg": sum(s.avg_chars_per_tool_msg for s in stats_list) / n,
        "tokens_per_char": sum(s.tokens_per_char for s in stats_list) / n,
        "total_chars": sum(s.total_chars for s in stats_list) / n,
    }


def _build_characteristics_markdown(
    valid_results: list[EvalResult],
    conv_stats_map: dict[str, ConversationStats],
) -> list[str]:
    """Build markdown for conversation characteristics analysis."""
    results_with_error = [
        (r, (r.tiktoken_estimate - r.anthropic_api) / r.anthropic_api * 100)
        for r in valid_results
        if r.tiktoken_estimate and r.anthropic_api
    ]
    if not results_with_error:
        return []

    results_with_error.sort(key=lambda x: x[1])  # Sort by error (most negative first)

    n_worst = max(1, len(results_with_error) // 10)  # worst 10%
    worst_results = [r for r, _ in results_with_error[:n_worst]]
    rest_results = [r for r, _ in results_with_error[n_worst:]]

    worst_avg = _avg_conv_stats(worst_results, conv_stats_map)
    rest_avg = _avg_conv_stats(rest_results, conv_stats_map)

    if not worst_avg or not rest_avg:
        return []

    lines = [
        "### Conversation Characteristics by Error Severity\n",
        "What distinguishes the worst-performing cases from the rest?\n",
        "| Metric | Worst 10% | Rest (90%) | Ratio |",
        "|--------|-----------|------------|-------|",
    ]

    metrics = [
        ("Avg messages", "num_messages", ".0f"),
        ("Avg tool calls", "num_tool_calls", ".0f"),
        ("Avg tool messages", "num_tool_messages", ".0f"),
        ("Avg chars/tool msg", "avg_chars_per_tool_msg", ",.0f"),
        ("Avg total chars", "total_chars", ",.0f"),
        ("Tokens/char ratio", "tokens_per_char", ".2f"),
    ]

    for label, key, fmt in metrics:
        w_val = worst_avg.get(key, 0)
        r_val = rest_avg.get(key, 0)
        ratio = w_val / r_val if r_val > 0 else 0
        lines.append(f"| {label} | {w_val:{fmt}} | {r_val:{fmt}} | {ratio:.1f}x |")

    lines.append("")
    return lines


def _build_markdown_report(
    args: argparse.Namespace,
    results: list[EvalResult],
    valid_results: list[EvalResult],
    conv_stats_map: dict[str, ConversationStats],
) -> str:
    """Build the complete markdown report."""
    md_lines = _build_header_markdown(args, len(valid_results), len(results))

    if args.use_anthropic_api:
        md_lines.extend(_build_api_validation_markdown(valid_results))

    if args.use_tiktoken:
        md_lines.extend(_build_tiktoken_vs_reported_markdown(valid_results))
        md_lines.extend(_build_tiktoken_vs_api_markdown(valid_results))
        md_lines.extend(_build_characteristics_markdown(valid_results, conv_stats_map))

    return "\n".join(md_lines)


def _load_and_evaluate(
    args: argparse.Namespace,
    anthropic_client: anthropic.Anthropic | None,
    tiktoken_counter: Any | None,
) -> tuple[list[EvalResult], dict[str, ConversationStats]]:
    """Load benchmark data and run evaluations."""
    results: list[EvalResult] = []
    conv_stats_map: dict[str, ConversationStats] = {}

    with open(args.input, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.limit and i >= args.limit:
                break

            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            inputs = record.get("inputs", {})
            outputs = record.get("outputs", {})

            if inputs.get("fetch_error"):
                print(f"Skipping {i+1}: fetch_error in record", file=sys.stderr)
                continue

            trace_id = inputs.get("trace_id", "unknown")
            print(f"Evaluating {i+1}: {trace_id[:16]}...", file=sys.stderr)

            conv_stats = extract_conversation_stats(inputs, outputs)
            conv_stats_map[trace_id] = conv_stats

            result = evaluate_record(
                inputs, outputs, anthropic_client, tiktoken_counter
            )
            results.append(result)
            _print_evaluation_feedback(result)

    return results, conv_stats_map


def main() -> int:
    load_dotenv()

    ap = argparse.ArgumentParser(description="Evaluate token counters")
    ap.add_argument(
        "--input",
        default="benchmark_dataset.jsonl",
        help="Benchmark dataset JSONL (LangSmith Dataset format)",
    )
    ap.add_argument(
        "--use-anthropic-api",
        action="store_true",
        help="Call Anthropic API to validate token counts",
    )
    ap.add_argument(
        "--use-tiktoken",
        action="store_true",
        help="Run TikTokenCounter evaluation",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit records to evaluate",
    )
    ap.add_argument(
        "--output",
        default="",
        help="Output JSONL for detailed results",
    )
    ap.add_argument(
        "--markdown",
        default="",
        help="Output markdown summary to file (in addition to stdout)",
    )
    args = ap.parse_args()

    # Initialize counters
    anthropic_client = anthropic.Anthropic() if args.use_anthropic_api else None
    tiktoken_counter = TikTokenCounter("executor") if args.use_tiktoken else None

    # Load and evaluate
    results, conv_stats_map = _load_and_evaluate(
        args, anthropic_client, tiktoken_counter
    )

    # Output detailed results to JSONL
    if args.output:
        _write_detailed_results(results, args.output)

    # Summary statistics
    print("\n" + "=" * 60, file=sys.stderr)
    print("SUMMARY", file=sys.stderr)
    print("=" * 60 + "\n", file=sys.stderr)

    valid_results = [r for r in results if not r.error]
    markdown_output = _build_markdown_report(
        args, results, valid_results, conv_stats_map
    )

    print(markdown_output)

    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write(markdown_output)
        print(f"\nMarkdown summary written to {args.markdown}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
