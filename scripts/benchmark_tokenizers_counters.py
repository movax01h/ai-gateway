import random
import sys
import time
from typing import Any, Protocol

import anthropic
import tiktoken
from dotenv import load_dotenv
from transformers import AutoTokenizer

from duo_workflow_service.token_counter.tiktoken_counter import TikTokenCounter


class TokenCounter(Protocol):
    def count_string_content(self, text: str) -> int: ...


# Simulate the old ApproximateTokenCounter
class ApproximateTokenCounter:
    def __init__(self, agent_name: str):
        self.tool_tokens = {"executor": 5650}.get(agent_name, 0)

    def count_string_content(self, text: str) -> int:
        return int(round(len(text) // 4 * 1.5))


# Pure tiktoken (ground truth for OpenAI models)
class PureTiktoken:
    def __init__(self):
        self._encoding = tiktoken.encoding_for_model("gpt-4o")

    def count_string_content(self, text: str) -> int:
        return len(self._encoding.encode(text))


# Anthropic official token counter (ground truth for Claude models)
class AnthropicAPICounter:
    def __init__(self, model: str = "claude-sonnet-4-5-20250929"):
        self._client = anthropic.Anthropic()
        self._model = model

    def count_string_content(self, text: str) -> int:
        response = self._client.messages.count_tokens(
            model=self._model,
            messages=[{"role": "user", "content": text}],
        )
        return response.input_tokens


# HuggingFace tokenizer wrapper
class HuggingFaceTokenizer:
    def __init__(self, model_id: str, display_name: str):
        self.display_name = display_name
        self._tokenizer = None
        self._error = None

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True, use_fast=True
            )
        except Exception as e:
            self._error = str(e)[:100]

    def is_available(self) -> bool:
        return self._tokenizer is not None

    def get_error(self) -> str:
        return self._error or "transformers not installed"

    def count_string_content(self, text: str) -> int:
        if not self._tokenizer:
            raise RuntimeError(f"Tokenizer not available: {self._error}")
        return len(self._tokenizer.encode(text, add_special_tokens=False))


# Tiktoken wrapper for different encodings
class TiktokenEncoding:
    def __init__(self, encoding_name: str, display_name: str):
        self.display_name = display_name
        self._encoding = tiktoken.get_encoding(encoding_name)

    def is_available(self) -> bool:
        return True

    def count_string_content(self, text: str) -> int:
        return len(self._encoding.encode(text))


# Define all tokenizers to test
TOKENIZERS: dict[str, TokenCounter] = {
    # Built-in counters
    "Approximate": ApproximateTokenCounter("executor"),
    "Sampling (current)": TikTokenCounter("executor", model="claude"),
    "Pure Tiktoken": PureTiktoken(),
    # Tiktoken encodings
    # "tiktoken-cl100k": TiktokenEncoding("cl100k_base", "tiktoken-cl100k"),
    # "tiktoken-o200k": TiktokenEncoding("o200k_base", "tiktoken-o200k"),
}

# HuggingFace tokenizers (only if transformers is available)
HF_TOKENIZERS = [
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama"),
    ("mistralai/Mistral-7B-v0.1", "Mistral-7B"),
    ("deepseek-ai/deepseek-llm-7b-base", "DeepSeek-7B"),
    ("Xenova/claude-tokenizer", "Claude-Old"),
    ("gpt2", "GPT-2"),
    ("microsoft/phi-2", "Phi-2"),
    ("tiiuae/falcon-7b", "Falcon-7B"),
    ("01-ai/Yi-6B", "Yi-6B"),
]


def load_hf_tokenizers():
    """Load HuggingFace tokenizers and add available ones to TOKENIZERS."""

    print("Loading HuggingFace tokenizers...")
    for model_id, display_name in HF_TOKENIZERS:
        print(f"  Loading {display_name}...", end=" ")
        tokenizer = HuggingFaceTokenizer(model_id, display_name)
        if tokenizer.is_available():
            TOKENIZERS[display_name] = tokenizer
            print("✓")
        else:
            print(f"✗ ({tokenizer.get_error()[:50]})")
    print()


# Helper functions to generate realistic test cases
def generate_code_file_with_dense_imports(size: int) -> str:
    imports = "\n".join(
        [
            f"from some_very_long_module_name_{i}.submodule.deeply.nested "
            f"import SomeClass{i}, AnotherClass{i}, ThirdClass{i}"
            for i in range(50)
        ]
    )
    sparse_code = "\n\n\n".join(
        [
            f"def function_{i}():\n    # TODO: implement\n    pass"
            for i in range(size // 100)
        ]
    )
    return imports + "\n\n" + sparse_code


def generate_mixed_density_file(size: int) -> str:
    dense_block = "αβγδεζηθικλμνξοπρστυφχψω" * 20
    sparse_block = "x" * 500
    pattern = (dense_block + sparse_block) * (
        size // (len(dense_block) + len(sparse_block))
    )
    return pattern[:size]


def generate_json_with_unicode_values(size: int) -> str:
    entries = []
    languages = ["你好世界", "こんにちは", "مرحبا", "Привет", "🚀🎉✨"]
    for i in range(size // 50):
        lang = languages[i % len(languages)]
        entries.append(f'"{lang}_{i}": "{lang * 3}"')
    return "{" + ", ".join(entries) + "}"


def generate_minified_js(size: int) -> str:
    code = (
        "var a=function(b,c){return b+c};"
        "var d=function(e){for(var i=0;i<e.length;i++){console.log(e[i])}};"
        "var obj={key1:'value1',key2:'value2',key3:[1,2,3,4,5]};"
    )
    return (code * (size // len(code) + 1))[:size]


def generate_log_file_with_timestamps(size: int) -> str:
    lines = []
    messages = [
        "Connection established successfully",
        "Processing request from user αβγ",
        "Error: 文件未找到 🚫",
        "Warning: High memory usage detected",
        "DEBUG: {" + '"key": "value", "nested": {"deep": "🔥"}}',
    ]
    for i in range(size // 80):
        msg = messages[i % len(messages)]
        lines.append(f"2024-12-11T10:30:{i % 60:02d}.{i % 1000:03d}Z [INFO] {msg}")
    return "\n".join(lines)


def generate_sparse_then_dense(size: int) -> str:
    sparse_part = "a " * (size // 2)
    dense_part = "🚀🎉✨💻🔥" * (size // 10)
    return sparse_part + dense_part


def generate_dense_then_sparse(size: int) -> str:
    dense_part = "你好世界！" * (size // 10)
    sparse_part = "x " * (size // 2)
    return dense_part + sparse_part


def generate_dense_middle_only(size: int) -> str:
    third = size // 3
    sparse = "a " * third
    dense = "🚀αβγ你好" * (third // 6)
    return sparse + dense + sparse


def generate_random_density_spikes(size: int) -> str:
    random.seed(42)
    result = []
    dense_chars = "🚀✨🎉💻🔥你好世界αβγδ"
    sparse_chars = "abcdefghijklmnopqrstuvwxyz      "

    for _ in range(size):
        if random.random() < 0.1:
            result.append(random.choice(dense_chars))
        else:
            result.append(random.choice(sparse_chars))
    return "".join(result)


def generate_adversarial_for_center_sampling(size: int) -> str:
    cell_size = size // 30
    sample_size = 300
    offset = (cell_size - sample_size) // 2

    result = []
    for _ in range(30):
        center_sparse = "a " * (sample_size // 2)
        edge_dense = "🚀你好" * (offset // 4) if offset > 0 else ""
        result.append(edge_dense + center_sparse + edge_dense)
    return "".join(result)[:size]


def generate_chat_conversation(size: int) -> str:
    messages: list[str] = []
    templates = [
        "User: Can you help me with this code?\n",
        "Assistant: Of course! The problem is in `calculate_αβγ()` where 你好 🚀\n",
        "User: Thanks! Error: TypeError: cannot convert '🎉' to int\n",
        "Assistant: " + "x" * 200 + "\n",
        "User: " + "αβγδ" * 50 + "\n",
    ]
    while len("".join(messages)) < size:
        messages.append(templates[len(messages) % len(templates)])
    return "".join(messages)[:size]


def generate_tool_call_json(size: int) -> str:
    tool_calls = []
    for i in range(size // 300):
        tool_calls.append(
            f'{{"tool_call_id": "call_{i}", "function": {{"name": "read_file", '
            f'"arguments": {{"path": "/src/components/файл_{i}.tsx", "content": "const x = 你好🚀"}}}}}}'
        )
    return "[" + ",\n".join(tool_calls) + "]"


def generate_git_diff(size: int) -> str:
    hunks = []
    for i in range(size // 200):
        hunks.append(
            f"@@ -{i * 10},{i % 5 + 1} +{i * 10},{i % 5 + 2} @@\n"
            f"-    old_value = 'hello'\n"
            f"+    new_value = '你好世界 🚀'\n"
            f"     unchanged_line = True\n"
        )
    return f"diff --git a/file.py b/file.py\n{''.join(hunks)}"[:size]


def generate_threshold_boundary_dense(size: int = 9001) -> str:
    return "🚀你好" * (size // 6)


def generate_threshold_boundary_sparse(size: int = 9001) -> str:
    return "a " * (size // 2)


# Test cases
TEST_CASES = [
    ("Short text (50 chars)", "Hello, how are you doing today? This is a test."),
    ("Medium text (1KB)", "x" * 1024),
    ("Large text (10KB)", "x" * 10240),
    ("Unicode heavy", "你好世界！" * 200),
    ("Emoji heavy", "🚀✨🎉💻🔥" * 200),
    ("JSON data", '{"users": [{"id": 1, "name": "Alice"}]}' * 50),
    ("Code w/ dense imports (10KB)", generate_code_file_with_dense_imports(10240)),
    ("Mixed density alternating (10KB)", generate_mixed_density_file(10240)),
    ("JSON with Unicode values (10KB)", generate_json_with_unicode_values(10240)),
    ("Minified JavaScript (10KB)", generate_minified_js(10240)),
    ("Log file with timestamps (10KB)", generate_log_file_with_timestamps(10240)),
    ("Sparse→Dense (10KB)", generate_sparse_then_dense(10240)),
    ("Dense→Sparse (10KB)", generate_dense_then_sparse(10240)),
    ("Dense middle only (10KB)", generate_dense_middle_only(10240)),
    ("Random density spikes (10KB)", generate_random_density_spikes(10240)),
    ("Large mixed density (50KB)", generate_mixed_density_file(51200)),
    ("Large random spikes (50KB)", generate_random_density_spikes(51200)),
    ("Large sparse→dense (50KB)", generate_sparse_then_dense(51200)),
    (
        "Adversarial center sampling (10KB)",
        generate_adversarial_for_center_sampling(10240),
    ),
    ("Chat conversation (10KB)", generate_chat_conversation(10240)),
    ("Tool call JSON (10KB)", generate_tool_call_json(10240)),
    ("Git diff output (10KB)", generate_git_diff(10240)),
    ("Just over threshold - dense (9001)", generate_threshold_boundary_dense(9001)),
    ("Just over threshold - sparse (9001)", generate_threshold_boundary_sparse(9001)),
    ("Just under threshold - dense (8999)", "🚀你好" * 1499),
]


def get_anthropic_counts(test_cases):
    """Get token counts from Anthropic API for all test cases."""
    print("Getting Anthropic API token counts (ground truth)...")
    anthropic_counter = AnthropicAPICounter()
    counts: dict[str, int | None] = {}

    for name, content in test_cases:
        try:
            counts[name] = anthropic_counter.count_string_content(content)
            print(f"  {name}: {counts[name]} tokens")
        except Exception as e:
            print(f"  {name}: ERROR - {e}")
            counts[name] = None
        time.sleep(0.05)  # Rate limiting

    return counts


def run_accuracy_benchmark(
    test_cases: list[tuple[str, str]], anthropic_counts: dict[str, int | None]
) -> dict[str, list[dict[str, Any]]]:
    """Run accuracy benchmark comparing all tokenizers against Anthropic API."""

    # Collect results for each tokenizer
    results: dict[str, list[dict[str, Any]]] = {name: [] for name in TOKENIZERS}

    for test_name, content in test_cases:
        actual: int | None = anthropic_counts.get(test_name)
        if actual is None:
            continue

        for tok_name, tokenizer in TOKENIZERS.items():
            try:
                if hasattr(tokenizer, "is_available") and not tokenizer.is_available():
                    continue
                estimated = tokenizer.count_string_content(content)
                diff = estimated - actual
                pct_err = (diff / actual * 100) if actual > 0 else 0
                results[tok_name].append(
                    {
                        "test": test_name,
                        "actual": actual,
                        "estimated": estimated,
                        "diff": diff,
                        "pct_err": pct_err,
                        "abs_pct_err": abs(pct_err),
                    }
                )
            except Exception:
                pass  # Skip failed tokenizations

    return results


def generate_accuracy_table_markdown(results, anthropic_counts):
    """Generate markdown table for accuracy benchmark."""
    active_tokenizers = {k: v for k, v in results.items() if v}
    tok_names = list(active_tokenizers.keys())

    lines = []
    lines.append("## Accuracy Benchmark")
    lines.append("")
    lines.append(
        "Comparison of tokenizer estimates vs Anthropic API (ground truth for Claude)"
    )
    lines.append("")

    # Header
    header = "| Test Case | Anthropic |"
    separator = "|:----------|----------:|"
    for name in tok_names:
        header += f" {name} |"
        separator += "----------:|"

    lines.append(header)
    lines.append(separator)

    # Data rows
    for test_name, _ in TEST_CASES:
        actual = anthropic_counts.get(test_name)
        if actual is None:
            continue

        row = f"| {test_name} | {actual} |"
        for tok_name in tok_names:
            tok_results = [r for r in results[tok_name] if r["test"] == test_name]
            if tok_results:
                r = tok_results[0]
                row += f" {r['estimated']} ({r['pct_err']:+.1f}%) |"
            else:
                row += " N/A |"
        lines.append(row)

    return "\n".join(lines)


def generate_summary_table_markdown(results):
    """Generate markdown table for summary statistics."""
    summaries = []
    for tok_name, tok_results in results.items():
        if not tok_results:
            continue

        pct_errors = [r["pct_err"] for r in tok_results]
        abs_errors = [r["abs_pct_err"] for r in tok_results]

        summaries.append(
            {
                "name": tok_name,
                "avg_err": sum(pct_errors) / len(pct_errors),
                "avg_abs_err": sum(abs_errors) / len(abs_errors),
                "min_err": min(pct_errors),
                "max_err": max(pct_errors),
                "std_dev": (
                    sum(
                        (e - sum(pct_errors) / len(pct_errors)) ** 2 for e in pct_errors
                    )
                    / len(pct_errors)
                )
                ** 0.5,
                "undercount_cases": len([e for e in pct_errors if e < 0]),
                "total_cases": len(pct_errors),
            }
        )

    # Sort
    summaries.sort(key=lambda x: x["min_err"], reverse=True)

    lines = []
    lines.append("## Summary Statistics")
    lines.append("")
    lines.append("Sorted by Worst Under, closest to zero is better")
    lines.append("")
    lines.append(
        "| Tokenizer | Avg Error | Avg Abs Error | Std Dev | Worst Under | Worst Over | Undercounts |"
    )
    lines.append(
        "|:----------|----------:|--------------:|--------:|------------:|-----------:|------------:|"
    )

    for s in summaries:
        lines.append(
            f"| {s['name']} | {s['avg_err']:+.1f}% | {s['avg_abs_err']:.1f}% | "
            f"{s['std_dev']:.1f}% | {s['min_err']:+.1f}% | {s['max_err']:+.1f}% | "
            f"{s['undercount_cases']}/{s['total_cases']} |"
        )
    return "\n".join(lines), summaries


def generate_category_table_markdown(results, best_tokenizer_name):
    """Generate markdown table for category analysis."""
    categories = {
        "Short/Simple": ["Short text", "Medium text", "Large text"],
        "Unicode (CJK)": ["Unicode heavy"],
        "Emoji": ["Emoji heavy"],
        "Code": ["Code w/", "Minified JavaScript"],
        "JSON": ["JSON data", "JSON with Unicode", "Tool call JSON"],
        "Mixed Density": [
            "Mixed density",
            "Sparse→Dense",
            "Dense→Sparse",
            "Dense middle",
            "Random density",
        ],
        "Real Workflow": ["Chat conversation", "Git diff", "Log file"],
        "Threshold Edge": ["threshold"],
    }

    tok_results = results.get(best_tokenizer_name, [])

    lines = []
    lines.append(f"## Category Analysis: {best_tokenizer_name}")
    lines.append("")
    lines.append("| Category | Avg Error | Avg Abs Error | Worst |")
    lines.append("|:---------|----------:|--------------:|------:|")

    for cat_name, patterns in categories.items():
        cat_results = [
            r
            for r in tok_results
            if any(p.lower() in r["test"].lower() for p in patterns)
        ]
        if cat_results:
            avg_err = sum(r["pct_err"] for r in cat_results) / len(cat_results)
            avg_abs = sum(r["abs_pct_err"] for r in cat_results) / len(cat_results)
            worst = min(r["pct_err"] for r in cat_results)
            lines.append(
                f"| {cat_name} | {avg_err:+.1f}% | {avg_abs:.1f}% | {worst:+.1f}% |"
            )

    return "\n".join(lines)


def generate_recommendations_markdown(summaries):
    """Generate markdown recommendations section."""
    if not summaries:
        return "## Recommendations\n\nNo results to analyze."

    best = summaries[0]

    # Calculate safe multiplier
    if best["min_err"] < 0:
        safe_multiplier = 1 / (1 + best["min_err"] / 100)
    else:
        safe_multiplier = 1.05

    # Add extra safety margin
    safe_multiplier *= 1.05

    lines = []
    lines.append("## Recommendations")
    lines.append("")
    lines.append(f"### Best Tokenizer: {best['name']}")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|:-------|------:|")
    lines.append(f"| Average Error | {best['avg_err']:+.1f}% |")
    lines.append(f"| Average Absolute Error | {best['avg_abs_err']:.1f}% |")
    lines.append(f"| Worst Undercount | {best['min_err']:+.1f}% |")
    lines.append(f"| Worst Overcount | {best['max_err']:+.1f}% |")
    lines.append("")
    lines.append("### Top 3 Tokenizers")
    lines.append("")
    lines.append("| Rank | Tokenizer | Avg Abs Error | Required Multiplier |")
    lines.append("|:-----|:----------|:-------------:|--------------------:|")

    for i, s in enumerate(summaries[:3], 1):
        mult = 1 / (1 + s["min_err"] / 100) * 1.05 if s["min_err"] < 0 else 1.05
        lines.append(f"| {i} | {s['name']} | {s['avg_abs_err']:.1f}% | {mult:.2f}x |")

    return "\n".join(lines)


def main():
    print("# Tokenizer Benchmark: Finding the Best Claude Token Estimator")
    print()
    load_dotenv()

    sys.stderr.write("Loading HuggingFace tokenizers...\n")
    for model_id, display_name in HF_TOKENIZERS:
        sys.stderr.write(f"  Loading {display_name}... ")
        tokenizer = HuggingFaceTokenizer(model_id, display_name)
        if tokenizer.is_available():
            TOKENIZERS[display_name] = tokenizer
            sys.stderr.write("✓\n")
        else:
            sys.stderr.write(f"✗ ({tokenizer.get_error()[:50]})\n")
        sys.stderr.write("\n")

    sys.stderr.write(
        f"Testing {len(TOKENIZERS)} tokenizers against {len(TEST_CASES)} test cases\n\n"
    )

    # Get ground truth from Anthropic API
    sys.stderr.write("Getting Anthropic API token counts (ground truth)...\n")
    anthropic_counter = AnthropicAPICounter()
    anthropic_counts: dict[str, int | None] = {}

    for name, content in TEST_CASES:
        try:
            anthropic_counts[name] = anthropic_counter.count_string_content(content)
            sys.stderr.write(f"  {name}: {anthropic_counts[name]} tokens\n")
        except Exception as e:
            sys.stderr.write(f"  {name}: ERROR - {e}\n")
            anthropic_counts[name] = None
        time.sleep(0.05)

    sys.stderr.write("\nGenerating markdown report...\n")

    # Run accuracy benchmark
    results = run_accuracy_benchmark(TEST_CASES, anthropic_counts)

    # Generate markdown output
    print(generate_accuracy_table_markdown(results, anthropic_counts))
    print()

    summary_md, summaries = generate_summary_table_markdown(results)
    print(summary_md)
    print()

    if summaries:
        print(generate_category_table_markdown(results, summaries[0]["name"]))
        print()
        print(generate_recommendations_markdown(summaries))


if __name__ == "__main__":
    main()
