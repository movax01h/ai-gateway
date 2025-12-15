#!/usr/bin/env python
"""Latency and Accuracy Benchmark: ApproximateTokenCounter vs TikTokenCounter vs Pure Tiktoken"""
import random
import time

import tiktoken

from duo_workflow_service.token_counter.tiktoken_counter import TikTokenCounter


# Simulate the old ApproximateTokenCounter
# Replaced in https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/4129/
class ApproximateTokenCounter:
    def __init__(self, agent_name: str):
        self.tool_tokens = {"executor": 5650}.get(agent_name, 0)

    def count_string_content(self, text: str) -> int:
        return int(round(len(text) // 4 * 1.5))


# Pure tiktoken (ground truth for accuracy)
class PureTiktoken:
    def __init__(self):
        self._encoding = tiktoken.encoding_for_model("gpt-4o")

    def count_string_content(self, text: str) -> int:
        return len(self._encoding.encode(text))


def benchmark(func, iterations: int = 100):
    start = time.perf_counter()
    for _ in range(iterations):
        result = func()
    elapsed = time.perf_counter() - start
    avg_ms = (elapsed / iterations) * 1000
    return avg_ms, result


# Helper functions to generate realistic test cases
def generate_code_file_with_dense_imports(size: int) -> str:
    """Simulates a Python file with dense imports at the top, sparse code below."""
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
    """File with alternating dense and sparse regions."""
    dense_block = "αβγδεζηθικλμνξοπρστυφχψω" * 20  # Greek letters (high token density)
    sparse_block = "x" * 500  # ASCII (low token density)
    pattern = (dense_block + sparse_block) * (
        size // (len(dense_block) + len(sparse_block))
    )
    return pattern[:size]


def generate_json_with_unicode_values(size: int) -> str:
    """JSON with Unicode strings in values - common in i18n data."""
    entries = []
    languages = ["你好世界", "こんにちは", "مرحبا", "Привет", "🚀🎉✨"]
    for i in range(size // 50):
        lang = languages[i % len(languages)]
        entries.append(f'"{lang}_{i}": "{lang * 3}"')
    return "{" + ", ".join(entries) + "}"


def generate_minified_js(size: int) -> str:
    """Minified JavaScript - no whitespace, high symbol density."""
    code = (
        "var a=function(b,c){return b+c};"
        "var d=function(e){for(var i=0;i<e.length;i++){console.log(e[i])}};"
        "var obj={key1:'value1',key2:'value2',key3:[1,2,3,4,5]};"
    )
    return (code * (size // len(code) + 1))[:size]


def generate_log_file_with_timestamps(size: int) -> str:
    """Log file with repetitive timestamps but variable message content."""
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
    """Sparse ASCII at start, dense Unicode at end - tests if sampling catches the end."""
    sparse_part = "a " * (size // 2)  # Very sparse
    dense_part = "🚀🎉✨💻🔥" * (size // 10)  # Very dense
    return sparse_part + dense_part


def generate_dense_then_sparse(size: int) -> str:
    """Dense Unicode at start, sparse ASCII at end."""
    dense_part = "你好世界！" * (size // 10)
    sparse_part = "x " * (size // 2)
    return dense_part + sparse_part


def generate_dense_middle_only(size: int) -> str:
    """Sparse edges, dense middle - tests center sampling."""
    third = size // 3
    sparse = "a " * third
    dense = "🚀αβγ你好" * (third // 6)
    return sparse + dense + sparse


def generate_random_density_spikes(size: int) -> str:
    """Random spikes of high-density content throughout."""
    random.seed(42)  # Reproducible
    result = []
    dense_chars = "🚀✨🎉💻🔥你好世界αβγδ"
    sparse_chars = "abcdefghijklmnopqrstuvwxyz      "

    for _ in range(size):
        if random.random() < 0.1:  # 10% chance of dense character
            result.append(random.choice(dense_chars))
        else:
            result.append(random.choice(sparse_chars))
    return "".join(result)


def generate_adversarial_for_center_sampling(size: int) -> str:
    """Dense content exactly where center sampling WON'T look."""
    cell_size = size // 30
    sample_size = 300
    offset = (cell_size - sample_size) // 2

    result = []
    for _ in range(30):
        # Sparse in the center (where we sample)
        center_sparse = "a " * (sample_size // 2)
        # Dense at the edges (where we don't sample)
        edge_dense = "🚀你好" * (offset // 4) if offset > 0 else ""
        result.append(edge_dense + center_sparse + edge_dense)
    return "".join(result)[:size]


def generate_chat_conversation(size: int) -> str:
    """Simulates actual chat messages - what the trimmer processes."""
    messages: list = []
    templates = [
        "User: Can you help me with this code?\n",
        "Assistant: Of course! The problem is in `calculate_αβγ()` where 你好 🚀\n",
        "User: Thanks! Error: TypeError: cannot convert '🎉' to int\n",
        "Assistant: " + "x" * 200 + "\n",  # Long sparse response
        "User: " + "αβγδ" * 50 + "\n",  # Dense Unicode input
    ]
    while len("".join(messages)) < size:
        messages.append(templates[len(messages) % len(templates)])
    return "".join(messages)[:size]


def generate_tool_call_json(size: int) -> str:
    """Simulates tool call responses with nested JSON - common in agent workflows."""
    tool_calls = []
    for i in range(size // 300):
        tool_calls.append(
            f'{{"tool_call_id": "call_{i}", "function": {{"name": "read_file", '
            f'"arguments": {{"path": "/src/components/файл_{i}.tsx", "content": "const x = 你好🚀"}}}}}}'
        )
    return "[" + ",\n".join(tool_calls) + "]"


def generate_git_diff(size: int) -> str:
    """Git diff output - mix of metadata and code changes."""
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
    """Just over 9000 char threshold with dense content - first case to use sampling."""
    return "🚀你好" * (size // 6)


def generate_threshold_boundary_sparse(size: int = 9001) -> str:
    """Just over 9000 char threshold with sparse content."""
    return "a " * (size // 2)


# Test cases - now much more challenging
test_cases = [
    ("Short text (50 chars)", "Hello, how are you doing today? This is a test."),
    ("Medium text (1KB)", "x" * 1024),
    ("Large text (10KB)", "x" * 10240),
    ("Unicode heavy", "你好世界！" * 200),
    ("Emoji heavy", "🚀✨🎉💻🔥" * 200),
    ("JSON data", '{"users": [{"id": 1, "name": "Alice"}]}' * 50),
    # Challenging cases that expose sampling weaknesses
    ("Code w/ dense imports (10KB)", generate_code_file_with_dense_imports(10240)),
    ("Mixed density alternating (10KB)", generate_mixed_density_file(10240)),
    ("JSON with Unicode values (10KB)", generate_json_with_unicode_values(10240)),
    ("Minified JavaScript (10KB)", generate_minified_js(10240)),
    ("Log file with timestamps (10KB)", generate_log_file_with_timestamps(10240)),
    ("Sparse→Dense (10KB)", generate_sparse_then_dense(10240)),
    ("Dense→Sparse (10KB)", generate_dense_then_sparse(10240)),
    ("Dense middle only (10KB)", generate_dense_middle_only(10240)),
    ("Random density spikes (10KB)", generate_random_density_spikes(10240)),
    # Larger files where sampling matters more
    ("Large mixed density (50KB)", generate_mixed_density_file(51200)),
    ("Large random spikes (50KB)", generate_random_density_spikes(51200)),
    ("Large sparse→dense (50KB)", generate_sparse_then_dense(51200)),
    # Algorithm-specific edge cases
    (
        "Adversarial center sampling (10KB)",
        generate_adversarial_for_center_sampling(10240),
    ),
    # Real workflow content types
    ("Chat conversation (10KB)", generate_chat_conversation(10240)),
    ("Tool call JSON (10KB)", generate_tool_call_json(10240)),
    ("Git diff output (10KB)", generate_git_diff(10240)),
    # Threshold boundary tests (9000 = 30 * 300)
    ("Just over threshold - dense (9001)", generate_threshold_boundary_dense(9001)),
    ("Just over threshold - sparse (9001)", generate_threshold_boundary_sparse(9001)),
    ("Just under threshold - dense (8999)", "🚀你好" * 1499),
]

approx_counter = ApproximateTokenCounter("executor")
tiktoken_counter = TikTokenCounter("executor")
pure_tiktoken = PureTiktoken()

print("=" * 110)
print("LATENCY BENCHMARK")
print("=" * 110)
print(
    f"{'Test Case':<35} {'Approximate (ms)':<18} {'Sampling (ms)':<18} {'Pure Tiktoken (ms)':<20} {'Sampling Speedup'}"
)
print("-" * 110)

for name, content in test_cases:
    approx_ms, _ = benchmark(lambda c=content: approx_counter.count_string_content(c))
    sampling_ms, _ = benchmark(
        lambda c=content: tiktoken_counter.count_string_content(c)
    )
    pure_ms, _ = benchmark(lambda c=content: pure_tiktoken.count_string_content(c))

    speedup = f"{pure_ms / sampling_ms:.1f}x" if sampling_ms > 0 else "N/A"
    print(
        f"{name:<35} {approx_ms:<18.4f} {sampling_ms:<18.4f} {pure_ms:<20.4f} {speedup}"
    )

print("=" * 110)
print("\n")
print("=" * 110)
print("ACCURACY BENCHMARK (difference from Pure Tiktoken)")
print("=" * 110)
print(
    f"{'Test Case':<35} {'Pure Tiktoken':<15} {'Approximate':<20} {'Sampling':<20} {'Sampling %Err'}"
)
print("-" * 110)

for name, content in test_cases:
    actual = pure_tiktoken.count_string_content(content)
    approx = approx_counter.count_string_content(content)
    sampling = tiktoken_counter.count_string_content(content)

    approx_diff = approx - actual
    sampling_diff = sampling - actual
    sampling_pct_err = (sampling_diff / actual * 100) if actual > 0 else 0

    approx_str = f"{approx} ({approx_diff:+d})"
    sampling_str = f"{sampling} ({sampling_diff:+d})"
    pct_str = f"{sampling_pct_err:+.1f}%"

    print(f"{name:<35} {actual:<15} {approx_str:<20} {sampling_str:<20} {pct_str}")

print("=" * 110)

# Summary statistics
print("\n")
print("=" * 110)
print("SUMMARY: Undercount Analysis (negative diff = DANGEROUS undercount)")
print("=" * 110)

undercounts = []
for name, content in test_cases:
    actual = pure_tiktoken.count_string_content(content)
    sampling = tiktoken_counter.count_string_content(content)
    diff = sampling - actual
    pct = (diff / actual * 100) if actual > 0 else 0
    if diff < 0:
        undercounts.append((name, diff, pct))

if undercounts:
    print("UNDERCOUNTED CASES (these could cause 'prompt is too long' errors):")
    for name, diff, pct in sorted(undercounts, key=lambda x: x[1]):
        print(f"   {name}: {diff:+d} tokens ({pct:+.1f}%)")
else:
    print("No undercounts detected - sampling is conservative enough")

print("=" * 110)
