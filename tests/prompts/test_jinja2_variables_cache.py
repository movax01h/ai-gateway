import random
import sys
import threading
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import Mock, patch

import pytest
import yaml

from ai_gateway.prompts import base, feature_roots
from ai_gateway.prompts.base import (
    _JINJA2_VARIABLES_CACHE_MAX_BYTES,
    _JINJA2_VARIABLES_CACHE_MAX_ENTRY_BYTES,
    _parse_jinja2_variables,
    _parse_jinja2_variables_recursively,
)
from ai_gateway.prompts.jinja2_variables_cache import (
    TemplateVariablesCache,
    entry_weight,
)
from duo_workflow_service.agent_platform.experimental.flows.flow_config import (
    FlowConfig as ExperimentalFlowConfig,
)
from duo_workflow_service.agent_platform.v1.flows.flow_config import FlowConfig
from lib.feature_roots import default_features_dir, default_project_root


def _names(count: int) -> frozenset[str]:
    return frozenset(f"name_{index}" for index in range(count))


class TestTemplateVariablesCache:
    def test_a_hit_returns_the_kept_variables_without_computing_them_again(self):
        compute = Mock(return_value=frozenset({"a"}))
        cache = TemplateVariablesCache(max_bytes=10_000, max_entry_bytes=10_000)

        assert cache.get("{{ a }}", compute) == {"a"}
        assert cache.get("{{ a }}", compute) == {"a"}

        compute.assert_called_once_with("{{ a }}")
        assert len(cache) == 1
        assert cache.bytes_held == entry_weight("{{ a }}", frozenset({"a"}))

    def test_the_least_recently_used_entry_leaves_first(self):
        variables = frozenset({"x"})
        # Same-length sources sharing one set, so every entry weighs the same.
        weight = entry_weight("{{ t0 }}", variables)
        cache = TemplateVariablesCache(max_bytes=2 * weight, max_entry_bytes=weight)
        compute = Mock(return_value=variables)

        cache.get("{{ t0 }}", compute)
        cache.get("{{ t1 }}", compute)
        cache.get("{{ t0 }}", compute)  # refreshes t0, so t2 evicts t1
        cache.get("{{ t2 }}", compute)
        compute.reset_mock()

        cache.get("{{ t0 }}", compute)
        compute.assert_not_called()
        cache.get("{{ t1 }}", compute)
        compute.assert_called_once_with("{{ t1 }}")

    def test_the_bytes_held_stay_within_the_budget(self):
        cache = TemplateVariablesCache(max_bytes=50_000, max_entry_bytes=50_000)

        for index in range(500):
            cache.get(f"{{{{ v{index} }}}}", lambda template: frozenset({template}))

        assert 0 < cache.bytes_held <= 50_000
        assert 0 < len(cache) < 500

    def test_an_entry_heavier_than_the_entry_limit_is_returned_but_not_kept(self):
        variables = _names(2000)
        compute = Mock(return_value=variables)
        cache = TemplateVariablesCache(
            max_bytes=10_000_000,
            max_entry_bytes=entry_weight("{{ x }}", variables) - 1,
        )

        assert cache.get("{{ x }}", compute) == variables
        assert cache.get("{{ x }}", compute) == variables

        assert compute.call_count == 2
        assert len(cache) == 0
        assert cache.bytes_held == 0

    def test_the_weight_counts_the_variables_not_only_the_source(self):
        variables = _names(5000)
        template = "{{ " + ",".join(sorted(variables)) + " }}"

        assert entry_weight(template, variables) > 5 * sys.getsizeof(template)

    def test_the_weight_counts_bytes_not_characters(self):
        ascii_template = "{{ a }}" + "x" * 1000
        emoji_template = "{{ a }}" + "\U0001f600" * 1000

        assert len(ascii_template) == len(emoji_template)
        assert entry_weight(emoji_template, frozenset()) > 3 * entry_weight(
            ascii_template, frozenset()
        )

    def test_errors_are_not_cached(self):
        compute = Mock(side_effect=[ValueError("unreadable include"), frozenset({"a"})])
        cache = TemplateVariablesCache(max_bytes=10_000, max_entry_bytes=10_000)

        with pytest.raises(ValueError, match="unreadable include"):
            cache.get("{{ a }}", compute)
        assert len(cache) == 0

        assert cache.get("{{ a }}", compute) == {"a"}
        assert len(cache) == 1

    def test_clear_removes_every_entry(self):
        cache = TemplateVariablesCache(max_bytes=10_000, max_entry_bytes=10_000)
        cache.get("{{ a }}", lambda template: frozenset({"a"}))

        cache.clear()

        assert len(cache) == 0
        assert cache.bytes_held == 0

    def test_concurrent_lookups_stay_correct_and_within_the_budget(self):
        def compute(template: str) -> frozenset[str]:
            return frozenset({template[3:-3]})

        keys = [f"{{{{ k{index:02d} }}}}" for index in range(40)]
        # Same-length keys, so every entry weighs the same.
        weight = entry_weight(keys[0], compute(keys[0]))
        cache = TemplateVariablesCache(max_bytes=10 * weight, max_entry_bytes=weight)
        wrong: list[str] = []

        def look_up(seed: int) -> None:
            pick = random.Random(seed)
            for _ in range(2000):
                key = pick.choice(keys)
                if cache.get(key, compute) != compute(key):
                    wrong.append(key)

        threads = [threading.Thread(target=look_up, args=(seed,)) for seed in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not wrong
        assert cache.bytes_held == len(cache) * weight
        assert cache.bytes_held <= 10 * weight


def _strings(node: Any) -> Iterator[str]:
    if isinstance(node, str):
        yield node
    elif isinstance(node, list):
        for item in node:
            yield from _strings(item)
    elif isinstance(node, dict):
        for value in node.values():
            yield from _strings(value)


def _authored_template_sources() -> set[str]:
    prompt_files = [
        *(Path(base.__file__).parent / "definitions").glob("**/*.yml"),
        *default_features_dir().glob("*/*/prompts/**/*.yml"),
        *(default_project_root() / "ai" / "shared").glob("*/prompts/**/*.yml"),
    ]
    flow_files = [
        *FlowConfig.DIRECTORY_PATH.glob("**/*.yml"),
        *default_features_dir().glob("*/*/config/*.yml"),
        *ExperimentalFlowConfig.DIRECTORY_PATH.glob("**/*.yml"),
    ]

    sources: set[str] = set()
    for path in prompt_files:
        data = yaml.safe_load(path.read_text())
        if isinstance(data, dict):
            sources.update(_strings(data.get("prompt_template", {})))
    for path in flow_files:
        data = yaml.safe_load(path.read_text())
        if isinstance(data, dict):
            sources.update(_strings(data.get("prompts", [])))

    return {source for source in sources if "{" in source}


@pytest.mark.parametrize(
    "parse",
    [
        pytest.param(_parse_jinja2_variables, id="from-template"),
        pytest.param(_parse_jinja2_variables_recursively, id="recursive"),
    ],
)
def test_every_authored_template_is_kept_and_all_of_them_fit_together(parse):
    with patch.dict(feature_roots._FEATURE_PROMPT_ROOTS):
        feature_roots.discover_feature_prompts()
        weights = {
            source: entry_weight(source, parse(source))
            for source in _authored_template_sources()
        }

    assert len(weights) > 300
    heaviest = max(weights, key=weights.__getitem__)
    assert weights[heaviest] <= _JINJA2_VARIABLES_CACHE_MAX_ENTRY_BYTES, heaviest[:80]
    assert sum(weights.values()) <= _JINJA2_VARIABLES_CACHE_MAX_BYTES / 2
