#!/usr/bin/env python
"""Keep the analytics agent tests' fake GLQL schema in step with the live one.

`agent_tests/analytics_agent/glql_schema_fake.json` stands in for `GET /api/v4/glql/schema` when the 2.0.0 analytics
agent runs in `agent_tests`. The agent prompt makes claims about that document's shape (modes, parameterized fields,
value kinds), so the fake has to keep the structure the instance serves, not just a plausible vocabulary.

    poetry run python scripts/check_glql_schema_fake.py  # compare with gitlab.com, exit 1 on drift

The fake is a representative subset, not a copy: it has to keep the structure the instance serves, and nothing else.
The structure leaves out field names, enum values, descriptions and which data source carries what, so a fake with
fewer fields and sources still passes as long as every mode, value kind, operator, parameter shape and display type
the live schema has is represented somewhere. When the structure drifts, add a representative example to the fake by
hand. `agent_tests/analytics_agent/test_glql_schema_fake_structure.py` runs the same comparison with the agent tests.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import requests

_REPO_ROOT = Path(__file__).resolve().parents[1]
FAKE_PATH = _REPO_ROOT / "agent_tests/analytics_agent/glql_schema_fake.json"
_SCHEMA_PATH = "/api/v4/glql/schema"
_DEFAULT_GITLAB_URL = "https://gitlab.com"

# Top-level vocabularies and the key that names an entry. These are the language
# itself rather than trimmable content, so their names are part of the structure.
_VOCABULARY_NAME_KEYS = {
    "operators": "symbol",
    "value_kinds": "name",
    "reference_types": "name",
    "functions": "name",
    "display_types": "name",
}
# Mode keys whose values describe the mode itself rather than list its fields.
_MODE_ATTRIBUTE_KEYS = ("mode", "allowed_scopes", "multi_scope")

Structure = dict[str, Any]


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _unique(values: list[Any]) -> list[Any]:
    """Deduplicate ``values`` and order them deterministically, whatever their type."""
    return [json.loads(text) for text in sorted({_canonical(v) for v in values})]


def _describe_entries(entries: list[Any]) -> Structure:
    """Describe a list by the key sets of its dict entries, or the types of its scalar ones."""
    if entries and all(isinstance(entry, dict) for entry in entries):
        return {"keys": _unique([sorted(entry) for entry in entries])}
    return {"kinds": _unique([type(entry).__name__ for entry in entries])}


def _item_shapes(items: list[dict[str, Any]]) -> list[Structure]:
    """Describe each List item by kind (qualified by what a Reference item refers to) and attribute names."""
    return _unique(
        [
            {
                "kind": item["kind"]
                + (f":{item['references']}" if "references" in item else ""),
                "keys": sorted(item),
            }
            for item in items
        ]
    )


def _value_type_shape(value_type: dict[str, Any]) -> Structure:
    shape: Structure = {
        "kind": value_type["kind"],
        "keys": sorted(value_type),
        "operators": sorted(value_type.get("operators", [])),
    }
    if "items" in value_type:
        shape["items"] = _item_shapes(value_type["items"])
    return shape


def _parameter_shape(parameter: dict[str, Any]) -> Structure:
    shape: Structure = {"kind": parameter["kind"], "keys": sorted(parameter)}
    if "items" in parameter:
        shape["items"] = _item_shapes(parameter["items"])
    return shape


def _mode_structure(mode: dict[str, Any]) -> Structure:
    structure: Structure = {
        "keys": sorted(mode),
        "allowed_scopes": sorted(mode.get("allowed_scopes", [])),
        "multi_scope": mode.get("multi_scope"),
    }
    for key, entries in mode.items():
        if key in _MODE_ATTRIBUTE_KEYS or not isinstance(entries, list):
            continue
        described = _describe_entries(entries)
        if key in ("filter_fields", "wildcard_filter_fields"):
            described["value_types"] = _unique(
                [_value_type_shape(vt) for f in entries for vt in f["value_types"]]
            )
        elif key == "parameterized_fields":
            described["parameters"] = _unique(
                [_parameter_shape(p) for f in entries for p in f["parameters"]]
            )
        structure[key] = described
    return structure


def _modes_structure(sources: list[dict[str, Any]]) -> Structure:
    """Describe each mode once, as the union of its shape over every source that offers it."""
    by_mode: dict[str, list[Structure]] = {}
    for source in sources:
        for mode in source["modes"]:
            by_mode.setdefault(mode["mode"], []).append(_mode_structure(mode))
    return {name: _union(shapes) for name, shapes in sorted(by_mode.items())}


def _union(shapes: list[Structure]) -> Structure:
    """Merge mode structures: lists are united, attributes of the mode itself are kept as a set of values."""
    merged: Structure = {}
    for shape in shapes:
        for key, value in shape.items():
            if isinstance(value, dict):
                merged[key] = _union([merged.get(key, {}), value])
            elif isinstance(value, list):
                merged[key] = _unique(merged.get(key, []) + value)
            else:
                merged[key] = _unique(merged.get(key, []) + [value])
    return merged


def structure_of(document: dict[str, Any]) -> Structure:
    """Reduce a GLQL schema document to the shape the agent prompt depends on.

    Field names, enum values, labels, descriptions and the gem ``version`` are left out, and a mode is described by the
    union of its shape over all sources, so two documents compare equal when one is a trimmed copy of the other that
    still covers every mode and every kind of entry, even if it leaves out whole sources.
    """
    structure: Structure = {"keys": sorted(document)}
    for key, value in document.items():
        if key == "version" or not isinstance(value, list):
            continue
        if key == "sources":
            structure[key] = {
                "keys": _unique([sorted(source) for source in value]),
                "modes": _modes_structure(value),
            }
            continue
        described = _describe_entries(value)
        name_key = _VOCABULARY_NAME_KEYS.get(key)
        if name_key:
            described["names"] = sorted(entry[name_key] for entry in value)
        if key == "display_types":
            described["selections"] = {
                entry["name"]: entry.get("selections") for entry in value
            }
        if key == "functions":
            described["args"] = _unique(
                [sorted(arg) for entry in value for arg in entry.get("args", [])]
            )
        structure[key] = described
    return structure


def _examples(document: dict[str, Any]) -> dict[tuple[str, str], str]:
    """Locate, for each mode entry shape, one entry of ``document`` that has it, as ``Source > Mode > key > name``."""
    found: dict[tuple[str, str], str] = {}

    def record(mode: str, key: str, part: str, value: Any, where: str) -> None:
        found.setdefault(
            (f"schema.sources.modes.{mode}.{key}.{part}", _canonical(value)), where
        )

    for source in document.get("sources", []):
        for mode in source["modes"]:
            for key, entries in mode.items():
                if key in _MODE_ATTRIBUTE_KEYS or not isinstance(entries, list):
                    continue
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    where = " > ".join(
                        [source["name"], mode["mode"], key, str(entry.get("name", "?"))]
                    )
                    record(mode["mode"], key, "keys", sorted(entry), where)
                    for value_type in entry.get("value_types", []):
                        record(
                            mode["mode"],
                            key,
                            "value_types",
                            _value_type_shape(value_type),
                            where,
                        )
                    for parameter in entry.get("parameters", []):
                        record(
                            mode["mode"],
                            key,
                            "parameters",
                            _parameter_shape(parameter),
                            where,
                        )
    return found


_DRIFT_LINE = re.compile(r"^(?P<path>\S+): (?P<kind>missing|unexpected) (?P<value>.+)$")


def drift_report(live: dict[str, Any], fake: dict[str, Any]) -> list[str]:
    """Turn the structure diff of ``fake`` against ``live`` into lines that say what to change in the fake.

    Empty when the structures match.
    """
    messages = diff_structure(structure_of(live), structure_of(fake))
    if not messages:
        return []
    examples = _examples(live)
    lines = []
    for message in messages:
        match = _DRIFT_LINE.match(message)
        if not match:
            lines.append(f"  {message}")
            continue
        path, kind, value = match["path"], match["kind"], match["value"]
        if not path.endswith(".keys"):
            what = value
        elif value.startswith("["):
            what = f"an entry with keys {value}"
        else:
            what = f"the key {value}"
        if kind == "missing":
            example = examples.get((path, value))
            copy_from = f" (copy a live entry such as {example})" if example else ""
            lines.append(f"  add to the fake: {path}: {what}{copy_from}")
        else:
            lines.append(f"  change or remove in the fake: {path}: {what}")
    return [
        *lines,
        "Each 'add' line is a shape the live schema has and no fake entry has. Each 'change or remove' line is a "
        "shape only the fake has; a changed entry shows up as both, so fix the entry rather than adding a new one. "
        "Edit agent_tests/analytics_agent/glql_schema_fake.json, then re-check with "
        "`poetry run python scripts/check_glql_schema_fake.py`.",
    ]


def diff_structure(expected: Any, actual: Any, path: str = "schema") -> list[str]:
    """List every way ``actual`` departs from ``expected``, one message per difference."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        messages: list[str] = []
        for key in sorted(set(expected) | set(actual)):
            child = f"{path}.{key}"
            if key not in actual:
                messages.append(f"{child}: missing")
            elif key not in expected:
                messages.append(f"{child}: unexpected")
            else:
                messages.extend(diff_structure(expected[key], actual[key], child))
        return messages

    if isinstance(expected, list) and isinstance(actual, list):
        expected_set = {_canonical(v) for v in expected}
        actual_set = {_canonical(v) for v in actual}
        return [f"{path}: missing {v}" for v in sorted(expected_set - actual_set)] + [
            f"{path}: unexpected {v}" for v in sorted(actual_set - expected_set)
        ]

    if expected != actual:
        return [f"{path}: expected {expected!r}, got {actual!r}"]
    return []


def fetch_schema(gitlab_url: str) -> dict[str, Any]:
    """Read the schema the instance at ``gitlab_url`` serves.

    The endpoint is public.
    """
    response = requests.get(gitlab_url.rstrip("/") + _SCHEMA_PATH, timeout=30)
    response.raise_for_status()
    return response.json()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    """Check the fake against the live schema and return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--gitlab-url",
        default=_DEFAULT_GITLAB_URL,
        help=f"instance to read the live schema from (default: {_DEFAULT_GITLAB_URL})",
    )
    parser.add_argument(
        "--from-file",
        type=Path,
        metavar="PATH",
        help="read the live schema from a saved JSON document instead of the instance",
    )
    args = parser.parse_args(argv)

    live = (
        load_json(args.from_file) if args.from_file else fetch_schema(args.gitlab_url)
    )
    fake = load_json(FAKE_PATH)
    print(f"live schema {live.get('version')}, fake schema {fake.get('version')}")
    messages = drift_report(live, fake)
    if messages:
        print(
            "The fake GLQL schema has drifted from the live one:\n"
            + "\n".join(messages),
            file=sys.stderr,
        )
    return 1 if messages else 0


if __name__ == "__main__":
    sys.exit(main())
