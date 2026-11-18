"""Offline tests for the fake GLQL schema check.

The comparison with the live schema runs in `agent_tests/analytics_agent/test_glql_schema_fake_structure.py`.
"""

import copy
import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from scripts.check_glql_schema_fake import (
    FAKE_PATH,
    diff_structure,
    drift_report,
    fetch_schema,
    load_json,
    main,
    structure_of,
)


@pytest.fixture(name="fake", scope="module")
def fake_fixture() -> dict:
    return load_json(FAKE_PATH)


def _modes(document: dict, mode_name: str) -> list[tuple[str, dict]]:
    """Every source carrying the mode, with that mode."""
    return [
        (source["name"], mode)
        for source in document["sources"]
        for mode in source["modes"]
        if mode["mode"] == mode_name
    ]


def test_fake_uses_only_declared_kinds_and_operators(fake):
    """Every kind and operator a field uses is one the document's own vocabularies declare."""
    kinds = {k["name"] for k in fake["value_kinds"]}
    operators = {o["symbol"] for o in fake["operators"]}

    for source in fake["sources"]:
        for mode in source["modes"]:
            fields = mode.get("filter_fields", []) + mode.get(
                "wildcard_filter_fields", []
            )
            for field in fields:
                for value_type in field["value_types"]:
                    assert value_type["kind"] in kinds, (source["name"], field["name"])
                    assert set(value_type.get("operators", [])) <= operators, (
                        source["name"],
                        field["name"],
                    )
                    for item in value_type.get("items", []):
                        assert item["kind"] in kinds, (source["name"], field["name"])
            for field in mode.get("parameterized_fields", []):
                for parameter in field["parameters"]:
                    assert parameter["kind"] in kinds, (source["name"], field["name"])


def test_structure_ignores_trimmed_content(fake):
    """A copy with fewer fields and enum values keeps the structure, so trimming the fake stays allowed."""
    trimmed = copy.deepcopy(fake)
    _, mode = next(
        (name, mode)
        for name, mode in _modes(trimmed, "Standard")
        if mode.get("filter_fields")
    )
    for field in mode["filter_fields"]:
        for value_type in field["value_types"]:
            for enum in [value_type, *value_type.get("items", [])]:
                if len(enum.get("values", [])) > 1:
                    enum["values"] = enum["values"][:1]
    mode["sort_fields"] = mode["sort_fields"][:1]

    assert structure_of(trimmed) == structure_of(fake)


def test_structure_ignores_which_source_carries_a_mode(fake):
    """A mode is the union over its sources, so a source can be renamed, duplicated or dropped when others cover it."""
    drifted = copy.deepcopy(fake)
    original = drifted["sources"][0]
    drifted["sources"].append({**copy.deepcopy(original), "name": "Duplicate"})
    assert structure_of(drifted) == structure_of(fake)

    drifted["sources"].remove(original)
    assert structure_of(drifted) == structure_of(fake)


def test_structure_notices_a_dropped_mode_key(fake):
    drifted = copy.deepcopy(fake)
    analytics = _modes(drifted, "Analytics")
    assert analytics, "the fake has no Analytics mode"
    for _, mode in analytics:
        mode.pop("multi_scope", None)

    messages = diff_structure(structure_of(fake), structure_of(drifted))

    assert 'schema.sources.modes.Analytics.keys: missing "multi_scope"' in messages


def test_structure_notices_a_dropped_mode(fake):
    drifted = copy.deepcopy(fake)
    for source in drifted["sources"]:
        source["modes"] = [m for m in source["modes"] if m["mode"] != "Analytics"]

    messages = diff_structure(structure_of(fake), structure_of(drifted))

    assert "schema.sources.modes.Analytics: missing" in messages


def test_structure_notices_a_dropped_display_type_and_operator(fake):
    drifted = copy.deepcopy(fake)
    display_type = drifted["display_types"].pop()["name"]
    operator = drifted["operators"].pop()["symbol"]

    messages = diff_structure(structure_of(fake), structure_of(drifted))

    assert f'schema.display_types.names: missing "{display_type}"' in messages
    assert f"schema.display_types.selections.{display_type}: missing" in messages
    assert f'schema.operators.names: missing "{operator}"' in messages


def test_structure_notices_a_dropped_value_type_operator(fake):
    drifted = copy.deepcopy(fake)
    for source in drifted["sources"]:
        for mode in source["modes"]:
            for field in mode.get("filter_fields", []):
                for value_type in field["value_types"]:
                    value_type["operators"] = [
                        op for op in value_type["operators"] if op != "not in"
                    ]

    messages = diff_structure(structure_of(fake), structure_of(drifted))

    assert any(
        ".filter_fields.value_types: missing" in m and '"not in"' in m for m in messages
    )


def test_structure_notices_a_dropped_parameter_shape(fake):
    drifted = copy.deepcopy(fake)
    for source in drifted["sources"]:
        for mode in source["modes"]:
            for field in mode.get("parameterized_fields", []):
                field["parameters"] = [
                    p for p in field["parameters"] if p["kind"] != "List"
                ]

    messages = diff_structure(structure_of(fake), structure_of(drifted))

    assert any(
        ".parameterized_fields.parameters: missing" in m and '"kind":"List"' in m
        for m in messages
    )


def test_structure_notices_dropped_value_lists(fake):
    drifted = copy.deepcopy(fake)
    for _, mode in _modes(drifted, "Standard"):
        for field in mode.get("filter_fields", []):
            for value_type in field["value_types"]:
                for enum in [value_type, *value_type.get("items", [])]:
                    enum.pop("values", None)

    messages = diff_structure(structure_of(fake), structure_of(drifted))

    assert any(".filter_fields.value_types: missing" in m for m in messages)


def test_structure_notices_a_new_value_type_attribute(fake):
    drifted = copy.deepcopy(fake)
    mode = next(
        mode for _, mode in _modes(drifted, "Standard") if mode.get("filter_fields")
    )
    mode["filter_fields"][0]["value_types"][0]["deprecated"] = True

    messages = diff_structure(structure_of(fake), structure_of(drifted))

    assert any(
        ".filter_fields.value_types: unexpected" in m and '"deprecated"' in m
        for m in messages
    )


def test_structure_notices_a_new_list_item_attribute(fake):
    drifted = copy.deepcopy(fake)
    for _, mode in _modes(drifted, "Standard"):
        for field in mode.get("filter_fields", []):
            for value_type in field["value_types"]:
                for item in value_type.get("items", []):
                    item["deprecated"] = True

    messages = diff_structure(structure_of(fake), structure_of(drifted))

    assert any(
        ".filter_fields.value_types: unexpected" in m and '"deprecated"' in m
        for m in messages
    )


def test_drift_report_says_what_to_do_with_missing_and_unexpected_lines(fake):
    live = copy.deepcopy(fake)
    mode = next(
        mode for _, mode in _modes(live, "Standard") if mode.get("display_fields")
    )
    mode["display_fields"][0]["deprecated"] = True
    stale = copy.deepcopy(fake)

    report = "\n".join(drift_report(live, stale))

    assert (
        "add to the fake: schema.sources.modes.Standard.display_fields.keys" in report
    )
    assert 'an entry with keys ["deprecated","name"]' in report
    assert "copy a live entry such as" in report
    assert " > Standard > display_fields > " in report
    assert "poetry run python scripts/check_glql_schema_fake.py" in report

    reverse = "\n".join(drift_report(stale, live))
    assert "change or remove in the fake:" in reverse


def test_drift_report_words_a_missing_mode_key_as_a_key(fake):
    live = copy.deepcopy(fake)
    mode = next(mode for _, mode in _modes(live, "Analytics"))
    mode["brand_new_attribute"] = True
    stale = copy.deepcopy(fake)

    report = "\n".join(drift_report(live, stale))

    assert (
        'add to the fake: schema.sources.modes.Analytics.keys: the key "brand_new_attribute"'
        in report
    )
    assert "an entry with keys" not in report


def test_drift_report_is_empty_when_structures_match(fake):
    assert drift_report(fake, copy.deepcopy(fake)) == []


def test_structure_notices_a_new_parameter_attribute(fake):
    drifted = copy.deepcopy(fake)
    mode = next(
        mode
        for _, mode in _modes(drifted, "Analytics")
        if mode.get("parameterized_fields")
    )
    mode["parameterized_fields"][0]["parameters"][0]["step"] = 1

    messages = diff_structure(structure_of(fake), structure_of(drifted))

    assert any(
        m.startswith(
            "schema.sources.modes.Analytics.parameterized_fields.parameters: unexpected "
        )
        and '"step"' in m
        for m in messages
    )


def test_main_passes_when_live_matches_the_fake(fake, tmp_path, capsys):
    live = tmp_path / "live.json"
    live.write_text(json.dumps(fake))

    assert main(["--from-file", str(live)]) == 0
    assert "fake:" not in capsys.readouterr().err


def test_main_fails_on_drift(fake, tmp_path, capsys):
    live_document = copy.deepcopy(fake)
    live_document["operators"].append(
        {"symbol": "~", "name": "Matches", "label": "matches"}
    )
    live = tmp_path / "live.json"
    live.write_text(json.dumps(live_document))

    assert main(["--from-file", str(live)]) == 1
    assert 'add to the fake: schema.operators.names: "~"' in capsys.readouterr().err


def test_diff_structure_reports_a_changed_scalar():
    assert diff_structure({"multi_scope": True}, {"multi_scope": False}) == [
        "schema.multi_scope: expected True, got False"
    ]


def test_fetch_schema_reads_the_schema_endpoint_of_the_instance():
    response = MagicMock()
    response.json.return_value = {"version": "9.9.9"}

    with patch(
        "scripts.check_glql_schema_fake.requests.get", return_value=response
    ) as get:
        assert fetch_schema("https://gitlab.example.com/") == {"version": "9.9.9"}

    get.assert_called_once_with(
        "https://gitlab.example.com/api/v4/glql/schema", timeout=30
    )
    response.raise_for_status.assert_called_once_with()


def test_fetch_schema_raises_on_an_http_error():
    response = MagicMock()
    response.raise_for_status.side_effect = requests.HTTPError(
        "503 Service Unavailable"
    )

    with patch("scripts.check_glql_schema_fake.requests.get", return_value=response):
        with pytest.raises(requests.HTTPError):
            fetch_schema("https://gitlab.example.com")


def test_main_reads_the_schema_from_the_given_instance(fake, capsys):
    with patch(
        "scripts.check_glql_schema_fake.fetch_schema", return_value=fake
    ) as fetch:
        assert main(["--gitlab-url", "https://gitlab.example.com"]) == 0

    fetch.assert_called_once_with("https://gitlab.example.com")
    assert "live schema" in capsys.readouterr().out
