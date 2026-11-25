from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from ai_gateway.catalog.__main__ import main
from ai_gateway.model_selection.model_selection_config import (
    ModelSelectionConfig,
    shipped_catalog_provider,
)


@pytest.fixture(name="shipped_catalog")
def shipped_catalog_fixture() -> dict[str, Any]:
    """The shipped catalog files merged into one single-file catalog document."""
    documents = shipped_catalog_provider().load()
    return {
        **documents.models,
        **documents.unit_primitives,
        **documents.model_restrictions,
    }


def _write(tmp_path: Path, document: Any) -> str:
    path = tmp_path / "catalog.yml"
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    return str(path)


def test_validate_accepts_the_shipped_files(capsys):
    assert main(["validate"]) == 0
    assert "Catalog in the shipped catalog files is valid." in capsys.readouterr().out


def test_validate_accepts_the_shipped_files_as_one_catalog_file(
    tmp_path, shipped_catalog, capsys
):
    path = _write(tmp_path, shipped_catalog)

    assert main(["validate", path]) == 0
    assert f"Catalog in {path} is valid." in capsys.readouterr().out


def test_validate_reports_cross_checks(tmp_path, shipped_catalog, capsys):
    shipped_catalog["configurable_unit_primitives"][0]["default_models"] = [
        "no_such_model"
    ]
    path = _write(tmp_path, shipped_catalog)

    assert main(["validate", path]) == 1
    err = capsys.readouterr().err
    assert f"Catalog in {path} is invalid:" in err
    assert "no_such_model" in err


def test_validate_reports_schema_errors_with_their_location(
    tmp_path, shipped_catalog, capsys
):
    shipped_catalog["models"][0]["max_context_tokens"] = "lots"
    path = _write(tmp_path, shipped_catalog)

    assert main(["validate", path]) == 1
    assert "max_context_tokens" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ({"models": []}, "missing keys"),
        ("not a mapping", "must be a mapping"),
    ],
)
def test_validate_reports_a_catalog_with_the_wrong_shape(
    tmp_path, capsys, document, message
):
    assert main(["validate", _write(tmp_path, document)]) == 1
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    ("section", "entry", "message"),
    [
        (
            "models",
            "drop:gitlab_identifier",
            "models[0]: missing required key 'gitlab_identifier'",
        ),
        ("models", "not a mapping", "models[0]: must be a mapping"),
        (
            "configurable_unit_primitives",
            "drop:feature_setting",
            "configurable_unit_primitives[0]: missing required key 'feature_setting'",
        ),
        (
            "configurable_unit_primitives",
            "not a mapping",
            "configurable_unit_primitives[0]: must be a mapping",
        ),
    ],
)
def test_validate_names_the_entry_with_a_missing_key(
    tmp_path, shipped_catalog, capsys, section, entry, message
):
    if entry.startswith("drop:"):
        del shipped_catalog[section][0][entry.removeprefix("drop:")]
    else:
        shipped_catalog[section][0] = entry

    assert main(["validate", _write(tmp_path, shipped_catalog)]) == 1
    assert f"  {message}\n" in capsys.readouterr().err


def test_validate_reports_a_section_that_is_not_a_list(
    tmp_path, shipped_catalog, capsys
):
    shipped_catalog["models"] = {"gitlab_identifier": "model-1"}

    assert main(["validate", _write(tmp_path, shipped_catalog)]) == 1
    assert "models: must be a list" in capsys.readouterr().err


def test_validate_falls_back_to_the_exception(tmp_path, shipped_catalog, capsys):
    path = _write(tmp_path, shipped_catalog)

    with patch.object(
        ModelSelectionConfig, "validate", side_effect=KeyError("something_else")
    ) as validate:
        assert main(["validate", path]) == 1

    validate.assert_called_once_with()
    assert "KeyError: 'something_else'" in capsys.readouterr().err


def test_validate_reports_a_missing_file(tmp_path, capsys):
    assert main(["validate", str(tmp_path / "missing.yml")]) == 1
    assert "Could not load the catalog" in capsys.readouterr().err


def test_requires_a_command():
    with pytest.raises(SystemExit) as exc_info:
        main([])

    assert exc_info.value.code == 2
