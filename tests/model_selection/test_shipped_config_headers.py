# pylint: disable=file-naming-for-tests
from pathlib import Path

import pytest
import yaml

from ai_gateway.model_selection.models import _ALLOWED_CLIENT_HEADER_NAMES

_REPO_ROOT = Path(__file__).resolve().parents[2]

_CONFIG_DIRS = (
    _REPO_ROOT / "duo_workflow_service/agent_platform/v1/flows/configs",
    _REPO_ROOT / "duo_workflow_service/agent_platform/experimental/flows/configs",
    _REPO_ROOT / "ai_gateway/prompts/definitions",
    _REPO_ROOT / "ai_gateway/model_selection",
)

_HEADER_FIELDS = ("extra_headers", "default_headers")


def _shipped_config_files() -> list[Path]:
    files: list[Path] = []
    for directory in _CONFIG_DIRS:
        files.extend(directory.rglob("*.yml"))
        files.extend(directory.rglob("*.yaml"))
    return sorted(files)


def _find_header_keys(node: object) -> list[str]:
    keys: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key in _HEADER_FIELDS and isinstance(value, dict):
                keys.extend(value.keys())
            keys.extend(_find_header_keys(value))
    elif isinstance(node, list):
        for item in node:
            keys.extend(_find_header_keys(item))
    return keys


def test_config_dirs_exist():
    for directory in _CONFIG_DIRS:
        assert directory.is_dir(), f"Expected config directory missing: {directory}"


@pytest.mark.parametrize(
    "config_file", _shipped_config_files(), ids=lambda p: str(p.relative_to(_REPO_ROOT))
)
def test_shipped_config_headers_are_allowlisted(config_file: Path):
    documents = yaml.safe_load_all(config_file.read_text())

    violations = set()
    for document in documents:
        for header in _find_header_keys(document):
            if header.strip().lower() not in _ALLOWED_CLIENT_HEADER_NAMES:
                violations.add(header)

    assert not violations, (
        f"{config_file.relative_to(_REPO_ROOT)} sets non-allowlisted header(s) "
        f"{sorted(violations)}. Model header maps are restricted to "
        f"{sorted(_ALLOWED_CLIENT_HEADER_NAMES)}; update "
        "ai_gateway/model_selection/models.py:_ALLOWED_CLIENT_HEADER_NAMES to add one."
    )
