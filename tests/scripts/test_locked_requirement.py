"""Tests for scripts/locked_requirement.py."""

from pathlib import Path

import pytest

from scripts.locked_requirement import locked_requirement

_LOCK_FILE = """
[[package]]
name = "pytest"
version = "9.2.0"
files = [
    {file = "pytest-9.2.0-py3-none-any.whl", hash = "sha256:aaa"},
]

[[package]]
name = "ruff"
version = "0.17.5"
files = [
    {file = "ruff-0.17.5-py3-none-linux_armv6l.whl", hash = "sha256:bbb"},
    {file = "ruff-0.17.5.tar.gz", hash = "sha256:ccc"},
]
"""


@pytest.fixture(name="lock_file")
def lock_file_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "poetry.lock"
    path.write_text(_LOCK_FILE)
    return path


@pytest.mark.parametrize(
    "package,expected",
    [
        ("ruff", "ruff==0.17.5 --hash=sha256:bbb --hash=sha256:ccc"),
        ("pytest", "pytest==9.2.0 --hash=sha256:aaa"),
    ],
)
def test_returns_the_version_and_the_hashes(
    lock_file: Path, package: str, expected: str
):
    assert locked_requirement(package, lock_file) == expected


def test_exits_when_the_package_is_not_locked(lock_file: Path):
    with pytest.raises(SystemExit, match="mypy is not in"):
        locked_requirement("mypy", lock_file)
