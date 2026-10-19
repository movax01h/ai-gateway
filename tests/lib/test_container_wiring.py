"""Boot-time wiring validation: fail loud on unresolved injection markers."""

import os
import re
import subprocess
import sys
import warnings
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from dependency_injector import containers, providers

from lib.container_wiring import UnwiredDependencyError, wire_and_validate
from tests.lib import wiring_fixtures, wiring_fixtures_unwired

REPO_ROOT = Path(__file__).resolve().parents[2]


class _Container(containers.DeclarativeContainer):
    present = providers.Object("resolved")


@pytest.fixture
def container():
    c = _Container()
    yield c
    c.unwire()


def test_raises_when_marker_unresolved(container):
    # needs_missing references a provider the container lacks.
    with pytest.raises(UnwiredDependencyError, match="Unresolved dependency-injection"):
        wire_and_validate(container, modules=[wiring_fixtures_unwired])


def test_lists_each_unresolved_marker_once(container):
    # wire() warns twice per marker from the same call site; the message must
    # not repeat it.
    with pytest.raises(UnwiredDependencyError) as exc_info:
        wire_and_validate(container, modules=[wiring_fixtures_unwired])
    assert str(exc_info.value).count("needs_missing") == 1


def test_succeeds_when_all_markers_resolve(container):
    # needs_present resolves against the container's `present` provider.
    wire_and_validate(container, modules=[wiring_fixtures])
    assert wiring_fixtures.needs_present() == "resolved"


def _validate_wiring_in_subprocess(code: str) -> None:
    """Run a real-container wiring validation in a fresh interpreter.

    wire()/unwire() patch @inject targets at the module level, so doing either in-process would clobber wiring set up by
    other tests in the same session (e.g. the module-scoped container fixture in tests/duo_workflow_service).
    """
    env = {
        **os.environ,  # pylint: disable=direct-environment-variable-reference
        "PYTHONPATH": str(REPO_ROOT),
        "HF_HUB_OFFLINE": "1",
        "LANGCHAIN_TRACING_V2": "false",
    }
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_real_dws_container_wires_clean():
    # Every @inject target reachable from the DWS container resolves at boot.
    _validate_wiring_in_subprocess(
        "from ai_gateway.container import ContainerApplication\n"
        "from duo_workflow_service.server import CONTAINER_APPLICATION_PACKAGES\n"
        "from lib.container_wiring import wire_and_validate\n"
        "wire_and_validate(ContainerApplication(), packages=CONTAINER_APPLICATION_PACKAGES)\n"
    )


def test_real_aigw_container_wires_clean():
    # Every @inject target in AIGW's explicit module list resolves at boot.
    _validate_wiring_in_subprocess(
        "from ai_gateway.api.server import CONTAINER_APPLICATION_MODULES\n"
        "from ai_gateway.container import ContainerApplication\n"
        "from lib.container_wiring import wire_and_validate\n"
        "wire_and_validate(ContainerApplication(), modules=CONTAINER_APPLICATION_MODULES)\n"
    )


def test_every_inject_site_is_wired_by_the_app_that_serves_it():
    """A module outside the wired sets unwires silently, so pin the consumer side.

    ai_gateway modules must be in AIGW's explicit list. duo_workflow_service and ai are wired recursively by DWS.
    Anything else with a marker (lib/, or an ai/ module served by AIGW) must be classified below, which forces a
    conscious wiring decision instead of a silent no-op.
    """
    from ai_gateway.api.server import CONTAINER_APPLICATION_MODULES
    from duo_workflow_service.server import CONTAINER_APPLICATION_PACKAGES

    assert "duo_workflow_service" in CONTAINER_APPLICATION_PACKAGES
    assert "ai" in CONTAINER_APPLICATION_PACKAGES

    # module -> how it is wired: "aigw" (must be in the AIGW module list),
    # "dws" (covered by the recursive package wiring), or "doc-only" (the
    # marker text appears only in prose).
    classified = {
        "lib.container_wiring": "doc-only",
    }

    repo_root = Path(__file__).resolve().parents[2]
    marker = re.compile(r"^\s*@inject\b|Provide\[|Provider\[|Closing\[", re.M)
    aigw_wired = set(CONTAINER_APPLICATION_MODULES)

    problems = []
    for source_root in ("ai_gateway", "ai", "lib"):
        # rglob on a missing dir yields nothing, which would silently skip
        # part of the scan — fail instead if a scanned root disappears.
        assert (repo_root / source_root).is_dir(), f"scan root {source_root} missing"
        for path in (repo_root / source_root).rglob("*.py"):
            if "tests" in path.relative_to(repo_root).parts:
                continue
            if not marker.search(path.read_text(encoding="utf-8")):
                continue
            module = ".".join(path.relative_to(repo_root).with_suffix("").parts)
            module = module.removesuffix(".__init__")
            kind = classified.get(module)
            if source_root == "ai_gateway":
                if module not in aigw_wired:
                    problems.append(f"{module}: not in CONTAINER_APPLICATION_MODULES")
            elif kind == "doc-only":
                continue
            elif kind == "aigw":
                if module not in aigw_wired:
                    problems.append(f"{module}: classified aigw but not in the list")
            elif kind == "dws":
                continue  # covered by the recursive package wiring asserted above
            else:
                problems.append(
                    f"{module}: has an injection marker but is not classified; "
                    "add it to the map above and wire it in the app that serves it"
                )

    assert not problems, "\n".join(problems)


def test_replays_unrelated_warnings():
    # Warnings emitted while wiring must reach the caller, not vanish
    # inside the recording catch_warnings block.
    container = MagicMock()
    container.wire.side_effect = lambda **_: warnings.warn(
        "unrelated deprecation", DeprecationWarning
    )

    with pytest.warns(DeprecationWarning, match="unrelated deprecation"):
        wire_and_validate(container, modules=[wiring_fixtures])
