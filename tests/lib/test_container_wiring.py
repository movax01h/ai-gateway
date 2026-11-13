"""Boot-time wiring validation: fail loud on unresolved injection markers."""

import os
import re
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Sequence
from unittest.mock import MagicMock

import pytest
from dependency_injector import containers, providers

from duo_workflow_service.agent_platform.serving_surface import (
    collect_serving_surfaces,
)
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


def _unwired_inject_sites(
    repo_root: Path,
    aigw_wired: set[str],
    dws_packages: Sequence[str],
    classified: dict[str, str],
) -> list[str]:
    """Return a problem line for each module with an injection marker that no app wires."""
    dws_features = {
        feature
        for feature, surfaces in collect_serving_surfaces(
            repo_root / "ai" / "features"
        ).items()
        if all(surface.deployable == "dws" for surface in surfaces)
    }

    def wired_by_dws(module: str) -> bool:
        if not any(
            module == package or module.startswith(f"{package}.")
            for package in dws_packages
        ):
            return False
        parts = module.split(".")
        # The DWS walk wires all of ai.features, but AIGW serves some features.
        if parts[:2] == ["ai", "features"] and len(parts) > 3:
            return parts[3] in dws_features
        return True

    marker = re.compile(r"^\s*@inject\b|Provide\[|Provider\[|Closing\[", re.M)
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
            elif wired_by_dws(module):
                continue
            else:
                problems.append(
                    f"{module}: has an injection marker but no app wires it. If only DWS serves it, add its "
                    "package to CONTAINER_APPLICATION_PACKAGES, or for an ai.features module declare only dws "
                    "surfaces in the feature's serving.py. If AIGW serves it, classify it aigw in the map "
                    "above and add it to CONTAINER_APPLICATION_MODULES"
                )
    return problems


def test_every_inject_site_is_wired_by_the_app_that_serves_it():
    """A module outside the wired sets unwires silently, so pin the consumer side.

    ai_gateway modules must be in AIGW's explicit list. DWS wires duo_workflow_service and ai.features recursively, but
    an ai.features module passes only if its feature's serving.py declares only dws surfaces. Anything else with a
    marker (lib/, or an ai/ module served by AIGW) must be classified below, which forces a conscious wiring decision
    instead of a silent no-op.
    """
    from ai_gateway.api.server import CONTAINER_APPLICATION_MODULES
    from duo_workflow_service.server import CONTAINER_APPLICATION_PACKAGES

    assert "duo_workflow_service" in CONTAINER_APPLICATION_PACKAGES
    assert "ai.features" in CONTAINER_APPLICATION_PACKAGES
    # Wiring imports every module under a package, and ai/apps/ must stay out.
    assert not [
        package
        for package in CONTAINER_APPLICATION_PACKAGES
        if package == "ai.apps"
        or "ai.apps".startswith(f"{package}.")
        or package.startswith("ai.apps.")
    ]

    # module -> how it is wired: "aigw" (must be in the AIGW module list) or
    # "doc-only" (the marker text appears only in prose). Modules that the DWS
    # walk wires need no entry.
    classified = {
        "lib.container_wiring": "doc-only",
    }

    problems = _unwired_inject_sites(
        REPO_ROOT,
        set(CONTAINER_APPLICATION_MODULES),
        CONTAINER_APPLICATION_PACKAGES,
        classified,
    )
    assert not problems, "\n".join(problems)


def test_wiring_scan_flags_a_marker_in_a_feature_aigw_serves(tmp_path):
    for root in ("ai_gateway", "lib"):
        (tmp_path / root).mkdir()
    for feature, surfaces in (
        ("served_by_aigw", [("rest", "aigw")]),
        ("served_by_both", [("rest", "aigw"), ("grpc", "dws")]),
        ("served_by_dws", [("grpc", "dws")]),
    ):
        feature_dir = tmp_path / "ai" / "features" / "code" / feature
        feature_dir.mkdir(parents=True)
        declared = ", ".join(
            f"ServingSurface(transport={transport!r}, deployable={deployable!r})"
            for transport, deployable in surfaces
        )
        (feature_dir / "serving.py").write_text(
            "from duo_workflow_service.agent_platform.serving_surface import ServingSurface\n"
            f"SERVING_SURFACE = [{declared}]\n"
        )
        (feature_dir / "tool.py").write_text("@inject\ndef tool(): ...\n")

    problems = _unwired_inject_sites(
        tmp_path,
        aigw_wired=set(),
        dws_packages=["duo_workflow_service", "ai.features"],
        classified={},
    )

    assert sorted(problem.split(":")[0] for problem in problems) == [
        "ai.features.code.served_by_aigw.tool",
        "ai.features.code.served_by_both.tool",
    ]


def test_wiring_scan_flags_a_marker_outside_the_dws_packages(tmp_path):
    for root in ("ai_gateway", "lib"):
        (tmp_path / root).mkdir()
    adapter = tmp_path / "ai" / "serving" / "grpc" / "adapter.py"
    adapter.parent.mkdir(parents=True)
    adapter.write_text("@inject\ndef adapter(): ...\n")

    problems = _unwired_inject_sites(
        tmp_path,
        aigw_wired=set(),
        dws_packages=["duo_workflow_service", "ai.features"],
        classified={},
    )

    assert [problem.split(":")[0] for problem in problems] == [
        "ai.serving.grpc.adapter"
    ]


def test_replays_unrelated_warnings():
    # Warnings emitted while wiring must reach the caller, not vanish
    # inside the recording catch_warnings block.
    container = MagicMock()
    container.wire.side_effect = lambda **_: warnings.warn(
        "unrelated deprecation", DeprecationWarning
    )

    with pytest.warns(DeprecationWarning, match="unrelated deprecation"):
        wire_and_validate(container, modules=[wiring_fixtures])
