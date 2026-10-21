"""Tests that DeterministicStepComponent registers on a fresh import of the package.

DeterministicStepComponent must register when the experimental components package is imported once; pytest's shared
import order can mask a missing registration, so the check runs in a fresh interpreter.
"""

import subprocess
import sys
import textwrap


def test_deterministic_step_component_registers_on_fresh_import():
    script = textwrap.dedent(
        """
        from duo_workflow_service.agent_platform.experimental.components.registry import (
            ComponentRegistry,
        )
        import duo_workflow_service.agent_platform.experimental.components  # noqa: F401

        registry = ComponentRegistry.instance()
        assert "DeterministicStepComponent" in registry, (
            "DeterministicStepComponent did not register on a fresh import"
        )
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, (
        f"fresh-import registration check failed\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
