"""Guards for the shipped skillguard_scan flow config.

The SkillGuard DAP flow is a security scanner, so a silent regression in the
shipped config is worse than a loud failure: a broken scan that still reports
COMPLETED would pass a malicious skill through. These tests pin the structural
invariants of ``skillguard_scan/1.0.0.yml`` so an accidental edit fails CI here
rather than only in manual GDK validation.

What is guarded:
- Every deterministic step aborts on failure instead of continuing a broken scan to COMPLETED.
- Every deterministic step overrides the default timeout (long network pulls and large-repo scans would otherwise be killed at 120 s).
- The ``setup_tools`` install script stays POSIX-sh safe (``set -eu``, no ``pipefail`` / ``curl | tar`` bashisms that tie it to bash).
- The per-skill supervisors stay wired to the correct subagent prompt and keep a delegation cap.

Generic ``FlowConfig`` behavior is covered in ``test_flow_config.py``; these
tests assert on the shipped config content only.
"""

from pathlib import Path

import pytest
import yaml

from duo_workflow_service.agent_platform.v1.flows.flow_config import FlowConfig

FLOW_ID = "skillguard_scan"
FLOW_VERSION = "1.0.0"

# The shipped config lives next to this test's feature dir:
# ai/features/security/skillguard_scan/{config/1.0.0.yml,tests/<this file>}
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / f"{FLOW_VERSION}.yml"

# Deterministic steps that must abort the flow on failure.
DETERMINISTIC_STEPS = [
    "setup_tools",
    "discover_and_deterministic",
    "companion_scan",
    "generate_report",
]

# Each supervisor and the subagent prompt it must delegate to.
SUPERVISORS = [
    ("fact_check_supervisor", "skillguard_fact_check_supervisor", "fact_check_agent"),
    ("purpose_supervisor", "skillguard_purpose_supervisor", "purpose_agent"),
    ("audit_supervisor", "skillguard_audit_supervisor", "audit_agent"),
    ("judge_supervisor", "skillguard_judge_supervisor", "judge_agent"),
]


def _config() -> FlowConfig:
    # Load by path rather than FlowConfig.from_yaml_config(flow_id, version):
    # the latter resolves against the legacy DIRECTORY_PATH, but this flow now
    # lives under ai/features/. Mirrors tests/.../test_configs.py::_load_flow_config.
    return FlowConfig(**(yaml.safe_load(CONFIG_PATH.read_text()) or {}))


def _component(config: FlowConfig, name: str) -> dict:
    return next(c for c in config.components if c["name"] == name)


def _router_for(config: FlowConfig, from_component: str) -> dict:
    return next(r for r in config.routers if r["from"] == from_component)


def _literal_input(component: dict, as_name: str) -> str:
    return next(i["from"] for i in component["inputs"] if i.get("as") == as_name)


class TestSkillguardScanConfigLoads:
    """Guard 1 — the shipped YAML parses and the entry point is stable."""

    def test_config_loads_successfully(self):
        config = _config()
        assert config.flow.entry_point == "setup_tools"
        assert config.environment == "ambient"


class TestSkillguardScanAbortsOnFailedStep:
    """Guard 2 — a failed deterministic step must abort, not continue.

    ``DeterministicStepComponent`` records ``execution_result="failed"`` and returns normally, so without a conditional
    router a broken setup, scan, or report would flow straight to the next step (or to ``end``) and the scan would
    finish COMPLETED with no findings.
    """

    @pytest.mark.parametrize("from_component", DETERMINISTIC_STEPS)
    def test_step_routes_on_execution_result_not_unconditionally(self, from_component):
        config = _config()
        router = _router_for(config, from_component)
        assert "to" not in router, (
            f"{from_component} must not route unconditionally — a failed "
            "deterministic step would otherwise reach the next stage and the "
            "scan could finish COMPLETED on a broken run"
        )
        assert (
            router["condition"]["input"] == f"context:{from_component}.execution_result"
        )

    @pytest.mark.parametrize(
        "from_component,success_target",
        [
            ("setup_tools", "discover_and_deterministic"),
            ("discover_and_deterministic", "fact_check_supervisor"),
            ("companion_scan", "purpose_supervisor"),
            ("generate_report", "end"),
        ],
    )
    def test_success_continues_and_failure_aborts(self, from_component, success_target):
        config = _config()
        routes = _router_for(config, from_component)["condition"]["routes"]
        assert routes["success"] == success_target
        assert routes["default_route"] == "abort"


class TestSkillguardScanDeterministicTimeouts:
    """Guard 3 — every deterministic step overrides the default timeout.

    The install pulls ~70 MB over the network and the scan/report steps scale with the number of skills; the default 120
    s would kill them on slow registries or large monorepos. Each step pins ``timeout: "300"``.
    """

    @pytest.mark.parametrize("step_name", DETERMINISTIC_STEPS)
    def test_step_sets_extended_timeout(self, step_name):
        component = _component(_config(), step_name)
        assert _literal_input(component, "timeout") == "300", (
            f"{step_name} must pin timeout: 300 — the default 120 s would kill "
            "long network pulls / large-repo scans"
        )


class TestSkillguardScanSetupScriptIsPosixSafe:
    """Guard 4 — the install script must stay POSIX-sh compatible.

    ``run_command`` may execute under ``/bin/sh`` (dash/busybox), not bash. ``set -o pipefail`` and ``curl | tar`` are
    bashisms; reintroducing them would abort the flow at step 0 on a non-bash shell.
    """

    def _setup_command(self) -> str:
        return _literal_input(_component(_config(), "setup_tools"), "command")

    def _setup_code(self) -> str:
        """The setup script with comment lines stripped.

        The script documents *why* it avoids ``pipefail`` / ``curl | tar`` in comments, so the guards must inspect
        executable lines only — otherwise the explanatory comments would trip the very checks they describe.
        """
        lines = self._setup_command().splitlines()
        return "\n".join(line for line in lines if not line.lstrip().startswith("#"))

    def test_uses_set_eu(self):
        assert "set -eu" in self._setup_code()

    def test_no_pipefail_bashism(self):
        assert "pipefail" not in self._setup_code(), (
            "pipefail is a bashism — the setup script must run under POSIX sh"
        )

    def test_no_curl_piped_into_tar(self):
        # A curl|tar pipe masks a failed download under set -e without pipefail;
        # the script downloads to a file then extracts instead.
        code = self._setup_code()
        assert "| tar" not in code and "|tar" not in code, (
            "curl | tar hides download failures without pipefail; download to a "
            "file first, then extract"
        )


class TestSkillguardScanSupervisorsAreWired:
    """Guard 6 — each supervisor delegates to its subagent with a cap.

    The flow runs one subagent per skill behind a supervisor. A supervisor pointed at the wrong prompt, or without a
    delegation cap, would silently mis-scan or run unbounded.
    """

    @pytest.mark.parametrize("name,prompt_id,subagent", SUPERVISORS)
    def test_supervisor_prompt_and_subagent(self, name, prompt_id, subagent):
        component = _component(_config(), name)
        assert component["type"] == "AgentComponent"
        assert component["prompt_id"] == prompt_id
        subagent_names = [s["name"] for s in component["subagents"]]
        assert subagent in subagent_names

    @pytest.mark.parametrize("name,prompt_id,subagent", SUPERVISORS)
    def test_supervisor_has_delegation_cap(self, name, prompt_id, subagent):
        component = _component(_config(), name)
        assert isinstance(component.get("max_delegations"), int)
        assert component["max_delegations"] > 0

    @pytest.mark.parametrize("name,prompt_id,subagent", SUPERVISORS)
    def test_subagent_component_exists(self, name, prompt_id, subagent):
        # The delegated subagent must be a declared component in the flow.
        config = _config()
        component = _component(config, subagent)
        assert component["type"] == "AgentComponent"
