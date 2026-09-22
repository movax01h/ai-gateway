"""Guards for the shipped code_review flow configs.

RFH #5233 - if build_review_context fails to resolve a real merge_request_iid, the flow must abort instead of
silently continuing to a forced placeholder publish.
See https://gitlab.com/gitlab-com/request-for-help/-/work_items/5233

RFH #5210 - the publish step must not be free to answer in prose.
See https://gitlab.com/gitlab-com/request-for-help/-/work_items/5210

Both parametrize over every shipped config rather than a hardcoded list, so promoting a new version cannot skip the
guards - which is how 2.0.0 was first missed.
"""

from pathlib import Path

import pytest
import yaml

import ai_gateway.prompts
from ai_gateway.prompts.feature_roots import feature_prompt_root
from duo_workflow_service.agent_platform.v1.flows.flow_config import FlowConfig

FLOW = "code_review"
PUBLISH_STEP = "perform_code_review_and_publish"


def shipped_versions() -> list[str]:
    """Every version shipped for the flow, taken from the config filenames."""
    config_dir = FlowConfig.DIRECTORY_PATH / FLOW
    return sorted(path.stem for path in config_dir.glob("*.yml"))


def test_shipped_versions_are_discovered():
    """A bad glob would silently parametrize the guards below into nothing."""
    assert shipped_versions(), f"no shipped configs found for {FLOW}"


def publish_component(config: FlowConfig) -> dict:
    return next(c for c in config.components if c.get("name") == PUBLISH_STEP)


def prompt_definition_dir(prompt_id: str) -> Path:
    """Resolve a prompt id to its definitions dir the way LocalPromptRegistry does."""
    return feature_prompt_root(prompt_id) or (
        Path(ai_gateway.prompts.__file__).parent / "definitions" / prompt_id
    )


@pytest.mark.parametrize("version", shipped_versions())
class TestCodeReviewAbortsOnUnresolvedMergeRequest:
    def _build_review_context_router(self, config: FlowConfig) -> dict:
        return next(
            r for r in config.routers if r.get("from") == "build_review_context"
        )

    def test_routes_on_execution_result_not_unconditionally(self, version):
        config = FlowConfig.from_yaml_config("code_review", version)
        router = self._build_review_context_router(config)
        assert "to" not in router, (
            "build_review_context must not route unconditionally - a failed "
            "MR-context lookup would otherwise reach post_duo_code_review "
            "with no real merge_request_iid (RFH #5233)"
        )
        assert (
            router["condition"]["input"]
            == "context:build_review_context.execution_result"
        )

    def test_success_continues_and_failure_aborts(self, version):
        config = FlowConfig.from_yaml_config("code_review", version)
        routes = self._build_review_context_router(config)["condition"]["routes"]
        assert routes["success"] == "fetch_mr_diffs"
        assert routes["default_route"] == "abort"


@pytest.mark.parametrize("version", shipped_versions())
class TestCodeReviewPublishStepForcesToolCall:
    """The publish step must not be free to answer in prose (RFH #5210).

    perform_code_review_and_publish has exactly one tool, post_duo_code_review, so with tool_choice "auto" the model can
    return a review as text and the step completes having posted nothing. "any" is the semantic value;
    LocalPromptRegistry translates it into each client's wire format, so the config must never carry "required".
    """

    def _publish_component(self, config: FlowConfig) -> dict:
        return next(
            c
            for c in config.components
            if c.get("name") == "perform_code_review_and_publish"
        )

    def test_publish_step_forces_a_tool_call(self, version):
        config = FlowConfig.from_yaml_config("code_review", version)
        component = self._publish_component(config)
        assert component.get("tool_choice") == "any", (
            "perform_code_review_and_publish must set tool_choice: any, otherwise the "
            "review can be emitted as prose and no comments are posted (RFH #5210)"
        )

    def test_publish_step_surfaces_agent_reasoning(self, version):
        config = FlowConfig.from_yaml_config("code_review", version)
        component = self._publish_component(config)
        assert "on_agent_reasoning" in component.get("ui_log_events", []), (
            "without on_agent_reasoning a turn that produced no tool call leaves the "
            "user with no output at all, which is how RFH #5210 presented"
        )


@pytest.mark.parametrize("version", shipped_versions())
def test_publish_prompt_requests_no_extended_thinking(version):
    """Extended thinking silently cancels the forced tool call.

    `ChatAnthropic.bind_tools` drops a forced `tool_choice` with a warning when thinking is enabled, which would put
    the publish step straight back into the RFH #5210 behaviour - free to answer in prose - with nothing to catch it.
    No Claude entry in models.yml enables thinking today (only the Gemini ones carry `thinking_level`, and
    `google_genai` maps "required" to ANY correctly), so this guards the prompt side: whichever version the config
    resolves to must not ask for it.
    """
    config = FlowConfig.from_yaml_config(FLOW, version)
    prompt_id = publish_component(config)["prompt_id"]

    definitions = sorted(prompt_definition_dir(prompt_id).rglob("*.yml"))
    assert definitions, f"no prompt definitions found for {prompt_id}"

    for definition in definitions:
        params = yaml.safe_load(definition.read_text(encoding="utf-8")).get("params")
        offenders = [
            key for key in (params or {}) if "thinking" in key or "reasoning" in key
        ]
        assert not offenders, (
            f"{definition.name} sets {offenders} on {prompt_id}, which makes ChatAnthropic "
            "drop the forced tool_choice and lets the publish step answer in prose (RFH #5210)"
        )
