"""Guards for the shipped security_review flow config.

These tests assert on the content of ``security_review/1.0.0.yml`` (pinned tool
options, input declarations, routing) rather than on ``FlowConfig`` machinery —
generic ``FlowConfig`` behavior is covered in ``test_flow_config.py``.
"""

from duo_workflow_service.agent_platform.v1.flows.flow_config import FlowConfig
from duo_workflow_service.tools.mr_review import SubmitMrReviewInput


class TestSecurityReviewToolOptions:
    """Guard the security_review flow's pinned submit_mr_review tool options.

    These arguments are pinned at the flow level (rather than set by the LLM) so a hallucination or a prompt injection
    cannot flip them — flipping the fold/internal flags on a public project would expose inline security findings. See
    the validate_and_publish component in security_review/1.0.0.yml.
    """

    EXPECTED_OPTIONS = {
        "fold_inline_into_summary_when_public": True,
        "inline_findings_title": (
            "**Security Findings** (internal only — this project is public)"
        ),
        "summary_internal": True,
    }

    @staticmethod
    def _submit_mr_review_options(config: FlowConfig) -> dict:
        component = next(
            c for c in config.components if c.get("name") == "validate_and_publish"
        )
        for entry in component["toolset"]:
            if isinstance(entry, dict) and "submit_mr_review" in entry:
                return entry["submit_mr_review"]
        raise AssertionError(
            "submit_mr_review is not declared with pinned tool options in "
            "validate_and_publish"
        )

    def test_submit_mr_review_args_are_pinned(self):
        config = FlowConfig.from_yaml_config("security_review", "1.0.0")
        assert self._submit_mr_review_options(config) == self.EXPECTED_OPTIONS

    def test_pinned_option_keys_are_valid_tool_parameters(self):
        # Mirrors Toolset._validate_tool_options: every pinned key must be a real
        # parameter on the tool's input schema, so a typo/rename fails fast here.
        config = FlowConfig.from_yaml_config("security_review", "1.0.0")
        options = self._submit_mr_review_options(config)
        valid_fields = set(SubmitMrReviewInput.model_fields.keys())
        assert set(options).issubset(valid_fields)


class TestSecurityReviewTriggerContext:
    """Guard the security_review flow's trigger-context input declaration (#604317).

    Trigger metadata (event_type / triggering_conversation) rides in its own optional
    agent_platform_trigger_context envelope rather than in agent_platform_standard_context.
    Unknown categories are skipped with a warning, so using a separate category avoids
    breaking flows on Rails/service version skew.

    Envelopes now validate with additionalProperties: true so that adding a new field to
    an envelope no longer breaks flows that have not yet declared the field in their
    input_schema (see issue #2515).
    """

    def test_trigger_context_category_is_fully_optional(self):
        config = FlowConfig.from_yaml_config("security_review", "1.0.0")
        schema = config.input_json_schemas_by_category()[
            "agent_platform_trigger_context"
        ]
        assert schema["required"] == []
        assert schema["additionalProperties"] is True
        assert set(schema["properties"]) == {"event_type", "triggering_conversation"}

    def test_standard_context_stays_frozen(self):
        # The standard-context schema must not grow trigger fields again — that is
        # exactly the skew hazard the separate category exists to avoid.
        config = FlowConfig.from_yaml_config("security_review", "1.0.0")
        schema = config.input_json_schemas_by_category()[
            "agent_platform_standard_context"
        ]
        assert set(schema["properties"]) == {
            "workload_branch",
            "primary_branch",
            "session_owner_id",
            "service_account_name",
        }

    def test_mention_router_branches_on_optional_trigger_context(self):
        # The router condition must use the optional mapping-form input so an
        # absent trigger-context category falls to the full review (default
        # route) instead of raising a KeyError at routing time.
        config = FlowConfig.from_yaml_config("security_review", "1.0.0")
        router = next(r for r in config.routers if r["from"] == "check_existing_review")
        assert router["condition"]["input"] == {
            "from": "context:inputs.agent_platform_trigger_context.event_type",
            "optional": True,
        }
        assert router["condition"]["routes"] == {
            "mention": "respond_to_comment",
            "default_route": "apply_triggered_label",
        }
