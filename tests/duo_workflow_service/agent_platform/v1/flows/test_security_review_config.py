"""Guards for the shipped security_review flow config.

These tests assert on the content of the shipped ``security_review`` configs (pinned
tool options, input declarations, routing) rather than on ``FlowConfig`` machinery —
generic ``FlowConfig`` behavior is covered in ``test_flow_config.py``.
"""

import pytest

from duo_workflow_service.agent_platform.v1.flows.flow_config import FlowConfig
from duo_workflow_service.tools.mr_review import SubmitMrReviewInput

VERSIONS = ["1.0.0", "2.0.0"]


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

    @pytest.mark.parametrize("version", VERSIONS)
    def test_submit_mr_review_args_are_pinned(self, version: str):
        config = FlowConfig.from_yaml_config("security_review", version)
        assert self._submit_mr_review_options(config) == self.EXPECTED_OPTIONS

    @pytest.mark.parametrize("version", VERSIONS)
    def test_pinned_option_keys_are_valid_tool_parameters(self, version: str):
        # Mirrors Toolset._validate_tool_options: every pinned key must be a real
        # parameter on the tool's input schema, so a typo/rename fails fast here.
        config = FlowConfig.from_yaml_config("security_review", version)
        options = self._submit_mr_review_options(config)
        valid_fields = set(SubmitMrReviewInput.model_fields.keys())
        assert set(options).issubset(valid_fields)


@pytest.mark.parametrize("version", VERSIONS)
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

    def test_trigger_context_category_is_fully_optional(self, version):
        config = FlowConfig.from_yaml_config("security_review", version)
        schema = config.input_json_schemas_by_category()[
            "agent_platform_trigger_context"
        ]
        assert schema["required"] == []
        assert schema["additionalProperties"] is True
        assert set(schema["properties"]) == {"event_type", "triggering_conversation"}

    def test_standard_context_stays_frozen(self, version):
        # The standard-context schema must not grow trigger fields again — that is
        # exactly the skew hazard the separate category exists to avoid.
        config = FlowConfig.from_yaml_config("security_review", version)
        schema = config.input_json_schemas_by_category()[
            "agent_platform_standard_context"
        ]
        assert set(schema["properties"]) == {
            "workload_branch",
            "primary_branch",
            "session_owner_id",
            "service_account_name",
        }

    def test_mention_router_branches_on_optional_trigger_context(self, version):
        # The router condition must use the optional mapping-form input so an
        # absent trigger-context category falls to the full review (default
        # route) instead of raising a KeyError at routing time.
        config = FlowConfig.from_yaml_config("security_review", version)
        router = next(r for r in config.routers if r["from"] == "check_existing_review")
        assert router["condition"]["input"] == {
            "from": "context:inputs.agent_platform_trigger_context.event_type",
            "optional": True,
        }
        assert router["condition"]["routes"] == {
            "mention": "respond_to_comment",
            "default_route": "apply_triggered_label",
        }


class TestSecurityReviewResourceContext:
    """Guard how security_review 2.0.0 reads the merge request from the resource context.

    Version 1.0.0 reads the merge request URL from ``goal``. Version 2.0.0 reads it from the
    agent_platform_resource_context envelope. The graph compiles even if an input path has a
    typo, so nothing else fails until the flow runs.
    """

    VERSION = "2.0.0"
    RESOURCE_CONTEXT = "context:inputs.agent_platform_resource_context"

    @staticmethod
    def _input_sources(config: FlowConfig, alias: str) -> set[str]:
        return {
            i["from"]
            for c in config.components
            for i in c.get("inputs", [])
            if i["as"] == alias
        }

    @pytest.mark.parametrize(
        "alias, field",
        [
            # The tool steps read the URL, the agent steps read the IID.
            ("url", "merge_request_web_url"),
            ("merge_request_iid", "merge_request_id"),
        ],
    )
    def test_steps_read_the_merge_request_from_the_resource_context(
        self, alias: str, field: str
    ):
        config = FlowConfig.from_yaml_config("security_review", self.VERSION)

        assert self._input_sources(config, alias) == {
            f"{self.RESOURCE_CONTEXT}.{field}"
        }

    def test_no_step_reads_goal(self):
        config = FlowConfig.from_yaml_config("security_review", self.VERSION)
        sources = {i["from"] for c in config.components for i in c.get("inputs", [])}

        assert "context:goal" not in sources

    def test_resource_context_fields_are_required(self):
        config = FlowConfig.from_yaml_config("security_review", self.VERSION)
        [category] = [
            i
            for i in config.flow.inputs
            if i.category == "agent_platform_resource_context"
        ]

        assert not category.input_schema["merge_request_id"].optional
        assert not category.input_schema["merge_request_web_url"].optional

    def test_fetch_mr_data_failure_aborts_the_flow(self):
        # An empty merge request URL makes fetch_mr_data fail. Routing that failure to
        # abort keeps the agent steps from running with an empty merge request IID.
        config = FlowConfig.from_yaml_config("security_review", self.VERSION)
        router = next(r for r in config.routers if r["from"] == "fetch_mr_data")

        assert router["condition"]["input"] == "context:fetch_mr_data.execution_result"
        assert router["condition"]["routes"] == {
            "success": "build_review_context",
            "default_route": "abort",
        }


class TestSecurityReviewToolNames:
    """The context-gathering stages must actually declare repo-wide search.

    Kept deliberately narrow. The registry sweep in ``test_configs.py`` supersedes the other half
    of what this used to assert — that the name is not the misspelled ``blob_search``
    (gitlab-org/gitlab#627166) — because a misspelling resolves to nothing and the sweep fails on
    it. What the sweep cannot see is the name being *removed*: a toolset that simply drops
    ``gitlab_blob_search`` declares nothing unresolvable, so the sweep stays green while the agent
    silently loses repo-wide search. That is the assertion below.
    """

    CONTEXT_STAGES = ("build_review_context", "prescan_codebase")

    @pytest.mark.parametrize("version", VERSIONS)
    @pytest.mark.parametrize("stage", CONTEXT_STAGES)
    def test_context_stages_declare_gitlab_blob_search(self, stage: str, version: str):
        config = FlowConfig.from_yaml_config("security_review", version)
        component = next(c for c in config.components if c.get("name") == stage)
        declared = [
            name
            for entry in component["toolset"]
            for name in ([entry] if isinstance(entry, str) else entry)
        ]

        assert "gitlab_blob_search" in declared, (
            f"{stage} no longer declares 'gitlab_blob_search', so the security-review agent runs "
            f"without repo-wide search. The registry sweep cannot catch this — it only sees names "
            f"that resolve to nothing, and a removed name resolves to nothing to check."
        )
