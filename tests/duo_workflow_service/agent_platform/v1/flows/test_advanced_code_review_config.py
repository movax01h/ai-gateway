"""Guards for the shipped advanced_code_review flow config.

These tests assert on the content of ``advanced_code_review/1.0.0.yml`` (the
three-step pipeline, the reviewer's schema id, toolset,
``response_schema_tool_choice`` and ``max_cycles``, the fetch step's pinned
inputs, and the publish step's schema-validated answer and ``min_confidence``
literal), plus the reviewer response schema those inputs read from, rather than
on ``FlowConfig`` machinery, which is covered in ``test_flow_config.py``.
"""

from typing import get_args

from ai_gateway.response_schemas import ResponseSchemaRegistry
from duo_workflow_service.agent_platform.v1.flows.flow_config import FlowConfig


class TestAdvancedCodeReviewConfig:
    INVESTIGATION_TOOLSET = [
        "gitlab_blob_search",
        "get_repository_file",
        "get_repository_files",
        "list_repository_tree",
        "list_commits",
        "get_commit_diff",
    ]

    @classmethod
    def _components(cls) -> dict:
        config = FlowConfig.from_yaml_config("advanced_code_review", "1.0.0")
        return {component["name"]: component for component in config.components}

    @staticmethod
    def _input(component: dict, name: str) -> dict:
        return next(
            component_input
            for component_input in component["inputs"]
            if component_input["as"] == name
        )

    def test_flow_is_fetch_review_publish(self):
        """The reviewer is the only model step; everything after its final answer is deterministic code."""
        components = self._components()

        assert list(components) == ["fetch_mr_data", "review", "publish_review"]
        assert components["fetch_mr_data"]["type"] == "DeterministicStepComponent"
        assert components["review"]["type"] == "AgentComponent"
        assert components["publish_review"]["type"] == "DeterministicStepComponent"

    def test_reviewer_emits_structured_findings_with_read_only_tools(self):
        components = self._components()
        review = components["review"]

        assert review["response_schema_id"] == "code_review_phase_findings"
        assert review["toolset"] == self.INVESTIGATION_TOOLSET

    def test_reviewer_investigation_is_bounded_and_can_think(self):
        """A forced tool choice disables extended thinking and produced unbounded investigation loops on large MRs;
        "auto" plus a cycle cap keeps the reviewer converging and turns the worst case into a partial review, not an
        empty one."""
        components = self._components()
        review = components["review"]

        assert review["response_schema_tool_choice"] == "auto"
        assert review["max_cycles"] == 25

    def test_fetch_disables_the_instruction_format_hint(self):
        """The publish step renders the attribution; sending the hint too would attribute comments twice."""
        components = self._components()
        hint = self._input(
            components["fetch_mr_data"], "include_instruction_format_hint"
        )

        assert hint["from"] == "false"
        assert hint["literal"] is True

    def test_fetch_sends_diffs_without_file_contents(self):
        """The reviewer agent fetches files on demand; preloaded contents ride in every agent cycle's prefix at ~68% of
        the payload with no recall benefit."""
        components = self._components()
        only_diffs = self._input(components["fetch_mr_data"], "only_diffs")

        assert only_diffs["from"] == "true"
        assert only_diffs["literal"] is True

    def test_fetch_includes_the_changed_files_checklist(self):
        """The prompt's sweep step treats <changed_files> as the coverage contract, so the fetch must actually send
        it."""
        components = self._components()
        checklist = self._input(
            components["fetch_mr_data"], "include_changed_files_list"
        )

        assert checklist["from"] == "true"
        assert checklist["literal"] is True

    def test_publish_reads_the_reviewer_answer_directly(self):
        """No adapter sits between the reviewer and the publish step: the schema-validated final answer is the input."""
        components = self._components()
        publish = components["publish_review"]

        assert publish["tool_name"] == "post_duo_code_review_findings"
        findings = self._input(publish, "findings")
        summary = self._input(publish, "summary")
        assert findings["from"] == "context:review.final_answer.findings"
        assert findings.get("optional") is not True
        assert summary["from"] == "context:review.final_answer.summary"
        assert summary["optional"] is True

    @classmethod
    def _finding_model(cls) -> type:
        """The finding model the reviewer is bound to, resolved by id and version the way the runtime resolves it."""
        review = cls._components()["review"]
        schema = ResponseSchemaRegistry().get(
            review["response_schema_id"], review["response_schema_version"]
        )
        return get_args(schema.model_fields["findings"].annotation)[0]

    def test_reviewer_schema_keeps_suggestion_available_and_optional(self):
        """A comment is applicable in one click only when its finding carries `suggestion`, so the field has to exist;
        it has to stay optional because a finding whose fix reaches past the anchored line is still worth publishing
        without one."""
        fields = self._finding_model().model_fields

        assert "suggestion" in fields
        assert fields["suggestion"].is_required() is False

    def test_publish_confidence_gate_is_a_literal(self):
        """The reviewer never self-censors; the volume/precision operating point lives in config, where it is logged and
        counted."""
        components = self._components()
        gate = self._input(components["publish_review"], "min_confidence")

        assert gate["from"] == "0"
        assert gate["literal"] is True
