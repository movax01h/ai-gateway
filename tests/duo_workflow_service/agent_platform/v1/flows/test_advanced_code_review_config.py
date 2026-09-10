"""Guards for the shipped advanced_code_review flow config.

These tests assert on the content of ``advanced_code_review/1.0.0.yml`` (the
four-step pipeline and the router that skips the discussion fetch on a first
review, the reviewer's schema id, toolset and ``max_cycles``, the fetch step's
pinned inputs, and the publish step's schema-validated answer and
``min_confidence`` literal), plus the reviewer response schema those inputs
read from, rather than on ``FlowConfig`` machinery, which is covered in
``test_flow_config.py``.
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
    def _config(cls) -> FlowConfig:
        return FlowConfig.from_yaml_config("advanced_code_review", "1.0.0")

    @classmethod
    def _components(cls) -> dict:
        return {component["name"]: component for component in cls._config().components}

    @staticmethod
    def _input(component: dict, name: str) -> dict:
        return next(
            component_input
            for component_input in component["inputs"]
            if component_input["as"] == name
        )

    def test_flow_is_fetch_discussions_review_publish(self):
        """The reviewer is the only model step; everything after its final answer is deterministic code."""
        components = self._components()

        assert list(components) == [
            "fetch_mr_data",
            "fetch_existing_discussions",
            "review",
            "publish_review",
        ]
        assert components["fetch_mr_data"]["type"] == "DeterministicStepComponent"
        assert (
            components["fetch_existing_discussions"]["type"]
            == "DeterministicStepComponent"
        )
        assert components["review"]["type"] == "AgentComponent"
        assert components["publish_review"]["type"] == "DeterministicStepComponent"

    def test_flow_declares_the_code_review_context_input(self):
        """The service drops any additional-context category the flow does not declare, so without this the bot name and
        the previous review's head SHA never reach the steps that read them."""
        [flow_input] = self._config().flow.inputs

        assert flow_input.category == "code_review_context"
        assert set(flow_input.input_schema) == {
            "duo_code_review_bot_name",
            "last_reviewed_head_sha",
        }
        assert all(field.optional for field in flow_input.input_schema.values())

    def test_fetch_marks_lines_added_since_the_previous_review(self):
        components = self._components()
        baseline = self._input(components["fetch_mr_data"], "baseline_sha")

        assert (
            baseline["from"]
            == "context:inputs.code_review_context.last_reviewed_head_sha"
        )
        assert baseline["optional"] is True

    def test_discussions_are_the_bots_own_resolvable_threads(self):
        components = self._components()
        fetch = components["fetch_existing_discussions"]

        assert fetch["tool_name"] == "list_mr_discussions"
        author = self._input(fetch, "only_from_author")
        assert (
            author["from"]
            == "context:inputs.code_review_context.duo_code_review_bot_name"
        )
        assert author["optional"] is True
        resolvable = self._input(fetch, "only_resolvable")
        assert resolvable["from"] == "true"
        assert resolvable["literal"] is True

    def test_first_review_skips_the_discussion_fetch(self):
        """last_reviewed_head_sha is absent exactly when the bot has never published a comment, so there is nothing to
        fetch."""
        router = next(r for r in self._config().routers if r["from"] == "fetch_mr_data")
        condition = router["condition"]

        assert (
            condition["input"]["from"]
            == "context:inputs.code_review_context.last_reviewed_head_sha"
        )
        assert condition["input"]["optional"] is True
        assert condition["routes"] == {
            "None": "review",
            "default_route": "fetch_existing_discussions",
        }

    def test_reviewer_receives_the_re_review_inputs(self):
        components = self._components()
        review = components["review"]

        for name, source in {
            "existing_discussions": "context:fetch_existing_discussions.tool_responses",
            "duo_code_review_bot_name": "context:inputs.code_review_context.duo_code_review_bot_name",
            "last_reviewed_head_sha": "context:inputs.code_review_context.last_reviewed_head_sha",
        }.items():
            component_input = self._input(review, name)
            assert component_input["from"] == source
            assert component_input["optional"] is True

    def test_reviewer_emits_structured_findings_with_read_only_tools(self):
        components = self._components()
        review = components["review"]

        assert review["response_schema_id"] == "code_review_findings"
        assert review["toolset"] == self.INVESTIGATION_TOOLSET

    def test_reviewer_investigation_is_bounded(self):
        """A stuck investigation should degrade into a partial review rather than burn the platform default of 280
        cycles reading files."""
        components = self._components()
        review = components["review"]

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
        previous = self._input(publish, "previous_findings")
        assert previous["from"] == "context:review.final_answer.previous_findings"
        assert previous["optional"] is True

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

    def test_reviewer_schema_requires_end_line(self):
        """`end_line` is required so the reviewer decides the span on every finding.

        Left optional, the model skipped it on about half of the multi-line rewrites in E2E runs, publishing one-line
        patches that broke on Apply.
        """
        fields = self._finding_model().model_fields

        assert fields["end_line"].is_required() is True

    def test_reviewer_schema_keeps_previous_findings_optional(self):
        """A first review has no threads to reconcile, so the field must not be demanded of it."""
        review = self._components()["review"]
        schema = ResponseSchemaRegistry().get(
            review["response_schema_id"], review["response_schema_version"]
        )

        assert schema.model_fields["previous_findings"].is_required() is False

    def test_publish_confidence_gate_is_a_literal(self):
        """The reviewer never self-censors; the volume/precision operating point lives in config, where it is logged and
        counted."""
        components = self._components()
        gate = self._input(components["publish_review"], "min_confidence")

        assert gate["from"] == "0"
        assert gate["literal"] is True
