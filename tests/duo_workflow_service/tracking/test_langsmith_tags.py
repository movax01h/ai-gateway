from unittest.mock import Mock, patch

import pytest

from duo_workflow_service.tracking.langsmith_tags import tag_current_run


def _make_run_tree(*, id, trace_id, tags=None, client=None):  # pylint: disable=redefined-builtin
    return Mock(id=id, trace_id=trace_id, tags=tags or [], client=client or Mock())


class TestTagCurrentRun:
    @pytest.mark.asyncio
    async def test_tags_run_tree_directly_when_it_is_the_trace_root(self):
        """When the current run is the trace root (id == trace_id), only the local, in-memory tag applies."""
        mock_run_tree = _make_run_tree(id="root-id", trace_id="root-id", tags=[])

        with patch(
            "duo_workflow_service.tracking.langsmith_tags.get_current_run_tree",
            return_value=mock_run_tree,
        ):
            await tag_current_run("soft_limit_reached", {"cycle_count": 4})

        mock_run_tree.add_tags.assert_called_once_with(["soft_limit_reached"])
        mock_run_tree.add_metadata.assert_called_once_with({"cycle_count": 4})
        mock_run_tree.client.read_run.assert_not_called()
        mock_run_tree.client.update_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_also_tags_trace_root_via_api_when_current_run_is_nested(self):
        """A tag on a nested run (e.g. an agent-node run) is also written to the trace root via the API.

        LangGraph's node-tracing path never populates `RunTree.parent_run` (only
        `parent_run_id`/`trace_id`), so there's no in-memory object reference to
        the root to mutate directly — reaching it requires a read-modify-write
        API round trip.
        """
        leaf_run = _make_run_tree(id="leaf-id", trace_id="root-id", tags=[])
        leaf_run.client.read_run.return_value = Mock(
            tags=["existing_tag"], extra={"metadata": {"flow_id": "developer"}}
        )

        with patch(
            "duo_workflow_service.tracking.langsmith_tags.get_current_run_tree",
            return_value=leaf_run,
        ):
            await tag_current_run(
                "soft_limit_reached", {"cycle_count": 4, "max_cycles": 3}
            )

        leaf_run.add_tags.assert_called_once_with(["soft_limit_reached"])
        leaf_run.add_metadata.assert_called_once_with(
            {"cycle_count": 4, "max_cycles": 3}
        )

        leaf_run.client.read_run.assert_called_once_with("root-id")
        leaf_run.client.update_run.assert_called_once_with(
            "root-id",
            tags=["existing_tag", "soft_limit_reached"],
            extra={
                "metadata": {
                    "flow_id": "developer",
                    "cycle_count": 4,
                    "max_cycles": 3,
                }
            },
        )

    @pytest.mark.asyncio
    async def test_skips_duplicate_tag_on_root_but_still_updates_metadata(self):
        """If the root already has the tag (e.g. re-tagged after crossing the limit again), don't duplicate it."""
        leaf_run = _make_run_tree(id="leaf-id", trace_id="root-id", tags=[])
        leaf_run.client.read_run.return_value = Mock(
            tags=["soft_limit_reached"], extra={"metadata": {"cycle_count": 3}}
        )

        with patch(
            "duo_workflow_service.tracking.langsmith_tags.get_current_run_tree",
            return_value=leaf_run,
        ):
            await tag_current_run("soft_limit_reached", {"cycle_count": 4})

        leaf_run.client.update_run.assert_called_once_with(
            "root-id",
            tags=["soft_limit_reached"],
            extra={"metadata": {"cycle_count": 4}},
        )

    @pytest.mark.asyncio
    async def test_skips_duplicate_tag_locally_but_still_updates_metadata(self):
        """A run tagged twice locally (e.g. re-entered after already crossing its limit) shouldn't get duplicates."""
        mock_run_tree = _make_run_tree(
            id="root-id", trace_id="root-id", tags=["soft_limit_reached"]
        )

        with patch(
            "duo_workflow_service.tracking.langsmith_tags.get_current_run_tree",
            return_value=mock_run_tree,
        ):
            await tag_current_run("soft_limit_reached", {"cycle_count": 5})

        mock_run_tree.add_tags.assert_not_called()
        mock_run_tree.add_metadata.assert_called_once_with({"cycle_count": 5})

    @pytest.mark.asyncio
    async def test_root_tagging_read_run_failure_does_not_raise(self):
        """A LangSmith API failure on read_run while tagging the root is swallowed, not propagated."""
        leaf_run = _make_run_tree(id="leaf-id", trace_id="root-id", tags=[])
        leaf_run.client.read_run.side_effect = ConnectionError("boom")

        with patch(
            "duo_workflow_service.tracking.langsmith_tags.get_current_run_tree",
            return_value=leaf_run,
        ):
            # Should not raise even though the root-tagging API call fails.
            await tag_current_run("soft_limit_reached", {"cycle_count": 4})

        # The local, in-memory tag on the current run still succeeded.
        leaf_run.add_tags.assert_called_once_with(["soft_limit_reached"])
        leaf_run.client.update_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_root_tagging_update_run_failure_does_not_raise(self):
        """A LangSmith API failure on update_run (read_run succeeds) while tagging the root is swallowed."""
        leaf_run = _make_run_tree(id="leaf-id", trace_id="root-id", tags=[])
        leaf_run.client.read_run.return_value = Mock(
            tags=["existing_tag"], extra={"metadata": {"flow_id": "developer"}}
        )
        leaf_run.client.update_run.side_effect = ConnectionError("boom")

        with patch(
            "duo_workflow_service.tracking.langsmith_tags.get_current_run_tree",
            return_value=leaf_run,
        ):
            # Should not raise even though the root-tagging update_run call fails.
            await tag_current_run("soft_limit_reached", {"cycle_count": 4})

        # The local, in-memory tag on the current run still succeeded.
        leaf_run.add_tags.assert_called_once_with(["soft_limit_reached"])
        leaf_run.client.update_run.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("field", ["tags", "extra", "metadata"])
    async def test_root_tagging_handles_none_fields_from_read_run(self, field):
        """`root.tags` / `root.extra` / `root.extra["metadata"]` being None (not absent) falls back to empty."""
        leaf_run = _make_run_tree(id="leaf-id", trace_id="root-id", tags=[])
        extra = {"metadata": None if field == "metadata" else {"flow_id": "developer"}}
        defaults = {
            "tags": None if field == "tags" else ["existing_tag"],
            "extra": None if field == "extra" else extra,
        }
        leaf_run.client.read_run.return_value = Mock(**defaults)

        with patch(
            "duo_workflow_service.tracking.langsmith_tags.get_current_run_tree",
            return_value=leaf_run,
        ):
            await tag_current_run("soft_limit_reached", {"cycle_count": 4})

        expected_tags = (
            ["soft_limit_reached"]
            if field == "tags"
            else ["existing_tag", "soft_limit_reached"]
        )
        expected_metadata = (
            {"cycle_count": 4}
            if field in ("extra", "metadata")
            else {"flow_id": "developer", "cycle_count": 4}
        )
        leaf_run.client.update_run.assert_called_once_with(
            "root-id",
            tags=expected_tags,
            extra={"metadata": expected_metadata},
        )

    @pytest.mark.asyncio
    async def test_no_op_when_run_tree_is_none(self):
        with patch(
            "duo_workflow_service.tracking.langsmith_tags.get_current_run_tree",
            return_value=None,
        ):
            # Should not raise even when run tree is None.
            await tag_current_run("soft_limit_reached", {"cycle_count": 4})
