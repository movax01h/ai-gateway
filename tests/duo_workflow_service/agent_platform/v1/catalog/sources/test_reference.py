import pytest
from pydantic import ValidationError

from duo_workflow_service.agent_platform.v1.catalog.sources.reference import (
    CatalogItemRef,
    CatalogItemSource,
    CatalogItemType,
)


class TestTheSharedVocabulary:
    """The enums are parsed straight from YAML, so their values are the config's contract."""

    def test_source_values(self):
        assert {source.value for source in CatalogItemSource} == {
            "workspace",
            "ai-catalog",
        }

    def test_item_type_values(self):
        """Pinned as a set, so a kind cannot be added without saying so here.

        There is no bare ``agent``: what a workspace holds is an agent *template*, which
        its source builds into a component.
        """
        assert {item_type.value for item_type in CatalogItemType} == {
            "agent_template",
            "flow",
        }


class TestCatalogItemRef:
    """A reference is inert data. Whether it can be served is its source's answer.

    Nothing here knows about wildcards, prompts or versions: those differ per source, so they are tested against the
    source that defines them.
    """

    def test_the_four_fields_are_all_it_carries(self):
        assert set(CatalogItemRef.model_fields) == {
            "source",
            "item_type",
            "item_id",
            "version",
        }

    def test_a_reference_any_source_might_use_is_accepted(self, ref):
        """Even one no source serves: refusing that is the registry's job, not the model's."""
        unserved = ref(
            source="ai-catalog", item_type="flow", item_id="321", version="1.2.0"
        )

        assert unserved.source is CatalogItemSource.AI_CATALOG
        assert unserved.version == "1.2.0"

    def test_version_defaults_to_absent(self, ref):
        assert ref().version is None

    @pytest.mark.parametrize(
        "overrides",
        [
            {"source": "not_a_source"},
            {"item_type": "not_a_type"},
            {"extra_key": "value"},
        ],
        ids=["unknown_source", "unknown_item_type", "unknown_key"],
    )
    def test_malformed_references_are_rejected(self, overrides, ref):
        """Unknown keys are refused rather than ignored, so a typo is never silently dropped."""
        with pytest.raises(ValidationError):
            ref(**overrides)

    @pytest.mark.parametrize(
        "overrides,expected",
        [
            ({}, True),
            ({"item_type": "flow"}, False),
            ({"source": "ai-catalog"}, False),
            ({"item_id": "123"}, False),
            # A claim repeats a declaration exactly, so a version is part of it.
            ({"version": "1.2.0"}, False),
        ],
        ids=["identical", "other_kind", "other_source", "other_id", "only_version"],
    )
    def test_equality_is_what_links_a_claim_to_a_declaration(
        self, overrides, expected, ref
    ):
        """A component claims an include entry by repeating it, so equality is the link.

        Field-wise, which is pydantic's default, so the reference needs no comparison of its own.
        """
        assert (ref() == ref(**overrides)) is expected

    def test_str_names_the_reference_for_error_messages(self, ref):
        assert str(ref()) == "workspace/agent_template/*"
