import pytest

from duo_workflow_service.agent_platform.v1.catalog.sources.reference import (
    CatalogItemRef,
)


@pytest.fixture(name="ref")
def ref_fixture():
    """Build the one reference a source serves today: every workspace agent template.

    A factory rather than a value, because most tests here change one field and that
    difference is what they are about.
    """

    def build(**overrides) -> CatalogItemRef:
        return CatalogItemRef.model_validate(
            {
                "source": "workspace",
                "item_type": "agent_template",
                "item_id": "*",
                **overrides,
            }
        )

    return build
