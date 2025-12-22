from lib.events import GLReportingEventContext


class TestGLReportingEventContext:
    """Test suite for GLReportingEventContext class."""

    def test_legacy_workflow_type(self):
        """Test legacy workflow type properties and behavior."""
        flow = GLReportingEventContext.from_workflow_definition("software_development")

        assert flow.value == "software_development"
        assert flow.feature_qualified_name == "software_development"
        assert flow.feature_ai_catalog_item is False
        assert str(flow) == "software_development"

    def test_flow_registry_path(self):
        """Test Flow Registry path with version."""
        flow = GLReportingEventContext.from_workflow_definition("my_flow/v1")

        assert flow.value == "my_flow"
        assert flow.feature_qualified_name == "my_flow/v1"
        assert flow.feature_ai_catalog_item is False
        assert str(flow) == "my_flow"

    def test_ai_catalog_item_legacy_type(self):
        """Test AI Catalog item with legacy type."""
        flow = GLReportingEventContext.from_workflow_definition(
            "my_flow", has_flow_config=True
        )

        assert flow.value == "my_flow"
        assert flow.feature_qualified_name == "my_flow"
        assert flow.feature_ai_catalog_item is True

    def test_ai_catalog_item_flow_registry(self):
        """Test AI Catalog item with Flow Registry path."""
        flow = GLReportingEventContext.from_workflow_definition(
            "my_flow/v1", has_flow_config=True
        )

        assert flow.value == "my_flow"
        assert flow.feature_qualified_name == "my_flow/v1"
        assert flow.feature_ai_catalog_item is True

    def test_equality(self):
        """Test equality comparisons with FlowType instances and strings."""
        flow1 = GLReportingEventContext.from_workflow_definition("software_development")
        flow2 = GLReportingEventContext.from_workflow_definition("software_development")
        flow3 = GLReportingEventContext.from_workflow_definition("chat")

        assert flow1 == flow2
        assert flow1 == "software_development"
        assert flow1 != flow3
        assert flow1 != "chat"
