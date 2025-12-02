"""Tests for AI context."""

from dataclasses import asdict

from lib.internal_events.ai_context import AIContext


class TestAIContext:
    """Test AIContext dataclass."""

    def test_creates_ai_context_with_all_fields(self):
        """Test that AIContext can be instantiated with all fields."""
        ai_context = AIContext(
            session_id="session123",
            workflow_id="workflow456",
            flow_type="chat",
            agent_name="duo_chat",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            ephemeral_5m_input_tokens=25,
            ephemeral_1h_input_tokens=75,
            cache_read=5,
        )

        assert ai_context.session_id == "session123"
        assert ai_context.workflow_id == "workflow456"
        assert ai_context.flow_type == "chat"
        assert ai_context.agent_name == "duo_chat"
        assert ai_context.input_tokens == 100
        assert ai_context.output_tokens == 50
        assert ai_context.total_tokens == 150
        assert ai_context.ephemeral_5m_input_tokens == 25
        assert ai_context.ephemeral_1h_input_tokens == 75
        assert ai_context.cache_read == 5

    def test_creates_ai_context_with_minimal_fields(self):
        """Test that AIContext can be instantiated with no fields (all None)."""
        ai_context = AIContext()

        assert ai_context.session_id is None
        assert ai_context.workflow_id is None
        assert ai_context.flow_type is None
        assert ai_context.agent_name is None
        assert ai_context.input_tokens is None
        assert ai_context.output_tokens is None
        assert ai_context.total_tokens is None
        assert ai_context.ephemeral_5m_input_tokens is None
        assert ai_context.ephemeral_1h_input_tokens is None
        assert ai_context.cache_read is None

    def test_creates_ai_context_with_partial_fields(self):
        """Test that AIContext can be instantiated with some fields."""
        ai_context = AIContext(
            session_id="session789",
            input_tokens=200,
            output_tokens=100,
        )

        assert ai_context.session_id == "session789"
        assert ai_context.input_tokens == 200
        assert ai_context.output_tokens == 100
        assert ai_context.workflow_id is None
        assert ai_context.flow_type is None
        assert ai_context.total_tokens is None

    def test_ai_context_converts_to_dict(self):
        """Test that AIContext can be converted to a dict using asdict."""
        ai_context = AIContext(
            session_id="session123",
            workflow_id="workflow456",
            input_tokens=100,
            output_tokens=50,
        )

        context_dict = asdict(ai_context)

        assert context_dict["session_id"] == "session123"
        assert context_dict["workflow_id"] == "workflow456"
        assert context_dict["input_tokens"] == 100
        assert context_dict["output_tokens"] == 50
        assert context_dict["flow_type"] is None
        assert context_dict["agent_name"] is None

    def test_ai_context_dict_includes_all_schema_fields(self):
        """Test that asdict includes all schema fields even when None."""
        ai_context = AIContext()
        context_dict = asdict(ai_context)

        # Verify all schema fields are present in the dict
        expected_fields = {
            "session_id",
            "workflow_id",
            "flow_type",
            "agent_name",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "ephemeral_5m_input_tokens",
            "ephemeral_1h_input_tokens",
            "cache_read",
        }

        assert set(context_dict.keys()) == expected_fields
        assert all(value is None for value in context_dict.values())

    def test_ai_context_created_from_dict_get(self):
        """Test creating AIContext from dictionary using .get() as done in client.py."""
        context_data = {
            "session_id": "session123",
            "workflow_id": "workflow456",
            "flow_type": "software_development",
            "agent_name": "code_agent",
            "input_tokens": 150,
            "output_tokens": 75,
            "total_tokens": 225,
            "ephemeral_5m_input_tokens": 30,
            "ephemeral_1h_input_tokens": 90,
            "cache_read": 10,
        }

        ai_context = AIContext(
            session_id=context_data.get("session_id"),
            workflow_id=context_data.get("workflow_id"),
            flow_type=context_data.get("flow_type"),
            agent_name=context_data.get("agent_name"),
            input_tokens=context_data.get("input_tokens"),
            output_tokens=context_data.get("output_tokens"),
            total_tokens=context_data.get("total_tokens"),
            ephemeral_5m_input_tokens=context_data.get("ephemeral_5m_input_tokens"),
            ephemeral_1h_input_tokens=context_data.get("ephemeral_1h_input_tokens"),
            cache_read=context_data.get("cache_read"),
        )

        assert ai_context.session_id == "session123"
        assert ai_context.workflow_id == "workflow456"
        assert ai_context.flow_type == "software_development"
        assert ai_context.input_tokens == 150
        assert ai_context.total_tokens == 225

    def test_ai_context_handles_missing_keys_with_dict_get(self):
        """Test that .get() returns None for missing keys."""
        context_data = {
            "session_id": "session123",
            "input_tokens": 100,
        }

        ai_context = AIContext(
            session_id=context_data.get("session_id"),
            workflow_id=context_data.get("workflow_id"),
            input_tokens=context_data.get("input_tokens"),
            output_tokens=context_data.get("output_tokens"),
        )

        assert ai_context.session_id == "session123"
        assert ai_context.workflow_id is None
        assert ai_context.input_tokens == 100
        assert ai_context.output_tokens is None
