"""Tests for lib/prompts/utilities module."""

from langchain_core.prompts import MessagesPlaceholder

from lib.prompts.utilities import (
    TOOL_OUTPUT_SECURITY_INCLUDE,
    prompt_template_to_messages,
)


class TestPromptTemplateToMessages:
    """Tests for prompt_template_to_messages function."""

    def test_basic_message_conversion(self):
        """Test conversion of basic role -> content mapping."""
        tpl = {
            "system": "You are a helpful assistant",
            "user": "Hello!",
        }

        result = prompt_template_to_messages(tpl)

        assert len(result) == 2
        assert result[0] == (
            "system",
            TOOL_OUTPUT_SECURITY_INCLUDE + "You are a helpful assistant",
        )
        assert result[1] == ("user", "Hello!")

    def test_placeholder_conversion(self):
        """Test that 'placeholder' role creates a MessagesPlaceholder."""
        tpl = {
            "system": "You are a helpful assistant",
            "placeholder": "history",
            "user": "Hello!",
        }

        result = prompt_template_to_messages(tpl)

        assert len(result) == 3
        assert result[0] == (
            "system",
            TOOL_OUTPUT_SECURITY_INCLUDE + "You are a helpful assistant",
        )
        assert isinstance(result[1], MessagesPlaceholder)
        assert result[1].variable_name == "history"
        assert result[2] == ("user", "Hello!")

    def test_empty_template(self):
        """Test handling of empty template."""
        tpl: dict[str, str] = {}

        result = prompt_template_to_messages(tpl)

        assert len(result) == 0
        assert not list(result)

    def test_jinja_template_content(self):
        """Test that jinja template content is preserved."""
        tpl = {
            "system": "You are {{ role }}",
            "user": "{% if condition %}Hello{% else %}Goodbye{% endif %}",
        }

        result = prompt_template_to_messages(tpl)

        assert result[0] == (
            "system",
            TOOL_OUTPUT_SECURITY_INCLUDE + "You are {{ role }}",
        )
        assert result[1] == (
            "user",
            "{% if condition %}Hello{% else %}Goodbye{% endif %}",
        )

    def test_different_roles(self):
        """Test various role types."""
        tpl = {
            "system": "System message",
            "assistant": "Assistant message",
            "human": "Human message",
            "user": "User message",
        }

        result = prompt_template_to_messages(tpl)

        assert len(result) == 4
        assert result[0] == (
            "system",
            TOOL_OUTPUT_SECURITY_INCLUDE + "System message",
        )
        assert result[1] == ("assistant", "Assistant message")
        assert result[2] == ("human", "Human message")
        assert result[3] == ("user", "User message")

    def test_security_injected_only_once(self):
        """Test that security include is only prepended to the first system message."""
        tpl = {
            "system_static": "Static system content",
            "system_dynamic": "Dynamic system content",
            "user": "Hello!",
        }

        result = prompt_template_to_messages(tpl)

        assert len(result) == 3
        assert result[0] == (
            "system_static",
            TOOL_OUTPUT_SECURITY_INCLUDE + "Static system content",
        )
        assert result[1] == ("system_dynamic", "Dynamic system content")
        assert result[2] == ("user", "Hello!")

    def test_no_security_injection_without_system_role(self):
        """Test that security include is not added when there's no system role."""
        tpl = {
            "user": "Hello!",
            "assistant": "Hi there!",
        }

        result = prompt_template_to_messages(tpl)

        assert len(result) == 2
        assert result[0] == ("user", "Hello!")
        assert result[1] == ("assistant", "Hi there!")
