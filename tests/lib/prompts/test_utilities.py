"""Tests for lib/prompts/utilities module."""

import contextvars
import hashlib
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from lib.prompts.utilities import (
    _security_suffix,
    _security_suffix_var,
    prompt_template_to_messages,
    render_security_block,
)


class TestPromptTemplateToMessages:
    """Tests for prompt_template_to_messages function."""

    @pytest.fixture(autouse=True)
    def mock_security_suffix(self):
        with patch("lib.prompts.utilities._security_suffix", return_value="test"):
            yield

    def test_basic_message_conversion(self):
        """Test conversion of basic role -> content mapping."""
        tpl = {
            "system": "You are a helpful assistant",
            "user": "Hello!",
        }

        result = prompt_template_to_messages(tpl)

        assert len(result) == 3
        assert result[0] == (
            "system",
            render_security_block() + "You are a helpful assistant",
        )
        assert result[1] == ("user", "Hello!")
        assert isinstance(result[2], MessagesPlaceholder)
        assert result[2].variable_name == "history"
        assert result[2].optional is True

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
            render_security_block() + "You are a helpful assistant",
        )
        assert isinstance(result[1], MessagesPlaceholder)
        assert result[1].variable_name == "history"
        assert result[2] == ("user", "Hello!")

    def test_empty_template(self):
        """Test handling of empty template."""
        tpl: dict[str, str] = {}

        result = prompt_template_to_messages(tpl)

        assert len(result) == 1
        assert isinstance(result[0], MessagesPlaceholder)
        assert result[0].variable_name == "history"
        assert result[0].optional is True

    def test_jinja_template_content(self):
        """Test that jinja template content is preserved."""
        tpl = {
            "system": "You are {{ role }}",
            "user": "{% if condition %}Hello{% else %}Goodbye{% endif %}",
        }

        result = prompt_template_to_messages(tpl)

        assert result[0] == (
            "system",
            render_security_block() + "You are {{ role }}",
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

        assert len(result) == 5
        assert result[0] == (
            "system",
            render_security_block() + "System message",
        )
        assert result[1] == ("assistant", "Assistant message")
        assert result[2] == ("human", "Human message")
        assert result[3] == ("user", "User message")
        assert isinstance(result[4], MessagesPlaceholder)
        assert result[4].variable_name == "history"
        assert result[4].optional is True

    def test_security_injected_only_once(self):
        """Test that security include is only prepended to the first system message."""
        tpl = {
            "system_static": "Static system content",
            "system_dynamic": "Dynamic system content",
            "user": "Hello!",
        }

        result = prompt_template_to_messages(tpl)

        assert len(result) == 4
        assert result[0] == (
            "system_static",
            render_security_block() + "Static system content",
        )
        assert result[1] == ("system_dynamic", "Dynamic system content")
        assert result[2] == ("user", "Hello!")
        assert isinstance(result[3], MessagesPlaceholder)
        assert result[3].variable_name == "history"
        assert result[3].optional is True

    def test_no_security_injection_without_system_role(self):
        """Test that security include is not added when there's no system role."""
        tpl = {
            "user": "Hello!",
            "assistant": "Hi there!",
        }

        result = prompt_template_to_messages(tpl)

        assert len(result) == 3
        assert result[0] == ("user", "Hello!")
        assert result[1] == ("assistant", "Hi there!")
        assert isinstance(result[2], MessagesPlaceholder)
        assert result[2].variable_name == "history"
        assert result[2].optional is True


class TestSecuritySuffix:
    """Tests for the _security_suffix() internal function."""

    @pytest.fixture(autouse=True)
    def reset_security_suffix_var(self):
        """Reset the cached suffix before each test so tests are independent."""
        _security_suffix_var.set(None)
        yield

    def test_workflow_id_branch_returns_deterministic_suffix(self):
        """When workflow_id is present, suffix is derived deterministically from it."""
        with patch("lib.prompts.utilities.get_workflow_id", return_value="abc-123"):
            result = _security_suffix()

        expected = hashlib.sha256(b"abc-123").hexdigest()[:16]
        assert result == expected

    def test_workflow_id_branch_is_cached_within_context(self):
        """The derived suffix is cached so the hash is not recomputed on subsequent calls."""
        with patch("lib.prompts.utilities.get_workflow_id", return_value="abc-123"):
            first = _security_suffix()
            second = _security_suffix()

        assert first == second

    def test_random_fallback_branch_returns_hex_string(self):
        """When workflow_id is absent, a random 16-char hex suffix is returned."""
        with patch("lib.prompts.utilities.get_workflow_id", return_value=None):
            result = _security_suffix()

        assert len(result) == 16
        assert all(c in "0123456789abcdef" for c in result)

    def test_random_fallback_branch_is_cached_within_context(self):
        """The random suffix is cached so it remains stable across calls in the same context."""
        with patch("lib.prompts.utilities.get_workflow_id", return_value=None):
            first = _security_suffix()
            second = _security_suffix()

        assert first == second

    def test_context_isolation_produces_different_random_suffixes(self):
        """Each independent context (i.e. each request) gets its own random suffix."""
        results = []

        def run():
            # Reset the var so this context starts clean (simulates a fresh request)
            _security_suffix_var.set(None)
            with patch("lib.prompts.utilities.get_workflow_id", return_value=None):
                results.append(_security_suffix())

        ctx1 = contextvars.copy_context()
        ctx2 = contextvars.copy_context()
        ctx1.run(run)
        ctx2.run(run)

        assert len(results) == 2
        assert results[0] != results[1]


class TestAutoAddHistoryPlaceholder:
    """Tests for the auto-add history placeholder behavior.

    Without this behavior, templates missing ``placeholder: history`` silently
    drop conversation history, causing infinite agent loops.
    """

    def test_auto_adds_optional_history_placeholder_when_missing(self):
        tpl = {"system": "You are helpful", "user": "Hello!"}

        result = prompt_template_to_messages(tpl)

        assert isinstance(result[-1], MessagesPlaceholder)
        assert result[-1].variable_name == "history"
        assert result[-1].optional is True

    def test_does_not_duplicate_existing_history_placeholder(self):
        tpl = {
            "system": "You are helpful",
            "placeholder": "history",
            "user": "Hello!",
        }

        result = prompt_template_to_messages(tpl)

        history_placeholders = [
            m
            for m in result
            if isinstance(m, MessagesPlaceholder) and m.variable_name == "history"
        ]
        assert len(history_placeholders) == 1

    def test_history_injected_at_runtime_when_auto_added(self):
        """End-to-end: history must actually appear in rendered messages."""
        tpl = {"system": "You are helpful", "user": "Fix bug X"}
        messages = prompt_template_to_messages(tpl)
        ct = ChatPromptTemplate.from_messages(messages, template_format="jinja2")

        history = [
            AIMessage(
                content="I'll read the file",
                tool_calls=[{"name": "read_file", "args": {"path": "f"}, "id": "1"}],
            ),
            ToolMessage(content="file contents", tool_call_id="1"),
        ]
        rendered = ct.invoke({"history": history})

        msg_types = [type(m).__name__ for m in rendered.messages]
        assert msg_types == [
            "SystemMessage",
            "HumanMessage",
            "AIMessage",
            "ToolMessage",
        ]

    def test_history_empty_on_first_turn_when_auto_added(self):
        """Optional placeholder renders as zero messages when history is empty."""
        tpl = {"system": "You are helpful", "user": "Fix bug X"}
        messages = prompt_template_to_messages(tpl)
        ct = ChatPromptTemplate.from_messages(messages, template_format="jinja2")

        rendered = ct.invoke({"history": []})

        msg_types = [type(m).__name__ for m in rendered.messages]
        assert msg_types == ["SystemMessage", "HumanMessage"]

    def test_history_omitted_on_first_turn_when_auto_added(self):
        """Optional placeholder renders as zero messages when history is not provided."""
        tpl = {"system": "You are helpful", "user": "Fix bug X"}
        messages = prompt_template_to_messages(tpl)
        ct = ChatPromptTemplate.from_messages(messages, template_format="jinja2")

        # No history key at all — optional=True means this should not raise
        rendered = ct.invoke({})

        msg_types = [type(m).__name__ for m in rendered.messages]
        assert msg_types == ["SystemMessage", "HumanMessage"]
