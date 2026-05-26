# pylint: disable=file-naming-for-tests
"""Unit tests for AgentNode._extract_text."""

from langchain_core.messages import AIMessage

from duo_workflow_service.agent_platform.v1.components.agent.nodes.agent_node import (
    _LITELLM_EMPTY_CONTENT_PLACEHOLDER,
    AgentNode,
)


class TestExtractText:
    """Unit tests for AgentNode._extract_text covering all content branches."""

    def test_str_block_inside_list(self):
        """A plain str entry inside a list content block is included in extracted text."""
        message = AIMessage(
            content=["plain string block", {"type": "text", "text": " and dict block"}]
        )
        assert AgentNode._extract_text(message) == "plain string block and dict block"

    def test_non_str_non_list_content_returns_empty(self):
        """Content that is neither str nor list returns an empty string."""
        message = AIMessage(content="placeholder")
        # Bypass the normal str path by directly setting content to an unexpected type
        message.content = 42
        assert AgentNode._extract_text(message) == ""

    def test_litellm_placeholder_string_content_returns_empty(self):
        """LiteLLM placeholder as a plain string content is filtered out."""
        message = AIMessage(content=_LITELLM_EMPTY_CONTENT_PLACEHOLDER)
        assert AgentNode._extract_text(message) == ""

    def test_litellm_placeholder_in_list_str_block_is_filtered(self):
        """LiteLLM placeholder as a plain str entry inside list content is filtered out."""
        message = AIMessage(
            content=[
                _LITELLM_EMPTY_CONTENT_PLACEHOLDER,
                {"type": "text", "text": " real text"},
            ]
        )
        assert AgentNode._extract_text(message) == " real text"

    def test_litellm_placeholder_in_dict_text_block_is_filtered(self):
        """LiteLLM placeholder inside a dict text block within list content is filtered out."""
        message = AIMessage(
            content=[
                {"type": "text", "text": _LITELLM_EMPTY_CONTENT_PLACEHOLDER},
                {"type": "text", "text": " real text"},
            ]
        )
        assert AgentNode._extract_text(message) == " real text"
