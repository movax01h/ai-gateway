"""LangChain standard image blocks must reach LiteLLM in OpenAI's ``image_url`` shape.

The LiteLLM adapter copies ``message.content`` verbatim, so unlike the Anthropic
integration (which understands standard blocks natively) the conversion has to
happen on our side.
"""

import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from ai_gateway.models.v2._model_compat import normalize_image_blocks
from ai_gateway.models.v2.chat_litellm import ChatLiteLLM

B64 = "aVZCT1J5Qm1ZV3Rs"


def image_message(*blocks) -> dict:
    return {"role": "user", "content": list(blocks)}


class TestNormalizeImageBlocks:
    def test_base64_block_becomes_an_openai_data_url(self):
        messages = normalize_image_blocks(
            [image_message({"type": "image", "base64": B64, "mime_type": "image/png"})]
        )

        assert messages[0]["content"] == [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{B64}"},
            }
        ]

    def test_url_block_is_passed_through_as_image_url(self):
        messages = normalize_image_blocks(
            [image_message({"type": "image", "url": "https://example.com/a.png"})]
        )

        assert messages[0]["content"] == [
            {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}
        ]

    def test_text_blocks_keep_their_position(self):
        messages = normalize_image_blocks(
            [
                image_message(
                    {"type": "text", "text": "what is this?"},
                    {"type": "image", "base64": B64, "mime_type": "image/jpeg"},
                )
            ]
        )

        content = messages[0]["content"]
        assert content[0] == {"type": "text", "text": "what is this?"}
        assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_other_message_fields_are_preserved(self):
        messages = normalize_image_blocks(
            [
                {
                    "role": "user",
                    "name": "someone",
                    "content": [
                        {"type": "image", "base64": B64, "mime_type": "image/png"}
                    ],
                }
            ]
        )

        assert messages[0]["role"] == "user"
        assert messages[0]["name"] == "someone"

    @pytest.mark.parametrize(
        "message",
        [
            {"role": "user", "content": "a plain string"},
            {"role": "user", "content": [{"type": "text", "text": "no images"}]},
            {"role": "assistant", "content": None},
            {"role": "user", "content": []},
            # Already in OpenAI form (e.g. re-normalized) — must not be touched.
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,x"},
                    }
                ],
            },
        ],
    )
    def test_messages_without_standard_image_blocks_are_untouched(self, message):
        assert normalize_image_blocks([message])[0] is message

    def test_mixed_batch_only_rewrites_the_affected_message(self):
        plain = {"role": "user", "content": "hello"}
        with_image = image_message(
            {"type": "image", "base64": B64, "mime_type": "image/png"}
        )

        messages = normalize_image_blocks([plain, with_image])

        assert messages[0] is plain
        assert messages[1] is not with_image
        assert messages[1]["content"][0]["type"] == "image_url"

    def test_empty_batch(self):
        assert normalize_image_blocks([]) == []

    @pytest.mark.parametrize("role", ["user", "assistant", "tool", "system"])
    def test_every_role_is_rewritten(self, role):
        """An image can be produced by a tool as well as attached by the user, and LiteLLM silently drops a standard
        block whatever the role carrying it."""
        messages = normalize_image_blocks(
            [
                {
                    "role": role,
                    "content": [
                        {"type": "image", "base64": B64, "mime_type": "image/png"}
                    ],
                }
            ]
        )

        assert messages[0]["content"][0] == {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{B64}"},
        }
        assert messages[0]["role"] == role


class TestChatLiteLLMIntegration:
    """The normalization must actually be wired into the message pipeline."""

    def test_create_message_dicts_rewrites_image_blocks(self):
        model = ChatLiteLLM(model="claude-sonnet-4-5-20250929")
        messages = [
            HumanMessage(
                content=[
                    {"type": "text", "text": "what is this?"},
                    {"type": "image", "base64": B64, "mime_type": "image/png"},
                ]
            )
        ]

        message_dicts, _ = model._create_message_dicts(messages, stop=None)

        content = message_dicts[0]["content"]
        assert content[0] == {"type": "text", "text": "what is this?"}
        assert content[1] == {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{B64}"},
        }

    def test_tool_results_carrying_images_are_rewritten(self):
        """Not gated on self-hosted: the default duo_agent_platform models reach
        the provider through this same path."""
        model = ChatLiteLLM(model="claude-sonnet-4-5-20250929")
        messages = [
            ToolMessage(
                content=[
                    {"type": "text", "text": "Contents of diagram.png:"},
                    {"type": "image", "base64": B64, "mime_type": "image/png"},
                ],
                tool_call_id="call-1",
            )
        ]

        message_dicts, _ = model._create_message_dicts(messages, stop=None)

        assert message_dicts[0]["content"][1] == {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{B64}"},
        }

    def test_plain_text_messages_are_unchanged(self):
        model = ChatLiteLLM(model="claude-sonnet-4-5-20250929")

        message_dicts, _ = model._create_message_dicts(
            [HumanMessage(content="hello")], stop=None
        )

        assert message_dicts[0]["content"] == "hello"
