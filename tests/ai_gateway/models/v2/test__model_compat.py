"""LangChain standard image blocks must reach LiteLLM in OpenAI's ``image_url`` shape.

The LiteLLM adapter copies ``message.content`` verbatim, so unlike the Anthropic
integration (which understands standard blocks natively) the conversion has to
happen on our side.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ai_gateway.models.v2._model_compat import (
    TOOL_IMAGE_PLACEHOLDER,
    TOOL_IMAGES_BANNER,
    hoist_tool_result_images,
    normalize_image_blocks,
)
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


DATA_URL_BLOCK = {
    "type": "image_url",
    "image_url": {"url": f"data:image/png;base64,{B64}"},
}
HTTPS_URL_BLOCK = {
    "type": "image_url",
    "image_url": {"url": "https://example.com/a.png"},
}


def tool_message(*blocks, tool_call_id="call_1") -> dict:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": list(blocks) if blocks else "bare string result",
    }


class TestHoistToolResultImages:
    """Pure-function behavior of the OpenAI-shape tool-image hoist."""

    def hoist(self, messages, provider="openai", model="gpt-4o"):
        return hoist_tool_result_images(messages, provider, model)

    def test_text_stays_in_tool_slot_and_image_moves_to_user_message(self):
        messages = self.hoist(
            [tool_message({"type": "text", "text": "file contents:"}, DATA_URL_BLOCK)]
        )

        assert messages[0]["role"] == "tool"
        assert messages[0]["content"] == [{"type": "text", "text": "file contents:"}]
        assert messages[0]["tool_call_id"] == "call_1"
        assert messages[1] == {
            "role": "user",
            "content": [
                {"type": "text", "text": TOOL_IMAGES_BANNER},
                DATA_URL_BLOCK,
            ],
        }

    def test_images_only_result_gets_the_placeholder(self):
        messages = self.hoist([tool_message(DATA_URL_BLOCK)])

        assert messages[0]["content"] == [
            {"type": "text", "text": TOOL_IMAGE_PLACEHOLDER}
        ]

    def test_parallel_tool_results_batch_into_one_user_message(self):
        second = {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,YQ=="},
        }
        messages = self.hoist(
            [
                tool_message(DATA_URL_BLOCK, tool_call_id="call_1"),
                tool_message(second, tool_call_id="call_2"),
            ]
        )

        assert [m["role"] for m in messages] == ["tool", "tool", "user"]
        assert messages[2]["content"][1:] == [DATA_URL_BLOCK, second]

    def test_separate_runs_flush_separately(self):
        messages = self.hoist(
            [
                tool_message(DATA_URL_BLOCK, tool_call_id="call_1"),
                {"role": "assistant", "content": "looking at it"},
                tool_message(HTTPS_URL_BLOCK, tool_call_id="call_2"),
            ]
        )

        assert [m["role"] for m in messages] == [
            "tool",
            "user",
            "assistant",
            "tool",
            "user",
        ]
        assert messages[1]["content"][1] == DATA_URL_BLOCK
        assert messages[4]["content"][1] == HTTPS_URL_BLOCK

    def test_bare_string_tool_result_passes_through_without_breaking_the_run(self):
        bare = tool_message(tool_call_id="call_2")
        messages = self.hoist(
            [
                tool_message(DATA_URL_BLOCK, tool_call_id="call_1"),
                bare,
                tool_message(HTTPS_URL_BLOCK, tool_call_id="call_3"),
            ]
        )

        assert messages[1] is bare
        assert [m["role"] for m in messages] == ["tool", "tool", "tool", "user"]
        assert messages[3]["content"][1:] == [DATA_URL_BLOCK, HTTPS_URL_BLOCK]

    def test_plain_https_image_urls_are_hoisted_too(self):
        messages = self.hoist([tool_message(HTTPS_URL_BLOCK)])

        assert messages[1]["content"][1] == HTTPS_URL_BLOCK

    def test_non_image_blocks_keep_their_positions(self):
        messages = self.hoist(
            [
                tool_message(
                    {"type": "text", "text": "before"},
                    DATA_URL_BLOCK,
                    {"type": "text", "text": "after"},
                )
            ]
        )

        assert messages[0]["content"] == [
            {"type": "text", "text": "before"},
            {"type": "text", "text": "after"},
        ]

    def test_imageless_batches_are_returned_identically(self):
        batch = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_1", "type": "function"}],
            },
            tool_message({"type": "text", "text": "no images"}),
        ]

        assert self.hoist(batch) is batch

    @pytest.mark.parametrize(
        ("provider", "model"),
        [
            (None, "gpt-4o"),
            ("anthropic", "claude-sonnet-4-6"),
            ("vertex_ai", "claude-sonnet-4-5"),
            ("bedrock", "anthropic.claude-3-sonnet"),
            ("mistral", "mistral-large"),
        ],
    )
    def test_gate_stays_closed_for_native_or_unverified_providers(
        self, provider, model
    ):
        batch = [tool_message(DATA_URL_BLOCK)]

        assert self.hoist(batch, provider=provider, model=model) is batch

    @pytest.mark.parametrize(
        ("provider", "model"),
        [
            ("openai", "gpt-4o"),
            ("custom_openai", "my-model"),
            ("fireworks_ai", "accounts/gitlab/deployments/x"),
            ("hosted_vllm", "qwen-vl"),
            ("vertex_ai", "gemini-2.5-flash"),
            ("gemini", "gemini-2.5-pro"),
        ],
    )
    def test_gate_opens_for_verbatim_tool_transport_providers(self, provider, model):
        messages = self.hoist([tool_message(DATA_URL_BLOCK)], provider, model)

        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"][1] == DATA_URL_BLOCK

    def test_user_role_images_are_not_touched(self):
        batch = [{"role": "user", "content": [DATA_URL_BLOCK]}]

        assert self.hoist(batch) is batch

    def test_empty_batch(self):
        assert self.hoist([]) == []


class TestChatLiteLLMHoistIntegration:
    """The hoist must run inside the real message pipeline, after normalize."""

    STANDARD_BLOCKS = [
        {"type": "text", "text": "Contents of diagram.png:"},
        {"type": "image", "base64": B64, "mime_type": "image/png"},
    ]

    def convert(self, model, provider, messages):
        chat = ChatLiteLLM(model=model, custom_llm_provider=provider)
        message_dicts, _ = chat._create_message_dicts(messages, stop=None)
        return message_dicts

    def canonical_messages(self):
        return [
            HumanMessage(content="compare the two diagrams"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"file_path": "a.png"},
                        "id": "call_1",
                    },
                    {
                        "name": "read_file",
                        "args": {"file_path": "b.png"},
                        "id": "call_2",
                    },
                ],
            ),
            ToolMessage(content=list(self.STANDARD_BLOCKS), tool_call_id="call_1"),
            ToolMessage(content=list(self.STANDARD_BLOCKS), tool_call_id="call_2"),
            HumanMessage(content="so, any differences?"),
        ]

    def test_canonical_openai_shape_sequence_hoists_per_run(self):
        message_dicts = self.convert(
            "my-model", "custom_openai", self.canonical_messages()
        )

        assert [m["role"] for m in message_dicts] == [
            "user",
            "assistant",
            "tool",
            "tool",
            "user",
            "user",
        ]
        for idx, call_id in ((2, "call_1"), (3, "call_2")):
            assert message_dicts[idx]["tool_call_id"] == call_id
            assert message_dicts[idx]["content"] == [
                {"type": "text", "text": "Contents of diagram.png:"}
            ]
        hoisted = message_dicts[4]
        assert hoisted["content"][0] == {"type": "text", "text": TOOL_IMAGES_BANNER}
        assert (
            hoisted["content"][1:]
            == [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{B64}"},
                }
            ]
            * 2
        )
        assert message_dicts[5]["content"] == "so, any differences?"

    @pytest.mark.parametrize(
        ("model", "provider"),
        [
            ("claude-sonnet-4-5-20250929", None),
            ("claude-sonnet-4-5", "vertex_ai"),
            ("anthropic.claude-3-sonnet-20240229-v1:0", "bedrock"),
        ],
    )
    def test_anthropic_family_keeps_images_in_the_tool_slot(self, model, provider):
        message_dicts = self.convert(model, provider, self.canonical_messages())

        assert len(message_dicts) == 5
        tool_content = message_dicts[2]["content"]
        assert tool_content[1]["type"] == "image_url"
        assert tool_content[1]["image_url"]["url"].endswith(B64)

    def test_images_only_tool_result_through_the_pipeline(self):
        messages = [
            HumanMessage(content="read it"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"file_path": "a.png"},
                        "id": "call_1",
                    }
                ],
            ),
            ToolMessage(
                content=[{"type": "image", "base64": B64, "mime_type": "image/png"}],
                tool_call_id="call_1",
            ),
        ]

        message_dicts = self.convert("gpt-4o", "openai", messages)

        assert message_dicts[2]["content"] == [
            {"type": "text", "text": TOOL_IMAGE_PLACEHOLDER}
        ]
        assert message_dicts[3]["role"] == "user"
        assert message_dicts[3]["content"][1]["type"] == "image_url"

    def test_vertex_gemini_hoists_while_vertex_claude_does_not(self):
        gemini = self.convert(
            "gemini-2.5-flash", "vertex_ai", self.canonical_messages()
        )
        claude = self.convert(
            "claude-sonnet-4-5", "vertex_ai", self.canonical_messages()
        )

        assert [m["role"] for m in gemini] == [
            "user",
            "assistant",
            "tool",
            "tool",
            "user",
            "user",
        ]
        assert len(claude) == 5
        assert claude[2]["content"][1]["type"] == "image_url"
