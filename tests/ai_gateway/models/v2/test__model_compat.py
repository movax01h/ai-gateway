"""LangChain standard image blocks must reach LiteLLM in OpenAI's ``image_url`` shape.

The LiteLLM adapter copies ``message.content`` verbatim, so unlike the Anthropic
integration (which understands standard blocks natively) the conversion has to
happen on our side.
"""

from contextlib import contextmanager
from typing import Optional
from unittest.mock import MagicMock, call, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ai_gateway.model_metadata import ModelMetadata
from ai_gateway.model_selection.model_selection_config import (
    ChatLiteLLMDefinition,
    ModelSelectionConfig,
)
from ai_gateway.model_selection.models import ModelClassProvider
from ai_gateway.models.v2._model_compat import (
    IMAGE_OMITTED_NOTICE,
    TOOL_IMAGE_PLACEHOLDER,
    TOOL_IMAGES_BANNER,
    _model_supports_vision,
    current_model_supports_vision,
    hoist_tool_result_images,
    normalize_image_blocks,
    strip_image_blocks_for_non_vision_model,
)
from ai_gateway.models.v2.chat_litellm import ChatLiteLLM
from lib.context import (
    current_model_metadata_context,
    current_model_metadata_with_size_context,
)

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


# Carries an explicit supports_vision: false in litellm's registry; the
# precondition test below fails if a litellm bump changes that.
NON_VISION_MODEL = "o3-mini"


class TestStripImageBlocksForNonVisionModel:
    """The gate fails open: it strips only on a positive non-vision verdict."""

    def test_registry_preconditions_still_hold(self):
        # The unmocked tests below rely on these two registry facts.
        assert _model_supports_vision(NON_VISION_MODEL, None) is False
        assert _model_supports_vision("claude-sonnet-4-6", None) is True

    def test_known_non_vision_model_gets_a_text_notice(self):
        messages = strip_image_blocks_for_non_vision_model(
            [
                image_message(
                    {"type": "text", "text": "what is this?"},
                    {"type": "image", "base64": B64, "mime_type": "image/png"},
                )
            ],
            NON_VISION_MODEL,
            None,
        )

        assert messages[0]["content"] == [
            {"type": "text", "text": "what is this?"},
            {
                "type": "text",
                "text": IMAGE_OMITTED_NOTICE.format(model=NON_VISION_MODEL),
            },
        ]

    def test_an_already_normalized_image_block_is_stripped_too(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,QUJD"},
                    },
                ],
            }
        ]

        stripped = strip_image_blocks_for_non_vision_model(
            messages, NON_VISION_MODEL, None
        )

        assert stripped[0]["content"][1] == {
            "type": "text",
            "text": IMAGE_OMITTED_NOTICE,
        }

    def test_a_strip_is_logged_with_where_the_verdict_came_from(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "base64": "QUJD", "mime_type": "image/png"},
                ],
            }
        ]

        with patch("ai_gateway.models.v2._model_compat.log") as log_mock:
            strip_image_blocks_for_non_vision_model(messages, NON_VISION_MODEL, None)

        log_mock.info.assert_called_once()
        assert log_mock.info.call_args.kwargs == {
            "model": NON_VISION_MODEL,
            "custom_llm_provider": None,
            "verdict_source": "litellm",
        }

    def test_url_carrying_image_blocks_are_stripped_too(self):
        messages = strip_image_blocks_for_non_vision_model(
            [image_message({"type": "image", "url": "https://example.com/a.png"})],
            NON_VISION_MODEL,
            None,
        )

        assert messages[0]["content"][0]["type"] == "text"

    def test_known_vision_model_passes_through(self):
        batch = [image_message({"type": "image", "base64": B64})]

        assert (
            strip_image_blocks_for_non_vision_model(batch, "claude-sonnet-4-6", None)
            is batch
        )

    def test_unknown_model_fails_open(self):
        batch = [image_message({"type": "image", "base64": B64})]

        assert (
            strip_image_blocks_for_non_vision_model(
                batch, "custom_openai/my-model", None
            )
            is batch
        )

    def test_unannotated_registry_model_fails_open(self):
        # An absent flag means "not annotated", not "no vision".
        batch = [image_message({"type": "image", "base64": B64})]

        with patch("ai_gateway.models.v2._model_compat.litellm") as litellm_mock:
            litellm_mock.get_model_info.return_value = {"supports_vision": None}

            assert (
                strip_image_blocks_for_non_vision_model(batch, "some-model", None)
                is batch
            )

        litellm_mock.get_model_info.assert_called_once_with("some-model")

    def test_unexpected_registry_failure_fails_open_and_logs(self):
        # Anything other than litellm's "isn't mapped yet" must leave a trace.
        batch = [image_message({"type": "image", "base64": B64})]

        with (
            patch("ai_gateway.models.v2._model_compat.litellm") as litellm_mock,
            patch("ai_gateway.models.v2._model_compat.log") as log_mock,
        ):
            litellm_mock.get_model_info.side_effect = TypeError("boom")

            assert (
                strip_image_blocks_for_non_vision_model(batch, "some-model", None)
                is batch
            )

        log_mock.debug.assert_called_once()
        assert log_mock.debug.call_args.kwargs["error_type"] == "TypeError"

    @pytest.mark.parametrize("model", [None, ""])
    def test_missing_model_name_passes_through(self, model):
        batch = [image_message({"type": "image", "base64": B64})]

        with patch("ai_gateway.models.v2._model_compat.litellm") as litellm_mock:
            assert strip_image_blocks_for_non_vision_model(batch, model, None) is batch

        litellm_mock.get_model_info.assert_not_called()

    def test_imageless_batches_never_query_the_registry(self):
        batch = [{"role": "user", "content": "hello"}]

        with patch("ai_gateway.models.v2._model_compat.litellm") as litellm_mock:
            assert (
                strip_image_blocks_for_non_vision_model(batch, NON_VISION_MODEL, None)
                is batch
            )

        litellm_mock.get_model_info.assert_not_called()

    def test_create_message_dicts_strips_before_normalizing(self):
        model = ChatLiteLLM(model=NON_VISION_MODEL)
        messages = [
            HumanMessage(content="read the diagram"),
            ToolMessage(
                content=[
                    {"type": "text", "text": "Contents of diagram.png:"},
                    {"type": "image", "base64": B64, "mime_type": "image/png"},
                ],
                tool_call_id="call_1",
            ),
        ]

        message_dicts, _ = model._create_message_dicts(messages, stop=None)

        tool_content = message_dicts[1]["content"]
        assert all(block["type"] == "text" for block in tool_content)
        assert IMAGE_OMITTED_NOTICE.format(model=NON_VISION_MODEL) in [
            block["text"] for block in tool_content
        ]

    def test_tool_read_image_block_reaches_a_vision_model_as_a_data_url(self):
        """Cross-layer contract: a canonical image block must leave the LiteLLM
        boundary as a byte-identical data-URL.

        ``read_file`` emits exactly these blocks (built with
        ``image_content_block``; pinned in the envelope funnel's own suite), so
        this is the provider-boundary half of that contract without importing
        the funnel, which lives in a separate MR train.
        """
        # Only this cross-layer test needs the workflow service's dependency tree.
        from duo_workflow_service.entities.image_blocks import image_content_block

        blocks = [
            {"type": "text", "text": "Read image file: diagram.png (image/png)."},
            image_content_block(base64=B64, mime_type="image/png"),
        ]

        model = ChatLiteLLM(model="claude-sonnet-4-5-20250929")
        message_dicts, _ = model._create_message_dicts(
            [ToolMessage(content=blocks, tool_call_id="call_1")], stop=None
        )

        image_blocks = [
            block
            for block in message_dicts[0]["content"]
            if block["type"] == "image_url"
        ]
        assert len(image_blocks) == 1
        assert image_blocks[0]["image_url"]["url"] == f"data:image/png;base64,{B64}"

    def test_consecutive_images_collapse_to_one_notice(self):
        messages = strip_image_blocks_for_non_vision_model(
            [
                image_message(
                    {"type": "text", "text": "three screenshots:"},
                    {"type": "image", "base64": B64, "mime_type": "image/png"},
                    {"type": "image", "base64": B64, "mime_type": "image/png"},
                    {"type": "image", "base64": B64, "mime_type": "image/png"},
                    {"type": "text", "text": "and one more:"},
                    {"type": "image", "base64": B64, "mime_type": "image/png"},
                )
            ],
            NON_VISION_MODEL,
            None,
        )

        notice = IMAGE_OMITTED_NOTICE.format(model=NON_VISION_MODEL)
        assert messages[0]["content"] == [
            {"type": "text", "text": "three screenshots:"},
            {"type": "text", "text": notice},
            {"type": "text", "text": "and one more:"},
            {"type": "text", "text": notice},
        ]


MINIMAX = "accounts/fireworks/models/minimax-m3"


@contextmanager
def model_context(metadata):
    token = current_model_metadata_context.set(metadata)
    try:
        yield
    finally:
        current_model_metadata_context.reset(token)


def metadata_for(definition) -> ModelMetadata:
    return ModelMetadata(
        provider="gitlab",
        name=definition.gitlab_identifier,
        llm_definition=definition,
        friendly_name=definition.name,
    )


def fireworks_definition(supports_vision, model=MINIMAX, **params):
    return ChatLiteLLMDefinition(
        name="Fireworks Test",
        gitlab_identifier="fireworks_test",
        max_context_tokens=200_000,
        supports_vision=supports_vision,
        params={"model": model, "custom_llm_provider": "fireworks_ai", **params},
    )


class TestCurrentModelSupportsVision:
    @staticmethod
    def _metadata(model, provider, declared, request_params=None):
        metadata = MagicMock()
        metadata.to_params.return_value = request_params or {}
        metadata.llm_definition.supports_vision = declared
        metadata.llm_definition.params.model = model
        metadata.llm_definition.params.identifier = None
        metadata.llm_definition.params.custom_llm_provider = provider
        return metadata

    @pytest.fixture(autouse=True)
    def _no_by_tag_context(self):
        token = current_model_metadata_with_size_context.set(None)
        yield
        current_model_metadata_with_size_context.reset(token)

    def test_no_resolved_model_is_unknown(self):
        token = current_model_metadata_context.set(None)
        try:
            assert current_model_supports_vision() is None
        finally:
            current_model_metadata_context.reset(token)

    @pytest.mark.parametrize("declared", [True, False])
    def test_the_declared_flag_is_the_verdict(self, declared):
        token = current_model_metadata_context.set(
            self._metadata(
                "accounts/fireworks/models/minimax-m3", "fireworks_ai", declared
            )
        )
        try:
            assert current_model_supports_vision() is declared
        finally:
            current_model_metadata_context.reset(token)

    def test_a_self_hosted_request_is_judged_by_its_own_model(self):
        # The family template says "gpt"; the request names the real deployment.
        token = current_model_metadata_context.set(
            self._metadata(
                "gpt",
                None,
                None,
                request_params={"model": "gpt-4o", "custom_llm_provider": "openai"},
            )
        )
        try:
            with patch(
                "ai_gateway.models.v2._model_compat._litellm_vision_verdict",
                return_value=True,
            ) as verdict:
                assert current_model_supports_vision() is True
        finally:
            current_model_metadata_context.reset(token)

        verdict.assert_called_once_with("gpt-4o", "openai")

    def test_metadata_without_a_model_name_is_unknown(self):
        token = current_model_metadata_context.set(self._metadata(None, None, None))
        try:
            assert current_model_supports_vision() is None
        finally:
            current_model_metadata_context.reset(token)

    @pytest.mark.parametrize(
        "tagged_declared,expected",
        [(False, False), (True, None), (None, None)],
        ids=["all-blind", "a-tagged-model-sees", "a-tagged-model-unknown"],
    )
    def test_every_model_a_component_may_run_on_has_a_say(
        self, tagged_declared, expected
    ):
        by_tag = MagicMock()
        by_tag.default = self._metadata(
            "accounts/fireworks/models/glm-5p3", "fireworks_ai", False
        )
        by_tag.by_tag = {
            "large": self._metadata(
                "accounts/fireworks/models/kimi-k3", "fireworks_ai", tagged_declared
            )
        }
        if tagged_declared is None:
            by_tag.by_tag[
                "large"
            ].llm_definition.params.model = "accounts/gitlab/deployments/unknown"
        token = current_model_metadata_with_size_context.set(by_tag)
        try:
            assert current_model_supports_vision() is expected
        finally:
            current_model_metadata_with_size_context.reset(token)

    def test_a_lookup_failure_is_unknown(self):
        broken = MagicMock()
        broken.to_params.side_effect = RuntimeError("boom")
        token = current_model_metadata_context.set(broken)
        try:
            assert current_model_supports_vision() is None
        finally:
            current_model_metadata_context.reset(token)


class TestModelSupportsVision:
    """Layer order: definition flag, litellm with the provider, litellm bare, unknown."""

    @pytest.fixture(name="litellm_mock")
    def litellm_mock_fixture(self):
        with patch("ai_gateway.models.v2._model_compat.litellm") as mock:
            yield mock

    @pytest.mark.parametrize(
        ("declared", "registry", "expected"),
        [
            # Minimax M3: live-verified vision while litellm's registry says no.
            (True, False, True),
            (False, True, False),
        ],
        ids=["flag-true-beats-registry-false", "flag-false-beats-registry-true"],
    )
    def test_declared_flag_beats_litellm(
        self, litellm_mock, declared, registry, expected
    ):
        litellm_mock.get_model_info.return_value = {"supports_vision": registry}

        with model_context(metadata_for(fireworks_definition(declared))):
            assert _model_supports_vision(MINIMAX, "fireworks_ai") is expected

        litellm_mock.get_model_info.assert_not_called()

    def test_without_a_flag_litellm_is_asked_with_the_provider(self, litellm_mock):
        litellm_mock.get_model_info.return_value = {"supports_vision": False}

        with model_context(metadata_for(fireworks_definition(None))):
            assert _model_supports_vision(MINIMAX, "fireworks_ai") is False

        litellm_mock.get_model_info.assert_called_once_with(
            MINIMAX, custom_llm_provider="fireworks_ai"
        )

    def test_unannotated_provider_entry_falls_back_to_the_bare_name(self, litellm_mock):
        litellm_mock.get_model_info.side_effect = [
            {"supports_vision": None},
            {"supports_vision": True},
        ]

        assert _model_supports_vision("some-model", "fireworks_ai") is True

        assert litellm_mock.get_model_info.call_args_list == [
            call("some-model", custom_llm_provider="fireworks_ai"),
            call("some-model"),
        ]

    def test_nothing_known_returns_none(self, litellm_mock):
        litellm_mock.get_model_info.side_effect = Exception(
            "This model isn't mapped yet."
        )

        assert _model_supports_vision("my-model", "custom_openai") is None

    def test_flag_of_another_model_in_the_context_is_ignored(self, litellm_mock):
        # A tag-routed component runs on a model other than the context's default.
        litellm_mock.get_model_info.return_value = {"supports_vision": False}

        with model_context(metadata_for(fireworks_definition(True))):
            assert (
                _model_supports_vision(
                    "accounts/fireworks/models/glm-5p3", "fireworks_ai"
                )
                is False
            )

    def test_flag_matches_the_router_identifier_too(self, litellm_mock):
        definition = fireworks_definition(
            False, model="glm-x", identifier="accounts/gitlab/routers/glm-x"
        )

        with model_context(metadata_for(definition)):
            assert (
                _model_supports_vision("accounts/gitlab/routers/glm-x", "fireworks_ai")
                is False
            )

        litellm_mock.get_model_info.assert_not_called()

    def test_metadata_without_a_definition_reads_as_undeclared(self, litellm_mock):
        litellm_mock.get_model_info.return_value = {"supports_vision": True}

        with model_context(object()):
            assert _model_supports_vision(MINIMAX, "fireworks_ai") is True

    def test_create_message_dicts_strips_on_a_declared_false_flag(self):
        # litellm has no verdict for this deployment, so only the flag can strip.
        model = "accounts/gitlab/deployments/no-eyes"
        chat = ChatLiteLLM(model=model, custom_llm_provider="fireworks_ai")
        messages = [
            HumanMessage(content="read the diagram"),
            ToolMessage(
                content=[
                    {"type": "text", "text": "Contents of diagram.png:"},
                    {"type": "image", "base64": B64, "mime_type": "image/png"},
                ],
                tool_call_id="call_1",
            ),
        ]

        with model_context(metadata_for(fireworks_definition(False, model=model))):
            message_dicts, _ = chat._create_message_dicts(messages, stop=None)

        # No image survived to be hoisted into a trailing user message.
        assert [m["role"] for m in message_dicts] == ["user", "tool"]
        assert message_dicts[1]["content"] == [
            {"type": "text", "text": "Contents of diagram.png:"},
            {"type": "text", "text": IMAGE_OMITTED_NOTICE.format(model=model)},
        ]


_CHAT_CLASSES = frozenset(
    {
        ModelClassProvider.LITE_LLM,
        ModelClassProvider.ANTHROPIC,
        ModelClassProvider.OPENAI,
        ModelClassProvider.GOOGLE_GENAI,
    }
)

# Every chat model in models.yml is expected to see images unless listed here.
# Computed with the gate's own layers, so a litellm bump that flips a model, or a
# new model litellm does not know, fails by name. Once !7167 lands its flags,
# Minimax and Qwen leave this list.
_NON_VISION_OR_UNKNOWN: dict[str, Optional[bool]] = {
    "minimax_m3_fireworks": False,  # litellm is wrong; live-verified vision
    "glm_5_3_fireworks": False,
    "qwen_3_8_27b_fireworks": None,  # GitLab deployment, registered unannotated
    # Self-hosted family templates: no provider, unknown to litellm.
    **dict.fromkeys(
        [
            "claude_3",
            "codegemma",
            "codellama",
            "codestral",
            "deepseekcoder",
            "gemini",
            "general",
            "gpt",
            "llama3",
            "mistral",
            "mixtral",
            "qwen",
        ],
        None,
    ),
}


def _shipped_chat_definitions() -> dict:
    return {
        identifier: definition
        for identifier, definition in (
            ModelSelectionConfig.instance().get_llm_definitions().items()
        )
        if definition.model_class_provider in _CHAT_CLASSES
    }


class TestShippedCatalogueVisionVerdicts:
    """Every chat model in models.yml has the vision verdict we expect."""

    def test_every_exception_still_ships(self):
        gone = sorted(set(_NON_VISION_OR_UNKNOWN) - set(_shipped_chat_definitions()))

        assert not gone, f"listed but gone from models.yml: {gone}"

    @pytest.mark.parametrize("identifier", sorted(_shipped_chat_definitions()))
    def test_verdict_matches_the_expectation(self, identifier):
        definition = _shipped_chat_definitions()[identifier]
        model = definition.params.model
        provider = getattr(definition.params, "custom_llm_provider", None)
        with model_context(metadata_for(definition)):
            verdict = _model_supports_vision(model, provider) if model else None

        # A declared flag is the expectation by definition, so !7167's values need no edit here.
        expected = definition.supports_vision
        if expected is None:
            expected = _NON_VISION_OR_UNKNOWN.get(identifier, True)
        assert verdict is expected, (
            f"{identifier}: flag+litellm say {verdict!r}, expected {expected!r} "
            f"(model={model!r}, provider={provider!r})"
        )
