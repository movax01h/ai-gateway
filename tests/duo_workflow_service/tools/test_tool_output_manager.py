# pylint: disable=import-outside-toplevel
import copy
import json
from typing import Any
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from duo_workflow_service.tools.duo_base_tool import TruncationConfig
from duo_workflow_service.tools.tool_output_manager import (
    TruncationDirection,
    _add_truncation_instruction,
    truncate_tool_response,
)


def test_add_truncation_instruction():
    notice = _add_truncation_instruction(
        truncated_text="random ted unique",
        original_token_size=77,
        truncated_token_size=54,
    )
    assert "70.1%" in notice
    assert notice.endswith("\n</instructions>\n</truncation_notice>\n")
    assert "random ted unique" in notice


def test_truncate_string_reverse():
    """Test reverse truncation keeps the end of the content."""
    from duo_workflow_service.tools.tool_output_manager import (
        TruncationDirection,
        truncate_string,
    )

    # Create content where we can identify beginning vs end
    content = "START" + ("x" * 200 * 1024) + "END"

    reverse_config = TruncationConfig(
        max_bytes=200 * 1024,
        truncated_size=100 * 1024,
        direction=TruncationDirection.FROM_END,
    )

    result = truncate_string(content, "test_tool", reverse_config)

    # Should contain END but not START
    assert "END" in result
    assert "START" not in result
    assert "<truncation_notice>" in result


def test_truncate_string_default_direction():
    """Test that default truncation direction is FROM_START."""
    from duo_workflow_service.tools.tool_output_manager import (
        TruncationDirection,
        truncate_string,
    )

    # Create content where we can identify beginning vs end
    content = "START" + ("x" * 200 * 1024) + "END"

    default_config = TruncationConfig(
        max_bytes=200 * 1024,
        truncated_size=100 * 1024,
    )

    # Verify default is FROM_START
    assert default_config.direction == TruncationDirection.FROM_START

    result = truncate_string(content, "test_tool", default_config)

    # Should contain START but not END (same as FROM_START behavior)
    assert "START" in result
    assert "END" not in result
    assert "<truncation_notice>" in result


def test_truncate_string_forward():
    """Test forward truncation keeps the beginning of the content."""
    from duo_workflow_service.tools.tool_output_manager import (
        TruncationDirection,
        truncate_string,
    )

    # Create content where we can identify beginning vs end
    content = "START" + ("x" * 200 * 1024) + "END"

    forward_config = TruncationConfig(
        max_bytes=200 * 1024,
        truncated_size=100 * 1024,
        direction=TruncationDirection.FROM_START,
    )

    result = truncate_string(content, "test_tool", forward_config)

    # Should contain START but not END
    assert "START" in result
    assert "END" not in result
    assert "<truncation_notice>" in result


def test_truncate_string_reverse_no_truncation_needed():
    """Test reverse truncation when content is under limit."""
    from duo_workflow_service.tools.tool_output_manager import (
        TruncationDirection,
        truncate_string,
    )

    content = "Small content"

    reverse_config = TruncationConfig(
        max_bytes=200 * 1024,
        truncated_size=100 * 1024,
        direction=TruncationDirection.FROM_END,
    )

    result = truncate_string(content, "test_tool", reverse_config)

    # Should return unchanged
    assert result == content
    assert "<truncation_notice>" not in result


def test_truncate_tool_response_with_custom_config():
    """Test truncation with custom config (1MB/800KB)."""
    custom_config = TruncationConfig(
        max_bytes=1 * 1024 * 1024,
        truncated_size=800 * 1024,  # 1 MiB  # 800 KiB
    )

    # Response that would be truncated with default config but not with custom
    medium_response = "x" * (200 * 1024)  # 200KB
    result = truncate_tool_response(
        medium_response, "build_review_merge_request_context", custom_config
    )
    assert result == medium_response  # Should NOT be truncated

    # Response that exceeds even the custom limit
    huge_response = "x" * (1 * 1024 * 1024 + 1000)  # Exceeds 1MB
    result = truncate_tool_response(
        huge_response, "build_review_merge_request_context", custom_config
    )
    assert len(result) < len(huge_response)
    assert "<truncation_notice>" in result


@pytest.mark.parametrize(
    ("response", "should_truncated"),
    [
        ("A response under limit", False),
        (None, False),
        (1.0, False),
        ("", False),
        (
            ToolMessage(
                content="A response under limit",
                tool_call_id="call_id",
            ),
            False,
        ),
        ("This is a response that exceed the byte limit", True),
        (
            ToolMessage(
                content="This is a response that exceed the byte limit",
                tool_call_id="call_id",
            ),
            True,
        ),
        (
            ToolMessage(
                content=[{"data": "B" * 30}, {"more_data": list(range(20))}],
                tool_call_id="call_id",
            ),
            True,
        ),
        (
            {"key": "This is a response that exceed the byte limit", "data": [1, 2, 3]},
            True,
        ),
        ({"data": "B" * 30, "more_data": list(range(20))}, True),
    ],
)
@patch("duo_workflow_service.tools.tool_output_manager.token_counter")
@patch("duo_workflow_service.tools.tool_output_manager.logger")
def test_truncate_tool_response(
    mock_logger: Mock,
    mock_token_counter: Mock,
    response: Any,
    should_truncated: bool,
):
    test_config = TruncationConfig(max_bytes=30, truncated_size=10)

    if isinstance(response, ToolMessage):
        expected_json_str = (
            json.dumps(response.content)
            if not isinstance(response.content, str)
            else response.content
        )
    else:
        expected_json_str = (
            json.dumps(response) if not isinstance(response, str) else response
        )

    mock_token_counter.count_string_content.return_value = 1

    result = truncate_tool_response(
        response, tool_name="test_tool", truncation_config=test_config
    )

    if should_truncated:
        mock_logger.info.assert_called_once_with(
            "Tool response exceeds max size and will be truncated",
            tool_name="test_tool",
            original_token_size=1,
            truncated_token_size=1,
            max_bytes=30,
            truncated_size=10,
            direction="from_start",
        )
        result = result.content if isinstance(result, ToolMessage) else result
        assert expected_json_str[:10] in result
        if isinstance(result, str):
            assert result.startswith("\n<truncation_notice>")
            assert result.endswith("</instructions>\n</truncation_notice>\n")
    else:
        assert result == response


@patch("duo_workflow_service.tools.tool_output_manager.logger")
def test_truncate_tool_response_exception(
    mock_logger: Mock,
):
    default_config = TruncationConfig()

    tool_response: Command = Command(
        update={
            "tool_response": ToolMessage(
                content="",
                tool_call_id="call_id",
            )
        }
    )
    result = truncate_tool_response(
        tool_response, tool_name="test_tool", truncation_config=default_config
    )
    mock_logger.info.assert_called_once_with(
        "Skip truncation for Command tool response"
    )
    assert result == tool_response


class TestImageBlockTruncation:
    """Content-block lists carrying images: image blocks are exempt as a
    class, and the text parts share the one ``max_bytes`` budget that governs
    every other response shape."""

    IMAGE_BLOCK = {
        "type": "image",
        "base64": "QUFBQQ==" * 100,
        "mime_type": "image/png",
    }
    TINY_CONFIG = TruncationConfig(max_bytes=30, truncated_size=10)

    @staticmethod
    def _kept_text(blocks: list) -> str:
        """The text surviving in *blocks*, excluding the trailing notice."""
        return "".join(
            block if isinstance(block, str) else block.get("text", "")
            for block in blocks
            if isinstance(block, str) or isinstance(block.get("text"), str)
        ).replace(TestImageBlockTruncation._notice(blocks), "")

    @staticmethod
    def _notice(blocks: list) -> str:
        for block in blocks:
            text = block if isinstance(block, str) else block.get("text", "")
            if isinstance(text, str) and "<truncation_notice>" in text:
                return text
        return ""

    @staticmethod
    def _notice_count(blocks: list) -> int:
        return sum(
            "<truncation_notice>"
            in (block if isinstance(block, str) else block.get("text") or "")
            for block in blocks
        )

    @staticmethod
    def _assert_no_empty_blocks(blocks: list) -> None:
        """No empty string and no text block with empty text: providers reject those."""
        for block in blocks:
            if isinstance(block, str):
                assert block
            elif block.get("type") == "text":
                assert block["text"]

    def test_image_list_under_limit_returned_by_identity(self):
        response = [{"type": "text", "text": "short"}, self.IMAGE_BLOCK]
        result = truncate_tool_response(
            response, tool_name="read_file", truncation_config=TruncationConfig()
        )

        assert result is response

    def test_image_block_survives_over_limit(self):
        response = [self.IMAGE_BLOCK]
        result = truncate_tool_response(
            response, tool_name="read_file", truncation_config=self.TINY_CONFIG
        )

        assert result[0] is self.IMAGE_BLOCK

    def test_mixed_content_truncates_text_only(self):
        big_text = {"type": "text", "text": "A" * 100}
        result = truncate_tool_response(
            [big_text, self.IMAGE_BLOCK],
            tool_name="read_file",
            truncation_config=self.TINY_CONFIG,
        )

        assert result[0]["type"] == "text"
        assert result[0]["text"] == "A" * 10
        assert result[1] is self.IMAGE_BLOCK
        # The notice is its own trailing block, so the kept text stays in the
        # block it came from and keeps its place relative to the image.
        assert "<truncation_notice>" in result[-1]["text"]

    def test_bare_string_alongside_image_truncated(self):
        result = truncate_tool_response(
            ["B" * 100, self.IMAGE_BLOCK],
            tool_name="read_file",
            truncation_config=self.TINY_CONFIG,
        )

        assert result[0] == "B" * 10
        assert result[1] is self.IMAGE_BLOCK
        assert "<truncation_notice>" in result[-1]["text"]

    def test_unknown_block_within_the_budget_passes_through(self):
        """A non-text block that fits is kept whole, by identity, while a later block is cut.

        The list has to exceed ``max_bytes`` for the spend loop to run at all, or this pins nothing.
        """
        other = {"type": "audio"}  # 17 bytes of JSON
        result = truncate_tool_response(
            [other, {"type": "text", "text": "A" * 100}, self.IMAGE_BLOCK],
            tool_name="read_file",
            truncation_config=TruncationConfig(max_bytes=30, truncated_size=20),
        )

        assert result[0] is other
        assert result[1]["text"] == "A" * 3
        assert result[2] is self.IMAGE_BLOCK
        assert self._notice_count(result) == 1

    def test_oversized_non_text_block_is_cut_to_its_json(self):
        """A non-text block costs its JSON and is cut like text.

        Without this, a list with an image and a 2 MB web_search_result kept the 2 MB and appended a notice claiming the
        output was cut to 100 KiB.
        """
        big = {"type": "web_search_result", "content": "X" * 100}
        result = truncate_tool_response(
            [self.IMAGE_BLOCK, big],
            tool_name="read_file",
            truncation_config=self.TINY_CONFIG,
        )

        assert result[0] is self.IMAGE_BLOCK
        assert result[1] == {"type": "text", "text": json.dumps(big)[:10]}
        assert len(self._kept_text(result).encode("utf-8")) == 10
        assert self._notice_count(result) == 1

    def test_non_text_block_shares_the_budget_with_text(self):
        # Text first, then a non-text block: the text takes the budget and the
        # non-text block, left with nothing, is dropped rather than riding
        # along for free.
        text = {"type": "text", "text": "A" * 8}
        big = {"type": "web_search_result", "content": "X" * 100}
        result = truncate_tool_response(
            [text, big, self.IMAGE_BLOCK],
            tool_name="read_file",
            truncation_config=TruncationConfig(max_bytes=10, truncated_size=8),
        )

        assert result[0] == text
        assert result[1] is self.IMAGE_BLOCK
        assert len(result) == 3
        assert len(self._kept_text(result).encode("utf-8")) == 8
        self._assert_no_empty_blocks(result)

    def test_list_without_images_keeps_legacy_behavior(self):
        response = [{"type": "text", "text": "D" * 100}]
        result = truncate_tool_response(
            response, tool_name="read_file", truncation_config=self.TINY_CONFIG
        )

        assert isinstance(result, str)
        assert "<truncation_notice>" in result

    def test_tool_message_with_image_content_truncates_text_only(self):
        message = ToolMessage(
            content=[{"type": "text", "text": "E" * 100}, self.IMAGE_BLOCK],
            tool_call_id="call_id",
        )
        result = truncate_tool_response(
            message, tool_name="read_file", truncation_config=self.TINY_CONFIG
        )

        assert isinstance(result, ToolMessage)
        assert result is not message
        assert result.content[0]["text"] == "E" * 10
        # ToolMessage construction copies content dicts, so identity is
        # relative to the message's own content, not the class constant
        assert result.content[1] is message.content[1]
        assert result.content[1] == self.IMAGE_BLOCK
        assert "<truncation_notice>" in result.content[-1]["text"]

    def test_tool_message_with_image_content_under_limit_unchanged(self):
        message = ToolMessage(
            content=[{"type": "text", "text": "short"}, self.IMAGE_BLOCK],
            tool_call_id="call_id",
        )
        result = truncate_tool_response(
            message, tool_name="read_file", truncation_config=TruncationConfig()
        )

        assert result is message

    # ---------------------------------------------------------------- budget

    def test_many_text_blocks_share_one_budget(self):
        """The whole-response budget, not a per-block one.

        With a budget per block, twenty 150 KiB blocks kept ~2.9 MiB and no block tripped the limit on its own, so no
        notice fired and the model was never told its output had been cut.
        """
        blocks = [{"type": "text", "text": "A" * (150 * 1024)} for _ in range(20)]
        config = TruncationConfig()

        result = truncate_tool_response(
            blocks + [self.IMAGE_BLOCK],
            tool_name="read_file",
            truncation_config=config,
        )

        kept = self._kept_text(result)
        assert len(kept.encode("utf-8")) == config.truncated_size
        assert self._notice_count(result) == 1
        assert self.IMAGE_BLOCK in result
        # Blocks 2..20 had no budget left; they are gone, not left empty.
        assert len(result) == 3
        self._assert_no_empty_blocks(result)

    def test_image_does_not_loosen_the_budget(self):
        """The same text keeps the same number of bytes with and without an image in the list.

        The image path and the legacy string path have to agree on what ``max_bytes`` means, which is
        what regressed.
        """
        blocks = [{"type": "text", "text": "A" * (150 * 1024)} for _ in range(20)]
        config = TruncationConfig()

        with_image = truncate_tool_response(
            blocks + [self.IMAGE_BLOCK], tool_name="read_file", truncation_config=config
        )
        without_image = truncate_tool_response(
            list(blocks), tool_name="read_file", truncation_config=config
        )

        kept_with = len(self._kept_text(with_image).encode("utf-8"))
        assert kept_with == config.truncated_size
        # No image: the legacy path json.dumps the list and cuts the string to
        # the same budget, so the two agree on the byte count they keep.
        assert isinstance(without_image, str)
        assert config.truncated_size <= len(without_image.encode("utf-8"))
        assert kept_with == config.truncated_size

    def test_budget_is_spent_in_order(self):
        """Earlier blocks are kept whole; the budget runs out on a later one.

        The block that gets nothing is dropped, so the image moves up one slot.
        """
        blocks = [
            {"type": "text", "text": "A" * 6},
            {"type": "text", "text": "B" * 6},
            {"type": "text", "text": "C" * 6},
            self.IMAGE_BLOCK,
        ]
        result = truncate_tool_response(
            blocks,
            tool_name="read_file",
            truncation_config=TruncationConfig(max_bytes=10, truncated_size=8),
        )

        assert result[0]["text"] == "A" * 6
        assert result[1]["text"] == "BB"
        assert result[2] is self.IMAGE_BLOCK
        assert len(result) == 4
        self._assert_no_empty_blocks(result)

    def test_url_form_image_block_is_exempt_like_an_inline_one(self):
        """A block with a remote ``url`` and no payload costs nothing to keep.

        Charging it its JSON let it eat the whole budget and come back as a text block holding a JSON fragment, so the
        reference was destroyed and the real output dropped.
        """
        url_image = {"type": "image", "url": "https://example.com/a.png"}

        result = truncate_tool_response(
            [url_image, {"type": "text", "text": "A" * 100}, self.IMAGE_BLOCK],
            tool_name="read_file",
            truncation_config=self.TINY_CONFIG,
        )

        assert result[0] is url_image
        assert result[1]["text"] == "A" * 10
        assert result[2] is self.IMAGE_BLOCK

    def test_url_only_list_over_budget_stays_a_list(self):
        """The gate and the exemption share one definition of an image block.

        A list whose only image is a remote ``url`` has nothing inline, so it used to fall through to ``json.dumps`` and
        the string slice, which can cut the url in half. Now it is routed like any image-bearing list: the text is cut,
        the url block is kept by identity, and the url is intact.
        """
        url_image = {"type": "image", "url": "https://example.com/img.png"}

        result = truncate_tool_response(
            [{"type": "text", "text": "D" * 100}, url_image],
            tool_name="read_file",
            truncation_config=self.TINY_CONFIG,
        )

        assert isinstance(result, list)
        assert result[0]["text"] == "D" * 10
        assert result[1] is url_image
        assert url_image["url"] in json.dumps(result)
        assert self._notice_count(result) == 1

    def test_a_lone_url_image_block_is_returned_by_identity(self):
        response = {"type": "image", "url": "https://example.com/" + "x" * 300}

        result = truncate_tool_response(
            response, tool_name="read_file", truncation_config=self.TINY_CONFIG
        )

        assert result is response

    def test_an_image_lookalike_with_an_unknown_payload_key_pays_its_json(self):
        """Only the two standard shapes are exempt: inline ``base64`` and a remote ``url``.

        A block that says ``type: image`` but carries its payload under another key is not something this module can
        recognise as an image, so it is budgeted by its JSON like any other block. Otherwise a payload under an unknown
        key would ride past the budget for free, as long as a real image sat next to it.
        """
        lookalike = {"type": "image", "source": {"type": "base64", "data": "x" * 200}}

        result = truncate_tool_response(
            [self.IMAGE_BLOCK, lookalike, {"type": "text", "text": "A" * 100}],
            tool_name="read_file",
            truncation_config=self.TINY_CONFIG,
        )

        assert result[0] is self.IMAGE_BLOCK
        assert not any(
            isinstance(block, dict) and "source" in block for block in result
        )
        assert result[1] == {"type": "text", "text": json.dumps(lookalike)[:10]}
        assert len(self._kept_text(result).encode()) <= self.TINY_CONFIG.truncated_size
        assert self._notice_count(result) == 1

    def test_a_lone_image_block_is_never_sliced(self):
        # Outside a list the image used to reach json.dumps and get cut
        # mid-payload, which is the defect this path exists to remove.
        response = {
            "type": "image",
            "base64": "QUFBQQ==" * 40000,
            "mime_type": "image/png",
        }

        result = truncate_tool_response(
            response, tool_name="read_file", truncation_config=TruncationConfig()
        )

        assert result is response

    def test_a_block_that_arrived_empty_is_dropped_too(self):
        # The loop skips zero-size blocks, so an empty text block in the input
        # used to survive into the output that promises to have none.
        result = truncate_tool_response(
            [
                {"type": "text", "text": ""},
                {"type": "text", "text": "A" * 100},
                self.IMAGE_BLOCK,
            ],
            tool_name="read_file",
            truncation_config=self.TINY_CONFIG,
        )

        assert len(result) == 3
        assert result[0]["text"] == "A" * 10
        assert result[1] is self.IMAGE_BLOCK
        self._assert_no_empty_blocks(result)

    def test_from_end_drops_the_blocks_the_budget_never_reached(self):
        blocks = [
            {"type": "text", "text": "A" * 6},
            {"type": "text", "text": "B" * 6},
            {"type": "text", "text": "C" * 6},
            self.IMAGE_BLOCK,
        ]

        result = truncate_tool_response(
            blocks,
            tool_name="read_file",
            truncation_config=TruncationConfig(
                max_bytes=10, truncated_size=8, direction=TruncationDirection.FROM_END
            ),
        )

        # Budget spent from the tail: C survives whole, B keeps its last bytes,
        # A is gone, and the image keeps its place after the survivors.
        assert [b.get("text") for b in result[:2]] == ["BB", "C" * 6]
        assert result[2] is self.IMAGE_BLOCK
        assert self._kept_text(result) == "BB" + "C" * 6
        self._assert_no_empty_blocks(result)

    def test_total_exactly_at_max_bytes_is_returned_untouched(self):
        response = [{"type": "text", "text": "A" * 30}, self.IMAGE_BLOCK]

        result = truncate_tool_response(
            response,
            tool_name="read_file",
            truncation_config=TruncationConfig(max_bytes=30, truncated_size=10),
        )

        assert result is response

    def test_the_input_is_never_mutated(self):
        blocks = [{"type": "text", "text": "A" * 100}, self.IMAGE_BLOCK]
        before = copy.deepcopy(blocks)

        truncate_tool_response(
            blocks, tool_name="read_file", truncation_config=self.TINY_CONFIG
        )

        assert blocks == before

    def test_zero_budget_keeps_only_the_image_and_the_notice(self):
        result = truncate_tool_response(
            [{"type": "text", "text": "A" * 6}, "bare", self.IMAGE_BLOCK],
            tool_name="read_file",
            truncation_config=TruncationConfig(max_bytes=5, truncated_size=0),
        )

        assert result[0] is self.IMAGE_BLOCK
        assert self._notice_count(result) == 1
        assert len(result) == 2

    @pytest.mark.parametrize(
        ("truncated_size", "expected_text"),
        [
            (3, "\u00e9"),  # 3 bytes: one 2-byte char plus a stray byte
            (1, None),  # 1 byte: inside the first char, nothing decodes
        ],
    )
    def test_cut_inside_a_multibyte_character(self, truncated_size, expected_text):
        blocks = [{"type": "text", "text": "\u00e9" * 10}, self.IMAGE_BLOCK]
        result = truncate_tool_response(
            blocks,
            tool_name="read_file",
            truncation_config=TruncationConfig(
                max_bytes=10, truncated_size=truncated_size
            ),
        )

        if expected_text is None:
            assert result[0] is self.IMAGE_BLOCK
        else:
            assert result[0]["text"] == expected_text
            assert result[1] is self.IMAGE_BLOCK
        self._assert_no_empty_blocks(result)

    def test_from_end_keeps_the_tail(self):
        blocks = [
            {"type": "text", "text": "A" * 6},
            {"type": "text", "text": "B" * 6},
            self.IMAGE_BLOCK,
        ]
        result = truncate_tool_response(
            blocks,
            tool_name="read_file",
            truncation_config=TruncationConfig(
                max_bytes=10,
                truncated_size=8,
                direction=TruncationDirection.FROM_END,
            ),
        )

        # Budget spent from the tail: the last block survives whole and the
        # earlier one keeps only its final bytes, which is the text
        # `truncate_string` would have kept from the joined response.
        assert result[1]["text"] == "B" * 6
        assert result[0]["text"] == "AA"
        assert len(self._kept_text(result).encode("utf-8")) == 8
        assert result[2] is self.IMAGE_BLOCK

    def test_no_notice_when_nothing_was_cut(self):
        """A config with truncated_size above max_bytes cuts nothing.

        The notice must not claim otherwise, so the list comes back by identity.
        """
        response = [{"type": "text", "text": "A" * 100}, self.IMAGE_BLOCK]
        result = truncate_tool_response(
            response,
            tool_name="read_file",
            truncation_config=TruncationConfig(max_bytes=50, truncated_size=500),
        )

        assert result is response

    def test_notice_does_not_repeat_the_kept_output(self):
        """The notice points at the blocks instead of inlining them.

        Repeating the kept text inside the notice would double the payload the truncation exists to bound.
        """
        blocks = [{"type": "text", "text": "A" * 100}, self.IMAGE_BLOCK]
        result = truncate_tool_response(
            blocks, tool_name="read_file", truncation_config=self.TINY_CONFIG
        )

        notice = result[-1]["text"]
        assert "content blocks above" in notice
        assert "AAAA" not in notice
        assert self.IMAGE_BLOCK["base64"] not in notice
