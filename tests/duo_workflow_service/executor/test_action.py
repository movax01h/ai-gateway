"""Tests for `_read_file_fully`, the pagination-following executor read.

A paginated `runReadFile` returns one page plus a footer and raises nothing, so
a single read silently yields a prefix of the file. `_PaginatingExecutor`
truncates reads the same way; a fake that returns whole files cannot catch it.
"""

import pytest
from langchain_core.tools import ToolException
from structlog.testing import capture_logs

from contract import contract_pb2
from duo_workflow_service.executor.action import _read_file_fully, _split_read_page


class _PaginatingExecutor:
    """An `_execute_action` fake that truncates a read at `max_bytes` and says so."""

    def __init__(self, files: dict, max_bytes: int = 51_200):
        self.files = files
        self.max_bytes = max_bytes
        self.reads: list = []

    async def __call__(self, metadata, action):
        if not action.HasField("runReadFile"):
            return "ok"
        req = action.runReadFile
        lines = self.files[req.filepath].split("\n")
        start = req.offset if req.HasField("offset") else 0
        self.reads.append(start)
        if start >= len(lines):
            return f"[Offset {start} is beyond end of file ({len(lines)} lines).]"
        page: list = []
        size = 0
        for i in range(start, len(lines)):
            size += len(lines[i]) + 1
            if page and size > self.max_bytes:
                break
            page.append(lines[i])
        end = start + len(page)
        body = "\n".join(page)
        if end < len(lines):
            body += (
                f"\n\n[Showing lines {start}-{end - 1} of {len(lines)} total. "
                f"Use offset={end} to continue reading.]"
            )
        return body


def _spill(n_lines: int = 85, line_bytes: int = 1980) -> str:
    """A JSONL file large enough to span several 50 KiB pages."""
    return "\n".join(
        '{"id": %d, "pad": "%s"}' % (i, "x" * line_bytes) for i in range(n_lines)
    )


class TestSplitReadPage:
    def test_no_footer_means_complete(self):
        content, nxt, footer = _split_read_page('{"a": 1}\n{"b": 2}')
        assert content == '{"a": 1}\n{"b": 2}'
        assert nxt is None and footer is None

    def test_footer_with_offset_is_a_continuation(self):
        page = '{"a": 1}\n\n[Showing lines 0-0 of 9 total. Use offset=1 to continue reading.]'
        content, nxt, footer = _split_read_page(page)
        assert content == '{"a": 1}'
        assert nxt == 1
        assert footer.startswith("[Showing")

    def test_footer_without_offset_means_the_page_ran_to_eof(self):
        content, nxt, footer = _split_read_page(
            '{"a": 1}\n\n[Showing lines 5-5 of 6 total.]'
        )
        assert content == '{"a": 1}'
        assert nxt is None
        assert footer is not None

    def test_paren_style_footer_is_also_recognised(self):
        page = '{"a": 1}\n\n(Showing lines 0-0 of 9 total. Use offset=1 to continue reading.)'
        content, nxt, _ = _split_read_page(page)
        assert content == '{"a": 1}'
        assert nxt == 1

    def test_past_eof_marker_yields_no_content(self):
        content, nxt, _ = _split_read_page(
            "[Offset 99 is beyond end of file (9 lines).]"
        )
        assert content == ""
        assert nxt is None

    @pytest.mark.parametrize(
        "last", ["(see the note above)", "[Showing all rows]", "(Showing 3 of 9)"]
    )
    def test_only_the_exact_footer_shape_is_a_footer(self, last):
        page = f"a\n\n{last}"
        assert _split_read_page(page) == (page, None, None)

    def test_a_real_blank_line_at_the_page_boundary_survives(self):
        page = (
            "a\n\n\n[Showing lines 0-1 of 5 total. Use offset=2 to continue reading.]"
        )
        assert _split_read_page(page)[0] == "a\n"

    def test_a_json_line_is_not_mistaken_for_a_footer(self):
        content, nxt, footer = _split_read_page('[{"cwe": "CWE-862"}]')
        assert content == '[{"cwe": "CWE-862"}]'
        assert nxt is None and footer is None


class TestReadFileFully:
    @pytest.mark.asyncio
    async def test_the_fake_reproduces_the_production_shape(self):
        """Control: without pagination a single read loses most of the file.

        If this stops holding, every test below is vacuous.
        """
        body = _spill()
        ex = _PaginatingExecutor({"spill.json": body})
        page = await ex(
            {},
            contract_pb2.Action(
                runReadFile=contract_pb2.ReadFile(filepath="spill.json")
            ),
        )
        assert "Showing lines" in page
        assert len(page) < len(body)
        assert page.count("\n") < 35

    @pytest.mark.asyncio
    async def test_reassembles_every_line_across_pages(self):
        body = _spill()
        ex = _PaginatingExecutor({"spill.json": body})

        out = await _read_file_fully({}, "spill.json", execute=ex)

        assert out == body
        assert len(out.splitlines()) == 85
        assert "Showing lines" not in out
        assert len(ex.reads) > 1, "expected the read to paginate"

    @pytest.mark.asyncio
    async def test_the_footer_wording_is_logged(self):
        ex = _PaginatingExecutor({"spill.json": _spill()})
        with capture_logs() as logs:
            await _read_file_fully({}, "spill.json", execute=ex)
        footer_logs = [e for e in logs if "pagination footer" in e["event"]]
        assert footer_logs, "the footer wording was never captured"
        assert "Showing lines" in footer_logs[0]["footer"]

    @pytest.mark.asyncio
    async def test_small_file_reads_in_one_page_and_logs_nothing(self):
        ex = _PaginatingExecutor({"spill.json": '{"a": 1}\n{"b": 2}'})
        with capture_logs() as logs:
            out = await _read_file_fully({}, "spill.json", execute=ex)
        assert out == '{"a": 1}\n{"b": 2}'
        assert ex.reads == [0]
        assert not [e for e in logs if "pagination footer" in e["event"]]

    @pytest.mark.asyncio
    async def test_a_truncating_server_that_offers_no_resume_offset_raises(self):
        """Worst case: it truncates and does not say how to continue.

        The read is then genuinely partial, so it must fail rather than be
        returned silently short.
        """

        async def _no_offset(metadata, action):
            return '{"a": 1}\n\n[Showing lines 0-0 of 99 total.]'

        with pytest.raises(ToolException, match="no resume offset"):
            await _read_file_fully({}, "spill.json", execute=_no_offset)

    @pytest.mark.asyncio
    async def test_a_past_eof_marker_on_the_first_page_is_an_empty_file(self):
        """An executor may answer a read of an empty file with the past-EOF marker."""

        async def _empty(metadata, action):
            return "[Offset 0 is beyond end of file (0 lines).]"

        assert await _read_file_fully({}, "empty.json", execute=_empty) == ""

    @pytest.mark.asyncio
    async def test_page_limit_raises_rather_than_return_a_prefix(self):
        ex = _PaginatingExecutor({"spill.json": _spill(n_lines=500)})
        with pytest.raises(ToolException, match="2-page limit"):
            await _read_file_fully({}, "spill.json", execute=ex, max_pages=2)
        assert len(ex.reads) == 2

    @pytest.mark.asyncio
    async def test_a_stalled_pagination_raises_without_repeating_the_page(self):
        calls = []

        async def _stuck(metadata, action):
            calls.append(action)
            return "line\n\n[Showing lines 0-0 of 9 total. Use offset=1 to continue reading.]"

        # The stalled page is not appended, so only the first page's line counts.
        with pytest.raises(ToolException, match=r"stalled at offset 1 \(1 lines read"):
            await _read_file_fully({}, "spill.json", execute=_stuck)
        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_an_empty_page_that_advances_the_offset_raises(self):
        """The Node executor skips a line over 50 KiB with an empty page.

        Its footer still advances the offset by one, so following it would drop that line in silence.
        """
        pages = {
            0: "a\n\n(Showing lines 1-1 of 3 total. Use offset=1 to continue reading.)",
            1: "\n\n(Showing lines 2-2 of 3 total. Use offset=2 to continue reading.)",
            2: "c",
        }

        async def _skips_long_line(metadata, action):
            req = action.runReadFile
            return pages[req.offset if req.HasField("offset") else 0]

        with pytest.raises(ToolException, match=r"empty page at offset 1 \(1 lines"):
            await _read_file_fully({}, "big.js", execute=_skips_long_line)

    @pytest.mark.asyncio
    async def test_real_blank_lines_at_page_boundaries_are_kept(self):
        body = "a\n\nb\n\n\nc"
        ex = _PaginatingExecutor({"spill.json": body}, max_bytes=3)

        assert await _read_file_fully({}, "spill.json", execute=ex) == body
        assert len(ex.reads) > 2


@pytest.mark.parametrize("max_bytes", [4_096, 51_200, 200_000])
@pytest.mark.asyncio
async def test_result_is_independent_of_where_the_cap_falls(max_bytes):
    """The caller gets the same bytes wherever the cap falls."""
    body = _spill(n_lines=120)
    ex = _PaginatingExecutor({"spill.json": body}, max_bytes=max_bytes)

    assert await _read_file_fully({}, "spill.json", execute=ex) == body


class TestDegenerateResponses:
    def test_an_all_blank_page_is_returned_as_is(self):
        # No non-blank line at all -> there is no footer to find.
        content, nxt, footer = _split_read_page("\n\n   \n")
        assert content == "\n\n   \n"
        assert nxt is None and footer is None

    def test_an_empty_page_is_returned_as_is(self):
        content, nxt, footer = _split_read_page("")
        assert content == ""
        assert nxt is None and footer is None

    @pytest.mark.asyncio
    async def test_a_non_string_response_raises(self):
        """A malformed response must end the read, not be concatenated."""

        async def _not_a_string(metadata, action):
            return None

        with pytest.raises(ToolException, match="non-string NoneType"):
            await _read_file_fully({}, "spill.json", execute=_not_a_string)
