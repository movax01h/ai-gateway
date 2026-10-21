from typing import Any


def content_to_text(content: Any) -> str:
    """Flatten content to a plain string, as some providers (e.g. `google_genai`) return a list of blocks instead.

    See `react.py::_chunk_text_and_metadata` for the equivalent handling on the agent path.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)
