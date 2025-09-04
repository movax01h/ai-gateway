import re
from typing import Any, Callable, Dict, List, Union

import bleach

from duo_workflow_service.security.exceptions import SecurityException


def _apply_recursively(response: Any, func: Callable[[str], str]) -> Any:
    """Apply a function recursively to strings in dict/list structures.

    Args:
        response: The response data to process
        func: Function to apply to string values

    Returns:
        Response with function applied to all string values
    """
    if isinstance(response, dict):
        return {k: _apply_recursively(v, func) for k, v in response.items()}
    elif isinstance(response, list):
        return [_apply_recursively(item, func) for item in response]
    elif isinstance(response, str):
        return func(response)
    elif response is None:
        return None
    else:
        # Reject unsupported types for security
        raise SecurityException(
            f"Unsupported type for security processing: {type(response).__name__}. "
            f"All data must be explicitly validated for security."
        )


def strip_hidden_html_comments(
    response: Union[str, Dict[str, Any], List[Any]],
) -> Union[str, List[Union[str, Dict[str, Any]]]]:
    """Strip HTML comments using Bleach library while preserving other content.

    Args:
        response: The response data to process

    Returns:
        Response with HTML comments removed
    """

    def _strip_comments(text: str) -> str:
        if not text or not isinstance(text, str):
            return text

        # Focus on HTML comment removal only

        # Early exit if no HTML comments are present in any format
        has_regular_comments = "<!--" in text
        has_escaped_comments = (
            "\\u003c!--" in text  # JSON unicode escape (lowercase)
            or "\\u003C!--" in text  # JSON unicode escape (uppercase)
            or "\\<!--" in text  # Backslash escape
        )

        if not has_regular_comments and not has_escaped_comments:
            return text

        # Remove JSON-escaped HTML comments
        text = re.sub(r"\\+u003[cC]!--.*?--\\+u003[eE]", "", text, flags=re.DOTALL)

        # Remove backslash-escaped HTML comments
        text = re.sub(r"\\+<!--.*?--\\+>", "", text, flags=re.DOTALL)

        # Configure Bleach with extended allowed tags
        allowed_tags = list(bleach.ALLOWED_TAGS) + [
            "div",
            "span",
            "p",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "br",
            "hr",
            "img",
            "table",
            "tr",
            "td",
            "th",
            "thead",
            "tbody",
        ]

        # Configure allowed attributes
        allowed_attributes = bleach.ALLOWED_ATTRIBUTES
        allowed_attributes.update(
            {
                "*": ["class", "id"],
                "img": ["src", "alt", "width", "height"],
                "table": ["border", "cellpadding", "cellspacing"],
            }
        )
        result = bleach.clean(
            text,
            tags=allowed_tags,
            attributes=allowed_attributes,
            strip_comments=True,
            strip=False,
        )

        # Clean up any remaining escaped comment patterns
        result = re.sub(r"&lt;!--.*?--&gt;", "", result, flags=re.DOTALL)

        return result

    return _apply_recursively(response, _strip_comments)


def strip_mermaid_comments(
    response: Union[str, Dict[str, Any], List[Any]],
) -> Union[str, List[Union[str, Dict[str, Any]]]]:
    """Strip comments from mermaid diagrams to prevent prompt injection.

    Removes mermaid comment lines (starting with %%) from within mermaid code blocks
    while preserving the diagram structure.

    Args:
        response: The response data to process

    Returns:
        Response with mermaid comments removed
    """

    def _strip_mermaid_comments(text: str) -> str:
        if not text or not isinstance(text, str):
            return text

        # Find and process mermaid code blocks
        def process_mermaid_block(match):
            block_content = match.group(0)

            # Remove multi-line directive comments %%{ ... }%%
            processed = re.sub(r"%%\{.*?\}%%", "", block_content, flags=re.DOTALL)

            # Remove line comments starting with %%
            processed = re.sub(
                r"(^|\\+n)\s*%%.*?(?=\\+n|$)", r"\1", processed, flags=re.MULTILINE
            )

            # Clean up excessive blank lines
            processed = re.sub(r"(\\+n\s*){3,}", r"\1\1", processed)
            processed = re.sub(r"(\n\s*){3,}", "\n\n", processed)

            return processed

        # Process regular mermaid blocks (```mermaid ... ```)
        text = re.sub(
            r"```\s*mermaid\b.*?```",
            process_mermaid_block,
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )

        # Process JSON-escaped mermaid blocks
        text = re.sub(
            r"```\\+n\s*mermaid\\b.*?```",
            process_mermaid_block,
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )

        return text

    return _apply_recursively(response, _strip_mermaid_comments)
