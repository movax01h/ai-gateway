# flake8: noqa: W605
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

from duo_workflow_service.security.exceptions import SecurityException
from duo_workflow_service.security.markdown_content_security import (
    strip_hidden_html_comments,
    strip_mermaid_comments,
)


def run_from_args():
    args = sys.argv[1:]
    filename = args[0]
    content = Path(filename).read_text()

    return PromptSecurity.apply_security_to_tool_response(content, "test-tool")


def encode_dangerous_tags(
    response: Union[str, Dict[str, Any], List[Any]],
) -> Union[str, List[Union[str, Dict[str, Any]]]]:
    """Recursively encode dangerous HTML tags in the response.

    Args:
        response: The response data to encode

    Returns:
        Response with encoded dangerous tags, compatible with ToolMessage.content
    """

    def _encode_recursive(data: Any) -> Any:
        """Internal recursive function that doesn't change top-level structure."""
        DANGEROUS_TAGS = {
            "goal": "goal",
            "system": "system",
        }

        if isinstance(data, dict):
            return {k: _encode_recursive(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [_encode_recursive(item) for item in data]

        for tag_name, replacement in DANGEROUS_TAGS.items():
            data = re.sub(
                rf"<\s*(/?)\s*{re.escape(tag_name)}\s*>",
                f"&lt;\\1{replacement}&gt;",
                data,
                flags=re.IGNORECASE,
            )

            data = re.sub(
                rf"\\u003c\s*(/?)\s*{re.escape(tag_name)}\s*\\u003e",
                f"&lt;\\1{replacement}&gt;",
                data,
                flags=re.IGNORECASE,
            )

            data = re.sub(
                rf"\\\\u003c\s*(/?)\s*{re.escape(tag_name)}\s*\\\\u003e",
                f"&lt;\\1{replacement}&gt;",
                data,
                flags=re.IGNORECASE,
            )

        return data

    processed = _encode_recursive(response)

    if isinstance(processed, dict):
        return [processed]

    # Type assertion: processed is guaranteed to be str or list after security processing
    return processed  # type: ignore[no-any-return]


def strip_hidden_unicode_tags(
    response: Union[str, Dict[str, Any], List[Any]],
) -> Union[str, List[Union[str, Dict[str, Any]]]]:
    """Remove hidden Unicode tag characters that can be used for steganographic attacks.

    Strips Unicode Tag Characters (U+E0000-E007F) and Language Tag characters
    (U+E0100-E01EF) that are invisible but can carry hidden malicious content.
    These characters are often used in steganographic attacks to hide instructions
    within seemingly innocent text.

    Args:
        response: The response data to process

    Returns:
        Response with hidden Unicode tag characters removed
    """
    from duo_workflow_service.security.markdown_content_security import (
        _apply_recursively,
    )

    def _strip_unicode_tags(text: str) -> str:
        if not text or not isinstance(text, str):
            return text

        # First handle JSON-escaped Unicode tag characters
        # Unicode Tag Characters (U+E0000-E007F) get encoded as UTF-16 surrogates:
        # U+E0000-E007F -> surrogate pairs starting with \udb40
        # U+E0100-E01EF -> surrogate pairs starting with \udb40
        import re

        # Remove JSON-escaped Unicode tag characters (UTF-16 surrogate pairs)
        # These appear as \\udb40\\udc?? in JSON output
        text = re.sub(
            r"\\udb40\\ud[c-f][0-9a-f][0-9a-f]", "", text, flags=re.IGNORECASE
        )

        # Also remove direct Unicode Tag Characters if they exist
        # These ranges contain invisible characters that can be used for steganographic attacks
        return "".join(
            char
            for char in text
            if not (0xE0000 <= ord(char) <= 0xE007F or 0xE0100 <= ord(char) <= 0xE01EF)
        )

    return _apply_recursively(response, _strip_unicode_tags)


def apply_security_unicode_only(
    response: Union[str, Dict[str, Any], List[Any]],
) -> Union[str, List[Union[str, Dict[str, Any]]]]:
    """Dedicated function to test Unicode tag stripping only.

    Args:
        response: The response data to process

    Returns:
        Response with only Unicode tag stripping applied
    """
    return strip_hidden_unicode_tags(response)


class PromptSecurity:
    """Security class with configurable security functions."""

    # Default security functions to apply to ALL tools
    DEFAULT_SECURITY_FUNCTIONS: List[
        Callable[
            [Union[str, Dict[str, Any], List[Any]]],
            Union[str, List[Union[str, Dict[str, Any]]]],
        ]
    ] = [
        encode_dangerous_tags,
        strip_hidden_html_comments,
        strip_hidden_unicode_tags,
        strip_mermaid_comments,
    ]

    # Tool-specific additional security functions
    TOOL_SPECIFIC_FUNCTIONS: Dict[
        str,
        List[
            Callable[
                [Union[str, Dict[str, Any], List[Any]]],
                Union[str, List[Union[str, Dict[str, Any]]]],
            ]
        ],
    ] = {
        # Example: 'file_read': [validate_no_script_tags],
        # Add tools that need EXTRA security functions beyond the defaults
    }

    @staticmethod
    def apply_security_to_tool_response(
        response: Union[str, Dict[str, Any], List[Any]], tool_name: str
    ) -> Union[str, List[Union[str, Dict[str, Any]]]]:
        """Apply all configured security functions for a specific tool.

        Each security function should either:
        - Return the (possibly modified) response
        - Raise SecurityException if validation fails

        Args:
            response: The response to secure (compatible with LangChain ToolCall/ToolMessage)
            tool_name: Name of the tool being used

        Returns:
            Secured response compatible with ToolMessage.content (str | list[str | dict])

        Raises:
            SecurityException: If any security validation fails
        """
        all_functions = list(PromptSecurity.DEFAULT_SECURITY_FUNCTIONS)
        if tool_name in PromptSecurity.TOOL_SPECIFIC_FUNCTIONS:
            all_functions.extend(PromptSecurity.TOOL_SPECIFIC_FUNCTIONS[tool_name])

        secured_response = response
        for func in all_functions:
            try:
                secured_response = func(secured_response)

            except SecurityException:
                raise

            except Exception as e:
                raise SecurityException(
                    f"Security function {func.__name__} failed for tool '{tool_name}': {str(e)}"
                ) from e

        # Type assertion: security functions guarantee proper return type
        return secured_response  # type: ignore[return-value]
