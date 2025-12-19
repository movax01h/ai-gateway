# flake8: noqa: W605
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

import structlog

from duo_workflow_service.security.emoji_security import strip_emojis
from duo_workflow_service.security.exceptions import SecurityException
from duo_workflow_service.security.markdown_content_security import (
    strip_hidden_html_comments,
    strip_markdown_link_comments,
    strip_mermaid_comments,
)
from duo_workflow_service.security.security_utils import (
    compute_response_hash_with_length,
)
from duo_workflow_service.security.tool_output_security import SECURITY_DELIMITER_TAGS

log = structlog.stdlib.get_logger("security")


# Type alias for security functions
SecurityFunctionType = Callable[
    [Union[str, Dict[str, Any], List[Any]]],
    Union[str, List[Union[str, Dict[str, Any]]]],
]


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
        # Core dangerous tags that could be used for prompt injection
        DANGEROUS_TAGS = {
            "goal": "goal",
            "system": "system",
        }
        # Add security delimiter tags to prevent delimiter escape attacks
        for tag in SECURITY_DELIMITER_TAGS:
            DANGEROUS_TAGS[tag] = tag

        if isinstance(data, dict):
            return {k: _encode_recursive(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [_encode_recursive(item) for item in data]
        elif not isinstance(data, str):
            # Return non-string types (int, float, bool, None, etc.) as-is
            return data

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
    return processed


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

    # Default security functions applied to all tool responses
    DEFAULT_SECURITY_FUNCTIONS: List[SecurityFunctionType] = [
        encode_dangerous_tags,
        strip_hidden_html_comments,
        strip_markdown_link_comments,
        strip_hidden_unicode_tags,
        strip_mermaid_comments,
        # strip_emojis,
    ]

    # Tool-specific additional security functions
    TOOL_SPECIFIC_FUNCTIONS: Dict[str, List[SecurityFunctionType]] = {
        # Example: 'file_read': [validate_no_script_tags],
        # Add tools that need EXTRA security functions beyond the defaults
    }

    # Tool-specific security overrides (always applied regardless of feature flag)
    # Empty list means no security functions applied (pass-through)
    TOOL_SECURITY_OVERRIDES: Dict[
        str,
        List[SecurityFunctionType],
    ] = {
        "read_file": [],
        "read_repository_file": [],
        "build_review_merge_request_context": [
            encode_dangerous_tags,
            strip_markdown_link_comments,
            strip_hidden_unicode_tags,
        ],
        "flow_config_prompts": [
            encode_dangerous_tags,
            strip_hidden_unicode_tags,
            strip_markdown_link_comments,
            strip_hidden_html_comments,
        ],
    }

    @staticmethod
    def apply_security_to_tool_response(
        response: Union[str, Dict[str, Any], List[Any]],
        tool_name: str,
        validate_only: bool = False,
    ) -> Union[str, Dict[str, Any], List[Any]]:
        """Apply all configured security functions for a specific tool.

        Args:
            response: The response to secure.
            tool_name: Name of the tool being used.
            validate_only: If True, raises SecurityException if content would be
                modified instead of actually modifying it. Useful for validating
                that content doesn't contain dangerous patterns.

        Returns:
            The secured response.

        Raises:
            SecurityException: If validate_only=True and content contains
                dangerous patterns that would be modified.
        """
        if tool_name in PromptSecurity.TOOL_SECURITY_OVERRIDES:
            security_functions = PromptSecurity.TOOL_SECURITY_OVERRIDES[tool_name]
        else:
            security_functions = list(PromptSecurity.DEFAULT_SECURITY_FUNCTIONS)
            security_functions += PromptSecurity.TOOL_SPECIFIC_FUNCTIONS.get(
                tool_name, []
            )

        result = response
        for func in security_functions:
            try:
                processed = func(result)

                if validate_only:
                    # In validate_only mode, check if the function would modify content
                    if processed != result:
                        raise SecurityException(
                            f"Security validation failed for tool '{tool_name}': "
                            f"content contains dangerous patterns detected by {func.__name__}"
                        )
                else:
                    result = processed
            except SecurityException:
                raise
            except Exception as e:
                raise SecurityException(
                    f"Security function {func.__name__} failed for tool '{tool_name}': {e}"
                ) from e

        return result
