import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import structlog

from duo_workflow_service.security.exceptions import SecurityException
from duo_workflow_service.security.markdown_content_security import (
    _apply_recursively,
    strip_hidden_html_comments,
    strip_hidden_markdown_comments,
    strip_mermaid_comments,
)
from duo_workflow_service.security.security_utils import (
    compute_response_hash_with_length,
)
from duo_workflow_service.security.tool_output_security import SECURITY_DELIMITER_TAGS

log = structlog.stdlib.get_logger("security")

# Type alias for tool responses
ToolResponse = str | dict[str, Any] | list[Any]


# Type alias for security functions
# Functions preserve input structure: dict→dict, list→list, str→str
# Note: apply_security_to_tool_response only receives JSON strings from tools
# (truncation layer converts all dicts to JSON). The dict input path is for
# parsed JSON processing: string→parse→dict→process→reserialize→string.
SecurityFunctionType = Callable[[ToolResponse], ToolResponse]

# Dangerous tags that could be used for prompt injection
DANGEROUS_TAGS = {
    "goal": "goal",
    "system": "system",
    **{tag: tag for tag in SECURITY_DELIMITER_TAGS},
}


def run_from_args():
    args = sys.argv[1:]
    filename = args[0]
    content = Path(filename).read_text()

    return PromptSecurity.apply_security_to_tool_response(content, "test-tool")


def encode_dangerous_tags(
    response: ToolResponse,
) -> ToolResponse:
    """Recursively encode dangerous HTML tags in the response.

    Args:
        response: The response data to encode

    Returns:
        Response with encoded dangerous tags, compatible with ToolMessage.content
    """

    def _encode_recursive(data: Any) -> Any:
        """Internal recursive function that doesn't change top-level structure."""
        if isinstance(data, dict):
            return {k: _encode_recursive(v) for k, v in data.items()}
        if isinstance(data, list):
            return [_encode_recursive(item) for item in data]
        if not isinstance(data, str):
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

    return processed


def strip_hidden_unicode_tags(
    response: ToolResponse,
) -> ToolResponse:
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

    def _strip_unicode_tags(text: str) -> str:
        if not text or not isinstance(text, str):
            return text

        # First handle JSON-escaped Unicode tag characters
        # Unicode Tag Characters (U+E0000-E007F) get encoded as UTF-16 surrogates:
        # U+E0000-E007F -> surrogate pairs starting with \udb40
        # U+E0100-E01EF -> surrogate pairs starting with \udb40

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
    response: ToolResponse,
) -> ToolResponse:
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
    DEFAULT_SECURITY_FUNCTIONS: list[SecurityFunctionType] = [
        encode_dangerous_tags,
        strip_hidden_html_comments,
        strip_hidden_markdown_comments,
        strip_hidden_unicode_tags,
        strip_mermaid_comments,
        # strip_emojis,
    ]

    # Tool-specific additional security functions
    TOOL_SPECIFIC_FUNCTIONS: dict[str, list[SecurityFunctionType]] = {
        # Example: 'file_read': [validate_no_script_tags],
        # Add tools that need EXTRA security functions beyond the defaults
    }

    # Tool-specific security overrides (always applied regardless of feature flag)
    # Empty list means no security functions applied (pass-through)
    TOOL_SECURITY_OVERRIDES: dict[
        str,
        list[SecurityFunctionType],
    ] = {
        "read_file": [],
        "read_repository_file": [],
        "build_review_merge_request_context": [
            encode_dangerous_tags,
            strip_hidden_markdown_comments,
            strip_hidden_unicode_tags,
        ],
        "flow_config_prompts": [
            encode_dangerous_tags,
            strip_hidden_unicode_tags,
            strip_hidden_markdown_comments,
            strip_hidden_html_comments,
        ],
    }

    @staticmethod
    def _resolve_security_functions(tool_name: str) -> list[SecurityFunctionType]:
        """Return the list of security functions to apply for the given tool.

        Args:
            tool_name: Name of the tool being used.

        Returns:
            Ordered list of security functions to apply.
        """
        if tool_name in PromptSecurity.TOOL_SECURITY_OVERRIDES:
            security_functions = PromptSecurity.TOOL_SECURITY_OVERRIDES[tool_name]
            log.info(
                "Applying security override configuration",
                tool_name=tool_name,
                security_functions=[func.__name__ for func in security_functions],
                config_type="override",
            )
            return security_functions

        security_functions = list(PromptSecurity.DEFAULT_SECURITY_FUNCTIONS)
        tool_specific = PromptSecurity.TOOL_SPECIFIC_FUNCTIONS.get(tool_name, [])
        security_functions += tool_specific
        config_type = "default+tool_specific" if tool_specific else "default"
        log.info(
            "Applying security configuration",
            tool_name=tool_name,
            security_functions=[func.__name__ for func in security_functions],
            config_type=config_type,
        )
        return security_functions

    @staticmethod
    def _try_parse_json_str(response: str, tool_name: str) -> ToolResponse:
        """Attempt to parse a string response as JSON for per-field processing.

        Args:
            response: The string to attempt to parse.
            tool_name: Name of the tool (used for logging).

        Returns:
            Parsed dict or list if the string is valid JSON, otherwise the original string.
        """
        try:
            parsed = json.loads(response)
            if isinstance(parsed, (dict, list)):
                log.debug(
                    "Parsed JSON string for per-field security processing",
                    tool_name=tool_name,
                )
                return parsed
            # JSON primitive types (string, number, boolean, null) are not
            # supported for per-field processing. Log for observability.
            log.info(
                "JSON parse succeeded but returned non-container type; "
                "processing as plain string",
                tool_name=tool_name,
                parsed_type=type(parsed).__name__,
            )
        except ValueError as e:
            log.info(
                "Response is not valid JSON; processing as plain string",
                tool_name=tool_name,
                error=str(e),
            )

        return response

    @staticmethod
    def _apply_security_function(
        func: SecurityFunctionType,
        result: ToolResponse,
        tool_name: str,
        validate_only: bool,
    ) -> tuple[ToolResponse, Optional[dict[str, Any]]]:
        """Apply a single security function and return the updated result and modification info.

        Args:
            func: The security function to apply.
            result: The current response value.
            tool_name: Name of the tool (used for error messages and logging).
            validate_only: If True, raises SecurityException instead of modifying.

        Returns:
            Tuple of (new_result, modification_entry_or_None).

        Raises:
            SecurityException: On validation failure or unexpected errors.
        """
        try:
            before_hash, before_length = compute_response_hash_with_length(result)
            processed = func(result)

            after_hash, after_length = compute_response_hash_with_length(processed)

            if before_hash == after_hash:
                return processed, None

            if validate_only:
                raise SecurityException(
                    f"Security validation failed for tool '{tool_name}': "
                    f"content contains dangerous patterns detected by {func.__name__}"
                )

            modification = {
                "function": func.__name__,
                "chars_removed": before_length - after_length,
            }

            return processed, modification

        except SecurityException as e:
            log.error(
                "Security validation failed",
                tool_name=tool_name,
                security_function=func.__name__,
                error=str(e),
            )
            raise
        except Exception as e:
            log.error(
                "Security function execution failed",
                tool_name=tool_name,
                security_function=func.__name__,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise SecurityException(
                f"Security function {func.__name__} failed for tool '{tool_name}': {e}"
            ) from e

    @staticmethod
    def _run_security_functions(
        result: ToolResponse,
        security_functions: list[SecurityFunctionType],
        tool_name: str,
        validate_only: bool,
        original_length: int,
    ) -> ToolResponse:
        """Run all security functions over a result, logging modifications.

        Args:
            result: The response to sanitize.
            security_functions: Ordered list of security functions to apply.
            tool_name: Name of the tool (used for logging).
            validate_only: If True, raises SecurityException on modification.
            original_length: Character length of the original input for logging.

        Returns:
            The sanitized result.

        Raises:
            SecurityException: If validate_only=True and content would be modified.
        """
        functions_that_modified = []

        for func in security_functions:
            result, modification = PromptSecurity._apply_security_function(
                func, result, tool_name, validate_only
            )
            if modification is not None:
                functions_that_modified.append(modification)

        if functions_that_modified:
            log.warning(
                "Security functions modified the tool response",
                tool_name=tool_name,
                functions_applied=len(security_functions),
                modification_details=functions_that_modified,
                original_length=original_length,
            )
        else:
            log.info(
                "Security validation completed, no modifications needed",
                tool_name=tool_name,
                functions_applied=len(security_functions),
            )

        return result

    @staticmethod
    def apply_security_to_tool_response(
        response: ToolResponse,
        tool_name: str,
        validate_only: bool = False,
    ) -> ToolResponse:
        """Apply all configured security functions for a specific tool.

        Args:
            response: The response to secure.
            tool_name: Name of the tool being used.
            validate_only: If True, raises SecurityException if content would be
                modified instead of actually modifying it. Useful for validating
                that content doesn't contain dangerous patterns.

        Returns:
            The secured response. str in → str out, dict/list in → dict/list out.

        Raises:
            SecurityException: If validate_only=True and content contains
                dangerous patterns that would be modified.
        """
        security_functions = PromptSecurity._resolve_security_functions(tool_name)

        if isinstance(response, str):
            # Attempt to parse as JSON so security functions process individual
            # fields rather than the entire serialized blob. This prevents an
            # HTML-like construct in one field (e.g. an unclosed "<!--" in a title)
            # from destroying adjacent data when bleach.clean() treats the whole
            # string as a single HTML document.
            parsed = PromptSecurity._try_parse_json_str(response, tool_name)
            original_length = len(response)

            if isinstance(parsed, (dict, list)):
                # JSON string: sanitize field-by-field, re-serialize back to str
                result = PromptSecurity._run_security_functions(
                    parsed,
                    security_functions,
                    tool_name,
                    validate_only,
                    original_length,
                )
                if validate_only:
                    return response
                return json.dumps(result, ensure_ascii=False, default=str)

            # Plain string: sanitize directly
            return PromptSecurity._run_security_functions(
                parsed,
                security_functions,
                tool_name,
                validate_only,
                original_length,
            )

        # dict/list arriving directly from the agent_platform tool node, which
        # bypasses DuoBaseTool._arun() and its truncation/serialization step.
        # Sanitize field-by-field and return the same type.
        original_length = len(json.dumps(response, ensure_ascii=False, default=str))
        return PromptSecurity._run_security_functions(
            response, security_functions, tool_name, validate_only, original_length
        )
