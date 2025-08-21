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
        # Never allow unknown types to bypass filtering
        raise SecurityException(
            f"Unsupported type for security processing: {type(response).__name__}. "
            f"All data must be explicitly validated for security."
        )


def strip_hidden_html_comments(
    response: Union[str, Dict[str, Any], List[Any]],
) -> Union[str, List[Union[str, Dict[str, Any]]]]:
    """Strip HTML comments using Bleach, leave everything else unchanged.

    Uses Mozilla's Bleach library (https://github.com/mozilla/bleach) to safely
    remove HTML comments while preserving all other content exactly as it was.
    Other security measures (like dangerous tag encoding) are handled by other
    functions in PromptSecurity.

    Args:
        response: The response data to process

    Returns:
        Response with HTML comments removed using Bleach, everything else unchanged
    """

    def _strip_comments(text: str) -> str:
        if not text or not isinstance(text, str):
            return text

        # Dangerous tags are already encoded as &lt;system&gt; etc. by encode_dangerous_tags
        # This function focuses solely on removing HTML comments while preserving all other content

        # Early exit if no HTML comments are present in any format
        has_regular_comments = "<!--" in text
        has_escaped_comments = (
            "\\u003c!--" in text  # JSON unicode escape (lowercase)
            or "\\u003C!--" in text  # JSON unicode escape (uppercase)
            or "\\<!--" in text  # Backslash escape
        )

        if not has_regular_comments and not has_escaped_comments:
            return text

        # Remove JSON-escaped HTML comments (e.g., from json.dumps serialization)
        # Handles patterns like \\u003c!-- ... --\\u003e and \\\\u003c!-- ... --\\\\u003e
        text = re.sub(r"\\+u003[cC]!--.*?--\\+u003[eE]", "", text, flags=re.DOTALL)

        # Remove backslash-escaped HTML comments
        # Handles patterns like \\<!-- ... --\\>
        text = re.sub(r"\\+<!--.*?--\\+>", "", text, flags=re.DOTALL)

        # Use Bleach to strip comments while preserving common HTML tags
        # Extend Bleach's default allowed tags with commonly used HTML elements
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

        # Allow common attributes that tests expect
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

        # After Bleach processing, handle any remaining malformed comment patterns
        # that might have been escaped instead of removed
        result = re.sub(r"&lt;!--.*?--&gt;", "", result, flags=re.DOTALL)

        return result

    return _apply_recursively(response, _strip_comments)
