"""Security exceptions module."""


class SecurityException(Exception):
    """Custom exception raised when security validation fails."""

    def format_user_message(self, tool_name: str) -> str:
        """Format a user-friendly error message.

        Args:
            tool_name: Name of the tool that triggered the security exception.

        Returns:
            Formatted error message suitable for display to users.
        """
        return (
            f"Security scan detected potentially malicious content. "
            f"Tool '{tool_name}' response was blocked."
        )
