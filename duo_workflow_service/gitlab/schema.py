from enum import Enum
from typing import Optional


class PromptInjectionProtectionLevel(str, Enum):
    """Protection level for prompt injection scanning.

    Values:
        NO_CHECKS: No scanning performed
        LOG_ONLY: Evaluation mode - detections logged but content allowed through
        INTERRUPT: Block mode - content blocked when injection detected
    """

    NO_CHECKS = "no_checks"
    LOG_ONLY = "log_only"
    INTERRUPT = "interrupt"

    @classmethod
    def from_graphql(cls, value: Optional[str]) -> "PromptInjectionProtectionLevel":
        """Convert GraphQL value to enum, defaulting to LOG_ONLY for safety."""
        if value is None:
            return cls.LOG_ONLY
        # GraphQL returns uppercase values like "INTERRUPT", "LOG_ONLY", "NO_CHECKS"
        normalized = value.lower()
        try:
            return cls(normalized)
        except ValueError:
            return cls.LOG_ONLY
