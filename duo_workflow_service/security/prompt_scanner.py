"""Abstract interface for prompt scanning services.

This module provides an abstract base class for prompt security scanning services, along with shared data structures for
detection results.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

import structlog

log = structlog.stdlib.get_logger("prompt_scanner")


class DetectionType(str, Enum):
    """Types of detection from scanning services."""

    PROMPT_INJECTION = "prompt_injection"
    MALICIOUS_CONTENT = "malicious_content"
    PII_DETECTED = "pii_detected"
    BLOCKED = "blocked"
    SAFE = "safe"


@dataclass
class ScanResult:
    """Result from prompt scanning evaluation."""

    detected: bool
    detection_type: DetectionType
    confidence: float = 0.0
    details: Optional[str] = None
    blocked: bool = False
    sanitized_text: Optional[str] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)


class PromptScanner(ABC):
    """Abstract base class for prompt security scanning services.

    All scanner implementations must provide a scan() method that evaluates text for security threats and returns a
    ScanResult.
    """

    @abstractmethod
    def scan(self, text: str) -> ScanResult:
        """Scan text for security threats.

        Args:
            text: The text to evaluate for security threats.

        Returns:
            ScanResult with detection status and details.

        Raises:
            Exception: If the scan operation fails.
        """

    @property
    @abstractmethod
    def enabled(self) -> bool:
        """Check if the scanner is enabled and initialized."""


class DefaultScanner(PromptScanner):
    """Default scanner implementation that passes all content as safe.

    This scanner is used as a fallback when no specific scanner is configured or when scanning is disabled. It logs
    scans but always returns safe results.
    """

    def __init__(self) -> None:
        """Initialize the default scanner."""
        log.info("Using default scanner - all content will pass as safe")

    @property
    def enabled(self) -> bool:
        """Default scanner is always enabled."""
        return True

    def scan(self, text: str) -> ScanResult:
        """Return safe result without performing actual scanning.

        Args:
            text: The text to evaluate (logged but not scanned).

        Returns:
            ScanResult indicating content is safe.
        """
        log.debug(
            "Default scanner called - returning safe result", text_length=len(text)
        )
        return ScanResult(
            detected=False,
            detection_type=DetectionType.SAFE,
            details="Default scanner - no actual scanning performed",
        )
