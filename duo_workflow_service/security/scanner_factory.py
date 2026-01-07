# pylint: disable=direct-environment-variable-reference,too-many-return-statements
"""Prompt scanner factory and security scanning utilities.

This module provides:
- Factory functions for creating prompt scanner instances
- Security scanning functions that integrate with HiddenLayer for threat detection
- Protection level-based behavior controlled by namespace settings

Protection Levels:
- NO_CHECKS: PromptSecurity sanitization only, no HiddenLayer scanning
- LOG_ONLY: Non-blocking HiddenLayer scanning, threats logged but allowed through
- INTERRUPT: Blocking HiddenLayer scanning, workflow stopped on threat detection
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Optional, Union

import structlog

from duo_workflow_service.security.exceptions import SecurityException
from duo_workflow_service.security.prompt_scanner import (
    DefaultScanner,
    PromptScanner,
    ScanResult,
)

log = structlog.stdlib.get_logger("scanner_factory")


class PromptInjectionDetectedError(SecurityException):
    """Raised when prompt injection is detected and workflow must be stopped."""

    def __init__(self, scan_result: ScanResult, tool_name: str):
        self.scan_result = scan_result
        self.tool_name = tool_name
        details = scan_result.details or scan_result.detection_type.value
        super().__init__(f"Prompt injection detected in '{tool_name}': {details}")

    def format_user_message(self, tool_name: str) -> str:
        """Format a user-friendly error message.

        Args:
            tool_name: Name of the tool that triggered the security exception.

        Returns:
            Formatted error message for the UI.
        """
        return (
            f"Security scan detected potentially malicious content. "
            f"Tool '{tool_name}' response was blocked."
        )


def create_scanner(scanner_type: Optional[str] = None) -> PromptScanner:
    """Create a prompt scanner instance based on configuration.

    Args:
        scanner_type: Type of scanner to create. If None, reads from
            PROMPT_SCANNER environment variable. Supported values:
            - "hidden_layer": Use Hidden Layer
            - "default": Use default pass-through scanner
            If not specified or invalid, uses default scanner.

    Returns:
        PromptScanner instance configured based on the scanner type.

    Examples:
        >>> # Using environment variable
        >>> scanner = create_scanner()
        >>>
        >>> # Explicitly specifying scanner type
        >>> scanner = create_scanner("hidden_layer")
    """
    # Determine scanner type from parameter or environment
    if scanner_type is None:
        scanner_type = os.getenv("PROMPT_SCANNER", "default").lower()
    else:
        scanner_type = scanner_type.lower()

    # Log configuration for debugging
    log_kwargs: dict[str, Any] = {"scanner_type": scanner_type}
    if scanner_type == "hidden_layer":
        log_kwargs.update(
            {
                "hiddenlayer_client_id_set": bool(os.getenv("HL_CLIENT_ID")),
                "hiddenlayer_client_secret_set": bool(os.getenv("HL_CLIENT_SECRET")),
                "hiddenlayer_environment": os.getenv(
                    "HIDDENLAYER_ENVIRONMENT", "not set"
                ),
            }
        )
    log.info("Creating prompt scanner", **log_kwargs)

    try:
        if scanner_type == "hidden_layer":
            from duo_workflow_service.security.hidden_layer_scanner import (
                HiddenLayerScanner,
            )

            try:
                scanner = HiddenLayerScanner()
                log.info(
                    "Successfully created Hidden Layer scanner",
                    scanner_enabled=scanner.enabled,
                )
                return scanner
            except Exception as e:
                log.error(
                    "Failed to create Hidden Layer scanner, falling back to default",
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True,
                )
                return DefaultScanner()

        elif scanner_type == "default":
            log.info("Using default scanner (no scanning performed)")
            return DefaultScanner()

        else:
            log.warning(
                "Unknown scanner type, using default scanner",
                scanner_type=scanner_type,
                supported_types=["hidden_layer", "default"],
            )
            return DefaultScanner()

    except ImportError as e:
        log.error(
            "Failed to import scanner module, using default scanner",
            error=str(e),
            scanner_type=scanner_type,
        )
        return DefaultScanner()


# Singleton instance for the default scanner
_default_scanner_instance: Optional[PromptScanner] = None

# Singleton instance for the Hidden Layer scanner
_hiddenlayer_scanner_instance = None


def _get_hiddenlayer_scanner():
    """Get or create the HiddenLayer scanner singleton."""
    global _hiddenlayer_scanner_instance

    if _hiddenlayer_scanner_instance is None:
        try:
            from duo_workflow_service.security.hidden_layer_scanner import (
                HiddenLayerScanner,
            )

            _hiddenlayer_scanner_instance = HiddenLayerScanner()
        except Exception as e:
            log.error(
                "HiddenLayer scanner initialization failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            return None

    return _hiddenlayer_scanner_instance


def _log_scan_result(result: ScanResult, mode: str, elapsed_ms: float) -> None:
    """Log scan results with appropriate severity based on detection status."""
    log.info("HiddenLayer scan completed", mode=mode, elapsed_ms=round(elapsed_ms, 2))

    if not result.detected and not result.blocked:
        return

    log.warning(
        "HiddenLayer threat detected",
        mode=mode,
        detection_type=result.detection_type.value,
        blocked=result.blocked,
        confidence=result.confidence,
        details=result.details,
    )


async def _execute_scan(text: str, mode: str) -> Optional[ScanResult]:
    """Execute HiddenLayer scan and log results.

    Args:
        text: Text content to scan for threats.
        mode: Scanning mode for logging context ('log_only' or 'interrupt').

    Returns:
        ScanResult if scan completed, None if scanner unavailable or error occurred.
    """
    scanner = _get_hiddenlayer_scanner()
    if scanner is None or not scanner.enabled:
        return None

    start_time = time.time()
    try:
        result = await scanner.scan(text)
        elapsed_ms = (time.time() - start_time) * 1000
        _log_scan_result(result, mode, elapsed_ms)
        return result
    except Exception as e:
        log.error(
            "HiddenLayer scan failed",
            mode=mode,
            error=str(e),
            error_type=type(e).__name__,
        )
        return None


async def _fire_and_forget_scan(text: str) -> None:
    """Non-blocking scan for LOG_ONLY mode.

    Logs threats but allows content through.
    """
    await _execute_scan(text, mode="log_only")


def get_scanner() -> PromptScanner:
    """Get or create the global scanner instance.

    This function maintains a singleton scanner instance based on the
    PROMPT_SCANNER environment variable. The scanner is created lazily
    on first access.

    Returns:
        PromptScanner instance.

    Examples:
        >>> scanner = get_scanner()
        >>> result = scanner.scan("user input text")
    """
    global _default_scanner_instance

    if _default_scanner_instance is None:
        _default_scanner_instance = create_scanner()

    return _default_scanner_instance


def _schedule_fire_and_forget_scan(response: Union[str, dict, list, Any]) -> None:
    """Schedule non-blocking scans for all text in response."""
    from duo_workflow_service.security.markdown_content_security import (
        _apply_recursively,
    )

    def _schedule_scan(text: str) -> str:
        if not text or not isinstance(text, str):
            return text
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_fire_and_forget_scan(text))
        except RuntimeError:
            pass  # No event loop available
        return text

    _apply_recursively(response, _schedule_scan)


def _extract_text_for_scanning(response: Union[str, dict, list, Any]) -> str:
    """Extract all text content from a response for scanning.

    Args:
        response: The response data that may contain text.

    Returns:
        Concatenated text content from the response.
    """
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        return " ".join(_extract_text_for_scanning(v) for v in response.values() if v)
    if isinstance(response, list):
        return " ".join(_extract_text_for_scanning(item) for item in response if item)
    return str(response) if response else ""


def _run_blocking_scan(text: str, tool_name: str) -> None:
    """Execute blocking scan and raise exception if threat detected.

    Uses the synchronous HiddenLayer client directly for interrupt mode.
    This ensures the workflow is blocked until the scan completes.

    Args:
        text: Text content to scan for threats.
        tool_name: Name of the tool that produced the response.

    Raises:
        PromptInjectionDetectedError: If threat is detected in the content.
    """
    scanner = _get_hiddenlayer_scanner()
    if scanner is None or not scanner.enabled:
        return

    start_time = time.time()
    try:
        scan_result = scanner.scan_sync(text)
        elapsed_ms = (time.time() - start_time) * 1000
        _log_scan_result(scan_result, mode="interrupt", elapsed_ms=elapsed_ms)
    except Exception as e:
        log.error(
            "Blocking scan execution failed",
            tool_name=tool_name,
            error=str(e),
            error_type=type(e).__name__,
        )
        return

    if scan_result and (scan_result.blocked or scan_result.detected):
        raise PromptInjectionDetectedError(scan_result, tool_name)


def apply_security_scanning(
    response: Union[str, dict, list, Any],
    tool_name: str,
    trust_level: Optional[Any] = None,
) -> Union[str, dict, list, Any]:
    """Apply security scanning to tool responses.

    HiddenLayer scanning only runs for untrusted tools when protection level
    is not NO_CHECKS. Trusted tools (TRUSTED_INTERNAL) skip HiddenLayer scanning.

    Behavior by protection level (for untrusted tools):
    - NO_CHECKS: PromptSecurity sanitization only, no HiddenLayer
    - LOG_ONLY: Sanitization + background HiddenLayer scan (non-blocking, logs only)
    - INTERRUPT: Sanitization + blocking HiddenLayer scan (raises on threat)

    PromptSecurity sanitization always runs regardless of protection level or trust.

    Args:
        response: Tool response to scan/process.
        tool_name: Name of the tool that produced the response.
        trust_level: Tool trust level. TRUSTED_INTERNAL skips HiddenLayer scanning.

    Returns:
        Sanitized response.

    Raises:
        PromptInjectionDetectedError: If threat detected in INTERRUPT mode.
    """
    from duo_workflow_service.gitlab.gitlab_api import PromptInjectionProtectionLevel
    from duo_workflow_service.security.prompt_security import PromptSecurity
    from duo_workflow_service.security.tool_output_security import ToolTrustLevel
    from duo_workflow_service.tracking import current_monitoring_context

    sanitized_response = PromptSecurity.apply_security_to_tool_response(
        response=response,
        tool_name=tool_name,
    )

    monitoring_context = current_monitoring_context.get()
    protection_level = monitoring_context.prompt_injection_protection_level

    # Skip HiddenLayer scanning for NO_CHECKS or trusted tools
    if protection_level == PromptInjectionProtectionLevel.NO_CHECKS:
        return sanitized_response

    if trust_level == ToolTrustLevel.TRUSTED_INTERNAL:
        return sanitized_response

    if protection_level == PromptInjectionProtectionLevel.LOG_ONLY:
        _schedule_fire_and_forget_scan(sanitized_response)

    elif protection_level == PromptInjectionProtectionLevel.INTERRUPT:
        text_to_scan = _extract_text_for_scanning(sanitized_response)
        if text_to_scan:
            _run_blocking_scan(text_to_scan, tool_name)

    return sanitized_response
