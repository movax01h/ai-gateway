# pylint: disable=direct-environment-variable-reference,too-many-return-statements
"""Factory for creating prompt scanner instances.

This module provides a factory function that creates the appropriate prompt scanner based on environment configuration
and feature flags.

The prompt scanner runs asynchronously in fire-and-forget mode when the AI_PROMPT_SCANNING feature flag is enabled.
Detections are logged but content is allowed through (evaluation mode).
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Optional, Union

import structlog

from duo_workflow_service.security.prompt_scanner import DefaultScanner, PromptScanner

log = structlog.stdlib.get_logger("scanner_factory")


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
    log.info(
        "Creating prompt scanner",
        scanner_type=scanner_type,
        hiddenlayer_client_id_set=bool(os.getenv("HL_CLIENT_ID")),
        hiddenlayer_client_secret_set=bool(os.getenv("HL_CLIENT_SECRET")),
        hiddenlayer_environment=os.getenv("HIDDENLAYER_ENVIRONMENT", "not set"),
    )

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
    """Get or create the Hidden Layer scanner instance (lazy initialization)."""
    global _hiddenlayer_scanner_instance

    if _hiddenlayer_scanner_instance is None:
        try:
            from duo_workflow_service.security.hidden_layer_scanner import (
                HiddenLayerScanner,
            )

            _hiddenlayer_scanner_instance = HiddenLayerScanner()
            log.info(
                "Hidden Layer scanner initialized",
                enabled=_hiddenlayer_scanner_instance.enabled,
            )
        except Exception as e:
            log.error(
                "Failed to initialize Hidden Layer scanner",
                error=str(e),
                error_type=type(e).__name__,
            )
            return None

    return _hiddenlayer_scanner_instance


async def _async_prompt_scan(text: str) -> None:
    """Fire-and-forget async prompt scan (evaluation mode).

    This function runs asynchronously and logs results without blocking. In evaluation mode, detections are logged but
    content is always allowed through.
    """
    start_time = time.time()
    scanner = _get_hiddenlayer_scanner()
    if scanner is None or not scanner.enabled:
        return

    try:
        result = await scanner.scan(text)
        elapsed_ms = (time.time() - start_time) * 1000
        log.info("Prompt scan completed", elapsed_ms=round(elapsed_ms, 2))

        if result.blocked:
            log.warning(
                "Prompt scan would block content (evaluation mode)",
                detection_type=result.detection_type.value,
                confidence=result.confidence,
                details=result.details,
            )
        elif result.detected:
            log.warning(
                "Prompt scan detected potential threat (evaluation mode)",
                detection_type=result.detection_type.value,
                confidence=result.confidence,
                details=result.details,
            )
    except Exception as e:
        log.error(
            "Async prompt scan error",
            error=str(e),
            error_type=type(e).__name__,
        )


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


def scan_prompt(
    response: Union[str, dict, list, Any],
) -> Union[str, dict, list, Any]:
    """Scan response content using async prompt scanning (fire-and-forget).

    Schedules async scans for all text content found in the response.
    Detections are logged but content is always allowed through (evaluation mode).

    This function is non-blocking - it schedules the scan as an async task
    and returns immediately with the original content unchanged.

    Args:
        response: The response data to evaluate. Can be a string, dict, list,
            or nested structure containing strings.

    Returns:
        The original response unchanged.
    """
    from duo_workflow_service.security.markdown_content_security import (
        _apply_recursively,
    )

    def _schedule_async_scan(text: str) -> str:
        if not text or not isinstance(text, str):
            return text

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_async_prompt_scan(text))
        except RuntimeError:
            # No running event loop - this can happen in synchronous contexts.
            # Skip scheduling since we can't run async tasks without a loop.
            pass

        return text

    return _apply_recursively(response, _schedule_async_scan)


def apply_security_scanning(
    response: Union[str, dict, list, Any],
    tool_name: str,
    trust_level: Optional[Any] = None,
) -> Union[str, dict, list, Any]:
    """Apply security scanning to tool responses.

    This is the main entry point for tool response security. The behavior depends
    on the use_ai_prompt_scanning feature flag:

    When enabled (use_ai_prompt_scanning=True):
        1. Applies PromptSecurity sanitization (encoding dangerous tags, stripping
            hidden content, etc.)
        2. Schedules async HiddenLayer detection for untrusted tools (fire-and-forget)

    When disabled (use_ai_prompt_scanning=False):
        1. Applies PromptSecurity sanitization only
        2. No HiddenLayer detection

    PromptSecurity sanitization always runs regardless of feature flag state,
    using TOOL_SECURITY_OVERRIDES or DEFAULT_SECURITY_FUNCTIONS as configured.

    Args:
        response: The tool response to scan/process.
        tool_name: Name of the tool that produced the response.
        trust_level: The trust level of the tool. If not TRUSTED_INTERNAL,
            HiddenLayer scanning may be applied when the feature flag is enabled.

    Returns:
        The sanitized response from PromptSecurity.
    """
    from duo_workflow_service.security.prompt_security import PromptSecurity
    from duo_workflow_service.security.tool_output_security import ToolTrustLevel
    from duo_workflow_service.tracking import current_monitoring_context

    sanitized_response = PromptSecurity.apply_security_to_tool_response(
        response=response,
        tool_name=tool_name,
    )

    monitoring_context = current_monitoring_context.get()
    if monitoring_context.use_ai_prompt_scanning:
        if trust_level != ToolTrustLevel.TRUSTED_INTERNAL:
            scan_prompt(sanitized_response)

    return sanitized_response
