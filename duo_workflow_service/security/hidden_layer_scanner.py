# pylint: disable=direct-environment-variable-reference
"""Hidden Layer integration for prompt security.

This module provides integration with Hidden Layer's AI Security service for detecting prompt injection attacks and
other malicious content in AI applications.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, cast

import structlog

from duo_workflow_service.security.prompt_scanner import (
    DetectionType,
    PromptScanner,
    ScanResult,
)

log = structlog.stdlib.get_logger("hidden_layer_scanner")


class HiddenLayerError(Exception):
    """Exception raised when Hidden Layer operations fail."""


@dataclass
class HiddenLayerConfig:
    """Configuration for Hidden Layer service."""

    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    environment: str = "prod-us"
    base_url: Optional[str] = None
    project_id: Optional[str] = None

    @classmethod
    def from_environment(cls) -> "HiddenLayerConfig":
        """Create configuration from environment variables."""
        return cls(
            client_id=os.getenv("HL_CLIENT_ID"),
            client_secret=os.getenv("HL_CLIENT_SECRET"),
            environment=os.getenv("HIDDENLAYER_ENVIRONMENT", "prod-us"),
            base_url=os.getenv("HIDDENLAYER_BASE_URL"),
            project_id=os.getenv("HL_PROJECT_ID"),
        )


class HiddenLayerScanner(PromptScanner):
    """Scanner using Hidden Layer's AI Security SDK.

    This scanner uses the Hidden Layer SDK to detect prompt injection and other security threats in prompts. Provides
    both async and sync scanning methods.
    """

    def __init__(self, config: Optional[HiddenLayerConfig] = None) -> None:
        """Initialize the Hidden Layer scanner.

        Args:
            config: Configuration for the scanner. If None, loads from environment.
        """
        self._config = config or HiddenLayerConfig.from_environment()
        self._async_client: Any = None
        self._sync_client: Any = None
        self._initialized = False
        self._initialize()

    def _initialize(self) -> None:
        """Initialize Hidden Layer SDK clients (async and sync)."""
        if not self._config.client_id or not self._config.client_secret:
            log.error(
                "Hidden Layer requires HL_CLIENT_ID and HL_CLIENT_SECRET "
                "environment variables"
            )
            raise HiddenLayerError(
                "Hidden Layer authentication not configured. "
                "Set HL_CLIENT_ID and HL_CLIENT_SECRET"
            )

        try:
            log.info(
                "Initializing Hidden Layer SDK",
                environment=self._config.environment,
            )
            from hiddenlayer import AsyncHiddenLayer, HiddenLayer

            hl_environment = cast(
                Literal["prod-us", "prod-eu"], self._config.environment
            )
            # Build client kwargs
            client_kwargs: Dict[str, Any] = {
                "client_id": self._config.client_id,
                "client_secret": self._config.client_secret,
            }

            # Add default headers only if project_id is configured
            if self._config.project_id:
                client_kwargs["default_headers"] = {
                    "HL-Project-Id": self._config.project_id
                }

            # skip setting environment if base_url is set
            if self._config.base_url:
                client_kwargs["base_url"] = self._config.base_url
            else:
                client_kwargs["environment"] = hl_environment

            self._async_client = AsyncHiddenLayer(**client_kwargs)
            self._sync_client = HiddenLayer(**client_kwargs)

            self._initialized = True
            log.info(
                "Hidden Layer scanner initialized",
                environment=self._config.environment,
            )
        except ImportError as e:
            log.error(
                "Failed to import Hidden Layer SDK",
                error=str(e),
                exc_info=True,
            )
            raise HiddenLayerError(
                "Hidden Layer SDK not installed. Install with: pip install hiddenlayer-sdk"
            ) from e
        except Exception as e:
            log.error(
                "Failed to initialize Hidden Layer scanner",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            raise HiddenLayerError(f"Failed to initialize Hidden Layer: {e}") from e

    @property
    def enabled(self) -> bool:
        """Check if the scanner is initialized and ready."""
        return self._initialized

    async def scan(self, text: str) -> ScanResult:  # type: ignore[override]
        """Scan text for security threats using Hidden Layer SDK.

        Args:
            text: The text to evaluate for security threats.

        Returns:
            ScanResult with detection status and details.

        Raises:
            HiddenLayerError: If the scan operation fails.
        """
        if not self.enabled:
            log.debug("Hidden Layer scan skipped - scanner not enabled")
            return ScanResult(
                detected=False,
                detection_type=DetectionType.SAFE,
                details="Hidden Layer scanner is not enabled",
            )

        log.debug("Calling Hidden Layer API", text_length=len(text) if text else 0)

        try:
            response = await self._async_client.interactions.analyze(
                metadata={
                    "model": "duo",
                    "requester_id": "gitlab-duo-workflow",
                },
                input={
                    "messages": [
                        {
                            "role": "user",
                            "content": text,
                        }
                    ]
                },
            )
        except Exception as e:
            log.error(
                "Hidden Layer API request failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise HiddenLayerError(f"API request failed: {e}") from e

        return self._parse_response(response)

    def scan_sync(self, text: str) -> ScanResult:
        """Synchronously scan text for security threats using Hidden Layer SDK.

        This method uses the synchronous HiddenLayer client for blocking scans
        in interrupt mode where we need to stop the workflow on threat detection.

        Args:
            text: The text to evaluate for security threats.

        Returns:
            ScanResult with detection status and details.

        Raises:
            HiddenLayerError: If the scan operation fails.
        """
        if not self.enabled:
            log.debug("Hidden Layer scan skipped - scanner not enabled")
            return ScanResult(
                detected=False,
                detection_type=DetectionType.SAFE,
                details="Hidden Layer scanner is not enabled",
            )

        log.debug(
            "Calling Hidden Layer API (sync)", text_length=len(text) if text else 0
        )

        try:
            response = self._sync_client.interactions.analyze(
                metadata={
                    "model": "duo",
                    "requester_id": "gitlab-duo-workflow",
                },
                input={
                    "messages": [
                        {
                            "role": "user",
                            "content": text,
                        }
                    ]
                },
            )
        except Exception as e:
            log.error(
                "Hidden Layer API request failed (sync)",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise HiddenLayerError(f"API request failed: {e}") from e

        return self._parse_response(response)

    def _parse_response(self, response: Any) -> ScanResult:
        """Parse Hidden Layer SDK response into ScanResult.

        Args:
            response: The SDK response object.

        Returns:
            ScanResult with detection status and details.
        """
        raw_response = self._response_to_dict(response)

        evaluation = getattr(response, "evaluation", None)
        has_detections = False
        action = "Allow"

        if evaluation:
            has_detections = getattr(evaluation, "has_detections", False)
            action = getattr(evaluation, "action", "Allow")

        detection_type = DetectionType.SAFE
        if has_detections:
            # Log response only when there is a detection
            log.warning("Hidden Layer scan detects threats", response=raw_response)
            detection_type = DetectionType.PROMPT_INJECTION

        analysis_list = getattr(response, "analysis", []) or []
        findings_details = []
        for analysis in analysis_list:
            if getattr(analysis, "detected", False):
                findings = getattr(analysis, "findings", []) or []
                for finding in findings:
                    finding_type = getattr(finding, "type", None)
                    if finding_type:
                        findings_details.append(finding_type)

        details = None
        if has_detections:
            if findings_details:
                details = f"Detected threats: {', '.join(findings_details)}"
            else:
                details = f"Threat detected (action: {action})"

        blocked = action in ("Block", "Redact")

        if has_detections:
            log.warning(
                "Hidden Layer detected potential threat",
                detection_type=detection_type.value,
                action=action,
                details=details,
            )

        return ScanResult(
            detected=has_detections,
            detection_type=detection_type,
            details=details,
            blocked=blocked,
            raw_response=raw_response,
        )

    def _response_to_dict(self, response: Any) -> Dict[str, Any]:
        """Convert SDK response to dictionary for logging.

        Args:
            response: The SDK response object.

        Returns:
            Dictionary representation of the response.
        """
        if hasattr(response, "to_dict"):
            return response.to_dict()
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json")
        if hasattr(response, "__dict__"):
            return {k: v for k, v in response.__dict__.items() if not k.startswith("_")}
        return {"raw": str(response)}
