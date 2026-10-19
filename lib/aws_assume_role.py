"""STS AssumeRole credentials for Amazon Bedrock, cached and refreshed per role."""

import re
import threading
from collections import OrderedDict
from typing import Any, NamedTuple, Optional

import botocore.session
import structlog
from botocore.credentials import (
    AssumeRoleCredentialFetcher,
    DeferredRefreshableCredentials,
)
from botocore.exceptions import BotoCoreError, ClientError

log = structlog.stdlib.get_logger("aws_assume_role")

__all__ = [
    "DEFAULT_ROLE_SESSION_NAME",
    "EXTERNAL_ID_MAX_LENGTH",
    "EXTERNAL_ID_PATTERN",
    "IAM_ROLE_ARN_PATTERN",
    "AssumeRoleError",
    "AssumedRoleCredentials",
    "clear_credentials_cache",
    "get_assumed_role_credentials",
    "validate_external_id",
    "validate_iam_role_arn",
]

IAM_ROLE_ARN_PATTERN = (
    r"^arn:aws(-[a-z]+)*:iam::[0-9]{12}:role/[A-Za-z0-9+=,.@_/-]{1,512}$"
)
_IAM_ROLE_ARN_RE = re.compile(IAM_ROLE_ARN_PATTERN)
IAM_ROLE_ARN_MAX_LENGTH = 2048

ROLE_SESSION_NAME_PATTERN = r"^[A-Za-z0-9_+=,.@-]{2,64}$"
_ROLE_SESSION_NAME_RE = re.compile(ROLE_SESSION_NAME_PATTERN)
DEFAULT_ROLE_SESSION_NAME = "gitlab-ai-gateway"

EXTERNAL_ID_PATTERN = r"^[A-Za-z0-9_+=,.@:/-]{2,1224}$"
_EXTERNAL_ID_RE = re.compile(EXTERNAL_ID_PATTERN)
EXTERNAL_ID_MAX_LENGTH = 1224

_MAX_CACHED_ROLES = 256


class AssumeRoleError(Exception):
    """Raised when credentials for a role cannot be obtained."""

    def __init__(
        self,
        message: str,
        *,
        role_arn: Optional[str] = None,
        error_code: Optional[str] = None,
    ):
        super().__init__(message)
        self.role_arn = role_arn
        self.error_code = error_code


class AssumedRoleCredentials(NamedTuple):
    access_key: str
    secret_key: str
    token: Optional[str]

    def __repr__(self) -> str:
        # Keep the STS secret key and session token out of logs and tracebacks.
        return (
            f"AssumedRoleCredentials(access_key={self.access_key!r}, "
            "secret_key='***', token='***')"
        )

    def to_litellm_params(self) -> dict[str, str]:
        params = {
            "aws_access_key_id": self.access_key,
            "aws_secret_access_key": self.secret_key,
        }
        if self.token:
            params["aws_session_token"] = self.token
        return params


def validate_iam_role_arn(value: str) -> str:
    """Return ``value`` unchanged if it is a well-formed IAM role ARN, else raise ``ValueError``."""
    if len(value) > IAM_ROLE_ARN_MAX_LENGTH or not _IAM_ROLE_ARN_RE.match(value):
        raise ValueError(
            "Invalid IAM role ARN: expected the form "
            "'arn:aws:iam::<12-digit-account-id>:role/<role-name>'"
        )
    return value


def validate_external_id(value: str) -> str:
    """Return ``value`` unchanged if it is a valid STS ExternalId, else raise ``ValueError``."""
    if len(value) > EXTERNAL_ID_MAX_LENGTH or not _EXTERNAL_ID_RE.fullmatch(value):
        raise ValueError(
            "Invalid external ID: expected 2-1224 characters from [A-Za-z0-9_+=,.@:/-]"
        )
    return value


_cache: "OrderedDict[tuple[str, str, Optional[str]], DeferredRefreshableCredentials]" = OrderedDict()
_cache_lock = threading.Lock()


def _build_credentials(
    role_arn: str, session_name: str, external_id: Optional[str]
) -> DeferredRefreshableCredentials:
    session = botocore.session.get_session()
    source_credentials: Any = session.get_credentials()
    if source_credentials is None:
        raise AssumeRoleError(
            "Cannot assume IAM role: no AWS credentials were found in the default "
            "credential chain to call sts:AssumeRole with",
            role_arn=role_arn,
            error_code="NoCredentials",
        )

    extra_args = {"RoleSessionName": session_name}
    if external_id:
        extra_args["ExternalId"] = external_id

    fetcher = AssumeRoleCredentialFetcher(
        client_creator=session.create_client,
        source_credentials=source_credentials,
        role_arn=role_arn,
        extra_args=extra_args,
    )
    return DeferredRefreshableCredentials(
        method="assume-role", refresh_using=fetcher.fetch_credentials
    )


def get_assumed_role_credentials(
    role_arn: str,
    session_name: Optional[str] = None,
    external_id: Optional[str] = None,
) -> AssumedRoleCredentials:
    """Return current temporary credentials for ``role_arn``, assuming the role on first use.

    Raises:
        ValueError: ``role_arn``, ``session_name`` or ``external_id`` is malformed.
        AssumeRoleError: no source credentials are available.
        AssumeRoleError: STS rejected the AssumeRole call (``error_code`` holds its code).
    """
    validate_iam_role_arn(role_arn)
    session_name = session_name or DEFAULT_ROLE_SESSION_NAME
    if not _ROLE_SESSION_NAME_RE.match(session_name):
        raise ValueError(
            "Invalid role session name: expected 2-64 characters from [A-Za-z0-9_+=,.@-]"
        )

    external_id = external_id or None
    if external_id is not None:
        validate_external_id(external_id)

    key = (role_arn, session_name, external_id)
    with _cache_lock:
        credentials = _cache.get(key)
        if credentials is None:
            credentials = _build_credentials(role_arn, session_name, external_id)
            _cache[key] = credentials
            while len(_cache) > _MAX_CACHED_ROLES:
                _cache.popitem(last=False)
        else:
            _cache.move_to_end(key)

    try:
        frozen = credentials.get_frozen_credentials()
    except (ClientError, BotoCoreError) as exc:
        if isinstance(exc, ClientError):
            error_code = exc.response.get("Error", {}).get("Code", "Unknown")
        else:
            error_code = type(exc).__name__
        log.warning(
            "sts:AssumeRole failed", role_arn=role_arn, sts_error_code=error_code
        )
        raise AssumeRoleError(
            f"Failed to assume IAM role {role_arn}: {error_code}",
            role_arn=role_arn,
            error_code=error_code,
        ) from exc
    return AssumedRoleCredentials(frozen.access_key, frozen.secret_key, frozen.token)


def clear_credentials_cache() -> None:
    with _cache_lock:
        _cache.clear()
