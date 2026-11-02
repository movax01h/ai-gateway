# pylint: disable=direct-environment-variable-reference
"""Restricted-flow proofs signed by named GitLab Dedicated instances with their own keys.

Each issuer has its own key cache, and a proof's `kid` must belong to the issuer it claims; these keys are kept out of
the shared CompositeProvider pool so they never authenticate requests. A failing issuer affects only itself.

verify_proof checks a restricted-flow proof: a short-lived JWT, typed restricted_flow_proof, that names one flow and
one workflow. A proof never authenticates a request; a caller may only use it to add restricted access to a request
that is already authenticated.
"""

import asyncio
import json
import os
import re
import threading
import time
import urllib.parse
import uuid
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Callable, Mapping, Optional, TypeGuard

import httpx
import structlog
from gitlab_cloud_connector import CloudConnectorConfig, CompositeProvider
from jose import JWTError, jwk, jwt
from jose.exceptions import JWKError

__all__ = [
    "TRUSTED_RESTRICTED_ISSUERS_ENV",
    "IssuerKeyCache",
    "RestrictedFlowProof",
    "TrustedIssuer",
    "TrustedRestrictedIssuerVerifier",
    "parse_trusted_restricted_issuers",
    "trusted_restricted_issuers",
]

log = structlog.stdlib.get_logger("trusted_restricted_issuers")

# JSON object: {"https://<tenant>.gitlab-dedicated.com": {"uid": "<instance uuid>", "flows": ["bl_security"]}}.
# `uid` is the instance ID that the caller's signed main token vouches for, as returned by
# lib.jwt.trusted_instance_id: the CustomersDot `gitlab_instance_uid`, or `sub` for a self-signed instance token.
# Read on each call.
TRUSTED_RESTRICTED_ISSUERS_ENV = "DUO_WORKFLOW_AUTH__TRUSTED_RESTRICTED_ISSUERS"

# Host names GitLab owns. Custom domains are not accepted.
TRUSTED_ISSUER_HOST_SUFFIXES = (".gitlab-dedicated.com",)
_HOST_LABEL = re.compile(r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$")

_KEYS_PATH = "/-/cloud_connector/keys"
_KEYS_FETCH_TOTAL_SECONDS = 10.0
_MAX_KEYS_RESPONSE_BYTES = 64 * 1024
_MAX_KEYS = 20
_KEY_MAX_AGE_SECONDS = 3600.0
_REFETCH_INTERVAL_SECONDS = 60.0
_ALGORITHM = CompositeProvider.RS256_ALGORITHM
_RESTRICTED_FLOW_PROOF_TYPE = "restricted_flow_proof"
_PROOF_MAX_LIFETIME_SECONDS = 3600
_PROOF_CLOCK_LEEWAY_SECONDS = 60


@dataclass(frozen=True)
class TrustedIssuer:
    issuer: str
    uid: str
    flows: frozenset[str]

    @property
    def keys_url(self) -> str:
        return self.issuer + _KEYS_PATH


@dataclass(frozen=True)
class RestrictedFlowProof:
    """A restricted-flow proof verified against its issuer's own keys."""

    issuer: TrustedIssuer
    flow_config_id: str
    workflow_id: str


def _canonical_issuer(url: object) -> tuple[Optional[str], str]:
    """The issuer as `https://<host>`, or None and the reason it is rejected."""
    if not isinstance(url, str) or url != url.strip():
        return None, "bad shape"
    try:
        parts = urllib.parse.urlsplit(url)
        port = parts.port
    except ValueError:
        return None, "bad shape"
    if parts.scheme != "https":
        return None, "not https"
    host = parts.hostname or ""
    if (
        parts.username is not None
        or parts.password is not None
        or port is not None
        or parts.path not in ("", "/")
        or parts.query
        or parts.fragment
        or parts.netloc.lower() != host
    ):
        return None, "bad shape"
    for suffix in TRUSTED_ISSUER_HOST_SUFFIXES:
        if host.endswith(suffix) and _HOST_LABEL.match(host[: -len(suffix)]):
            return f"https://{host}", ""
    return None, "host not allowed"


def _is_canonical_uuid(uid: object) -> bool:
    try:
        return isinstance(uid, str) and str(uuid.UUID(uid)) == uid
    except ValueError:
        return False


def _parse_entry(url: object, value: object) -> tuple[Optional[TrustedIssuer], str]:
    """The trusted issuer of one config entry, or None and the reason it is rejected."""
    issuer, reason = _canonical_issuer(url)
    if issuer is None:
        return None, reason
    if not isinstance(value, dict) or set(value) != {"uid", "flows"}:
        return None, "bad shape"
    uid, flows = value["uid"], value["flows"]
    if not _is_canonical_uuid(uid):
        return None, "uid not a canonical UUID"
    if not isinstance(flows, list):
        return None, "bad shape"
    if not flows or not all(isinstance(flow, str) and flow for flow in flows):
        return None, "empty flows"
    return TrustedIssuer(issuer=issuer, uid=uid, flows=frozenset(flows)), ""


class _RepeatedKeys(tuple):
    """The (key, value) pairs of a JSON object that repeats a key, kept so the repeat can be rejected."""


def _json_object(pairs: list[tuple[str, object]]) -> object:
    return dict(pairs) if len(dict(pairs)) == len(pairs) else _RepeatedKeys(pairs)


@lru_cache(maxsize=8)
def parse_trusted_restricted_issuers(raw: str) -> Mapping[str, TrustedIssuer]:
    """Parse the trusted issuer map; malformed entries are dropped and logged, never trusted."""
    if not raw.strip():
        return MappingProxyType({})
    try:
        data = json.loads(raw, object_pairs_hook=_json_object)
    except ValueError:
        log.error("Trusted restricted-flow issuers config is not valid JSON")
        return MappingProxyType({})
    if not isinstance(data, (dict, _RepeatedKeys)):
        log.error("Trusted restricted-flow issuers config is not a JSON object")
        return MappingProxyType({})

    issuers: dict[str, TrustedIssuer] = {}
    rejected: set[str] = set()
    for url, value in data.items() if isinstance(data, dict) else data:
        entry, reason = _parse_entry(url, value)
        if entry is None or entry.issuer in issuers or entry.issuer in rejected:
            log.error(
                "Rejected trusted restricted-flow issuer entry",
                issuer=url[:200],
                reason=reason or "duplicate spelling",
            )
            if entry is not None:
                # Two spellings, or a repeat, of one issuer: keep neither.
                rejected.add(entry.issuer)
                issuers.pop(entry.issuer, None)
            continue
        issuers[entry.issuer] = entry
    return MappingProxyType(issuers)


def trusted_restricted_issuers() -> Mapping[str, TrustedIssuer]:
    return parse_trusted_restricted_issuers(
        os.environ.get(TRUSTED_RESTRICTED_ISSUERS_ENV, "")
    )


class KeyFetchError(Exception):
    pass


def _parse_keys(body: bytes) -> dict[str, dict]:
    data = json.loads(body)
    keys = data.get("keys") if isinstance(data, dict) else None
    if not isinstance(keys, list) or not keys or len(keys) > _MAX_KEYS:
        raise KeyFetchError("unexpected key set shape")

    parsed: dict[str, dict] = {}
    for key in keys:
        if not isinstance(key, dict):
            raise KeyFetchError("key is not an object")
        kid = key.get("kid")
        if (
            not isinstance(kid, str)
            or not kid
            or kid in parsed
            or key.get("kty") != "RSA"
            or key.get("alg", _ALGORITHM) != _ALGORITHM
            or key.get("use", "sig") != "sig"
        ):
            raise KeyFetchError("unusable key")
        try:
            jwk.construct(key, _ALGORITHM)
        except JWKError as e:
            raise KeyFetchError("key does not load") from e
        parsed[kid] = key
    return parsed


class IssuerKeyCache:
    """The signing keys of one issuer, fetched from its Cloud Connector keys endpoint only.

    Keys older than _KEY_MAX_AGE_SECONDS are not used. A fetch happens at most once per _REFETCH_INTERVAL_SECONDS: on
    first use, after the keys expire, or when a token names a kid the cache doesn't know. One fetch runs at a time and
    outside the lock, so known kids are served meanwhile and other misses wait for it. A failed fetch leaves this issuer
    without keys until the next allowed fetch; it never affects other issuers.
    """

    def __init__(
        self,
        keys_url: str,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._keys_url = keys_url
        self._transport = transport
        self._clock = clock
        self._lock = threading.Lock()
        self._keys: dict[str, dict] = {}
        self._fetched_at: Optional[float] = None
        self._last_attempt: Optional[float] = None
        self._fetching: Optional[threading.Event] = None

    def key_for(self, kid: str) -> Optional[dict]:
        with self._lock:
            now = self._clock()
            key = self._fresh_key(kid, now)
            if key is not None:
                return key
            fetch = self._fetching is None and (
                self._last_attempt is None
                or now - self._last_attempt >= _REFETCH_INTERVAL_SECONDS
            )
            if fetch:
                self._last_attempt = now
                self._fetching = threading.Event()
            in_flight = self._fetching

        if fetch:
            self._refresh(now)
        elif in_flight is not None:
            in_flight.wait(_KEYS_FETCH_TOTAL_SECONDS)

        with self._lock:
            return self._fresh_key(kid, self._clock())

    def _fresh_key(self, kid: str, now: float) -> Optional[dict]:
        if self._fetched_at is None or now - self._fetched_at >= _KEY_MAX_AGE_SECONDS:
            return None
        return self._keys.get(kid)

    def _refresh(self, started_at: float) -> None:
        keys = None
        try:
            keys = self._fetch()
        except Exception as e:  # pylint: disable=broad-exception-caught
            log.warning(
                "Trusted restricted-flow issuer keys fetch failed",
                keys_url=self._keys_url,
                error=type(e).__name__,
            )
        finally:
            with self._lock:
                if keys is not None:
                    self._keys, self._fetched_at = keys, started_at
                if self._fetching is not None:
                    self._fetching.set()
                self._fetching = None

    def _fetch(self) -> dict[str, dict]:
        # Callers are sync (run off the event loop), so the fetch gets its own loop. Closing the loop does not wait
        # for a stuck DNS lookup thread.
        loop = asyncio.new_event_loop()
        try:
            return _parse_keys(loop.run_until_complete(self._fetch_body()))
        finally:
            loop.close()

    async def _fetch_body(self) -> bytes:
        # One deadline for the whole fetch: connect, headers and body.
        async with asyncio.timeout(_KEYS_FETCH_TOTAL_SECONDS):
            async with httpx.AsyncClient(
                transport=self._transport, follow_redirects=False
            ) as client:
                async with client.stream(
                    "GET", self._keys_url, headers={"Accept": "application/json"}
                ) as response:
                    if response.status_code != 200:
                        raise KeyFetchError(f"HTTP {response.status_code}")
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body += chunk
                        if len(body) > _MAX_KEYS_RESPONSE_BYTES:
                            raise KeyFetchError("response too large")
                    return bytes(body)


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _time_problem(claims: dict) -> str:
    """The name of the time check a decoded proof fails, or "" if it passes."""
    exp, iat, nbf = claims.get("exp"), claims.get("iat"), claims.get("nbf")
    if not (_is_int(exp) and _is_int(iat) and _is_int(nbf)):
        return "iat/exp/nbf type"
    now = int(time.time())
    if (
        exp - iat > _PROOF_MAX_LIFETIME_SECONDS
        or exp - now > _PROOF_MAX_LIFETIME_SECONDS + _PROOF_CLOCK_LEEWAY_SECONDS
    ):
        return "lifetime"
    if iat > now + _PROOF_CLOCK_LEEWAY_SECONDS:
        return "future iat"
    return ""


def _binding_problem(claims: dict, issuer: TrustedIssuer) -> str:
    """The name of the flow or workflow check a decoded proof fails, or "" if it passes."""
    flow_config_id = claims.get("flow_config_id")
    workflow_id = claims.get("workflow_id")
    if not isinstance(flow_config_id, str) or not flow_config_id:
        return "flow_config_id"
    if flow_config_id not in issuer.flows:
        return "flow not enabled for issuer"
    if not isinstance(workflow_id, str) or not workflow_id:
        return "workflow_id"
    return ""


def _unexpected_claim(claims: dict, service_name: str, issuer: TrustedIssuer) -> str:
    """The name of the first check a decoded proof's claims fail, or "" if they pass."""
    sub = claims.get("sub")
    if claims.get("aud") not in (service_name, [service_name]):
        return "aud"
    if claims.get("typ") != _RESTRICTED_FLOW_PROOF_TYPE:
        return "typ"
    if not isinstance(sub, str) or not sub:
        return "sub"
    return _time_problem(claims) or _binding_problem(claims, issuer)


class TrustedRestrictedIssuerVerifier:
    """Verifies a restricted-flow proof signed by a trusted issuer against that issuer's keys only."""

    def __init__(
        self, key_cache_factory: Callable[[str], IssuerKeyCache] = IssuerKeyCache
    ):
        self._key_cache_factory = key_cache_factory
        self._caches: dict[str, IssuerKeyCache] = {}
        self._lock = threading.Lock()

    @staticmethod
    def claimed_issuer(token: str) -> Optional[TrustedIssuer]:
        """The trusted issuer the token claims to be from, without verifying it."""
        issuers = trusted_restricted_issuers()
        if not issuers:
            return None
        try:
            iss = jwt.get_unverified_claims(token).get("iss")
        except JWTError:
            return None
        return issuers.get(iss) if isinstance(iss, str) else None

    def verify_proof(self, token: str) -> Optional[RestrictedFlowProof]:
        """Verify a restricted-flow proof against the keys of the trusted issuer it claims; None if it fails.

        The proof is audienced to this service only, typed as a restricted-flow proof, has a non-empty subject, and is
        valid now and for at most _PROOF_MAX_LIFETIME_SECONDS. It names a flow enabled for its issuer and a non-empty
        workflow ID. Malformed claims fail verification. The proof's subject is not compared to the issuer's `uid`: the
        caller compares `uid` with the instance ID its signed main token vouches for. The flow and workflow claims are
        returned for the caller to check against the request.
        """
        issuer = self.claimed_issuer(token)
        key = self._proof_key(token, issuer) if issuer else None
        if issuer is None or key is None:
            return None

        service_name = CloudConnectorConfig().service_name
        try:
            claims = jwt.decode(
                token,
                key,
                algorithms=[_ALGORITHM],
                audience=service_name,
                issuer=issuer.issuer,
                options={
                    "require_aud": True,
                    "require_iat": True,
                    "require_nbf": True,
                    "require_exp": True,
                    "require_sub": True,
                    "require_jti": True,
                    "leeway": _PROOF_CLOCK_LEEWAY_SECONDS,
                },
            )
        except (JWTError, JWKError, TypeError, ValueError, OverflowError) as e:
            # jose raises TypeError or OverflowError on some malformed numeric claims, such as a list or 1e400 `iat`.
            log.warning(
                "Trusted restricted-flow issuer proof rejected",
                issuer=issuer.issuer,
                error=type(e).__name__,
            )
            return None

        reason = _unexpected_claim(claims, service_name, issuer)
        if reason:
            log.warning(
                "Trusted restricted-flow issuer proof has unexpected claims",
                issuer=issuer.issuer,
                reason=reason,
            )
            return None
        return RestrictedFlowProof(
            issuer=issuer,
            flow_config_id=claims["flow_config_id"],
            workflow_id=claims["workflow_id"],
        )

    def _proof_key(self, token: str, issuer: TrustedIssuer) -> Optional[dict]:
        """This issuer's own key for the proof's kid, if it has one."""
        try:
            header = jwt.get_unverified_header(token)
        except JWTError:
            return None
        kid = header.get("kid")
        if header.get("alg") != _ALGORITHM or not isinstance(kid, str) or not kid:
            return None
        key = self._cache_for(issuer).key_for(kid)
        if key is None:
            log.warning(
                "Unknown kid for trusted restricted-flow issuer proof",
                issuer=issuer.issuer,
                kid=kid[:64],
            )
        return key

    def _cache_for(self, issuer: TrustedIssuer) -> IssuerKeyCache:
        with self._lock:
            cache = self._caches.get(issuer.issuer)
            if cache is None:
                cache = self._key_cache_factory(issuer.keys_url)
                self._caches[issuer.issuer] = cache
            return cache
