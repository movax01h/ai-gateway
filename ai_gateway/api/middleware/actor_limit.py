import math
from typing import NamedTuple

import structlog
from fastapi import status
from gitlab_cloud_connector import UserClaims
from prometheus_client import Counter, Histogram
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from ai_gateway.config import ConfigCodeSuggestions
from lib.context import StarletteUser
from lib.flow_enablement import SAAS_REALM
from lib.jwt import root_namespace_id_from_claims_extra, trusted_instance_id

__all__ = ["ActorLimitMiddleware"]

log = structlog.stdlib.get_logger("code_suggestions_actor_limit")

_ACTOR_LIMIT_COUNTER = Counter(
    "code_suggestions_actor_limit_total",
    "Code suggestion requests checked against the per-actor in-flight limits",
    ["mode", "outcome", "bucket"],
)
_ACTOR_IN_FLIGHT_HISTOGRAM = Histogram(
    "code_suggestions_actor_in_flight",
    "In-flight code suggestion requests of one actor when a request is admitted",
    ["bucket"],
    buckets=(1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 60),
)
_SECURITY_EVENT_TYPE = "security.rate_limit.block"
_CONTROL_NAME = "code_suggestions_actor_limit"


class _Bucket(NamedTuple):
    scope: str
    actor: str
    limit: int

    @property
    def key(self) -> tuple[str, str]:
        return (self.scope, self.actor)


class ActorLimitMiddleware:
    """Caps the in-flight code suggestion requests of one actor on this process.

    Every request counts against its user, keyed within the instance its token vouches for. A
    request from a self-managed instance also counts against that instance, with a limit scaled by
    the token's Duo seat count, and a GitLab.com request against its root namespace. The request is
    admitted only when every counter it belongs to is under its limit. ``off`` disables the check,
    ``log`` serves every request but records the ones that would be rejected, and ``enforce``
    answers 429. A request without a user id counts only against its instance or namespace.
    """

    def __init__(
        self,
        app: ASGIApp,
        limits: ConfigCodeSuggestions,
        path_prefixes: tuple[str, ...],
    ):
        self.app = app
        self.limits = limits
        self.path_prefixes = path_prefixes
        self._in_flight: dict[tuple[str, str], int] = {}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        user = scope.get("user")
        mode = self.limits.actor_limit_mode
        if (
            mode == "off"
            or scope["type"] != "http"
            or not scope["path"].startswith(self.path_prefixes)
            or not isinstance(user, StarletteUser)
            or not user.is_authenticated
        ):
            await self.app(scope, receive, send)
            return

        buckets = self._buckets(user)
        if not buckets:
            await self.app(scope, receive, send)
            return

        exceeded = [b for b in buckets if self._count(b) >= b.limit]
        if not exceeded:
            _ACTOR_LIMIT_COUNTER.labels(mode=mode, outcome="allowed", bucket="").inc()
        elif mode == "enforce":
            self._record_exceeded(
                user, exceeded, scope["path"], outcome="failure", result="rejected"
            )
            await self._reject(scope, receive, send)
            return
        else:
            self._record_exceeded(
                user, exceeded, scope["path"], outcome="success", result="would_reject"
            )

        for bucket in buckets:
            in_flight = self._count(bucket) + 1
            _ACTOR_IN_FLIGHT_HISTOGRAM.labels(bucket=bucket.scope).observe(in_flight)
            self._in_flight[bucket.key] = in_flight
        try:
            await self.app(scope, receive, send)
        finally:
            for bucket in buckets:
                self._release(bucket)

    def _buckets(self, user: StarletteUser) -> list[_Bucket]:
        claims = user.claims
        buckets = []
        if user.global_user_id:
            tenant = self._tenant(claims)
            actor = f"{tenant}:{user.global_user_id}" if tenant else user.global_user_id
            buckets.append(_Bucket("user", actor, self.limits.user_limit))
        if claims is None:
            return buckets
        if claims.gitlab_realm == SAAS_REALM:
            namespace_id = root_namespace_id_from_claims_extra(claims.extra)
            if namespace_id is not None:
                buckets.append(
                    _Bucket("namespace", str(namespace_id), self.limits.namespace_limit)
                )
        elif instance_id := trusted_instance_id(claims):
            buckets.append(
                _Bucket("instance", instance_id, self._instance_limit(claims))
            )
        return buckets

    @staticmethod
    def _tenant(claims: UserClaims | None) -> str:
        if claims is None or claims.gitlab_realm == SAAS_REALM:
            return ""
        return trusted_instance_id(claims) or claims.gitlab_realm or "self-managed"

    def _instance_limit(self, claims: UserClaims) -> int:
        seats = str(claims.duo_seat_count)
        per_seats = (
            math.ceil(int(seats) * self.limits.instance_limit_per_seat)
            if seats.isdecimal()
            else 0
        )
        return max(self.limits.instance_limit, per_seats)

    def _record_exceeded(
        self,
        user: StarletteUser,
        exceeded: list[_Bucket],
        path: str,
        *,
        outcome: str,
        result: str,
    ) -> None:
        for bucket in exceeded:
            self._log_limit_exceeded(user, bucket, path, outcome=outcome)
            _ACTOR_LIMIT_COUNTER.labels(
                mode=self.limits.actor_limit_mode, outcome=result, bucket=bucket.scope
            ).inc()

    def _log_limit_exceeded(
        self, user: StarletteUser, exceeded: _Bucket, path: str, outcome: str
    ) -> None:
        log.warning(
            "Code suggestions actor limit exceeded",
            event_type=_SECURITY_EVENT_TYPE,
            author_id=user.global_user_id,
            entity_id=exceeded.actor,
            bucket=exceeded.scope,
            in_flight=self._count(exceeded),
            path=path,
            details={"outcome": outcome, "reason": "rate_limited"},
            gitlab={
                "control_name": _CONTROL_NAME,
                "control_type": "rate_limit",
                "threshold": exceeded.limit,
                "mode": self.limits.actor_limit_mode,
            },
        )

    def _count(self, bucket: _Bucket) -> int:
        return self._in_flight.get(bucket.key, 0)

    def _release(self, bucket: _Bucket) -> None:
        remaining = self._in_flight[bucket.key] - 1
        if remaining:
            self._in_flight[bucket.key] = remaining
        else:
            del self._in_flight[bucket.key]

    @staticmethod
    async def _reject(scope: Scope, receive: Receive, send: Send) -> None:
        response = JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "detail": "Too many concurrent code suggestion requests",
                "reason": "rate_limited",
            },
            headers={
                "retry-after": "1",
                "connection": "close",
                "x-gitlab-rate-limit": _CONTROL_NAME,
            },
        )
        await response(scope, receive, send)
