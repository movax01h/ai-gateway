# pylint: disable=direct-environment-variable-reference
import asyncio
import base64
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import patch

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from jose import jwk, jws, jwt

from duo_workflow_service.interceptors import trusted_restricted_issuers as tri
from duo_workflow_service.interceptors.trusted_restricted_issuers import (
    TRUSTED_RESTRICTED_ISSUERS_ENV,
    IssuerKeyCache,
    TrustedRestrictedIssuerVerifier,
    parse_trusted_restricted_issuers,
)

SERVICE_NAME = "gitlab-duo-workflow-service"
TENANT_A = "https://tenant-a.gitlab-dedicated.com"
TENANT_B = "https://tenant-b.gitlab-dedicated.com"
UID_A = "aaaaaaaa-1111-4111-8111-111111111111"
UID_B = "22222222-2222-4222-8222-222222222222"
CONFIG = json.dumps(
    {
        TENANT_A: {"uid": UID_A, "flows": ["bl_security"]},
        TENANT_B: {"uid": UID_B, "flows": ["bl_security"]},
    }
)
ENV = {
    "CLOUD_CONNECTOR_SERVICE_NAME": SERVICE_NAME,
    TRUSTED_RESTRICTED_ISSUERS_ENV: CONFIG,
}


def _pem() -> str:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()


KEY_A, KEY_B = _pem(), _pem()


def _jwk(pem: str, kid: str) -> dict:
    public = jwk.RSAKey(algorithm="RS256", key=pem).public_key().to_dict()
    return {**public, "kid": kid, "use": "sig", "alg": "RS256"}


JWKS = {
    TENANT_A + tri._KEYS_PATH: {"keys": [_jwk(KEY_A, "kid-a")]},
    TENANT_B + tri._KEYS_PATH: {"keys": [_jwk(KEY_B, "kid-b")]},
}


class FakeResponse:
    def __init__(self, status_code: int = 200, body: bytes = b"", headers=None):
        self.status_code = status_code
        self.body = body
        self.headers = headers or {}


class FakeTransport(httpx.AsyncBaseTransport):
    """Serves the JWKS above; per-URL overrides simulate failures."""

    def __init__(self, overrides: Optional[dict] = None):
        self.overrides = overrides or {}
        self.calls: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        url = str(request.url)
        override = self.overrides.get(url)
        if isinstance(override, Exception):
            raise override
        if callable(override):
            return await override(request)
        if isinstance(override, FakeResponse):
            return httpx.Response(
                override.status_code, headers=override.headers, content=override.body
            )
        if url not in JWKS:
            return httpx.Response(404)
        return httpx.Response(200, content=json.dumps(JWKS[url]).encode())


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _verifier(transport: FakeTransport, clock: Optional[Clock] = None):
    clock = clock or Clock()
    return TrustedRestrictedIssuerVerifier(
        lambda url: IssuerKeyCache(url, transport=transport, clock=clock)
    )


@pytest.fixture(autouse=True)
def _env():
    with patch.dict(os.environ, ENV):
        yield


# Config


def test_parses_trusted_issuers():
    issuers = parse_trusted_restricted_issuers(
        json.dumps({TENANT_A + "/": {"uid": UID_A, "flows": ["bl_security"]}})
    )
    assert list(issuers) == [TENANT_A]
    assert issuers[TENANT_A].uid == UID_A
    assert issuers[TENANT_A].flows == frozenset({"bl_security"})
    assert issuers[TENANT_A].keys_url == TENANT_A + "/-/cloud_connector/keys"


def test_accepts_gitlab_private_test_tenants():
    url = "https://dapcurrent.gitlab-private.org"
    raw = json.dumps({url: {"uid": UID_A, "flows": ["bl_security"]}})

    assert list(parse_trusted_restricted_issuers(raw)) == [url]


@pytest.mark.parametrize(
    "url",
    [
        "http://tenant-a.gitlab-dedicated.com",
        "https://tenant-a.gitlab-dedicated.com/gitlab",
        "https://tenant-a.gitlab-dedicated.com?x=1",
        "https://tenant-a.gitlab-dedicated.com#x",
        "https://tenant-a.gitlab-dedicated.com:8443",
        "https://tenant-a.gitlab-dedicated.com:99999",
        "https://user@tenant-a.gitlab-dedicated.com",
        "https://gitlab.com",
        "https://customers.gitlab.com",
        "https://gitlab-dedicated.com",
        "https://a.b.gitlab-dedicated.com",
        "https://tenant.gitlab-dedicated.com.evil.example",
        "https://gitlab.example.com",
        " https://tenant-a.gitlab-dedicated.com",
        "https://gitlab-private.org",
        "https://a.b.gitlab-private.org",
        "https://tenant.gitlab-private.org.evil.example",
    ],
)
def test_rejects_issuer_urls(url):
    raw = json.dumps({url: {"uid": UID_A, "flows": ["bl_security"]}})
    assert not parse_trusted_restricted_issuers(raw)


@pytest.mark.parametrize(
    "value",
    [
        {"uid": "not-a-uuid", "flows": ["bl_security"]},
        {"uid": UID_A.upper(), "flows": ["bl_security"]},
        {"uid": UID_A, "flows": []},
        {"uid": UID_A, "flows": "bl_security"},
        {"uid": UID_A, "flows": [""]},
        {"uid": UID_A},
        {"uid": UID_A, "flows": ["bl_security"], "realm": "saas"},
        ["bl_security"],
    ],
)
def test_rejects_malformed_entries_only(value):
    raw = json.dumps({TENANT_A: value, TENANT_B: {"uid": UID_B, "flows": ["x"]}})
    assert list(parse_trusted_restricted_issuers(raw)) == [TENANT_B]


@pytest.mark.parametrize("raw", ["", "not json", "[]", '"x"'])
def test_unusable_config_trusts_nothing(raw):
    assert parse_trusted_restricted_issuers(raw) == {}


@pytest.mark.parametrize(
    "raw", ["", json.dumps({TENANT_B: {"uid": UID_B, "flows": ["x"]}})]
)
def test_parsed_issuers_are_read_only(raw):
    issuers = parse_trusted_restricted_issuers(raw)

    with pytest.raises(TypeError):
        issuers[TENANT_A] = None


def test_two_spellings_of_one_issuer_trust_neither():
    raw = json.dumps(
        {
            TENANT_A: {"uid": UID_A, "flows": ["bl_security"]},
            TENANT_A + "/": {"uid": UID_B, "flows": ["bl_security"]},
        }
    )
    assert not parse_trusted_restricted_issuers(raw)


def test_third_spelling_of_duplicated_issuer_stays_rejected():
    raw = json.dumps(
        {
            TENANT_A: {"uid": UID_A, "flows": ["bl_security"]},
            TENANT_A + "/": {"uid": UID_A, "flows": ["bl_security"]},
            "https://TENANT-A.gitlab-dedicated.com": {
                "uid": UID_A,
                "flows": ["bl_security"],
            },
        }
    )
    assert not parse_trusted_restricted_issuers(raw)


def test_uppercase_host_is_lowercased():
    raw = json.dumps(
        {"https://TENANT-A.Gitlab-Dedicated.com": {"uid": UID_A, "flows": ["x"]}}
    )
    assert list(parse_trusted_restricted_issuers(raw)) == [TENANT_A]


@pytest.mark.parametrize(
    "url,value,reason",
    [
        ("http://tenant-a.gitlab-dedicated.com", None, "not https"),
        ("https://gitlab.com", None, "host not allowed"),
        ("https://tenant-a.gitlab-dedicated.com/gitlab", None, "bad shape"),
        (TENANT_A, ["bl_security"], "bad shape"),
        (TENANT_A, {"uid": "not-a-uuid", "flows": ["x"]}, "uid not a canonical UUID"),
        (TENANT_A, {"uid": UID_A, "flows": []}, "empty flows"),
    ],
)
def test_rejected_entry_logs_reason(url, value, reason):
    value = value if value is not None else {"uid": UID_A, "flows": ["x"]}
    with patch.object(tri, "log") as log:
        assert not parse_trusted_restricted_issuers(json.dumps({url: value}))
    log.error.assert_called_once_with(
        "Rejected trusted restricted-flow issuer entry", issuer=url, reason=reason
    )


def test_duplicate_spelling_logs_reason():
    raw = json.dumps(
        {
            TENANT_A: {"uid": UID_A, "flows": ["x"]},
            TENANT_A + "/": {"uid": UID_A, "flows": ["x"]},
        }
    )
    with patch.object(tri, "log") as log:
        parse_trusted_restricted_issuers(raw)
    log.error.assert_called_once_with(
        "Rejected trusted restricted-flow issuer entry",
        issuer=TENANT_A + "/",
        reason="duplicate spelling",
    )


def test_repeated_issuer_trusts_neither_and_logs_reason():
    entry = json.dumps({"uid": UID_A, "flows": ["x"]})
    raw = f'{{"{TENANT_A}": {entry}, "{TENANT_B}": {entry}, "{TENANT_A}": {entry}}}'
    with patch.object(tri, "log") as log:
        assert list(parse_trusted_restricted_issuers(raw)) == [TENANT_B]
    log.error.assert_called_once_with(
        "Rejected trusted restricted-flow issuer entry",
        issuer=TENANT_A,
        reason="duplicate spelling",
    )


def test_entry_repeating_a_key_is_rejected():
    raw = f'{{"{TENANT_A}": {{"uid": "{UID_B}", "uid": "{UID_A}", "flows": ["x"]}}}}'
    with patch.object(tri, "log") as log:
        assert not parse_trusted_restricted_issuers(raw)
    log.error.assert_called_once_with(
        "Rejected trusted restricted-flow issuer entry",
        issuer=TENANT_A,
        reason="bad shape",
    )


# Key fetching


def test_fetches_only_the_keys_endpoint_without_redirects():
    transport = FakeTransport()
    cache = IssuerKeyCache(
        TENANT_A + tri._KEYS_PATH, transport=transport, clock=Clock()
    )

    assert cache.key_for("kid-a")["kid"] == "kid-a"
    assert [str(r.url) for r in transport.calls] == [
        "https://tenant-a.gitlab-dedicated.com/-/cloud_connector/keys"
    ]


def test_does_not_follow_redirects():
    url = TENANT_A + tri._KEYS_PATH
    transport = FakeTransport(
        {url: FakeResponse(302, headers={"Location": TENANT_B + tri._KEYS_PATH})}
    )
    cache = IssuerKeyCache(url, transport=transport, clock=Clock())

    assert cache.key_for("kid-b") is None
    assert len(transport.calls) == 1


def _key_set(*keys: object) -> bytes:
    return json.dumps({"keys": list(keys)}).encode()


VALID_KEY_SET = _key_set(_jwk(KEY_A, "kid-a"))


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(FakeResponse(302), id="redirect"),
        pytest.param(FakeResponse(302, VALID_KEY_SET), id="redirect-with-valid-keys"),
        pytest.param(
            FakeResponse(
                200,
                b'{"keys": '
                + b" " * tri._MAX_KEYS_RESPONSE_BYTES
                + json.dumps([_jwk(KEY_A, "kid-a")]).encode()
                + b"}",
            ),
            id="valid-keys-padded-past-size-cap",
        ),
        pytest.param(
            FakeResponse(
                200,
                _key_set(
                    _jwk(KEY_A, "kid-a"),
                    *(_jwk(KEY_A, f"kid-{i}") for i in range(tri._MAX_KEYS)),
                ),
            ),
            id="too-many-keys",
        ),
        pytest.param(
            FakeResponse(
                200,
                _key_set(
                    {
                        "kid": "kid-a",
                        "kty": "oct",
                        "k": "c2VjcmV0LXNlY3JldC1zZWNyZXQ",
                        "alg": "RS256",
                        "use": "sig",
                    }
                ),
            ),
            id="oct-key-with-k",
        ),
        pytest.param(
            FakeResponse(200, _key_set(_jwk(KEY_A, "kid-a"), _jwk(KEY_B, "kid-a"))),
            id="duplicate-kid",
        ),
        pytest.param(FakeResponse(200, _key_set("kid-a")), id="key-not-an-object"),
        pytest.param(
            FakeResponse(
                200, _key_set({**_jwk(KEY_A, "kid-a"), "d": "AQAB", "p": "AQAB"})
            ),
            id="key-does-not-load",
        ),
        pytest.param(FakeResponse(500), id="server-error"),
        pytest.param(
            FakeResponse(200, b" " * (tri._MAX_KEYS_RESPONSE_BYTES + 1)), id="oversized"
        ),
        pytest.param(FakeResponse(200, b"not json"), id="not-json"),
        pytest.param(FakeResponse(200, b'{"keys": []}'), id="no-keys"),
        pytest.param(
            FakeResponse(
                200, json.dumps({"keys": [{"kid": "x", "kty": "oct"}]}).encode()
            ),
            id="not-rsa",
        ),
        pytest.param(httpx.ReadTimeout("slow"), id="timeout"),
    ],
)
def test_unusable_key_responses_fail_closed(response):
    url = TENANT_A + tri._KEYS_PATH
    cache = IssuerKeyCache(url, transport=FakeTransport({url: response}), clock=Clock())
    assert cache.key_for("kid-a") is None


def test_accepts_up_to_max_keys():
    url = TENANT_A + tri._KEYS_PATH
    body = _key_set(
        _jwk(KEY_A, "kid-a"),
        *(_jwk(KEY_A, f"kid-{i}") for i in range(tri._MAX_KEYS - 1)),
    )
    transport = FakeTransport({url: FakeResponse(200, body)})
    cache = IssuerKeyCache(url, transport=transport, clock=Clock())
    assert cache.key_for("kid-a")


def test_slow_drip_key_response_is_cut_off_at_the_overall_deadline():
    url = TENANT_A + tri._KEYS_PATH

    async def drip():
        # A valid key set, one byte every 10ms for about 3 seconds: each byte comes well within any read timeout.
        yield b'{"keys": '
        for _ in range(300):
            await asyncio.sleep(0.01)
            yield b" "
        yield json.dumps([_jwk(KEY_A, "kid-a")]).encode() + b"}"

    async def slow(_request):
        return httpx.Response(200, content=drip())

    cache = IssuerKeyCache(url, transport=FakeTransport({url: slow}), clock=Clock())
    started = time.monotonic()
    with patch.object(tri, "_KEYS_FETCH_TOTAL_SECONDS", 0.2):
        assert cache.key_for("kid-a") is None
    assert time.monotonic() - started < 1.0


def test_slow_key_response_headers_are_cut_off_at_the_overall_deadline():
    url = TENANT_A + tri._KEYS_PATH

    async def hang(_request):
        await asyncio.sleep(30)

    cache = IssuerKeyCache(url, transport=FakeTransport({url: hang}), clock=Clock())
    started = time.monotonic()
    with patch.object(tri, "_KEYS_FETCH_TOTAL_SECONDS", 0.2):
        assert cache.key_for("kid-a") is None
    assert time.monotonic() - started < 1.0


def test_cached_kid_does_not_wait_for_an_in_flight_fetch():
    url = TENANT_A + tri._KEYS_PATH
    transport = FakeTransport()
    clock = Clock()
    cache = IssuerKeyCache(url, transport=transport, clock=clock)
    assert cache.key_for("kid-a")

    fetching, release = threading.Event(), threading.Event()

    async def blocked(_request):
        fetching.set()
        while not release.is_set():
            await asyncio.sleep(0.01)
        return httpx.Response(200, content=_key_set(_jwk(KEY_A, "kid-new")))

    transport.overrides[url] = blocked
    clock.now += tri._REFETCH_INTERVAL_SECONDS
    with ThreadPoolExecutor(max_workers=3) as pool:
        try:
            refetch = pool.submit(cache.key_for, "kid-new")
            assert fetching.wait(5)

            # Served from the cache while the fetch is blocked.
            assert pool.submit(cache.key_for, "kid-a").result(1)
            waiter = pool.submit(cache.key_for, "kid-new")
            time.sleep(0.05)
            assert not waiter.done()  # a miss waits for the fetch in flight
        finally:
            release.set()
        assert refetch.result(5)["kid"] == "kid-new"
        assert waiter.result(5)["kid"] == "kid-new"
    assert len(transport.calls) == 2  # the waiter started no second fetch


def test_unknown_kid_refetches_once_per_interval():
    transport = FakeTransport()
    clock = Clock()
    cache = IssuerKeyCache(TENANT_A + tri._KEYS_PATH, transport=transport, clock=clock)
    assert cache.key_for("kid-a")
    assert len(transport.calls) == 1

    assert cache.key_for("kid-new") is None
    assert cache.key_for("kid-other") is None
    assert len(transport.calls) == 1  # inside the interval: no refetch

    clock.now += tri._REFETCH_INTERVAL_SECONDS
    assert cache.key_for("kid-new") is None
    assert len(transport.calls) == 2
    assert cache.key_for("kid-a")  # known kids keep working meanwhile


def test_keys_past_max_age_are_not_used_when_refetch_fails():
    url = TENANT_A + tri._KEYS_PATH
    transport = FakeTransport()
    clock = Clock()
    cache = IssuerKeyCache(url, transport=transport, clock=clock)
    assert cache.key_for("kid-a")

    transport.overrides[url] = FakeResponse(503)
    clock.now += tri._KEY_MAX_AGE_SECONDS
    assert cache.key_for("kid-a") is None
    assert len(transport.calls) == 2


# Restricted-flow proofs


def _now() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def _proof(
    iss: str = TENANT_A,
    key: str = KEY_A,
    kid: str = "kid-a",
    lifetime: int = 3600,
    **overrides,
) -> str:
    now = int(datetime.now(timezone.utc).timestamp())
    claims = {
        "iss": iss,
        "sub": UID_A,
        "aud": SERVICE_NAME,
        "typ": "restricted_flow_proof",
        "flow_config_id": "bl_security",
        "workflow_id": "123",
        "iat": now,
        "nbf": now,
        "exp": now + lifetime,
        "jti": "proof-1",
        **overrides,
    }
    claims = {k: v for k, v in claims.items() if v is not None}
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": kid})


def test_verifies_proof_against_its_issuer_keys():
    proof = _verifier(FakeTransport()).verify_proof(_proof())

    assert proof is not None
    assert proof.issuer.issuer == TENANT_A
    assert proof.issuer.uid == UID_A
    assert proof.flow_config_id == "bl_security"
    assert proof.workflow_id == "123"


@pytest.mark.parametrize(
    "token",
    [
        pytest.param(
            _proof(iss="https://gitlab.com"), id="tenant-key-spoofing-gitlab-com"
        ),
        pytest.param(
            _proof(iss="https://customers.gitlab.com"),
            id="tenant-key-spoofing-customers-dot",
        ),
        pytest.param(_proof(iss=SERVICE_NAME), id="tenant-key-spoofing-dws"),
        pytest.param(
            _proof(iss="https://other.gitlab-dedicated.com"), id="untrusted-issuer"
        ),
        pytest.param(_proof(key=KEY_B, kid="kid-b"), id="kid-of-other-issuer"),
        pytest.param(_proof(key=KEY_B, kid="kid-a"), id="wrong-key-for-kid"),
        pytest.param(_proof(kid="kid-unknown"), id="unknown-kid"),
        pytest.param(_proof(aud="gitlab-ai-gateway"), id="wrong-aud"),
        pytest.param(
            _proof(aud=[SERVICE_NAME, "gitlab-ai-gateway"]), id="aud-not-only-dws"
        ),
        pytest.param(_proof(typ="access_token"), id="wrong-typ"),
        pytest.param(_proof(typ=None), id="no-typ"),
        pytest.param(
            _proof(typ=None, token_use="restricted_flow_proof"), id="token-use-not-typ"
        ),
        pytest.param(_proof(sub=""), id="empty-sub"),
        pytest.param(_proof(lifetime=3601), id="lifetime-over-an-hour"),
        pytest.param(
            _proof(lifetime=-tri._PROOF_CLOCK_LEEWAY_SECONDS - 10), id="expired"
        ),
        pytest.param(
            _proof(nbf=int(datetime.now(timezone.utc).timestamp()) + 600),
            id="not-yet-valid",
        ),
        pytest.param(_proof(nbf=None), id="no-nbf"),
        pytest.param(_proof(jti=None), id="no-jti"),
        pytest.param(_proof(flow_config_id=None), id="no-flow-claim"),
        pytest.param(_proof(flow_config_id=""), id="empty-flow-claim"),
        pytest.param(_proof(flow_config_id=5), id="non-string-flow-claim"),
        pytest.param(_proof(iat=_now() + 0.5), id="non-int-iat"),
        pytest.param(_proof(exp=None), id="no-exp"),
        pytest.param(_proof(exp=str(_now() + 600)), id="numeric-string-exp"),
        pytest.param(_proof(iat=[_now()]), id="list-iat"),
        pytest.param(_proof(nbf=[_now()]), id="list-nbf"),
        pytest.param(_proof(nbf=True), id="bool-nbf"),
        pytest.param("not-a-jwt", id="garbage"),
    ],
)
def test_rejects_proofs(token):
    assert _verifier(FakeTransport()).verify_proof(token) is None


def test_accepts_aud_list_of_only_this_service():
    # Rails sends `aud` as a list.
    assert _verifier(FakeTransport()).verify_proof(_proof(aud=[SERVICE_NAME]))


def test_does_not_compare_sub_to_issuer_uid():
    # The caller binds the issuer's uid to its signed main token instead.
    proof = _verifier(FakeTransport()).verify_proof(_proof(sub=UID_B))
    assert proof is not None
    assert proof.issuer.uid == UID_A


def test_rejects_proof_issued_in_the_future():
    now = _now()
    iat = now + 10 * 365 * 24 * 3600
    token = _proof(iat=iat, nbf=now, exp=iat + 3600)
    assert _verifier(FakeTransport()).verify_proof(token) is None


@pytest.mark.parametrize(
    ("claim", "offset", "accepted"),
    [
        ("nbf", 10, True),
        ("nbf", tri._PROOF_CLOCK_LEEWAY_SECONDS + 10, False),
        ("exp", -10, True),
        ("exp", -tri._PROOF_CLOCK_LEEWAY_SECONDS - 10, False),
    ],
)
def test_nbf_and_exp_get_the_clock_leeway(claim, offset, accepted):
    proof = _proof(**{claim: _now() + offset})
    assert (_verifier(FakeTransport()).verify_proof(proof) is not None) is accepted


def test_rejects_overflowing_iat():
    claims = json.dumps({**jwt.get_unverified_claims(_proof()), "iat": "IAT"})
    payload = claims.replace('"IAT"', "1e400").encode()
    token = jws.sign(payload, KEY_A, headers={"kid": "kid-a"}, algorithm="RS256")
    assert _verifier(FakeTransport()).verify_proof(token) is None


def test_rejects_proof_expiring_past_max_lifetime_from_now():
    now = _now()
    token = _proof(iat=now, nbf=now, exp=now + tri._PROOF_MAX_LIFETIME_SECONDS + 120)
    assert _verifier(FakeTransport()).verify_proof(token) is None


def test_accepts_iat_within_clock_leeway():
    now = _now()
    token = _proof(iat=now + 30, nbf=now, exp=now + 600)
    assert _verifier(FakeTransport()).verify_proof(token) is not None


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"typ": "access_token"}, "typ"),
        ({"sub": ""}, "sub"),
        ({"iat": _now() + 0.5}, "iat/exp/nbf type"),
        ({"lifetime": 3601}, "lifetime"),
        ({"iat": _now() + 600, "exp": _now() + 1200}, "future iat"),
        ({"flow_config_id": ""}, "flow_config_id"),
        ({"flow_config_id": "other_flow"}, "flow not enabled for issuer"),
        ({"workflow_id": None}, "workflow_id"),
        ({"workflow_id": ""}, "workflow_id"),
        ({"workflow_id": 123}, "workflow_id"),
    ],
)
def test_unexpected_claims_log_reason(overrides, reason):
    with patch.object(tri, "log") as log:
        assert _verifier(FakeTransport()).verify_proof(_proof(**overrides)) is None
    log.warning.assert_called_once_with(
        "Trusted restricted-flow issuer proof has unexpected claims",
        issuer=TENANT_A,
        reason=reason,
    )


def test_unexpected_aud_claim_logs_reason():
    with patch.object(tri.jwt, "decode", return_value={"aud": "other"}):
        with patch.object(tri, "log") as log:
            assert _verifier(FakeTransport()).verify_proof(_proof()) is None
    log.warning.assert_called_once_with(
        "Trusted restricted-flow issuer proof has unexpected claims",
        issuer=TENANT_A,
        reason="aud",
    )


def test_two_proofs_share_one_key_fetch():
    transport = FakeTransport()
    verifier = _verifier(transport)
    assert verifier.verify_proof(_proof())
    assert verifier.verify_proof(_proof(jti="proof-2"))
    assert len(transport.calls) == 1


def test_rejects_kid_of_other_issuer_with_warm_cache():
    verifier = _verifier(FakeTransport())
    assert verifier.verify_proof(_proof(iss=TENANT_B, key=KEY_B, kid="kid-b"))
    assert verifier.verify_proof(_proof(key=KEY_B, kid="kid-b")) is None


def test_rejects_proof_of_removed_issuer():
    transport = FakeTransport()
    verifier = _verifier(transport)
    assert verifier.verify_proof(_proof()) is not None

    with patch.dict(
        os.environ,
        {
            TRUSTED_RESTRICTED_ISSUERS_ENV: json.dumps(
                {TENANT_B: {"uid": UID_B, "flows": ["bl_security"]}}
            )
        },
    ):
        assert verifier.verify_proof(_proof()) is None


def test_rejects_hs256_proof():
    token = jwt.encode(
        {"iss": TENANT_A, "sub": UID_A},
        "secret",
        algorithm="HS256",
        headers={"kid": "kid-a"},
    )
    assert _verifier(FakeTransport()).verify_proof(token) is None


def test_rejects_hs256_proof_even_with_a_key_for_it():
    token = jwt.encode(
        {"iss": TENANT_A, "sub": UID_A},
        "secret",
        algorithm="HS256",
        headers={"kid": "kid-a"},
    )
    verifier = _verifier(FakeTransport())
    with patch.object(verifier, "_proof_key", return_value=_jwk(KEY_A, "kid-a")):
        assert verifier.verify_proof(token) is None


def test_fetch_failure_for_one_issuer_leaves_others_working():
    transport = FakeTransport({TENANT_A + tri._KEYS_PATH: httpx.ConnectError("down")})
    verifier = _verifier(transport)

    assert verifier.verify_proof(_proof()) is None
    assert (
        verifier.verify_proof(_proof(iss=TENANT_B, sub=UID_B, key=KEY_B, kid="kid-b"))
        is not None
    )


def test_no_config_trusts_nothing_and_fetches_nothing():
    transport = FakeTransport()
    with patch.dict(os.environ, {TRUSTED_RESTRICTED_ISSUERS_ENV: ""}):
        assert TrustedRestrictedIssuerVerifier.claimed_issuer(_proof()) is None
        assert _verifier(transport).verify_proof(_proof()) is None
    assert not transport.calls


def test_proof_key_rejects_unreadable_header():
    real = _proof().split(".")
    header = base64.urlsafe_b64encode(b"not json").rstrip(b"=").decode()
    token = ".".join([header, real[1], real[2]])
    issuer = tri.trusted_restricted_issuers()[TENANT_A]
    assert _verifier(FakeTransport())._proof_key(token, issuer) is None
