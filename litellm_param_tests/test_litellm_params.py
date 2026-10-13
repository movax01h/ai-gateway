"""Check that litellm accepts the parameters our code sends, for every litellm entry in models.yml.

A case fails when litellm raises, when the request goes somewhere unexpected, or when something tries to open a real
connection. A pass proves litellm accepts the parameters. It does not prove the provider does.
"""

import json
import time

import litellm
import pytest

from ai_gateway.model_selection import ModelSelectionConfig
from litellm_param_tests import cases, fakenet

pytestmark = pytest.mark.asyncio

ENTRIES = cases.load_entries()
_CASE_TIMEOUT_SECONDS = 30


def _matrix() -> list:
    params = []
    for entry in ENTRIES:
        for pset in cases.SETS_BY_KIND[entry.kind]:
            marks = []
            if entry.ident in cases.SKIPS:
                marks.append(
                    pytest.mark.skip(
                        reason=f"{entry.ident}: {cases.SKIPS[entry.ident]}"
                    )
                )
            elif (entry.provider, pset.name) in cases.SET_SKIPS:
                reason = cases.SET_SKIPS[(entry.provider, pset.name)]
                marks.append(
                    pytest.mark.skip(reason=f"{entry.provider}/{pset.name}: {reason}")
                )
            params.append(
                pytest.param(entry, pset, id=f"{entry.ident}-{pset.name}", marks=marks)
            )
    return params


@pytest.mark.parametrize("entry,pset", _matrix())
async def test_litellm_accepts_parameters(harness, net, entry, pset):
    label = f"model={entry.ident} param_set={pset.name} ({pset.why})"

    try:
        await fakenet.run_guarded(
            cases.run_case(harness, entry, pset), net, _CASE_TIMEOUT_SECONDS
        )
    except (fakenet.OutboundConnectionBlocked, Exception) as error:  # pylint: disable=broad-exception-caught
        # langchain can re-raise the blocked-connection error as a different type, so always report the addresses.
        pytest.fail(
            f"{label}: {type(error).__name__}: {error} | blocked connections: {net.socket_attempts} "
            f"| unexpected requests: {net.unexpected}"
        )

    assert not net.unexpected, f"{label}: unexpected requests {net.unexpected}"
    assert not net.socket_attempts, (
        f"{label}: blocked connections {net.socket_attempts}"
    )

    for call in net.calls:
        assert cases.EMPTY_TEXT_PLACEHOLDER not in json.dumps(call.body), (
            f"{label}: the request to {call.url} contains litellm's empty-content placeholder (#2335)"
        )

    method, host, path = cases.expected_route(entry)
    sent = [f"{call.method} {call.url}" for call in net.calls]
    assert net.matching(method, host, path), (
        f"{label}: expected {method} {host} {path}, litellm sent {sent}"
    )


async def test_every_models_yml_entry_is_a_case_or_a_listed_skip():
    in_models_yml = {
        ident
        for ident, definition in ModelSelectionConfig.instance()
        .get_llm_definitions()
        .items()
        if str(
            getattr(
                definition.model_class_provider,
                "value",
                definition.model_class_provider,
            )
        )
        in cases.SETS_BY_KIND
    }

    assert {entry.ident for entry in ENTRIES} == in_models_yml
    assert in_models_yml, (
        "models.yml has no litellm entries, so the matrix would check nothing"
    )


async def test_skips_name_entries_that_still_exist():
    idents = {entry.ident for entry in ENTRIES}

    assert set(cases.SKIPS) <= idents, (
        f"stale skips: {sorted(set(cases.SKIPS) - idents)}"
    )
    providers = {entry.provider for entry in ENTRIES}
    set_names = {pset.name for psets in cases.SETS_BY_KIND.values() for pset in psets}
    for provider, set_name in cases.SET_SKIPS:
        assert provider in providers, (
            f"stale set skip: no entry uses provider {provider!r}"
        )
        assert set_name in set_names, (
            f"stale set skip: no parameter set named {set_name!r}"
        )
    assert all(cases.SKIPS.values()) and all(cases.SET_SKIPS.values()), (
        "every skip needs a reason"
    )


async def test_every_entry_has_an_expected_route():
    for entry in ENTRIES:
        cases.expected_route(entry)


async def test_removing_the_fireworks_registration_brings_back_2587(
    harness, net, monkeypatch
):
    """Without register_fireworks_models, litellm rejects tool_choice for the Fireworks deployment (#2587)."""
    entry = next(entry for entry in ENTRIES if entry.ident == "qwen_3_8_27b_fireworks")
    pset = next(pset for pset in cases.CHAT_SETS if pset.name == "tools_tool_choice")
    assert entry.model, "the entry must name its deployment in params.model"
    deployment = entry.model.split("/")[-1]

    await cases.run_case(harness, entry, pset)
    assert net.matching("POST", "api.fireworks.ai", r"/chat/completions$"), (
        "the registered deployment must work"
    )

    keys = [key for key in litellm.model_cost if deployment in key]
    assert keys, (
        f"no model_cost key mentions {deployment}, so register_fireworks_models did nothing"
    )

    monkeypatch.setattr(
        litellm,
        "model_cost",
        {k: v for k, v in litellm.model_cost.items() if k not in keys},
    )
    net.reset()
    litellm.in_memory_llm_clients_cache.flush_cache()

    with pytest.raises(litellm.UnsupportedParamsError, match="tool_choice"):
        await cases.run_case(harness, entry, pset)
    assert not net.calls, "litellm must reject the request before sending it"


async def test_a_real_connection_attempt_fails_fast_with_its_address(
    harness, net, monkeypatch
):
    """With aiohttp on, litellm bypasses the fake transport.

    The socket guard must fail the case, not hang.
    """
    entry = next(entry for entry in ENTRIES if entry.ident == "kimi_k3_fireworks")
    pset = cases.CHAT_SETS[0]
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", False)
    litellm.in_memory_llm_clients_cache.flush_cache()

    started = time.monotonic()
    with pytest.raises(fakenet.OutboundConnectionBlocked):
        await fakenet.run_guarded(
            cases.run_case(harness, entry, pset), net, _CASE_TIMEOUT_SECONDS
        )

    assert net.socket_attempts, (
        "the connection attempt must be recorded with its address"
    )
    assert time.monotonic() - started < 5, (
        "a blocked connection must not wait for retries"
    )


async def test_an_unmatched_url_is_reported_with_the_url(harness, net, monkeypatch):
    """A request to a URL with no fake route fails fast with a 400 that names the URL."""
    entry = next(entry for entry in ENTRIES if entry.ident == "kimi_k3_fireworks")
    pset = cases.CHAT_SETS[0]
    monkeypatch.setattr(
        fakenet,
        "ROUTES",
        [route for route in fakenet.ROUTES if route.host != "api.fireworks.ai"],
    )

    started = time.monotonic()
    with pytest.raises(Exception):  # litellm raises its own error type for a 400
        await fakenet.run_guarded(
            cases.run_case(harness, entry, pset), net, _CASE_TIMEOUT_SECONDS
        )

    assert time.monotonic() - started < 10
    assert any("api.fireworks.ai" in url for url in net.unexpected), net.unexpected
