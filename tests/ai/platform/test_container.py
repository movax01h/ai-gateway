import pytest

from ai.platform.container import ContainerPlatform
from ai_gateway.container import ContainerApplication
from tests.fresh_interpreter import run_in_fresh_interpreter

PLATFORM_PROVIDERS = {
    "billing_event",
    "config",
    "integrations",
    "interceptor",
    "internal_event",
    "pkg_models_v2",
    "pkg_prompts",
    "pkg_schemas",
    "searches",
    "usage_quota",
}


def test_platform_providers():
    # Not `providers`: wiring an instance adds `__self__` to its class's `providers`.
    assert set(ContainerPlatform.cls_providers) == PLATFORM_PROVIDERS


@pytest.mark.parametrize("name", sorted(PLATFORM_PROVIDERS))
def test_app_container_shares_the_platform_provider(name):
    # Markers resolve by provider identity, so a marker on ContainerApplication
    # must point at the same object as the platform provider.
    assert ContainerApplication.providers[name] is ContainerPlatform.providers[name]


@pytest.mark.parametrize("module", ["ai.platform.container", "ai_gateway.container"])
def test_fresh_import(module):
    result = run_in_fresh_interpreter(f"import {module}")

    assert result.returncode == 0, result.stderr
