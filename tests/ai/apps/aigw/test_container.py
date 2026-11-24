import pytest

from ai.apps.aigw.container import ContainerApplication
from ai.platform.container import ContainerPlatform
from tests.fresh_interpreter import run_in_fresh_interpreter

# Not `providers`: wiring an instance adds `__self__` to its class's `providers`.


def test_aigw_adds_only_its_providers():
    assert set(ContainerApplication.inherited_providers) == set(
        ContainerPlatform.cls_providers
    )
    assert set(ContainerApplication.cls_providers) == {
        "chat",
        "code_suggestions",
        "pkg_models",
        "self_signed_jwt",
        "snowplow",
    }


@pytest.mark.parametrize("name", sorted(ContainerPlatform.cls_providers))
def test_aigw_shares_the_platform_provider(name):
    assert (
        ContainerApplication.inherited_providers[name]
        is ContainerPlatform.cls_providers[name]
    )


def test_fresh_import():
    result = run_in_fresh_interpreter("import ai.apps.aigw.container")

    assert result.returncode == 0, result.stderr
