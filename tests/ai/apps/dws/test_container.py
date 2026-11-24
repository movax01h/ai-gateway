import pytest

from ai.apps.dws.container import ContainerApplication
from ai.platform.container import ContainerPlatform
from tests.fresh_interpreter import run_in_fresh_interpreter

# Not `providers`: wiring an instance adds `__self__` to its class's `providers`.


def test_dws_adds_only_its_providers():
    assert set(ContainerApplication.inherited_providers) == set(
        ContainerPlatform.cls_providers
    )
    assert set(ContainerApplication.cls_providers) == {"audit_event"}


@pytest.mark.parametrize("name", sorted(ContainerPlatform.cls_providers))
def test_dws_shares_the_platform_provider(name):
    assert (
        ContainerApplication.inherited_providers[name]
        is ContainerPlatform.cls_providers[name]
    )


def test_fresh_import():
    result = run_in_fresh_interpreter("import ai.apps.dws.container")

    assert result.returncode == 0, result.stderr
