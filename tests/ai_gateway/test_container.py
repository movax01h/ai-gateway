import pytest

from ai.apps.aigw import container as aigw_container
from ai.apps.dws import container as dws_container
from ai_gateway.container import ContainerApplication

APP_CONTAINERS = [
    aigw_container.ContainerApplication,
    dws_container.ContainerApplication,
]


def _declared(container):
    # Not `providers`: wiring an instance adds `__self__` to its class's `providers`.
    return {**container.inherited_providers, **container.cls_providers}


def test_union_has_the_providers_of_both_apps():
    assert not ContainerApplication.cls_providers
    assert set(_declared(ContainerApplication)) == {
        name for app in APP_CONTAINERS for name in _declared(app)
    }


@pytest.mark.parametrize(
    ("app", "name"),
    [(app, name) for app in APP_CONTAINERS for name in sorted(_declared(app))],
    ids=lambda value: getattr(value, "__module__", value),
)
def test_union_shares_each_app_provider(app, name):
    # Markers resolve by provider identity. Issue #2501 can switch an entrypoint
    # to its app container only while this holds.
    assert _declared(ContainerApplication)[name] is _declared(app)[name]
