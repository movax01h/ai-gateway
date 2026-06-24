import pytest
from gitlab_cloud_connector import CloudConnectorUser, UserClaims

from ai_gateway.api.v1 import api_router
from ai_gateway.api.v1.proxy.request import (
    EXTENDED_FEATURE_CATEGORIES_FOR_PROXY_ENDPOINTS,
)


@pytest.fixture(name="fast_api_router", scope="class")
def fast_api_router_fixture():
    return api_router


@pytest.fixture(name="auth_user")
def auth_user_fixture():
    return CloudConnectorUser(
        authenticated=True,
        claims=UserClaims(
            scopes=EXTENDED_FEATURE_CATEGORIES_FOR_PROXY_ENDPOINTS.keys(),
            gitlab_instance_id="1",
            extra={
                "gitlab_project_id": "1",
                "gitlab_namespace_id": "1",
                "gitlab_root_namespace_id": "1",
            },
        ),
    )


@pytest.fixture(name="unit_primitive")
def unit_primitive_fixture():
    return next(iter(EXTENDED_FEATURE_CATEGORIES_FOR_PROXY_ENDPOINTS.keys()))


@pytest.fixture(name="proxy_headers")
def proxy_headers_fixture():
    """Common headers for proxy requests."""
    return {
        "Authorization": "Bearer 12345",
        "X-Gitlab-Authentication-Type": "oidc",
        "X-Gitlab-Instance-Id": "1",
        "X-Gitlab-Project-Id": "1",
        "X-Gitlab-Namespace-Id": "1",
        "x-gitlab-root-namespace-id": "1",
    }
