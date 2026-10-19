import pytest

from ai_gateway.models.user_identity_header import inject_user_identity_header
from lib.context import gitlab_user_id


class TestInjectUserIdentityHeader:
    def test_adds_extra_headers_when_absent(self, gitlab_user_id_in_context):
        params: dict = {"model": "m"}

        inject_user_identity_header(params, "x-gitlab-user-id")

        assert params == {
            "model": "m",
            "extra_headers": {"x-gitlab-user-id": gitlab_user_id_in_context},
        }

    def test_preserves_existing_headers(self, gitlab_user_id_in_context):
        params: dict = {
            "extra_headers": {
                "X-Api-Subscription": "sub",
                "x-session-affinity": "sess",
            }
        }

        inject_user_identity_header(params, "x-gitlab-user-id")

        assert params["extra_headers"] == {
            "X-Api-Subscription": "sub",
            "x-session-affinity": "sess",
            "x-gitlab-user-id": gitlab_user_id_in_context,
        }

    @pytest.mark.parametrize(
        "existing_name", ["x-gitlab-user-id", "X-Gitlab-User-Id", "X-GITLAB-USER-ID"]
    )
    def test_replaces_same_name_header_case_insensitively(
        self, gitlab_user_id_in_context, existing_name
    ):
        params: dict = {"extra_headers": {existing_name: "static"}}

        inject_user_identity_header(params, "x-gitlab-user-id")

        assert params["extra_headers"] == {
            "x-gitlab-user-id": gitlab_user_id_in_context
        }

    def test_does_not_mutate_shared_headers_dict(self, gitlab_user_id_in_context):
        static_headers = {"X-Api-Subscription": "sub"}
        params: dict = {"extra_headers": static_headers}

        inject_user_identity_header(params, "x-gitlab-user-id")

        assert static_headers == {"X-Api-Subscription": "sub"}

    @pytest.mark.usefixtures("gitlab_user_id_in_context")
    @pytest.mark.parametrize("header_name", [None, ""])
    def test_leaves_params_untouched_when_not_configured(self, header_name):
        params: dict = {"extra_headers": {"X-Api-Subscription": "sub"}}

        inject_user_identity_header(params, header_name)

        assert params == {"extra_headers": {"X-Api-Subscription": "sub"}}

    @pytest.mark.usefixtures("no_gitlab_user_id_in_context")
    def test_leaves_params_untouched_without_user_id(self):
        params: dict = {"model": "m"}

        inject_user_identity_header(params, "x-gitlab-user-id")

        assert params == {"model": "m"}

    @pytest.mark.parametrize(
        "user_id",
        ["", " 42", "42\r\nX-Injected: 1", "42\u00e9", "abc", "4" * 21, "-1", "4.2"],
    )
    def test_drops_non_numeric_user_ids(self, user_id):
        token = gitlab_user_id.set(user_id)
        try:
            params: dict = {"model": "m"}

            inject_user_identity_header(params, "x-gitlab-user-id")

            assert params == {"model": "m"}
        finally:
            gitlab_user_id.reset(token)
