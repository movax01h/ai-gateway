import contextvars

import pytest

from duo_workflow_service.gitlab.help_links import gitlab_web_url, help_page_url


@pytest.mark.parametrize(
    ("oidc_url", "gateway_url", "instance"),
    [
        (None, None, "https://gitlab.com"),
        (None, "https://gitlab.example.test/", "https://gitlab.example.test"),
        (
            "https://oidc.example.test",
            "https://gateway.example.test",
            "https://oidc.example.test",
        ),
    ],
    ids=["default", "gateway_url_trailing_slash", "oidc_url_wins"],
)
def test_the_link_follows_the_configured_instance(
    monkeypatch, oidc_url, gateway_url, instance
):
    for key, value in (
        ("DUO_WORKFLOW_AUTH__OIDC_GITLAB_URL", oidc_url),
        ("AIGW_GITLAB_URL", gateway_url),
    ):
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)

    assert (
        help_page_url("/user/project/x", anchor="a-section")
        == f"{instance}/help/user/project/x#a-section"
    )


def test_no_anchor_means_no_fragment(monkeypatch):
    monkeypatch.delenv("DUO_WORKFLOW_AUTH__OIDC_GITLAB_URL", raising=False)
    monkeypatch.setenv("AIGW_GITLAB_URL", "http://gdk.test:3000")

    assert help_page_url("user/project/x") == "http://gdk.test:3000/help/user/project/x"


@pytest.mark.parametrize(
    ("web_url", "instance"),
    [
        (
            "https://gitlab.example.test:8443/group/project",
            "https://gitlab.example.test:8443",
        ),
        ("", "http://gitlab.internal:8080"),
        ("not a url", "http://gitlab.internal:8080"),
    ],
    ids=["browser_url_wins", "nothing_fetched", "unparsable"],
)
def test_the_link_follows_the_url_the_browser_opened(monkeypatch, web_url, instance):
    """The configured URL is how the service reaches GitLab; a browser may not be able to open it."""
    monkeypatch.delenv("DUO_WORKFLOW_AUTH__OIDC_GITLAB_URL", raising=False)
    monkeypatch.setenv("AIGW_GITLAB_URL", "http://gitlab.internal:8080")

    def link() -> str:
        gitlab_web_url.set(web_url)
        return help_page_url("user/project/x")

    assert contextvars.copy_context().run(link) == f"{instance}/help/user/project/x"
