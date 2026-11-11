"""Links to GitLab documentation through the instance's own ``/help`` route, as Rails does."""

import os
from contextvars import ContextVar
from typing import Optional
from urllib.parse import urlparse

# The session's project or namespace URL, as the user's browser sees it. The workflow
# sets it once it has fetched the container; empty when nothing was fetched.
gitlab_web_url: ContextVar[str] = ContextVar("gitlab_web_url", default="")


def gitlab_instance_url() -> str:
    """Return the GitLab instance this service serves, as configured for OIDC."""
    # pylint: disable=direct-environment-variable-reference
    return (
        os.environ.get("DUO_WORKFLOW_AUTH__OIDC_GITLAB_URL")
        or os.environ.get("AIGW_GITLAB_URL")
        or "https://gitlab.com"
    )


def help_page_url(path: str, anchor: Optional[str] = None) -> str:
    """Return the instance's ``/help`` URL for ``path`` (a ``doc/`` path without ``.md``).

    The origin of the session's ``gitlab_web_url`` wins over the configured instance URL, which the service uses to
    reach GitLab and can be an internal address a browser cannot open.
    """
    origin = _origin(gitlab_web_url.get()) or gitlab_instance_url().rstrip("/")
    url = f"{origin}/help/{path.strip('/')}"
    return f"{url}#{anchor}" if anchor else url


def _origin(web_url: str) -> str:
    parsed = urlparse(web_url)
    if not (parsed.scheme and parsed.netloc):
        return ""
    return f"{parsed.scheme}://{parsed.netloc}"
