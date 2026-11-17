from importlib import resources

from jinja2 import BaseLoader, Environment

from lib.session_owner.context import get_session_owner_username

__all__ = ["render_session_owner_block"]

# Template source is package-internal and not user-controlled. The username is validated when it is stored.
_template = Environment(loader=BaseLoader()).from_string(
    resources.files("lib.session_owner").joinpath("session_owner.jinja").read_text()
)


def render_session_owner_block(suffix: str) -> str:
    """Render the session owner block, or an empty string when GitLab did not name a human owner.

    Args:
        suffix: The per-session suffix of the security block, so tool output cannot forge the tags.
    """
    username = get_session_owner_username()
    if not username:
        return ""

    return _template.render(suffix=suffix, username=username)
