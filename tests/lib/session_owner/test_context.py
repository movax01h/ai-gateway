import contextvars

import pytest

from lib.session_owner import (
    get_session_owner_username,
    set_session_owner_username,
)


def _in_fresh_context(fn):
    return contextvars.copy_context().run(fn)


@pytest.mark.parametrize(
    "username",
    [
        # Current GitLab rules
        "jane.doe",
        "jane_doe",
        "Jane-Doe1",
        "a" * 255,
        # Older usernames, allowed by the rules at the time
        ".jane",
        "_bot",
        "jane-",
        "jane..doe",
        "a",
    ],
)
def test_stores_a_gitlab_username(username):
    def run():
        set_session_owner_username(username)
        return get_session_owner_username()

    assert _in_fresh_context(run) == username


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "jane doe",
        "jane\ndoe",
        "jane{{ x }}",
        "jane{% if %}",
        "jane</session_owner>",
        "jäne",
        "a" * 256,
        123,
    ],
)
def test_rejects_a_value_that_could_not_come_from_rails(value):
    def run():
        set_session_owner_username("previous")
        set_session_owner_username(value)
        return get_session_owner_username()

    assert _in_fresh_context(run) is None


def test_defaults_to_none():
    assert _in_fresh_context(get_session_owner_username) is None
