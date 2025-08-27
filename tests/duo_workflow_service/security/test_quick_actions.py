import pytest

from duo_workflow_service.security.quick_actions import validate_no_quick_actions


@pytest.mark.parametrize(
    "text,should_err",
    [
        # negatives (should NOT error)
        (None, False),
        ("", False),
        ("regular text", False),
        ("inline /merge is fine", False),
        ("/etc/hosts", False),
        ("/foo.bar", False),
        ("/123", False),
        ("// comment", False),
        (". /merge", False),
        ("/close-issue", False),
        # positives (should error)
        ("/merge", True),
        ("   /approve", True),
        ("\t/close", True),
        ("first line\n/label bug", True),
        ("```\n/close in code block\n```", True),
        ("/health onTrack", True),
        ("\n\n/close\n", True),
    ],
)
def test_validate_no_quick_actions(text, should_err):
    err = validate_no_quick_actions(text)
    assert (err is not None) == should_err


@pytest.mark.parametrize(
    "field,expected_prefix",
    [
        ("description", "Description contains GitLab quick actions"),
        ("body", "Body contains GitLab quick actions"),
    ],
)
def test_validate_no_quick_actions_message_field_prefix(field, expected_prefix):
    err = validate_no_quick_actions("/close", field=field)
    assert err is not None
    assert err.startswith(expected_prefix)
