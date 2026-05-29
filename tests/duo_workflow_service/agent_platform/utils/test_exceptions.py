from duo_workflow_service.agent_platform.utils.exceptions import (
    NotifiableAgentException,
)


class TestNotifiableAgentException:
    def test_ui_message_is_accessible(self):
        exc = NotifiableAgentException("Something went wrong")
        assert exc.ui_message == "Something went wrong"

    def test_str_returns_ui_message(self):
        exc = NotifiableAgentException("Something went wrong")
        assert str(exc) == "Something went wrong"

    def test_internal_detail_defaults_to_none(self):
        exc = NotifiableAgentException("Something went wrong")
        assert exc.internal_detail is None

    def test_internal_detail_is_accessible(self):
        exc = NotifiableAgentException(
            "Something went wrong", internal_detail="token=abc123 host=internal.svc"
        )
        assert exc.internal_detail == "token=abc123 host=internal.svc"

    def test_ui_message_does_not_contain_internal_detail(self):
        """Internal detail must never leak into the user-facing message."""
        exc = NotifiableAgentException(
            "Something went wrong", internal_detail="secret-token-xyz"
        )
        assert "secret-token-xyz" not in exc.ui_message
        assert "secret-token-xyz" not in str(exc)

    def test_is_exception_subclass(self):
        exc = NotifiableAgentException("msg")
        assert isinstance(exc, Exception)

    def test_cause_chain_preserved(self):
        original = ValueError("original cause")
        exc = NotifiableAgentException("user-facing message")
        exc.__cause__ = original
        assert exc.__cause__ is original
