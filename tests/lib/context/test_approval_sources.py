from lib.context.approval_sources import (
    approval_sources,
    get_approval_policy_ref,
    get_approval_source,
    init_approval_sources,
    record_approval_policy_ref,
    record_approval_source,
)


class TestApprovalSourcesRegistry:
    def teardown_method(self):
        # Reset the ContextVar so tests don't leak state into each other.
        approval_sources.set(None)

    def test_record_and_get_source(self):
        init_approval_sources()
        record_approval_source("call-1", "user_explicit")
        assert get_approval_source("call-1") == "user_explicit"

    def test_record_and_get_policy_ref(self):
        init_approval_sources()
        record_approval_policy_ref("call-1", {"origin": "file", "file": ".gitlab/duo"})
        assert get_approval_policy_ref("call-1") == {
            "origin": "file",
            "file": ".gitlab/duo",
        }

    def test_source_and_policy_ref_coexist_on_same_call(self):
        init_approval_sources()
        record_approval_source("call-1", "auto_mode")
        record_approval_policy_ref("call-1", {"origin": "policy"})
        assert get_approval_source("call-1") == "auto_mode"
        assert get_approval_policy_ref("call-1") == {"origin": "policy"}

    def test_missing_id_returns_none(self):
        init_approval_sources()
        assert get_approval_source("nope") is None
        assert get_approval_policy_ref("nope") is None

    def test_uninitialized_registry_is_noop(self):
        # No init: recording is a safe no-op and lookups return None.
        approval_sources.set(None)
        record_approval_source("call-1", "user_explicit")
        record_approval_policy_ref("call-1", {"origin": "x"})
        assert get_approval_source("call-1") is None
        assert get_approval_policy_ref("call-1") is None

    def test_record_ignores_empty_inputs(self):
        init_approval_sources()
        record_approval_source(None, "user_explicit")
        record_approval_source("call-1", None)
        record_approval_source("call-1", "")
        record_approval_policy_ref("call-1", None)
        record_approval_policy_ref("call-1", {})
        assert get_approval_source("call-1") is None
        assert get_approval_policy_ref("call-1") is None

    def test_get_with_none_id_returns_none(self):
        init_approval_sources()
        assert get_approval_source(None) is None
        assert get_approval_policy_ref(None) is None

    def test_init_resets_registry(self):
        init_approval_sources()
        record_approval_source("call-1", "user_explicit")
        init_approval_sources()
        assert get_approval_source("call-1") is None
