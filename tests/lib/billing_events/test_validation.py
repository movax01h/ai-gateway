from unittest.mock import MagicMock, patch

import pytest
from gitlab_cloud_connector import GitLabUnitPrimitive

from lib.billing_events.validation import (
    billable_feature_for_unit_primitive,
    is_billing_validation_enforced,
    validate_billing_event_context,
)
from lib.internal_events.context import EventContext


@pytest.fixture(name="logger")
def logger_fixture():
    return MagicMock()


@pytest.mark.parametrize(
    ("context", "expected_missing"),
    [
        pytest.param(
            EventContext(realm="saas", ultimate_parent_namespace_id=1),
            [],
            id="valid_saas",
        ),
        pytest.param(
            EventContext(
                realm="self-managed", instance_id="inst", unique_instance_id="uid"
            ),
            [],
            id="valid_sm",
        ),
        pytest.param(
            EventContext(realm="saas"),
            ["ultimate_parent_namespace_id"],
            id="saas_missing_root_namespace_id",
        ),
        pytest.param(
            EventContext(realm="self-managed", unique_instance_id="uid"),
            ["instance_id"],
            id="sm_missing_instance_id",
        ),
        pytest.param(
            EventContext(realm="self-managed", instance_id=""),
            ["instance_id", "unique_instance_id"],
            id="sm_empty_instance_id_and_no_claims",
        ),
        pytest.param(
            EventContext(realm="self-managed", instance_id="inst"),
            ["unique_instance_id"],
            id="sm_missing_unique_instance_id",
        ),
        pytest.param(EventContext(), ["realm"], id="missing_realm"),
        pytest.param(EventContext(realm=""), ["realm"], id="empty_realm"),
        pytest.param(
            EventContext(realm="dedicated"),
            ["ultimate_parent_namespace_id"],
            id="dedicated_treated_as_saas_missing",
        ),
        pytest.param(
            EventContext(realm="dedicated", ultimate_parent_namespace_id=1),
            [],
            id="dedicated_treated_as_saas_valid",
        ),
    ],
)
def test_validate_billing_event_context(context, expected_missing, logger):
    assert (
        validate_billing_event_context(
            context, "code_suggestions", enforce=False, logger=logger
        )
        == expected_missing
    )

    if expected_missing:
        logger.warning.assert_called_once()
    else:
        logger.warning.assert_not_called()
    logger.error.assert_not_called()


@pytest.mark.parametrize(
    ("enforce", "expected_log_level"),
    [(False, "warning"), (True, "error")],
)
def test_validate_billing_event_context_logs_structured_message(
    enforce, expected_log_level, logger
):
    context = EventContext(
        realm="saas",
        deployment_type=".com",
        instance_version="18.5.0",
        client_type="vscode",
        client_name="gitlab-lsp",
        correlation_id="corr-123",
    )

    validate_billing_event_context(
        context, "code_suggestions", enforce=enforce, logger=logger, endpoint="/v2/x"
    )

    getattr(logger, expected_log_level).assert_called_once_with(
        "Billing event context missing mandatory fields",
        missing_fields=["ultimate_parent_namespace_id"],
        feature_qualified_name="code_suggestions",
        realm="SaaS",
        deployment_type=".com",
        instance_version="18.5.0",
        client_type="vscode",
        client_name="gitlab-lsp",
        enforced=enforce,
        correlation_id="corr-123",
        endpoint="/v2/x",
    )


def test_validate_billing_event_context_uses_module_logger_by_default():
    with patch("lib.billing_events.validation.log") as mock_log:
        validate_billing_event_context(
            EventContext(), "code_suggestions", enforce=False
        )

    mock_log.warning.assert_called_once()


@pytest.mark.parametrize(
    ("unit_primitive", "user_can", "expected"),
    [
        (GitLabUnitPrimitive.COMPLETE_CODE, True, "code_suggestions"),
        (GitLabUnitPrimitive.GENERATE_CODE, True, "code_suggestions"),
        (
            GitLabUnitPrimitive.AI_GATEWAY_MODEL_PROVIDER_PROXY,
            True,
            "ai_gateway_proxy_use",
        ),
        (GitLabUnitPrimitive.COMPLETE_CODE, False, None),
        (GitLabUnitPrimitive.DUO_CHAT, True, None),
        (GitLabUnitPrimitive.DUO_AGENT_PLATFORM, True, None),
        ("unknown_unit_primitive", True, None),
        (None, True, None),
    ],
)
def test_billable_feature_for_unit_primitive(unit_primitive, user_can, expected):
    user = MagicMock()
    user.can.return_value = user_can

    assert billable_feature_for_unit_primitive(user, unit_primitive) == expected


def test_billable_feature_for_unit_primitive_without_user():
    assert (
        billable_feature_for_unit_primitive(None, GitLabUnitPrimitive.COMPLETE_CODE)
        is None
    )


@pytest.mark.parametrize(
    ("mode", "feature_qualified_name", "expected"),
    [
        ("enforce", "code_suggestions", True),
        ("enforce", "fix_pipeline/v1", True),
        ("enforce", "fix_pipeline/experimental", True),
        ("enforce", "ai_gateway_proxy_use", False),
        ("enforce", "software_development", False),
        ("log", "code_suggestions", False),
        ("log", "fix_pipeline/v1", False),
    ],
)
def test_is_billing_validation_enforced(mode, feature_qualified_name, expected):
    with patch(
        "lib.billing_events.validation._ENFORCED_FEATURE_QUALIFIED_NAMES",
        frozenset({"code_suggestions", "fix_pipeline"}),
    ):
        assert is_billing_validation_enforced(mode, feature_qualified_name) is expected


def test_no_feature_is_enforced_by_default():
    assert not is_billing_validation_enforced("enforce", "code_suggestions")
