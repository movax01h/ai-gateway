# tests/api/test_context_utils.py
from typing import Optional

import pytest
from starlette_context import context, request_cycle_context

from ai_gateway.api.context_utils import (
    GitLabAiRequestType,
    populate_ai_metadata_in_context,
)
from ai_gateway.model_metadata import (
    AmazonQModelMetadata,
    ModelMetadata,
    TypeModelMetadata,
)
from ai_gateway.models.base import KindModelProvider


def create_mock_model_metadata(
    provider: str, name: str, identifier: Optional[str] = None
) -> TypeModelMetadata:
    if provider == KindModelProvider.AMAZON_Q.value:
        return AmazonQModelMetadata(
            provider=KindModelProvider.AMAZON_Q.value,
            name=KindModelProvider.AMAZON_Q.value,
            role_arn="test_arn",
        )
    return ModelMetadata(provider=provider, name=name, identifier=identifier or name)


@pytest.mark.parametrize(
    "model_metadata_input_args, request_type, feature_id_input, context_unit_primitive, expected_context",
    [
        # ModelMetadata with identifier, feature_id provided
        (
            {
                "provider": "test_provider_1",
                "name": "test_model_1",
                "identifier": "id_1",
            },
            "chat",
            "feat_1",
            None,
            {
                "model_provider": "test_provider_1",
                "model_name": "test_model_1",
                "model_identifier": "id_1",
                "request_type": GitLabAiRequestType.CHAT,
                "feature_id": "feat_1",
            },
        ),
        # ModelMetadata without identifier, feature_id provided
        (
            {"provider": "test_provider_2", "name": "test_model_2"},
            "completions",
            "feat_2",
            None,
            {
                "model_provider": "test_provider_2",
                "model_name": "test_model_2",
                "model_identifier": "test_model_2",
                "request_type": GitLabAiRequestType.COMPLETIONS,
                "feature_id": "feat_2",
            },
        ),
        # AmazonQModelMetadata (no identifier attribute), feature_id provided
        (
            {"provider": KindModelProvider.AMAZON_Q.value, "name": "amazon_q"},
            "generations",
            "feat_3",
            None,
            {
                "model_provider": KindModelProvider.AMAZON_Q.value,
                "model_name": "amazon_q",
                "model_identifier": "amazon_q",
                "request_type": GitLabAiRequestType.GENERATIONS,
                "feature_id": "feat_3",
            },
        ),
        # model_metadata is None, feature_id provided
        (
            None,
            "suggestions",
            "feat_4",
            None,
            {
                "model_provider": "unknown",
                "model_name": "unknown",
                "model_identifier": "unknown",
                "request_type": GitLabAiRequestType.SUGGESTIONS,
                "feature_id": "feat_4",
            },
        ),
        # model_metadata provided, feature_id is None
        (
            {"provider": "test_provider_5", "name": "test_model_5"},
            "chat",
            None,
            "unit_primitive_from_context",
            {
                "model_provider": "test_provider_5",
                "model_name": "test_model_5",
                "model_identifier": "test_model_5",
                "request_type": GitLabAiRequestType.CHAT,
                "feature_id": "unit_primitive_from_context",
            },
        ),
        # model_metadata provided, feature_id is None
        (
            {"provider": "test_provider_6", "name": "test_model_6"},
            "completions",
            None,
            None,
            {
                "model_provider": "test_provider_6",
                "model_name": "test_model_6",
                "model_identifier": "test_model_6",
                "request_type": GitLabAiRequestType.COMPLETIONS,
                "feature_id": "unknown",
            },
        ),
        # model_metadata is None, feature_id is None
        (
            None,
            "generations",
            None,
            "another_unit_primitive",
            {
                "model_provider": "unknown",
                "model_name": "unknown",
                "model_identifier": "unknown",
                "request_type": GitLabAiRequestType.GENERATIONS,
                "feature_id": "another_unit_primitive",
            },
        ),
        # model_metadata is None, feature_id is None
        (
            None,
            "suggestions",
            None,
            None,
            {
                "model_provider": "unknown",
                "model_name": "unknown",
                "model_identifier": "unknown",
                "request_type": "suggestions",
                "feature_id": "unknown",
            },
        ),
    ],
)
def test_populate_ai_metadata_in_context(
    model_metadata_input_args: Optional[dict],
    request_type: str,
    feature_id_input: Optional[str],
    context_unit_primitive: Optional[str],
    expected_context: dict,
):
    actual_model_metadata_input = None
    if model_metadata_input_args:
        actual_model_metadata_input = create_mock_model_metadata(
            **model_metadata_input_args
        )

    initial_context_setup = {}
    if context_unit_primitive:
        initial_context_setup["meta.unit_primitive"] = context_unit_primitive

    with request_cycle_context(initial_context_setup):
        populate_ai_metadata_in_context(
            model_metadata=actual_model_metadata_input,
            request_type=request_type,
            feature_id=feature_id_input,
        )

        assert context.get("model_provider") == expected_context["model_provider"]
        assert context.get("model_name") == expected_context["model_name"]
        assert context.get("model_identifier") == expected_context["model_identifier"]
        assert context.get("request_type") == expected_context["request_type"]
        assert context.get("feature_id") == expected_context["feature_id"]
