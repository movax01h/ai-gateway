from typing import Optional
from unittest.mock import Mock, call

import pytest
from gitlab_cloud_connector import GitLabUnitPrimitive, WrongUnitPrimitives
from pydantic import AnyUrl

from ai_gateway.api.auth_utils import StarletteUser
from ai_gateway.prompts import BasePromptRegistry, Prompt
from ai_gateway.prompts.typing import (
    AmazonQModelMetadata,
    ModelMetadata,
    TypeModelMetadata,
)


@pytest.fixture
def registry(internal_event_client: Mock, prompt: Prompt):
    class Registry(BasePromptRegistry):
        def __init__(self):
            self.internal_event_client = internal_event_client

        def get(self, *args, **kwargs):
            return prompt

    yield Registry()


class TestBaseRegistry:
    @pytest.mark.parametrize(
        (
            "unit_primitives",
            "scopes",
            "model_metadata",
            "success",
            "expected_internal_events",
        ),
        [
            (
                [GitLabUnitPrimitive.COMPLETE_CODE],
                ["complete_code"],
                None,
                True,
                [call("request_complete_code", category="ai_gateway.prompts.base")],
            ),
            (
                [GitLabUnitPrimitive.COMPLETE_CODE, GitLabUnitPrimitive.ASK_BUILD],
                ["complete_code", "ask_build"],
                None,
                True,
                [
                    call("request_complete_code", category="ai_gateway.prompts.base"),
                    call("request_ask_build", category="ai_gateway.prompts.base"),
                ],
            ),
            (
                [GitLabUnitPrimitive.COMPLETE_CODE],
                ["complete_code"],
                ModelMetadata(
                    name="mistral",
                    provider="litellm",
                    endpoint=AnyUrl("http://localhost:4000"),
                    api_key="token",
                ),
                True,
                [
                    call("request_complete_code", category="ai_gateway.prompts.base"),
                ],
            ),
            (
                [GitLabUnitPrimitive.AMAZON_Q_INTEGRATION],
                ["amazon_q_integration"],
                AmazonQModelMetadata(
                    name="amazon_q",
                    provider="amazon_q",
                    role_arn="role-arn",
                ),
                True,
                [
                    call(
                        "request_amazon_q_integration",
                        category="ai_gateway.prompts.base",
                    ),
                ],
            ),
            ([GitLabUnitPrimitive.COMPLETE_CODE], [], None, False, []),
            (
                [
                    GitLabUnitPrimitive.COMPLETE_CODE,
                    GitLabUnitPrimitive.ASK_BUILD,
                ],
                ["complete_code"],
                None,
                False,
                [],
            ),
        ],
    )
    def test_get_on_behalf(
        self,
        internal_event_client: Mock,
        registry: BasePromptRegistry,
        user: StarletteUser,
        prompt: Prompt,
        model_metadata: Optional[TypeModelMetadata],
        unit_primitives: list[GitLabUnitPrimitive],
        scopes: list[str],
        success: bool,
        expected_internal_events,
    ):
        if success:
            assert registry.get_on_behalf(user, "test", None, model_metadata) == prompt

            if model_metadata:
                assert model_metadata._user == user

            internal_event_client.track_event.assert_has_calls(expected_internal_events)
        else:
            with pytest.raises(WrongUnitPrimitives):
                registry.get_on_behalf(user=user, prompt_id="test")

            internal_event_client.track_event.assert_not_called()

    @pytest.mark.parametrize(
        (
            "unit_primitives",
            "scopes",
            "internal_event_category",
            "expected_internal_events",
        ),
        [
            (
                [GitLabUnitPrimitive.COMPLETE_CODE, GitLabUnitPrimitive.ASK_BUILD],
                ["complete_code", "ask_build"],
                "my_category",
                [
                    call("request_complete_code", category="my_category"),
                    call("request_ask_build", category="my_category"),
                ],
            ),
        ],
    )
    def test_get_on_behalf_with_internal_event_category(
        self,
        internal_event_client: Mock,
        registry: BasePromptRegistry,
        user: StarletteUser,
        prompt: Prompt,
        unit_primitives: list[GitLabUnitPrimitive],
        scopes: list[str],
        internal_event_category: str,
        expected_internal_events,
    ):
        registry.get_on_behalf(
            user=user, prompt_id="test", internal_event_category=internal_event_category
        )

        internal_event_client.track_event.assert_has_calls(expected_internal_events)
