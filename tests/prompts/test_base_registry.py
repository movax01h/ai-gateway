from unittest.mock import Mock, call

import pytest
from gitlab_cloud_connector import GitLabUnitPrimitive, WrongUnitPrimitives

from ai_gateway.api.auth_utils import StarletteUser
from ai_gateway.prompts import BasePromptRegistry, Prompt


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
        ("unit_primitives", "scopes", "success", "expected_internal_events"),
        [
            (
                [GitLabUnitPrimitive.COMPLETE_CODE],
                ["complete_code"],
                True,
                [call("request_complete_code", category="ai_gateway.prompts.base")],
            ),
            (
                [GitLabUnitPrimitive.COMPLETE_CODE, GitLabUnitPrimitive.ASK_BUILD],
                ["complete_code", "ask_build"],
                True,
                [
                    call("request_complete_code", category="ai_gateway.prompts.base"),
                    call("request_ask_build", category="ai_gateway.prompts.base"),
                ],
            ),
            ([GitLabUnitPrimitive.COMPLETE_CODE], [], False, []),
            (
                [
                    GitLabUnitPrimitive.COMPLETE_CODE,
                    GitLabUnitPrimitive.ASK_BUILD,
                ],
                ["complete_code"],
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
        unit_primitives: list[GitLabUnitPrimitive],
        scopes: list[str],
        success: bool,
        expected_internal_events,
    ):
        if success:
            assert registry.get_on_behalf(user=user, prompt_id="test") == prompt

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
