from datetime import datetime
from typing import Any
from unittest.mock import patch

import pytest
from gitlab_cloud_connector import CloudConnectorUser, GitLabUnitPrimitive, UserClaims
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import BaseMessage


class FakeModel(SimpleChatModel):
    expected_message: str
    response: str

    @property
    def _llm_type(self) -> str:
        return "fake-provider"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": "fake-model"}

    def _call(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        assert self.expected_message == messages[0].content

        return self.response


@pytest.fixture(name="model_factory")
def model_factory_fixture():
    return lambda model, **kwargs: FakeModel(
        expected_message="Hi, I'm John and I'm 20 years old. It's now July 12, 2025.",
        response="Hi John!",
    )


@pytest.fixture(name="prompt_template")
def prompt_template_fixture():
    return {
        "system": "Hi, I'm {{name}} and I'm {{age}} years old. It's now {{current_date}}."
    }


@pytest.fixture(name="compatible_versions")
def compatible_versions_fixture():
    return ["1.0.0"]


@pytest.fixture(name="mock_registry_get")
def mock_registry_get_fixture(
    request,
    compatible_versions: list[str],
):
    with patch("ai_gateway.prompts.registry.LocalPromptRegistry.get") as mock:
        if compatible_versions is None:
            mock.side_effect = KeyError()
        elif not compatible_versions:
            mock.side_effect = ValueError("No prompt version found matching the query:")
        else:
            mock.return_value = request.getfixturevalue("prompt")

        yield mock


@pytest.fixture(name="frozen_datetime_now")
def frozen_datetime_now_fixture():
    frozen = datetime(2025, 7, 12, 12, 0, 0)
    with patch("ai_gateway.api.v1.prompts.invoke.datetime") as mock_datetime:
        mock_datetime.now.return_value = frozen
        yield mock_datetime


@pytest.fixture(name="unit_primitives")
def unit_primitives_fixture():
    return ["explain_vulnerability"]


@pytest.fixture(name="auth_user")
def auth_user_fixture(unit_primitives: list[GitLabUnitPrimitive]):
    return CloudConnectorUser(
        authenticated=True, claims=UserClaims(scopes=unit_primitives)
    )
