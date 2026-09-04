import pytest
import yaml
from langchain_core.prompts import ChatPromptTemplate

from ai_gateway.model_metadata import ModelMetadata
from ai_gateway.model_selection.models import ModelClassProvider
from ai_gateway.prompts import Prompt
from ai_gateway.prompts.model_variables import (
    MODEL_TEMPLATE_VARIABLES,
    model_friendly_name,
)
from duo_workflow_service.agent_platform.v1.flows.flow_config import FlowConfig
from lib.prompts.caching import (
    current_prompt_cache_context,
    set_prompt_caching_enabled_to_current_request,
)


@pytest.fixture(autouse=True)
def reset_prompt_cache_context():
    token = current_prompt_cache_context.set(None)
    yield
    current_prompt_cache_context.reset(token)


@pytest.fixture(name="prompt_template")
def prompt_template_fixture():
    return {
        "system": "model={{ model_friendly_name }} show_time={{ should_show_current_time }}",
        "user": "{{content}}",
    }


@pytest.fixture(name="model_provider")
def model_provider_fixture(request):
    return request.param


def _render(prompt: Prompt, **template_vars) -> str:
    prompt_tpl: ChatPromptTemplate = prompt.prompt_tpl  # type: ignore[assignment]
    messages = prompt_tpl.invoke({"content": "hi", **template_vars}).to_messages()

    return "\n".join(message.text() for message in messages)


@pytest.mark.parametrize(
    ("model_provider", "caching_enabled", "expected"),
    [
        (ModelClassProvider.OPENAI, "false", "model=Mistral show_time=True"),
        (ModelClassProvider.OPENAI, "true", "model=Mistral show_time=False"),
        (ModelClassProvider.ANTHROPIC, "false", "model=Mistral show_time=False"),
    ],
    indirect=["model_provider"],
)
def test_model_variables_rendered_without_caller_input(
    prompt: Prompt, caching_enabled: str, expected: str
):
    set_prompt_caching_enabled_to_current_request(caching_enabled)

    assert expected in _render(prompt)


@pytest.mark.parametrize("model_provider", [ModelClassProvider.OPENAI], indirect=True)
def test_model_variables_resolved_per_invocation(prompt: Prompt):
    """The caching header is per request, so a single ``Prompt`` must not freeze the resolved value."""
    set_prompt_caching_enabled_to_current_request("true")
    assert "show_time=False" in _render(prompt)

    set_prompt_caching_enabled_to_current_request("false")
    assert "show_time=True" in _render(prompt)


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        (None, "Unknown"),
        ("no_friendly_name", "Unknown"),
    ],
)
def test_model_friendly_name_falls_back_to_unknown(metadata, expected, llm_definition):
    if metadata == "no_friendly_name":
        metadata = ModelMetadata(
            provider="gitlab", name="mistral", llm_definition=llm_definition
        )

    assert model_friendly_name(metadata) == expected


class TestShippedAgenticChatPrompt:
    """The agentic chat flow config declares no input for these variables, so nothing else would catch the shipped
    template losing its binding."""

    @pytest.fixture(name="prompt_template")
    def prompt_template_fixture(self):
        config_path = FlowConfig.DIRECTORY_PATH / "agentic_chat" / "1.0.0.yml"
        config = yaml.safe_load(config_path.read_text())
        prompt = next(
            p for p in config["prompts"] if p["prompt_id"] == "chat_agent_prompt"
        )

        return prompt["prompt_template"]

    @pytest.mark.parametrize(
        "model_provider", [ModelClassProvider.OPENAI], indirect=True
    )
    def test_model_name_rendered_from_bound_model(self, prompt: Prompt):
        prompt_tpl: ChatPromptTemplate = prompt.prompt_tpl  # type: ignore[assignment]
        # The variables the flow config does declare as component inputs; the model ones
        # must not be among them.
        declared_inputs = dict.fromkeys(prompt_tpl.input_variables, "")

        assert not MODEL_TEMPLATE_VARIABLES & set(declared_inputs)
        assert "<model_name>Mistral</model_name>" in _render(prompt, **declared_inputs)

    @pytest.mark.parametrize(
        ("model_provider", "caching_enabled", "time_shown"),
        [
            (ModelClassProvider.OPENAI, "false", True),
            (ModelClassProvider.OPENAI, "true", False),
            (ModelClassProvider.ANTHROPIC, "false", False),
        ],
        indirect=["model_provider"],
    )
    def test_current_time_gated_by_bound_model(
        self, prompt: Prompt, caching_enabled: str, time_shown: bool
    ):
        set_prompt_caching_enabled_to_current_request(caching_enabled)
        prompt_tpl: ChatPromptTemplate = prompt.prompt_tpl  # type: ignore[assignment]
        declared_inputs = dict.fromkeys(prompt_tpl.input_variables, "")

        rendered = _render(prompt, **{**declared_inputs, "current_time": "12:00:00"})

        assert ("<current_time>12:00:00</current_time>" in rendered) is time_shown
