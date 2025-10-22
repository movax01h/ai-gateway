"""Test for model_class_provider override fix.

Validates that when a base prompt specifies anthropic but user selects another model like GPT, the request succeeds by
using the OpenAI provider from models.yml.
"""

# pylint: disable=file-naming-for-tests

from pathlib import Path
from unittest.mock import Mock

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from pyfakefs.fake_filesystem import FakeFilesystem

from ai_gateway.config import ConfigModelLimits
from ai_gateway.model_metadata import create_model_metadata
from ai_gateway.prompts import LocalPromptRegistry
from ai_gateway.prompts.config import ModelClassProvider


@pytest.fixture(autouse=True)
def stub_api_keys(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")


@pytest.fixture(name="mock_fs")
def mock_fs_fixture(fs: FakeFilesystem):
    ai_gateway_dir = Path(__file__).parent.parent.parent / "ai_gateway"
    model_selection_dir = ai_gateway_dir / "model_selection"
    prompts_dir = ai_gateway_dir / "prompts" / "definitions"

    fs.create_file(
        model_selection_dir / "models.yml",
        contents="""---
models:
    -
        name: "OpenAI GPT-5"
        gitlab_identifier: "gpt_5"
        family:
            - gpt_5
        params:
            model: "gpt-5-2025-08-07"
            model_class_provider: "openai"
            max_tokens: 4096
""",
    )

    fs.create_file(
        model_selection_dir / "unit_primitives.yml",
        contents="""---
configurable_unit_primitives:
    -
        feature_setting: "workflow"
        unit_primitives:
            - "duo_workflow_execute_workflow"
        default_model: "gpt_5"
        selectable_models:
            - "gpt_5"
""",
    )

    fs.create_dir(prompts_dir / "workflow" / "test" / "base")
    fs.create_file(
        prompts_dir / "workflow" / "test" / "base" / "1.0.0.yml",
        contents="""---
name: test
model:
    params:
        model_class_provider: anthropic
unit_primitives:
    - duo_workflow_execute_workflow
prompt_template:
    system: "Test"
    user: "{{goal}}"
""",
    )


@pytest.mark.usefixtures("mock_fs")
def test_gpt5_with_base_anthropic_prompt_does_not_error():
    """Base prompt has anthropic provider, no gpt_5 prompt exists, but request with gpt_5 model should not error."""
    registry = LocalPromptRegistry.from_local_yaml(
        class_overrides={},
        prompt_template_factories={},
        model_factories={
            ModelClassProvider.ANTHROPIC: lambda model, **kwargs: ChatAnthropic(
                model=model, **kwargs
            ),
            ModelClassProvider.OPENAI: lambda model, **kwargs: ChatOpenAI(
                model=model, **kwargs
            ),
        },
        internal_event_client=Mock(),
        model_limits=ConfigModelLimits({}),
        custom_models_enabled=True,
        disable_streaming=True,
    )

    model_metadata = create_model_metadata({"provider": "gitlab", "name": "gpt_5"})

    prompt = registry.get("workflow/test", "^1.0.0", model_metadata=model_metadata)

    # Verify OpenAI model was created with the expected model id
    assert isinstance(prompt.model, ChatOpenAI)
    assert prompt.model.model_name == "gpt-5-2025-08-07"
