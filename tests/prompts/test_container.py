import sys
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pytest
from dependency_injector import containers, providers
from pydantic import AnyUrl

from ai_gateway.config import ConfigModelLimits
from ai_gateway.model_metadata import ModelMetadata
from ai_gateway.prompts.registry import LocalPromptRegistry
from duo_workflow_service import agents as workflow
from duo_workflow_service.gitlab.http_client import GitlabHttpClient


@pytest.fixture
def config_values(assets_dir):
    return {
        "custom_models": {"enabled": True, "disable_streaming": True},
        "self_signed_jwt": {
            "signing_key": open(assets_dir / "keys" / "signing_key.pem").read()
        },
        "amazon_q": {
            "region": "us-west-2",
        },
        "model_engine_limits": {
            "anthropic-chat": {
                "claude-3-7-sonnet-20250219": {
                    "input_tokens": 5,
                    "output_tokens": 10,
                    "concurrency": 15,
                }
            }
        },
    }


def _kwargs_for_class(klass):
    match klass:
        case workflow.AgentV2:
            return {
                "workflow_id": "123",
                "http_client": Mock(spec=GitlabHttpClient),
            }

    return {}


def test_container(mock_ai_gateway_container: containers.DeclarativeContainer):
    prompts = cast(providers.Container, mock_ai_gateway_container.pkg_prompts)
    registry = cast(LocalPromptRegistry, prompts.prompt_registry())

    assert registry.model_limits == ConfigModelLimits(
        {
            "anthropic-chat": {
                "claude-3-7-sonnet-20250219": {
                    "input_tokens": 5,
                    "output_tokens": 10,
                    "concurrency": 15,
                }
            }
        }
    )

    prompts_dir = Path(
        sys.modules[LocalPromptRegistry.__module__].__file__ or ""
    ).parent
    prompts_definitions_dir = prompts_dir / "definitions"
    # Iterate over every file in the prompts definitions directory. Make sure
    # they're loaded into the registry and that the resulting Prompts are valid.
    for path in prompts_definitions_dir.glob("**"):
        versions = [version.stem for version in path.glob("*.yml")]

        if not versions:
            continue

        prompt_id_with_model_name = path.relative_to(prompts_definitions_dir)
        prompt_id = prompt_id_with_model_name.parent
        model_name = prompt_id_with_model_name.name

        model_metadata = ModelMetadata(
            name=str(model_name),
            endpoint=AnyUrl("http://localhost:4000"),
            provider="",
        )

        klass = registry.prompts_registered[str(prompt_id_with_model_name)].klass
        kwargs = _kwargs_for_class(klass)

        for version in versions:
            prompt = registry.get(
                str(prompt_id),
                f"={version}",  # Make a strict constraint so we can check every existing version
                model_metadata=model_metadata,
                **kwargs,
            )
            assert isinstance(prompt, klass)
            assert prompt.model.disable_streaming
