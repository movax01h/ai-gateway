"""Which flow versions opt into the chat engine by declaring their prompts with `template`."""

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import pytest
import yaml

from duo_workflow_service.agent_platform.v1.flows.flow_config import (
    PartialFlowConfig,
)
from duo_workflow_service.agent_platform.v1.flows.prompt_format import (
    opts_into_chat_engine,
)

_FIXTURE = (
    Path(__file__).parents[4] / "prompts" / "config" / "chat_partial_template.yml"
)


class TestTheFlowVersionOptsIn:
    def test_a_version_whose_prompts_use_template_opts_in(self, fixture_config):
        assert opts_into_chat_engine(fixture_config) is True

    def test_template_is_refused_for_a_flow_written_outside_this_repository(self):
        # An AI Catalog flow reaches the gateway as an inline config, with no registry identity. Until the
        # validator, the docs and Rails's prompt builder handle the new format, it can't opt in.
        inline = PartialFlowConfig(**yaml.safe_load((_FIXTURE).read_text())).to_config()

        with pytest.raises(ValueError, match="outside the AI Gateway repository"):
            opts_into_chat_engine(inline)

    def test_a_version_that_opts_in_declares_its_prompts_inline(self):
        # A repository prompt's format can't be read from the config, so it can't count as opted in.
        data = yaml.safe_load(_FIXTURE.read_text())
        component = dict(data["components"][0], prompt_version="^1.0.0")
        config = _load_fixture(components=[component])

        with pytest.raises(ValueError, match="declares its prompts inline"):
            opts_into_chat_engine(config)

    def test_a_version_does_not_mix_the_two_formats(self):
        data = yaml.safe_load(_FIXTURE.read_text())
        data["prompts"] = list(data["prompts"])
        data["prompts"].append(
            {
                "name": "other",
                "prompt_id": "other",
                "unit_primitives": ["duo_agent_platform"],
                "prompt_template": {"system": "a"},
            }
        )
        config = _load_fixture(prompts=data["prompts"])

        with pytest.raises(ValueError, match="all its prompts with 'template'"):
            opts_into_chat_engine(config)


@pytest.fixture(name="fixture_config")
def fixture_config_fixture():
    return _load_fixture()


def _load_fixture(**overrides: Any):
    """The fixture config as the registry would load it, identity included."""
    if not overrides:
        return PartialFlowConfig.from_registry_file(
            _FIXTURE, "developer", "4.0.0"
        ).to_config()

    with NamedTemporaryFile("w", suffix=".yml") as edited:
        data = yaml.safe_load(_FIXTURE.read_text())
        yaml.safe_dump(data | overrides, edited)
        edited.flush()

        return PartialFlowConfig.from_registry_file(
            Path(edited.name), "developer", "4.0.0"
        ).to_config()
