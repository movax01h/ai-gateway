"""The `template` prompt format: a prompt declares it, and the model receives what it would from `prompt_template`.

These run against the real prompt pipeline, with only the model faked, because what matters is the bytes the model would
receive.
"""

from itertools import takewhile
from typing import Any, cast
from unittest.mock import patch

import jsonschema
import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableSequence
from pydantic import ValidationError

from ai_gateway.prompts.base import Prompt
from ai_gateway.prompts.config import ModelClassProvider
from ai_gateway.prompts.config.base import PromptConfig, Template
from ai_gateway.prompts.config.formats import parse_format
from ai_gateway.prompts.feature_roots import discover_feature_prompts
from duo_workflow_service.agent_platform.v1.flows.flow_config import FlowConfig
from duo_workflow_service.schemas.v1.flow_config_schema import (
    FlowConfigJsonSchemaGenerator,
)


def _parts(*texts: str) -> dict[str, Any]:
    """A system prompt in the advanced form, one part per text."""
    return {"type": "system_message", "parts": [{"text": text} for text in texts]}


class _FakeModel(FakeMessagesListChatModel):
    """A fake that also answers what the prompt layer asks of a real model."""

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": "fake-model"}


def _prompt(
    template: dict[str, Any], responses: list[BaseMessage] | None = None
) -> Prompt:
    config = PromptConfig(
        name="test_prompt",
        unit_primitive="duo_agent_platform",
        template=Template(**template),
    )
    model = _FakeModel(responses=responses or [AIMessage(content="answer")])

    return Prompt(
        model_provider=ModelClassProvider.ANTHROPIC,
        model_factory=lambda **_: model,  # type: ignore[arg-type]
        config=config,
    )


# Everything a prompt config needs besides its template.
_PROMPT_FIELDS: dict[str, Any] = {
    "name": "cache_comparison",
    "unit_primitive": "duo_agent_platform",
}


def _prompt_for(provider: ModelClassProvider, config: PromptConfig) -> Prompt:
    return Prompt(
        model_provider=provider,
        model_factory=lambda **_: _FakeModel(responses=[]),  # type: ignore[arg-type]
        config=config,
    )


def _config(**template_field: Any) -> PromptConfig:
    return PromptConfig(**_PROMPT_FIELDS, **template_field)


def _is_cached(message: tuple[str, Any]) -> bool:
    """Whether a rendered message carries the cache breakpoint the prefix is read up to."""
    _, content = message

    return isinstance(content, list) and any(
        isinstance(block, dict) and "cache_control" in block for block in content
    )


async def _prefix(prompt: Prompt, variables: dict[str, Any]) -> list[tuple[str, Any]]:
    """What the model receives before the first user message, cache markers included.

    Renders through everything `Prompt` chains in front of the model, so the cache-control annotations are on the
    messages, and stops at the first human message: that is where the cached prefix ends.
    """
    # Everything the prompt chains in front of the model: the template, the per-model system merge, and the
    # cache-control annotations. The model itself is the last step.
    chain = cast(RunnableSequence, prompt.bound)
    before_the_model: RunnableSequence = RunnableSequence(*chain.steps[:-1])
    rendered = await before_the_model.ainvoke({"history": [], **variables})
    prefix = takewhile(
        lambda message: not isinstance(message, HumanMessage), rendered.to_messages()
    )

    return [(type(message).__name__, message.content) for message in prefix]


class TestTheFormat:
    def test_a_prompt_declares_one_format(self):
        with pytest.raises(ValueError, match="not both"):
            PromptConfig(
                name="x",
                unit_primitive="duo_agent_platform",
                prompt_template={"system": "a"},
                template=Template(system="a"),
            )

        with pytest.raises(
            ValueError, match="declares 'template' or 'prompt_template'"
        ):
            PromptConfig(name="x", unit_primitive="duo_agent_platform")

    def test_an_empty_prompt_template_is_still_a_declared_format(self):
        # Accepted on `main`, so still accepted, and still exclusive with `template`.
        PromptConfig(name="x", unit_primitive="duo_agent_platform", prompt_template={})

        with pytest.raises(ValueError, match="not both"):
            PromptConfig(
                name="x",
                unit_primitive="duo_agent_platform",
                prompt_template={},
                template=Template(system="a"),
            )

    @pytest.mark.parametrize(
        ("written", "texts"),
        [
            ("one part", ("one part",)),
            (_parts("instructions", "context"), ("instructions", "context")),
        ],
        ids=["a string", "the advanced form"],
    )
    def test_system_is_a_string_or_the_advanced_form(self, written, texts):
        assert Template(system=written).system_texts == texts

    @pytest.mark.parametrize(
        ("written", "error"),
        [
            (["one part"], "valid string"),
            ({"parts": [{"text": "a"}]}, "type"),
            (_parts(), "at least 1 item"),
            ("  ", "needs text"),
            (_parts("instructions", "  "), "needs text"),
            (
                {"type": "system_message", "parts": [{"text": "a", "cache": True}]},
                "Extra inputs are not permitted",
            ),
        ],
        ids=[
            "a list",
            "the advanced form without its type",
            "no parts",
            "a blank string",
            "a blank part",
            "an unknown key on a part",
        ],
    )
    def test_what_system_refuses(self, written, error):
        # One string or one object, with `type` always named, so a later kind never has to be guessed.
        with pytest.raises(ValidationError, match=error):
            Template(system=written)

    def test_a_prompt_with_neither_format_is_refused(self):
        with pytest.raises(
            ValueError, match="declares 'template' or 'prompt_template'"
        ):
            parse_format(None, None)


class TestWhatTheModelSees:
    def test_the_parts_are_built_as_a_prompt_template_system_list_is(self):
        prompt = _prompt({"system": _parts("stable instructions", "session context")})

        value = prompt.prompt_tpl.invoke({"history": []})
        system = [m for m in value.to_messages() if isinstance(m, SystemMessage)]

        # Inside `Prompt`, one LangChain system message per part, then the security block: the layout
        # `merges_system_messages` and the cache breakpoint on the first part both expect.
        assert [m.text for m in system[:2]] == [
            "stable instructions",
            "session context",
        ]
        assert "tool_output_policy" in system[2].text


class TestCachingDoesNotRegress:
    """Moving a prompt to the new format must not change the cached prefix the model receives.

    `system`'s parts are built as a `prompt_template.system` list is today, so the breakpoint keeps sitting on the
    first part and the per-session parts sit behind it.
    """

    @pytest.mark.parametrize(
        "provider",
        [ModelClassProvider.ANTHROPIC, ModelClassProvider.LITE_LLM],
        ids=["claude keeps its breakpoints", "models without breakpoints merge"],
    )
    @pytest.mark.asyncio
    async def test_the_cached_prefix_is_unchanged_by_the_move(self, provider):
        system = ["Stable instructions.", "Session context: {{ id }}."]
        old = _prompt_for(provider, _config(prompt_template={"system": system}))
        new = _prompt_for(provider, _config(template=Template(system=_parts(*system))))

        before = await _prefix(old, {"id": "a"})
        after = await _prefix(new, {"id": "a"})

        assert before == after

    @pytest.mark.asyncio
    async def test_the_breakpoint_still_sits_on_the_first_message(self):
        # Parity alone would also hold if both formats lost their caching, so check the breakpoint is there.
        prompt = _prompt_for(
            ModelClassProvider.ANTHROPIC,
            _config(
                template=Template(
                    system=_parts("Stable instructions.", "Session {{ id }}.")
                )
            ),
        )

        prefix = await _prefix(prompt, {"id": "a"})

        assert _is_cached(prefix[0]), prefix[0]
        assert not _is_cached(prefix[1]), prefix[1]

    @pytest.mark.asyncio
    async def test_the_cached_part_is_identical_across_sessions(self):
        prompt = _prompt({"system": _parts("stable instructions", "session {{ id }}")})

        first = await _prefix(prompt, {"id": "a"})
        second = await _prefix(prompt, {"id": "b"})

        # The first message carries the breakpoint, so it has to match byte for byte for anything to be read from
        # the cache; the per-session part sits behind it.
        assert first[0] == second[0]
        assert first[1] != second[1]

    @pytest.mark.parametrize(
        ("declared", "has_prompt"),
        [
            ({"prompt_template": {}}, False),
            ({"prompt_template": {"system": "a"}}, True),
            ({"template": Template(system="a")}, True),
        ],
        ids=["an empty prompt_template", "a prompt_template", "a template"],
    )
    def test_the_default_breakpoint_goes_to_a_prompt_that_renders_something(
        self, declared, has_prompt
    ):
        # An empty `prompt_template: {}` renders nothing of its own, so it gets no default breakpoint, as on `main`.
        with patch.object(
            Prompt, "_build_model_kwargs", wraps=Prompt._build_model_kwargs
        ) as build:
            _prompt_for(ModelClassProvider.ANTHROPIC, _config(**declared))

        assert build.call_args.args[2] is has_prompt


class TestWhatTheOldFormatStillSees:
    def test_a_template_prompt_refuses_a_consumer_that_reads_only_prompt_template(self):
        config = PromptConfig(
            name="x",
            unit_primitive="duo_agent_platform",
            template=Template(system="a"),
        )

        # An empty mapping here would send the model a prompt with no instructions and no error.
        with pytest.raises(ValueError, match="only agent flows render"):
            _ = config.legacy_prompt_template

    @pytest.mark.parametrize(
        "written",
        ["one part", _parts("a", "b")],
        ids=["a string", "the advanced form"],
    )
    def test_the_published_schema_accepts_what_pydantic_accepts(self, written):
        schema = FlowConfig.model_json_schema(
            schema_generator=FlowConfigJsonSchemaGenerator
        )
        block = schema["$defs"]["Template"]["properties"]["system"]

        # Authors validate against this schema in the AI Catalog and in their editor.
        jsonschema.validate(written, {**block, "$defs": schema["$defs"]})

    @pytest.mark.parametrize(
        "prompt",
        [
            {},
            {"prompt_template": {"system": "a"}, "template": {"system": "a"}},
        ],
        ids=["neither format", "both formats"],
    )
    def test_the_published_schema_requires_exactly_one_format(self, prompt):
        schema = FlowConfig.model_json_schema(
            schema_generator=FlowConfigJsonSchemaGenerator
        )
        prompt_schema = {
            **schema["$defs"]["InMemoryPromptConfig"],
            "$defs": schema["$defs"],
        }
        fields = {"prompt_id": "p", "name": "p", "unit_primitives": ["duo_chat"]}

        jsonschema.validate(fields | {"template": {"system": "a"}}, prompt_schema)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(fields | prompt, prompt_schema)

    @pytest.mark.parametrize(
        "written",
        [
            ["one part"],
            {"parts": [{"text": "a"}]},
            {"type": "system_message", "parts": [{"text": "a", "cache": True}]},
            "  ",
            {"type": "system_message", "parts": [{"text": "\n"}]},
        ],
        ids=[
            "a list",
            "the advanced form without its type",
            "an unknown key on a part",
            "a blank string",
            "a blank part",
        ],
    )
    def test_the_published_schema_and_pydantic_refuse_the_same(self, written):
        schema = FlowConfig.model_json_schema(
            schema_generator=FlowConfigJsonSchemaGenerator
        )
        block = schema["$defs"]["Template"]["properties"]["system"]

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(written, {**block, "$defs": schema["$defs"]})
        with pytest.raises(ValidationError):
            Template(system=written)


@pytest.fixture(autouse=True, scope="module")
def _feature_prompts():
    discover_feature_prompts()
