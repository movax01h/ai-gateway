"""Tests for the one object a prompt's format resolves to, whichever syntax it was written in."""

import pytest

from ai_gateway.prompts.config.base import InMemoryPromptConfig, PromptConfig
from ai_gateway.prompts.config.formats import (
    LegacyTemplate,
    parse_format,
)
from ai_gateway.prompts.config.template import DEFAULT_MESSAGE_TEMPLATE, Template
from lib.prompts.utilities import (
    PromptSource,
    prompt_template_to_messages,
)


class TestLegacyTemplate:
    def test_sources_keep_every_string_with_its_role_in_order(self):
        sources = LegacyTemplate(
            {
                "system": ["static", "dynamic"],
                "user": "{{ goal }}",
                "placeholder": "history",
            }
        ).sources()

        # A list expands into one source per entry; roles that aren't text are the caller's to skip.
        assert sources == (
            PromptSource("system", "static"),
            PromptSource("system", "dynamic"),
            PromptSource("user", "{{ goal }}"),
            PromptSource("placeholder", "history"),
        )

    def test_sources_skip_empty_strings(self):
        assert LegacyTemplate({"system": ["", "kept"], "user": ""}).sources() == (
            PromptSource("system", "kept"),
        )

    @pytest.mark.parametrize("tool_output_security", [True, False])
    def test_resolve_renders_the_whole_mapping(self, tool_output_security):
        roles = {"system": ["static", "dynamic"], "user": "{{ goal }}"}

        resolved = LegacyTemplate(roles).resolve(tool_output_security)

        assert resolved.prefix == prompt_template_to_messages(
            roles, tool_output_security=tool_output_security
        )

    def test_resolve_has_no_tail(self):
        # The old format renders whole on every call and stores no message of its own.
        assert LegacyTemplate({"user": "{{ goal }}"}).resolve(True).tail is None


class TestTemplate:
    @pytest.mark.parametrize(
        "system",
        [
            "instructions",
            {
                "type": "system_message",
                "parts": [{"text": "static"}, {"text": "dynamic"}],
            },
        ],
        ids=["simple", "advanced"],
    )
    @pytest.mark.parametrize("tool_output_security", [True, False])
    def test_the_prefix_is_what_the_old_format_renders_for_the_same_parts(
        self, system, tool_output_security
    ):
        template = Template(system=system)

        resolved = template.resolve(tool_output_security)

        # One system message per part, then the security block and the history placeholder: the layout the cache
        # breakpoint relies on, unchanged from `prompt_template: {system: [...]}`.
        assert resolved.prefix == prompt_template_to_messages(
            {"system": list(template.system_texts)},
            tool_output_security=tool_output_security,
        )

    @pytest.mark.parametrize(
        ("message", "tail"),
        [
            (
                "<m>{{ inbound_messages.text }}</m>",
                ("user", "<m>{{ inbound_messages.text }}</m>"),
            ),
            (None, ("user", DEFAULT_MESSAGE_TEMPLATE)),
        ],
        ids=["the author's block", "the platform default"],
    )
    def test_the_tail_is_one_user_message(self, message, tail):
        # One message, not a list of them: a structured block renders as its content blocks later (ADR-004 §3).
        assert Template(system="a", message=message).resolve(True).tail == tail

    def test_sources_mark_inbound_messages_as_engine_provided_in_message_only(self):
        template = Template(
            system={
                "type": "system_message",
                "parts": [{"text": "static"}, {"text": "{{ project }}"}],
            },
            message="{{ inbound_messages.text }}",
        )

        assert template.sources() == (
            PromptSource("system", "static"),
            PromptSource("system", "{{ project }}"),
            PromptSource(
                "message",
                "{{ inbound_messages.text }}",
                provided_by_engine=frozenset({"inbound_messages"}),
            ),
        )

    def test_sources_include_the_platform_default_message(self):
        # What `resolve` renders is what the checks read, the default block included.
        assert Template(system="a").sources() == (
            PromptSource("system", "a"),
            PromptSource(
                "message",
                DEFAULT_MESSAGE_TEMPLATE,
                provided_by_engine=frozenset({"inbound_messages"}),
            ),
        )


class TestParseFormat:
    def test_the_old_format(self):
        parsed = parse_format({"user": "hi"}, None)

        assert parsed == LegacyTemplate({"user": "hi"})

    @pytest.mark.parametrize(
        "template",
        [Template(system="a"), {"system": "a"}],
        ids=["the model", "the mapping it was written as"],
    )
    def test_the_new_format(self, template):
        parsed = parse_format(None, template)

        assert parsed == Template(system="a")

    def test_an_empty_prompt_template_is_a_declared_format(self):
        # `is None`, not emptiness, as on `main`.
        assert parse_format({}, None) == LegacyTemplate({})

    def test_neither_is_refused(self):
        with pytest.raises(
            ValueError, match="declares 'template' or 'prompt_template'"
        ):
            parse_format(None, None)

    def test_both_are_refused(self):
        with pytest.raises(ValueError, match="not both"):
            parse_format({"user": "hi"}, {"system": "a"})


class TestConfigsExposeOneFormat:
    def test_prompt_config(self):
        legacy = PromptConfig(
            name="x", unit_primitive="duo_chat", prompt_template={"user": "hi"}
        )
        new = PromptConfig(
            name="x", unit_primitive="duo_chat", template={"system": "a"}
        )

        assert legacy.prompt_format == LegacyTemplate({"user": "hi"})
        assert legacy.legacy_prompt_template == {"user": "hi"}
        assert new.prompt_format == Template(system="a")
        with pytest.raises(ValueError, match="declares 'template'"):
            _ = new.legacy_prompt_template

    def test_a_copy_reads_its_own_format(self):
        # `model_copy(update=...)` skips validation, so a format stored at validation would be the original's.
        config = PromptConfig(
            name="x", unit_primitive="duo_chat", template={"system": "a"}
        )

        copy = config.model_copy(update={"template": Template(system="b")})

        assert copy.prompt_format == Template(system="b")

    def test_in_memory_prompt_config_round_trips_through_its_prompt_data(self):
        config = InMemoryPromptConfig(
            prompt_id="p",
            name="p",
            unit_primitives=["duo_chat"],
            template={"system": "a", "message": "{{ inbound_messages.text }}"},
        )

        data = config.to_prompt_data()

        assert parse_format(data.get("prompt_template"), data.get("template")) == (
            config.prompt_format
        )
