import pytest
from pydantic import ValidationError
from structlog.testing import capture_logs

from duo_workflow_service.agent_platform.core.inbox import TextPart, build_part


def _text(text: str) -> dict[str, str]:
    # What a writer passes for text.
    return {"type": "text", "text": text}


# A part with a key that isn't a string, annotated because mypy can't infer mixed key types.
_NON_STRING_KEY: dict[object, object] = {"type": "text", "text": "Hi", 1: "a"}


def _reason(error: ValueError) -> str:
    # What a validator said, which Pydantic wraps in a `ValidationError`.
    if isinstance(error, ValidationError):
        return str(error.errors()[0]["ctx"]["error"])
    return str(error)


class TestBuildPart:
    def test_a_mapping_is_built_as_its_type(self):
        assert build_part(_text("Hi"), "row-1") == TextPart(type="text", text="Hi")

    def test_a_built_part_is_kept(self):
        part = TextPart(type="text", text="Hi")

        assert build_part(part, "row-1") is part

    def test_a_part_keeps_fields_it_does_not_read(self):
        part = {"type": "text", "text": "Hi", "meta": {"scale": 1.5, "caption": None}}

        built = build_part(part, "row-1")

        assert built.model_extra == {"meta": {"scale": 1.5, "caption": None}}

    def test_a_part_is_a_copy_of_what_the_writer_sent(self):
        part = {**_text("Hi"), "label": "original", "items": [{"value": "original"}]}
        built = build_part(part, "row-1")

        part["label"] = "Changed by the producer"
        part["items"][0]["value"] = "Changed by the producer"

        assert built.model_extra == {
            "label": "original",
            "items": [{"value": "original"}],
        }

    def test_an_unsupported_part_type_is_refused_and_logged(self):
        reference = {"type": "file_reference", "upload_id": "upload-1"}

        with capture_logs() as logs, pytest.raises(ValueError) as refusal:
            build_part(reference, "row-1")

        assert _reason(refusal.value) == (
            "This AI Gateway doesn't support `file_reference` parts. Upgrade it, or"
            " send the input without them."
        )
        assert logs == [
            {
                "event": "Refused an unsupported part type",
                "log_level": "warning",
                "part_type": "file_reference",
                "input_id": "row-1",
            }
        ]

    @pytest.mark.parametrize(
        ("part", "reason"),
        [
            ({"text": "Hi"}, "Every part needs a `type`."),
            ({"type": 1, "text": "Hi"}, "Every part needs a `type`."),
            ("Hi", "Every part needs a `type`."),
            (_text("  \n"), "A text part needs text that isn't blank."),
            ({"type": "text"}, "A text part needs text that isn't blank."),
            (
                {**_text("Hi"), "data": bytearray(b"a")},
                "A part holds only JSON values.",
            ),
            ({**_text("Hi"), "tags": {"a"}}, "A part holds only JSON values."),
            (_NON_STRING_KEY, "A part holds only JSON values."),
            (
                {**_text("Hi"), "width": float("nan")},
                "A part holds only JSON values.",
            ),
            (
                {**_text("Hi"), "sizes": [1.5, float("inf")]},
                "A part holds only JSON values.",
            ),
        ],
        ids=[
            "a part without a type",
            "a part whose type isn't a string",
            "a part that isn't a mapping",
            "a blank text part",
            "a text part without text",
            "a part holding bytes",
            "a part holding a set",
            "a part with a key that isn't a string",
            "a part holding NaN",
            "a part holding infinity",
        ],
    )
    def test_an_invalid_part_says_what_to_fix(self, part, reason):
        with pytest.raises(ValueError) as refusal:
            build_part(part, "row-1")

        assert _reason(refusal.value) == reason


class TestTextPart:
    @pytest.mark.parametrize(
        ("part", "reason"),
        [
            (
                {"type": "image", "text": "Hi"},
                "A `TextPart` is a part of type `text`.",
            ),
            ("Hi", "A `TextPart` is a part of type `text`."),
        ],
        ids=["a text part of another type", "a text part that isn't a mapping"],
    )
    def test_refuses_invalid_fields(self, part, reason):
        with pytest.raises(ValidationError) as refusal:
            TextPart.model_validate(part)

        assert _reason(refusal.value) == reason
