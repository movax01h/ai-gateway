import pytest
from pydantic import ValidationError

from duo_workflow_service.agent_platform.experimental.components.for_each import (
    MAX_CONCURRENCY_CEILING,
    MAX_ITEMS_CEILING,
    ForEachConfig,
)


def cfg(**kwargs) -> dict:
    base = {"items": "context:discover.files", "as": "context:item"}
    base.update(kwargs)
    return base


def test_the_documented_shape_is_accepted():
    config = ForEachConfig(**cfg())

    assert config.items == "context:discover.files"
    assert config.as_ == "context:item"
    assert config.max_items == MAX_ITEMS_CEILING
    assert config.max_concurrency == 10


def test_config_rejects_bad_iokey():
    with pytest.raises(ValueError, match="Invalid target"):
        ForEachConfig(**cfg(items="nonsense:foo"))


@pytest.mark.parametrize("value", [0, MAX_ITEMS_CEILING + 1])
def test_config_rejects_out_of_range(value):
    """The ceiling binds, so a typo cannot fan out unbounded."""
    with pytest.raises(ValueError):
        ForEachConfig(**cfg(max_items=value))


@pytest.mark.parametrize("value", [0, MAX_CONCURRENCY_CEILING + 1])
def test_config_rejects_an_out_of_range_concurrency(value):
    with pytest.raises(ValueError):
        ForEachConfig(**cfg(max_concurrency=value))


def test_max_items_defaults_to_its_ceiling():
    """Nothing is dropped unless a flow asks for it: the default runs the list."""
    assert ForEachConfig(**cfg()).max_items == MAX_ITEMS_CEILING


def test_a_configured_concurrency_is_kept():
    assert ForEachConfig(**cfg(max_concurrency=3)).max_concurrency == 3


@pytest.mark.parametrize("typo,value", [("max_itmes", 5), ("maxitems", 5)])
def test_a_mistyped_for_each_key_is_rejected_and_named(typo, value):
    """An unknown key is an error, not a silent default."""
    with pytest.raises(ValidationError) as exc:
        ForEachConfig(**cfg(**{typo: value}))

    assert typo in str(exc.value), "the error does not name the offending key"


@pytest.mark.parametrize(
    "bad_as",
    [
        "ui_chat_log",  # every item dies on TypeError: list + str
        "conversation_history",  # AttributeError: 'str' has no 'items'
        "status",  # silently overwrites the branch's workflow status
        "context",  # replaces the whole context channel for every branch
    ],
)
def test_as_is_a_write_target_and_must_be_a_context_namespace(bad_as):
    with pytest.raises(ValidationError) as exc:
        ForEachConfig(**cfg(**{"as": bad_as}))

    message = str(exc.value)
    assert "for_each.as" in message, "the error does not name the field to fix"
    assert "context:item" in message, "the error does not say what to write instead"


def test_an_as_with_its_own_namespace_is_accepted():
    """The guard above must not reject the shape authors are told to use."""
    assert ForEachConfig(**cfg(**{"as": "context:item"})).as_ == "context:item"


def test_as_may_not_land_on_the_namespace_items_reads_from():
    """Silent data loss otherwise: each item overwrites the list being iterated."""
    with pytest.raises(ValidationError) as exc:
        ForEachConfig(**cfg(**{"as": "context:discover.item"}))

    message = str(exc.value)
    assert "context:discover" in message, "the error does not name the collision"
    assert "context:item" in message, "the error does not say what to write instead"


def test_a_sibling_subkey_of_a_different_namespace_is_fine():
    """Only the FIRST subkey is the namespace, so a deeper path may repeat a name."""
    config = ForEachConfig(**cfg(**{"as": "context:item.discover"}))
    assert config.item_key.subkeys == ["item", "discover"]


def test_items_from_a_non_context_target_has_no_namespace_to_collide_with():
    config = ForEachConfig(**cfg(items="conversation_history"))
    assert config.items_key.target == "conversation_history"
    assert config.as_ == "context:item"


@pytest.mark.parametrize(
    "bad_for_each,expected_loc",
    [
        pytest.param({"items": "context:discover.files"}, "as", id="missing-as"),
        pytest.param(cfg(**{"as": "nonsense:foo"}), "as", id="unparseable-as"),
        pytest.param(cfg(items="nonsense:foo"), "items", id="unparseable-items"),
    ],
)
def test_a_malformed_block_points_at_the_field_that_is_wrong(
    bad_for_each, expected_loc
):
    with pytest.raises(ValidationError) as exc:
        ForEachConfig(**bad_for_each)

    locations = {".".join(str(part) for part in e["loc"]) for e in exc.value.errors()}
    assert locations == {expected_loc}, f"error points at {locations}"


def test_the_parsed_keys_match_the_strings():
    config = ForEachConfig(**cfg())

    assert (config.items_key.target, config.items_key.subkeys) == (
        "context",
        ["discover", "files"],
    )
    assert (config.item_key.target, config.item_key.subkeys) == ("context", ["item"])


def test_the_field_name_is_usable_in_python_too():
    """``as`` is a keyword, so the field has to be assignable by its alias-free name."""
    assert ForEachConfig(items="context:d.files", as_="context:item").as_ == (
        "context:item"
    )


def test_a_config_cannot_be_mutated_after_construction():
    """A frozen config is safe to share across every branch of one fan-out."""
    config = ForEachConfig(**cfg())
    with pytest.raises(ValidationError):
        config.max_items = 1
