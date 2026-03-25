from unittest.mock import MagicMock

import pytest

from lib.context.model import (
    current_model_metadata_with_size_context,
    get_model_metadata,
)


@pytest.fixture(autouse=True)
def reset_context():
    token = current_model_metadata_with_size_context.set(None)
    yield
    current_model_metadata_with_size_context.reset(token)


class TestGetModelMetadata:
    def test_returns_none_when_no_context(self):
        assert get_model_metadata() is None

    @pytest.mark.parametrize(
        "args,expected_key",
        [
            ([], None),
            (["small"], "small"),
            (["large"], "large"),
        ],
    )
    def test_forwards_size_to_config(self, args, expected_key):
        model_size_config = MagicMock()
        current_model_metadata_with_size_context.set(model_size_config)

        result = get_model_metadata(*args)

        model_size_config.get.assert_called_once_with(expected_key)
        assert result is model_size_config.get.return_value
