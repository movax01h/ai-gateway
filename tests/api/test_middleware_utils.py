import pytest

from ai_gateway.api.middleware_utils import get_valid_namespace_ids


class TestGetValidNamespaceIds:
    @pytest.mark.parametrize(
        "input_ids, expected",
        [
            (["1", "2", "3"], [1, 2, 3]),
        ],
    )
    def test_valid_ids(self, input_ids, expected):
        result = get_valid_namespace_ids(input_ids)
        assert result == expected

    @pytest.mark.parametrize("input_ids, expected", [(["abc", "xyz"], [])])
    def test_invalid_ids(self, input_ids, expected):
        result = get_valid_namespace_ids(input_ids)
        assert result == expected

    @pytest.mark.parametrize("input_ids, expected", [(["1", "invalid", "007"], [1, 7])])
    def test_mixed_valid_and_invalid_ids(self, input_ids, expected):
        result = get_valid_namespace_ids(input_ids)
        assert result == expected

    @pytest.mark.parametrize("input_ids, expected", [(["", ""], [])])
    def test_empty_string_values(self, input_ids, expected):
        result = get_valid_namespace_ids(input_ids)
        assert result == expected

    @pytest.mark.parametrize("input_ids, expected", [([], [])])
    def test_empty_list(self, input_ids, expected):
        result = get_valid_namespace_ids(input_ids)
        assert result == expected

    @pytest.mark.parametrize("input_ids, expected", [(None, [])])
    def test_none_input(self, input_ids, expected):
        result = get_valid_namespace_ids(input_ids)
        assert result == expected
