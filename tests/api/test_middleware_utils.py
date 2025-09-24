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

    @pytest.mark.parametrize(
        "input_ids, expected",
        [
            (["1", "2", "2", "3", "1"], [1, 2, 3]),
            (["5", "1", "2", "1", "3", "2", "5", "4"], [5, 1, 2, 3, 4]),
            (["7", "7", "7", "7"], [7]),
            (["1", "1"], [1]),
        ],
    )
    def test_duplicate_removal(self, input_ids, expected):
        """Test that duplicates are removed."""
        result = get_valid_namespace_ids(input_ids)
        assert result == expected
