import pytest

from eval.routing.common import load_routing_configs
from eval.routing.schema import EvalDataset


@pytest.mark.parametrize(
    "member, expected_value, expected_full_path",
    [
        (EvalDataset.LOCAL, "local", "routing_eval.dataset.local"),
        (EvalDataset.MR, "mr", "routing_eval.dataset.mr"),
        (EvalDataset.MAIN, "main", "routing_eval.dataset.main"),
    ],
)
def test_full_path_per_member(member, expected_value, expected_full_path):
    assert isinstance(member.value, str)
    assert member.value == expected_value
    assert member.full_path == expected_full_path
    assert EvalDataset(expected_value) == member


def test_yaml_configs_are_valid():
    configs = load_routing_configs(base_dir="config/routing")
    for config in configs:
        assert config.tool_name is not None
        assert len(config.cases) > 0
