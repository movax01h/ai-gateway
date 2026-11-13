from pathlib import Path
from typing import Optional

import pytest

from ai.serving.grpc.flows import register_served_flows
from duo_workflow_service.agent_platform.v1.flows import flow_config
from lib.feature_roots import FOUNDATIONAL_AGENTS_DOMAIN

_IMPORT = (
    "from duo_workflow_service.agent_platform.serving_surface import ServingSurface\n"
)
_GRPC_DWS = (
    _IMPORT + "SERVING_SURFACE = [ServingSurface(transport='grpc', deployable='dws')]\n"
)
_REST_AIGW = (
    _IMPORT
    + "SERVING_SURFACE = [ServingSurface(transport='rest', deployable='aigw')]\n"
)


@pytest.fixture(autouse=True)
def empty_flow_roots():
    saved = dict(flow_config._FEATURE_FLOW_ROOTS)
    flow_config._FEATURE_FLOW_ROOTS.clear()
    yield
    flow_config._FEATURE_FLOW_ROOTS.clear()
    flow_config._FEATURE_FLOW_ROOTS.update(saved)


def _write_flow(
    features: Path, domain: str, feature: str, serving: Optional[str]
) -> Path:
    feature_dir = features / domain / feature
    (feature_dir / "config").mkdir(parents=True)
    if serving is not None:
        (feature_dir / "serving.py").write_text(serving)
    return feature_dir


class TestRegisterServedFlows:
    def test_registers_a_flow_that_declares_grpc_dws(self, tmp_path: Path):
        features = tmp_path / "ai" / "features"
        _write_flow(features, "insights", "served", _GRPC_DWS)

        register_served_flows(features)

        assert set(flow_config._FEATURE_FLOW_ROOTS) == {"served"}

    @pytest.mark.parametrize(
        "serving",
        [_REST_AIGW, None, "raise RuntimeError('broken')\n"],
        ids=["rest_aigw", "no_serving_py", "serving_py_fails_to_import"],
    )
    def test_raises_for_a_flow_that_does_not_declare_grpc_dws(
        self, tmp_path: Path, serving: Optional[str]
    ):
        features = tmp_path / "ai" / "features"
        feature_dir = _write_flow(features, "insights", "unserved", serving)

        with pytest.raises(ValueError, match="does not declare grpc/dws") as error:
            register_served_flows(features)

        assert str(feature_dir) in str(error.value)
        assert not flow_config._FEATURE_FLOW_ROOTS

    def test_registers_a_fetched_catalog_flow_without_serving_py(self, tmp_path: Path):
        features = tmp_path / "ai" / "features"
        _write_flow(features, FOUNDATIONAL_AGENTS_DOMAIN, "catalog_flow", None)

        register_served_flows(features)

        assert set(flow_config._FEATURE_FLOW_ROOTS) == {"catalog_flow"}

    def test_serves_every_bundled_flow(self):
        # Sweeps the real ai/features tree, so a flow that drops its grpc/dws
        # declaration fails here instead of at boot.
        flow_config.discover_feature_flow_configs()
        discovered = dict(flow_config._FEATURE_FLOW_ROOTS)
        flow_config._FEATURE_FLOW_ROOTS.clear()

        register_served_flows()

        assert discovered
        assert flow_config._FEATURE_FLOW_ROOTS == discovered
