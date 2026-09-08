# pylint: disable=file-naming-for-tests
"""Multi-root flow-config discovery for flows moved under ai/features/ (Layout B)."""

from pathlib import Path
from unittest import mock

import pytest
import yaml

from duo_workflow_service.agent_platform.experimental.flows import (
    flow_config as experimental_flow_config,
)
from duo_workflow_service.agent_platform.v1.flows import flow_config

_YAML = (
    'version: "v1"\n'
    "environment: chat-partial\n"
    'name: "{name}"\n'
    "components: []\n"
    "routers: []\n"
    "flow:\n"
    '  entry_point: "x"\n'
)


@pytest.fixture(autouse=True)
def restore_flow_roots():
    saved = dict(flow_config._FEATURE_FLOW_ROOTS)
    yield
    flow_config._FEATURE_FLOW_ROOTS.clear()
    flow_config._FEATURE_FLOW_ROOTS.update(saved)


def _write_flow(features_dir: Path, domain: str, feature: str, version: str) -> Path:
    config_dir = features_dir / domain / feature / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / f"{version}.yml").write_text(_YAML.format(name=feature))
    return config_dir


class TestDiscovery:
    def test_registers_config_dir_and_skips_features_without_one(self, tmp_path: Path):
        features = tmp_path / "ai" / "features"
        _write_flow(features, "insights", "with_config", "1.0.0")
        # a prompt-only feature: prompts/ but no config/
        (features / "cli" / "prompt_only" / "prompts").mkdir(parents=True)

        flow_config.discover_feature_flow_configs(features)

        roots = flow_config.FlowConfig.feature_config_roots()
        assert (
            roots["with_config"]
            == (features / "insights" / "with_config" / "config").resolve()
        )
        assert "prompt_only" not in roots

    def test_default_features_dir_is_the_repo_tree(self):
        # parents[4] from flow_config.py must land on the repo root.
        default = flow_config._default_features_dir()
        assert default.name == "features"
        assert default.parent.name == "ai"
        assert default.is_dir()

    def test_default_features_dir_falls_back_without_marker(
        self, monkeypatch, tmp_path: Path
    ):
        # No marker (wheel install, faked filesystem): fall back to the
        # fixed-depth derivation instead of failing the caller.
        orphan = tmp_path / "a" / "b" / "c" / "d" / "e" / "flow_config.py"
        orphan.parent.mkdir(parents=True)
        monkeypatch.setattr(flow_config, "__file__", str(orphan))

        assert (
            flow_config._default_features_dir()
            == (tmp_path / "a" / "ai" / "features").resolve()
        )

    def test_missing_tree_is_silent(self, tmp_path: Path):
        flow_config.discover_feature_flow_configs(tmp_path / "nope")
        assert "anything" not in flow_config.FlowConfig.feature_config_roots()

    def test_reregistering_same_root_is_idempotent(self, tmp_path: Path):
        flow_config.register_flow_config_root("a_flow", tmp_path / "config")
        flow_config.register_flow_config_root("a_flow", tmp_path / "config")

        assert (
            flow_config.FlowConfig.feature_config_roots()["a_flow"]
            == (tmp_path / "config").resolve()
        )

    def test_discovery_raises_on_duplicate_feature_dirs(self, tmp_path: Path):
        # The on-disk failure mode: two domains ship a feature with the same
        # name, and one discovery scan must refuse the second registration.
        features = tmp_path / "ai" / "features"
        _write_flow(features, "insights", "dupe_flow", "1.0.0")
        _write_flow(features, "cli", "dupe_flow", "1.0.0")

        with pytest.raises(ValueError, match="Duplicate flow config root"):
            flow_config.discover_feature_flow_configs(features)

    def test_conflicting_root_raises(self, tmp_path: Path):
        flow_config.register_flow_config_root("a_flow", tmp_path / "one" / "config")

        with pytest.raises(ValueError, match="Duplicate flow config root"):
            flow_config.register_flow_config_root("a_flow", tmp_path / "two" / "config")


class TestFromYamlConfig:
    def test_loads_from_feature_root(self, tmp_path: Path):
        features = tmp_path / "ai" / "features"
        _write_flow(features, "insights", "moved_flow", "1.0.0")
        flow_config.discover_feature_flow_configs(features)

        config = flow_config.FlowConfig.from_yaml_config("moved_flow", "^1.0.0")
        assert config.name == "moved_flow"
        assert config.resolved_version == "1.0.0"

    def test_legacy_flow_still_loads(self):
        # A flow still under the legacy configs/ root resolves unchanged.
        config = flow_config.FlowConfig.from_yaml_config("agentic_chat")
        assert config.resolved_version is not None

    def test_legacy_wins_on_collision(self, tmp_path: Path, monkeypatch):
        # Same (flow_id, version) in both roots: legacy wins, matching list_flow_configs,
        # so a config fetched into the writable legacy root overrides the bundled copy.
        legacy = tmp_path / "configs"
        (legacy / "dupe_flow").mkdir(parents=True)
        (legacy / "dupe_flow" / "1.0.0.yml").write_text(_YAML.format(name="Legacy"))
        monkeypatch.setattr(flow_config.FlowConfig, "DIRECTORY_PATH", legacy)

        features = tmp_path / "ai" / "features"
        _write_flow(features, "insights", "dupe_flow", "1.0.0")
        flow_config.discover_feature_flow_configs(features)

        config = flow_config.FlowConfig.from_yaml_config("dupe_flow", "1.0.0")
        assert config.name == "Legacy"

    def test_escaping_symlink_is_skipped_and_logged(self, tmp_path: Path):
        outside = tmp_path / "outside.yml"
        outside.write_text(_YAML.format(name="outside"))
        features = tmp_path / "ai" / "features"
        config_dir = _write_flow(features, "insights", "moved_flow", "1.0.0")
        (config_dir / "9.9.9.yml").symlink_to(outside)
        flow_config.discover_feature_flow_configs(features)

        with mock.patch.object(flow_config, "logger") as logger:
            config = flow_config.FlowConfig.from_yaml_config("moved_flow", "*")

        # The escaping symlink is excluded from the candidates, so the
        # constraint resolves to the legitimate version, and the skip is logged.
        assert config.resolved_version == "1.0.0"
        assert logger.warning.called

    def test_symlinked_legacy_dir_still_loads_feature_copy(
        self, tmp_path: Path, monkeypatch
    ):
        # A legacy flow dir symlinked outside the legacy root must not discard
        # the feature candidates: ListFlows lists the feature copy, so
        # ExecuteWorkflow must load it too.
        outside = tmp_path / "outside_config"
        outside.mkdir()
        legacy = tmp_path / "configs"
        legacy.mkdir()
        (legacy / "moved_flow").symlink_to(outside)
        monkeypatch.setattr(flow_config.FlowConfig, "DIRECTORY_PATH", legacy)

        features = tmp_path / "ai" / "features"
        _write_flow(features, "insights", "moved_flow", "1.0.0")
        flow_config.discover_feature_flow_configs(features)

        config = flow_config.FlowConfig.from_yaml_config("moved_flow", "1.0.0")
        assert config.name == "moved_flow"

    def test_real_tree_discovery_is_clean(self):
        # Discovery over the repo's real ai/features tree must not raise, so a
        # duplicate feature id is caught in CI, not at production boot.
        flow_config.discover_feature_flow_configs()

    def test_version_constraint_resolves_across_both_roots(
        self, tmp_path: Path, monkeypatch
    ):
        # Legacy holds 1.0.0, the feature root holds 2.0.0; a range constraint
        # resolves over the union.
        legacy = tmp_path / "configs"
        (legacy / "moved_flow").mkdir(parents=True)
        (legacy / "moved_flow" / "1.0.0.yml").write_text(_YAML.format(name="Legacy"))
        monkeypatch.setattr(flow_config.FlowConfig, "DIRECTORY_PATH", legacy)

        features = tmp_path / "ai" / "features"
        _write_flow(features, "insights", "moved_flow", "2.0.0")
        flow_config.discover_feature_flow_configs(features)

        config = flow_config.FlowConfig.from_yaml_config("moved_flow", ">=1.0.0")
        assert config.resolved_version == "2.0.0"
        assert config.name == "moved_flow"


class TestListFlowConfigs:
    def test_includes_feature_and_legacy_flows(self, tmp_path: Path):
        features = tmp_path / "ai" / "features"
        _write_flow(features, "insights", "moved_flow", "1.0.0")
        flow_config.discover_feature_flow_configs(features)

        ids = {c["flow_identifier"] for c in flow_config.list_configs()}
        assert "moved_flow" in ids  # from the feature root
        assert "agentic_chat" in ids  # from the legacy configs/ root

    def test_escaping_symlink_is_excluded_from_listing(self, tmp_path: Path):
        outside = tmp_path / "outside.yml"
        outside.write_text(_YAML.format(name="outside"))
        features = tmp_path / "ai" / "features"
        config_dir = _write_flow(features, "insights", "moved_flow", "1.0.0")
        (config_dir / "9.9.9.yml").symlink_to(outside)
        flow_config.discover_feature_flow_configs(features)

        versions = {
            c["flow_version"]
            for c in flow_config.list_configs()
            if c["flow_identifier"] == "moved_flow"
        }
        assert versions == {"1.0.0"}

    def test_legacy_wins_on_collision(self, tmp_path: Path, monkeypatch):
        # Same (flow_id, version) in the legacy root and a feature root: legacy wins.
        legacy = tmp_path / "configs"
        (legacy / "dupe_flow").mkdir(parents=True)
        (legacy / "dupe_flow" / "1.0.0.yml").write_text(_YAML.format(name="Legacy"))
        monkeypatch.setattr(flow_config.FlowConfig, "DIRECTORY_PATH", legacy)

        features = tmp_path / "ai" / "features"
        config_dir = features / "insights" / "dupe_flow" / "config"
        config_dir.mkdir(parents=True)
        (config_dir / "1.0.0.yml").write_text(_YAML.format(name="Feature"))
        flow_config.discover_feature_flow_configs(features)

        dupes = [
            c
            for c in flow_config.list_configs()
            if c["flow_identifier"] == "dupe_flow" and c["flow_version"] == "1.0.0"
        ]
        assert len(dupes) == 1
        assert '"Legacy"' in dupes[0]["config"]

    def test_malformed_legacy_copy_still_shadows_feature_copy(
        self, tmp_path: Path, monkeypatch
    ):
        # A legacy file that fails to parse still wins the (flow, version) slot:
        # the entry is dropped from the listing instead of falling back to the
        # feature copy, so ListFlows agrees with from_yaml_config, which raises.
        legacy = tmp_path / "configs"
        (legacy / "dupe_flow").mkdir(parents=True)
        (legacy / "dupe_flow" / "1.0.0.yml").write_text("{ broken yaml: [")
        monkeypatch.setattr(flow_config.FlowConfig, "DIRECTORY_PATH", legacy)

        features = tmp_path / "ai" / "features"
        _write_flow(features, "insights", "dupe_flow", "1.0.0")
        flow_config.discover_feature_flow_configs(features)

        listed = [
            c for c in flow_config.list_configs() if c["flow_identifier"] == "dupe_flow"
        ]
        assert listed == []
        with pytest.raises(yaml.YAMLError):
            flow_config.FlowConfig.from_yaml_config("dupe_flow", "1.0.0")


def test_experimental_flows_ignore_v1_feature_roots(tmp_path: Path):
    features = tmp_path / "ai" / "features"
    _write_flow(features, "insights", "moved_flow", "1.0.0")
    flow_config.discover_feature_flow_configs(features)

    assert experimental_flow_config.FlowConfig.feature_config_roots() == {}
