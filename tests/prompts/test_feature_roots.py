"""Cross-root prompt discovery for features moved under ai/features/ (Layout B)."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from jinja2.exceptions import TemplateNotFound

from ai_gateway.prompts import base, feature_roots
from ai_gateway.prompts.registry import LocalPromptRegistry


@pytest.fixture(autouse=True)
def restore_feature_roots():
    """Keep the module-global feature-root registry isolated per test."""
    saved = dict(feature_roots._FEATURE_PROMPT_ROOTS)
    yield
    feature_roots._FEATURE_PROMPT_ROOTS.clear()
    feature_roots._FEATURE_PROMPT_ROOTS.update(saved)


def _make_registry() -> LocalPromptRegistry:
    # get_required_variables only needs discovery + Jinja; the model machinery is unused.
    return LocalPromptRegistry(
        prompt_template_factories={},
        model_factories={},
        internal_event_client=Mock(),
        model_limits=Mock(),
        custom_models_enabled=False,
    )


def _write_moved_feature(features_dir: Path, domain: str, feature: str) -> Path:
    prompts = features_dir / domain / feature / "prompts"
    (prompts / "base").mkdir(parents=True)
    (prompts / "system").mkdir(parents=True)
    (prompts / "base" / "1.0.0.yml").write_text(
        "---\n"
        "name: Moved\n"
        "unit_primitive: duo_chat\n"
        "prompt_template:\n"
        f"  system: \"{{% include '{feature}/system/1.0.0.jinja' %}}\"\n"
    )
    (prompts / "system" / "1.0.0.jinja").write_text("Hello {{ target }}")
    return prompts


class TestDiscovery:
    def test_registers_each_feature_prompts_dir(self, tmp_path: Path):
        features = tmp_path / "ai" / "features"
        _write_moved_feature(features, "cli", "my_feature")
        _write_moved_feature(features, "code", "other_feature")

        feature_roots.discover_feature_prompts(features)

        assert feature_roots.feature_prompt_root("my_feature") == (
            features / "cli" / "my_feature" / "prompts"
        )
        assert feature_roots.feature_prompt_root("other_feature") == (
            features / "code" / "other_feature" / "prompts"
        )

    def test_missing_tree_is_silent(self, tmp_path: Path):
        feature_roots.discover_feature_prompts(tmp_path / "does_not_exist")
        assert feature_roots.feature_prompt_root("anything") is None

    def test_duplicate_feature_id_raises(self, tmp_path: Path):
        feature_roots.register_prompt_root("dup", tmp_path / "a")
        feature_roots.register_prompt_root("dup", tmp_path / "a")  # same path is fine
        with pytest.raises(ValueError, match="Duplicate prompt feature id"):
            feature_roots.register_prompt_root("dup", tmp_path / "b")

    def test_default_features_dir_anchors_on_pyproject(self):
        features = feature_roots.default_features_dir()

        assert features.name == "features"
        assert features.parent.name == "ai"
        assert (features.parent.parent / "pyproject.toml").is_file()

    def test_default_features_dir_falls_back_without_marker(
        self, monkeypatch, tmp_path: Path
    ):
        # No marker (wheel install, faked filesystem): fall back to the
        # fixed-depth derivation instead of failing registry construction.
        orphan = tmp_path / "a" / "b" / "feature_roots.py"
        orphan.parent.mkdir(parents=True)
        monkeypatch.setattr(feature_roots, "__file__", str(orphan))

        assert feature_roots.default_features_dir() == tmp_path / "ai" / "features"

    def test_feature_without_prompts_dir_is_skipped(self, tmp_path: Path):
        features = tmp_path / "ai" / "features"
        _write_moved_feature(features, "cli", "with_prompts")
        # a flow-only feature: has config/ but no prompts/
        (features / "insights" / "flow_only" / "config").mkdir(parents=True)

        feature_roots.discover_feature_prompts(features)

        assert feature_roots.feature_prompt_root("with_prompts") is not None
        assert feature_roots.feature_prompt_root("flow_only") is None


class TestResolveId:
    def test_resolves_moved_feature_family_and_base(self, tmp_path: Path):
        features = tmp_path / "ai" / "features"
        prompts = _write_moved_feature(features, "cli", "my_feature")
        (prompts / "amazon_q").mkdir()
        (prompts / "amazon_q" / "1.0.0.yml").write_text(
            "---\nname: Q\nunit_primitive: duo_chat\nprompt_template:\n  system: hi\n"
        )
        feature_roots.discover_feature_prompts(features)
        registry = _make_registry()

        assert registry._resolve_id("my_feature", family=["amazon_q"]) == (
            prompts / "amazon_q"
        )
        # falls back to base when the family folder is absent
        assert registry._resolve_id("my_feature", family=["missing"]) == (
            prompts / "base"
        )

    def test_unregistered_id_uses_legacy_definitions_root(self):
        registry = _make_registry()
        with pytest.raises(FileNotFoundError, match="definitions"):
            registry._resolve_id("not_a_real_prompt", family=[])


class TestJinjaIncludeLoader:
    def test_self_namespaced_include_resolves_from_moved_root(self, tmp_path: Path):
        features = tmp_path / "ai" / "features"
        _write_moved_feature(features, "cli", "my_feature")
        feature_roots.discover_feature_prompts(features)

        rendered = base.jinja_env.from_string(
            "{% include 'my_feature/system/1.0.0.jinja' %}"
        ).render(target="world")
        assert rendered == "Hello world"

    def test_missing_file_under_registered_root_raises(self, tmp_path: Path):
        features = tmp_path / "ai" / "features"
        _write_moved_feature(features, "cli", "my_feature")
        feature_roots.discover_feature_prompts(features)

        with pytest.raises(TemplateNotFound):
            base.FeatureRootLoader().get_source(
                base.jinja_env, "my_feature/system/does_not_exist.jinja"
            )

    def test_traversal_outside_registered_root_raises(self, tmp_path: Path):
        features = tmp_path / "ai" / "features"
        _write_moved_feature(features, "cli", "my_feature")
        (tmp_path / "secret.txt").write_text("secret")
        feature_roots.discover_feature_prompts(features)

        with pytest.raises(TemplateNotFound):
            base.FeatureRootLoader().get_source(
                base.jinja_env, "my_feature/../../../../secret.txt"
            )

    def test_absolute_fragment_outside_registered_root_raises(self, tmp_path: Path):
        # Path("/root") / "/etc/x" discards the left side entirely, so an
        # absolute fragment escapes without any "..". The containment check
        # must catch it regardless of check ordering in get_source.
        features = tmp_path / "ai" / "features"
        _write_moved_feature(features, "cli", "my_feature")
        secret = tmp_path / "secret.txt"
        secret.write_text("secret")
        feature_roots.discover_feature_prompts(features)

        with pytest.raises(TemplateNotFound):
            base.FeatureRootLoader().get_source(base.jinja_env, f"my_feature/{secret}")

    def test_legacy_include_still_resolves(self):
        # An existing definitions/ template resolves through the same ChoiceLoader.
        source, _, _ = base.jinja_loader.get_source(
            base.jinja_env, "generate_commit_message/system/1.0.0.jinja"
        )
        assert source


class TestMovedFeatureEndToEnd:
    """Exercises the real moved glab_ask_git_command feature files."""

    def test_get_required_variables_follows_moved_include(self):
        feature_roots.discover_feature_prompts()
        registry = _make_registry()

        variables = registry.get_required_variables(
            "glab_ask_git_command", prompt_version="^1.0.0"
        )
        # Variables come from the system/user includes under the moved root.
        assert isinstance(variables, set)
        assert variables
