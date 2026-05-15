import pytest

from lib.version import resolve_version


class TestResolveVersion:
    def test_exact_match(self):
        assert resolve_version(["1.0.0", "2.0.0"], "1.0.0") == "1.0.0"

    def test_caret_picks_highest_compatible(self):
        assert resolve_version(["1.0.0", "1.2.0", "2.0.0"], "^1.0.0") == "1.2.0"

    def test_tilde_stays_within_minor(self):
        assert resolve_version(["1.0.0", "1.0.5", "1.1.0"], "~1.0.0") == "1.0.5"

    def test_no_match_raises(self):
        with pytest.raises(ValueError, match="No version matching"):
            resolve_version(["1.0.0"], "^2.0.0")

    def test_range_excludes_prerelease(self):
        assert resolve_version(["1.0.0", "1.1.0-rc1"], "^1.0.0") == "1.0.0"

    def test_exact_prerelease_works(self):
        assert resolve_version(["1.1.0-rc1"], "1.1.0-rc1") == "1.1.0-rc1"

    def test_unparseable_versions_skipped(self):
        assert resolve_version(["1.0.0", "not-a-version"], "1.0.0") == "1.0.0"

    def test_empty_available_raises(self):
        with pytest.raises(ValueError, match="No version matching"):
            resolve_version([], "1.0.0")

    def test_non_pep440_exact_match(self):
        assert resolve_version(["1.0.0", "2.0.0-orbit"], "2.0.0-orbit") == "2.0.0-orbit"

    def test_non_pep440_no_match_raises(self):
        with pytest.raises(ValueError, match="not a valid PEP 440 constraint"):
            resolve_version(["1.0.0"], "2.0.0-orbit")
