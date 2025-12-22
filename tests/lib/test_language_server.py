import pytest
from packaging.version import Version

from lib.language_server import LanguageServerVersion


@pytest.mark.asyncio
class TestLanguageServer:
    @pytest.mark.parametrize(
        ("semver", "expected_version", "supports_advanced_context"),
        [
            (None, Version("0.0.0"), False),
            ("invalid version", Version("0.0.0"), False),
            ("0.0.0", Version("0.0.0"), False),
            ("4.15.0", Version("4.15.0"), False),
            ("6.21.0", Version("6.21.0"), False),
            ("7.17.1", Version("7.17.1"), True),
            ("7.18.0", Version("7.18.0"), True),
            ("8.0.0-beta.1", Version("8.0.0-beta.1"), True),
            ("999.99.9", Version("999.99.9"), True),
        ],
    )
    async def test_supports_advanced_context(
        self, semver, expected_version, supports_advanced_context
    ):
        subject = LanguageServerVersion.from_string(semver)
        assert subject.version == expected_version
        assert subject.supports_advanced_context() == supports_advanced_context

    @pytest.mark.parametrize(
        ("semver", "expected_version", "supports_node_executor_tools"),
        [
            (None, Version("0.0.0"), False),
            ("invalid version", Version("0.0.0"), False),
            ("0.0.0", Version("0.0.0"), False),
            ("4.15.0", Version("4.15.0"), False),
            ("6.21.0", Version("6.21.0"), False),
            ("7.17.1", Version("7.17.1"), False),
            ("7.42.9", Version("7.42.9"), False),
            ("7.42.99", Version("7.42.99"), False),
            ("7.43.0", Version("7.43.0"), True),
            ("7.43.1", Version("7.43.1"), True),
            ("7.44.0", Version("7.44.0"), True),
            ("8.0.0", Version("8.0.0"), True),
            ("8.0.0-beta.1", Version("8.0.0-beta.1"), True),
            ("999.99.9", Version("999.99.9"), True),
        ],
    )
    async def test_supports_node_executor_tools(
        self, semver, expected_version, supports_node_executor_tools
    ):
        subject = LanguageServerVersion.from_string(semver)
        assert subject.version == expected_version
        assert subject.supports_node_executor_tools() == supports_node_executor_tools

    @pytest.mark.parametrize(
        ("semver", "ignore_broken_flow_schema_version"),
        [
            (None, True),
            ("invalid version", True),
            ("0.0.0", True),
            ("7.18.0", True),
            ("8.50.99999", True),
            ("8.51.0", False),
            ("8.51.0-beta.1", False),
            ("8.52.0", False),
            ("999.99.9", False),
        ],
    )
    async def ignore_broken_flow_schema_version(
        self, semver, ignore_broken_flow_schema_version
    ):
        subject = LanguageServerVersion.from_string(semver)
        assert (
            subject.ignore_broken_flow_schema_version()
            == ignore_broken_flow_schema_version
        )
