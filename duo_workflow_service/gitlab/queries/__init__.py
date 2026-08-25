from pathlib import Path

from packaging.version import InvalidVersion, Version

from duo_workflow_service.tracking.errors import log_exception

_VERSIONED_DIR = Path(__file__).parent / "versioned"

# Loaded and sorted once at module import time — no per-call overhead.
_QUERIES: list[tuple[Version, str]] = sorted(
    [
        (Version(f.stem.replace("_", ".")), f.read_text())
        for f in _VERSIONED_DIR.glob("*.graphql")
    ],
    reverse=True,
)

# Oldest query — used when the GitLab version is unknown or unparsable.
_FALLBACK_QUERY: str = _QUERIES[-1][1]


def fetch_query_for_version(gitlab_version: str | None) -> str:
    """Return the GraphQL query string appropriate for the given GitLab version.

    Compares on the release number alone. PEP 440 sorts a pre-release below its
    release, so GitLab.com reporting "19.4.0-pre" would otherwise be served the
    19.3 query and stay a milestone behind on every new field. A pre-release
    build carries merged master code, so it has the fields of its own milestone.

    Args:
        gitlab_version: The GitLab version string (e.g. "18.8.0"), or None.

    Returns:
        GraphQL query string compatible with the detected GitLab version.
    """
    try:
        version = Version(Version(gitlab_version).base_version)  # type: ignore[arg-type]
    except (InvalidVersion, TypeError) as ex:
        log_exception(ex)
        return _FALLBACK_QUERY

    for version_key, query in _QUERIES:
        if version_key <= version:
            return query

    return _FALLBACK_QUERY
