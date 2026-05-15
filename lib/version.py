"""Shared version resolution using Poetry's constraint syntax.

Supports exact versions ("1.0.0"), caret ("^1.0.0"), tilde ("~1.0"), and range constraints.  See
https://python-poetry.org/docs/dependency-specification/#version-constraints

Used by prompt registries, response schema registries, and flow config loading.
"""

from collections.abc import Iterable

import structlog
from poetry.core.constraints.version import Version, parse_constraint

logger = structlog.stdlib.get_logger(__name__)


def resolve_version(available: Iterable[str], version_query: str) -> str:
    """Resolve a version constraint against a set of available version strings.

    When *version_query* is a range (not a simple/exact reference), only stable
    versions are considered so that dev/rc versions are not auto-served.

    If the query is not a valid PEP 440 constraint, an exact string match
    against *available* is attempted so that non-PEP 440 names such as
    ``2.0.0-orbit`` still resolve.

    Args:
        available: Version strings to match against (e.g. from YAML filenames or dict keys).
        version_query: A Poetry-compatible version constraint, or an exact version string.

    Returns:
        The highest compatible version string.

    Raises:
        ValueError: If no compatible version is found.
    """
    available_list = list(available)

    try:
        constraint = parse_constraint(version_query)
    except ValueError:
        # version_query is not a valid PEP 440 constraint (e.g. "2.0.0-orbit").
        # Fall back to an exact string match so that arbitrarily-named config
        # files can still be selected.
        if version_query in available_list:
            logger.info(
                "Resolved version via exact match (non-PEP 440)",
                requested=version_query,
                resolved=version_query,
            )
            return version_query
        raise ValueError(
            f"Version '{version_query}' is not a valid PEP 440 constraint "
            f"and no exact match found (available: {sorted(available_list)})"
        )

    parsed: dict[str, Version] = {}
    for v in available_list:
        try:
            parsed[v] = Version.parse(v)
        except ValueError:
            logger.debug("Skipping unparsable version string", version=v)

    candidates = list(parsed.values())
    if not constraint.is_simple():
        candidates = [v for v in candidates if v.is_stable()]

    compatible = sorted(filter(constraint.allows, candidates), reverse=True)
    if not compatible:
        logger.info(
            "No compatible versions found",
            version_query=version_query,
            available=sorted(parsed.keys()),
        )
        raise ValueError(
            f"No version matching '{version_query}' "
            f"(available: {sorted(parsed.keys())})"
        )

    # Find the original string that corresponds to the highest compatible version
    chosen_version = compatible[0]
    resolved = next(
        orig for orig, parsed_ver in parsed.items() if parsed_ver == chosen_version
    )
    logger.info("Resolved version", requested=version_query, resolved=resolved)
    return resolved
