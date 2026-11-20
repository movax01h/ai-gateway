"""Parse the ``tools:`` key of an experimental flow component.

Every entry is a selector. ``server`` and ``name`` select tools of one server by exact name. A native tool is selected
under ``server: gitlab``, like a tool of the GitLab MCP server.
"""

from typing import Any, Optional

from duo_workflow_service.components.tools_registry import ToolSelector

# DWS matches an MCP tool by the prefix of its name, because the tool message carries no server name yet.
_SERVERS = ("gitlab", "orbit", "slack")
_SELECTOR_KEYS = {"server", "name", "origin", "include", "exclude", "ops", "version"}
_SUPPORTED_SELECTOR_KEYS = {"server", "name", "origin"}


def parse_component_tools(component: dict) -> Optional[list[ToolSelector]]:
    """Return the selectors of a component, or ``None`` when the component has no ``tools:`` key.

    Raises:
        ValueError: An entry is not valid.
    """
    if "tools" not in component:
        return None

    try:
        return _parse_entries(component["tools"])
    except ValueError as exc:
        raise ValueError(f'Component "{component.get("name")}": {exc}') from exc


def _parse_entries(entries: Any) -> list[ToolSelector]:
    if not isinstance(entries, list):
        raise ValueError('"tools" must be a list, use [] for no tools')

    selectors = []
    for entry in entries:
        if not isinstance(entry, dict) or not _SELECTOR_KEYS & entry.keys():
            raise ValueError(
                f'"tools" entry must be a selector with "server" and "name", got {entry!r}. '
                'A tool that needs options stays on "toolset:"'
            )
        selectors.append(_parse_selector(entry))
    return selectors


def _parse_selector(entry: dict) -> ToolSelector:
    unsupported = sorted(map(str, entry.keys() - _SUPPORTED_SELECTOR_KEYS))
    if unsupported:
        raise ValueError(f'"tools" selector key "{unsupported[0]}" is not supported')

    server = entry.get("server")
    if server not in _SERVERS:
        raise ValueError(
            f'"server" must be "gitlab", "orbit" or "slack", got {server!r}'
        )

    if entry.get("origin", "instance") != "instance":
        raise ValueError(f'"origin" can only be "instance", got {entry["origin"]!r}')

    names = entry.get("name")
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise ValueError('"name" is required and must be a list of tool names')
    if not names:
        raise ValueError('"name" must list at least one tool, use "tools: []" for none')
    for name in names:
        if "*" in name:
            raise ValueError(f'"name" takes exact tool names, got "{name}"')

    return ToolSelector(server=server, names=names)
