"""Boot-time dependency-injection wiring validation.

dependency_injector leaves an unresolved ``Provide[...]`` marker as a parameter
default when an ``@inject`` target is outside the wired packages/modules. The
call then fails confusingly at request time. ``wire_and_validate`` escalates the
``UnresolvedMarkerWarning`` into a startup error so the app refuses to start
instead of failing later.
"""

import warnings
from types import ModuleType
from typing import Optional, Sequence, Union

from dependency_injector import containers
from dependency_injector.wiring import UnresolvedMarkerWarning

__all__ = ["UnwiredDependencyError", "wire_and_validate"]


class UnwiredDependencyError(RuntimeError):
    """Raised at startup when an @inject target has an unresolved provider marker."""


def wire_and_validate(
    container: containers.Container,
    *,
    modules: Optional[Sequence[Union[str, ModuleType]]] = None,
    packages: Optional[Sequence[Union[str, ModuleType]]] = None,
) -> None:
    """Wire ``container`` and raise if any injection marker stays unresolved.

    Note:
        Warnings emitted during ``container.wire()`` are intercepted; the ones
        unrelated to wiring are re-emitted afterwards instead of reaching
        ``showwarning`` directly.

    Args:
        container: The dependency_injector container to wire.
        modules: Explicit modules to wire (AIGW style).
        packages: Packages to wire recursively (DWS style).

    Raises:
        UnwiredDependencyError: If a wired ``@inject`` target has no matching provider.
    """
    with warnings.catch_warnings(record=True) as caught:
        # "default", not "always": wire() emits each unresolved marker twice from
        # the same call site, and the explicit entry still overrides host -W filters.
        warnings.simplefilter("default", UnresolvedMarkerWarning)
        container.wire(modules=modules, packages=packages, warn_unresolved=True)

    # catch_warnings(record=True) intercepts every warning, so replay the
    # unrelated ones instead of dropping them from boot logs.
    # Replay is per call: catch_warnings mutates the filter state version, which
    # invalidates any cross-call dedup registry, so repeats are not suppressed.
    unresolved = []
    for w in caught:
        if issubclass(w.category, UnresolvedMarkerWarning):
            unresolved.append(str(w.message))
        else:
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    if unresolved:
        raise UnwiredDependencyError(
            "Unresolved dependency-injection markers after wiring "
            f"(a wired @inject target has no matching provider): {'; '.join(unresolved)}"
        )
