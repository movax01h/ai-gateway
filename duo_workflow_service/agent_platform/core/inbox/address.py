from dataclasses import dataclass
from typing import Optional, Self

__all__ = ["Address"]

# Joins a component to the component that runs it, as in `supervisor/reviewer`.
_PLACE_SEPARATOR = "/"
# Joins a component to the ID of one running instance of it, as in `reviewer#call-7`.
_INSTANCE_SEPARATOR = "#"
# The escape itself comes first, so escaping twice never reads as escaping once.
_ESCAPES = (("%", "%25"), (_PLACE_SEPARATOR, "%2F"), (_INSTANCE_SEPARATOR, "%23"))


def _escape(text: str) -> str:
    for raw, escaped in _ESCAPES:
        text = text.replace(raw, escaped)
    return text


@dataclass(frozen=True, slots=True)
class _Segment:
    name: str
    instance: Optional[str] = None

    def __str__(self) -> str:
        if self.instance is None:
            return _escape(self.name)
        return f"{_escape(self.name)}{_INSTANCE_SEPARATOR}{_escape(self.instance)}"


@dataclass(frozen=True, slots=True)
class Address:
    """Where a component claims: a top-level component, or one run of a component that another one dispatched.

    Only the engine makes one. A top-level component's is its name, and a dispatcher makes a run's with `child` when
    it dispatches, from the ID of the call that started it. The names and the ID are escaped, so a component named
    `dev#2` or `a/b` never reads as an instance or as a nested place. A client gets the string, and sends it back
    unchanged.
    """

    _segments: tuple[_Segment, ...]

    @classmethod
    def of(cls, component_name: str) -> Self:
        """The address of a top-level component, which runs once per flow."""
        return cls((_segment(component_name, None),))

    def child(self, component_name: str, instance: str) -> Self:
        """The address of one run of a component this one dispatches.

        Args:
            component_name: The dispatched component's name.
            instance: What tells this run from the others of the same component, such as the ID of the
                `delegate_task` call that dispatched it.
        """
        return type(self)((*self._segments, _segment(component_name, instance)))

    def __str__(self) -> str:
        return _PLACE_SEPARATOR.join(str(segment) for segment in self._segments)


def _segment(component_name: str, instance: Optional[str]) -> _Segment:
    if not component_name or instance == "":
        raise ValueError(
            "An address needs a component name, and an instance needs an ID."
        )
    return _Segment(component_name, instance)
