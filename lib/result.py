"""Result module for handling operation outcomes.

This module defines a type-safe approach to handling operation results inspired by Rust's Result type. It provides Ok
and Error types to represent successful and failed outcomes, and utilities for working with these types.
"""

from typing import Generic, Literal, TypeGuard, TypeVar, Union

T_co = TypeVar("T_co", covariant=True)
E_co = TypeVar("E_co", covariant=True, bound=Exception)


class Ok(Generic[T_co]):
    """Represents Ok outcome of an operation."""

    def __init__(self, value: T_co):
        """Initializes Ok with a success value."""
        self.value = value

    @property
    def error(self) -> None:
        """Returns None as Ok represents success."""
        return None

    def is_ok(self) -> Literal[True]:
        """Ok result always returns True."""
        return True

    def is_err(self) -> Literal[False]:
        """Ok result always returns False."""
        return False


class Error(Generic[E_co]):
    """Represents Error outcome of an operation."""

    def __init__(self, error: E_co):
        """Initializes Ok with a success value."""
        self.error = error

    @property
    def value(self) -> None:
        """Returns None as Error represents failure."""
        return None

    def is_ok(self) -> Literal[False]:
        """Error result always returns False."""
        return False

    def is_err(self) -> Literal[True]:
        """Error result always returns True."""
        return True


Result = Union[Ok[T_co], Error[E_co]]


def ok(result: Result[T_co, E_co]) -> TypeGuard[Ok[T_co]]:
    """Type guard for Ok result."""
    return isinstance(result, Ok)
