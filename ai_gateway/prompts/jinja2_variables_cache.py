import sys
import threading
from collections import OrderedDict
from typing import Callable

__all__ = ["TemplateVariablesCache", "entry_weight"]


def entry_weight(template: str, variables: frozenset[str]) -> int:
    # Not len(): CPython stores 1, 2 or 4 bytes per character, and the names can weigh many times the source.
    return (
        sys.getsizeof(template)
        + sys.getsizeof(variables)
        + sum(sys.getsizeof(name) for name in variables)
    )


class TemplateVariablesCache:
    """LRU bounded by bytes, not entries: some sources come from the client, so entries have no fixed size."""

    def __init__(self, max_bytes: int, max_entry_bytes: int) -> None:
        self._max_bytes = max_bytes
        self._max_entry_bytes = max_entry_bytes
        self._entries: OrderedDict[str, tuple[frozenset[str], int]] = OrderedDict()
        self._bytes_held = 0
        self._lock = threading.Lock()

    def get(
        self, template: str, compute: Callable[[str], frozenset[str]]
    ) -> frozenset[str]:
        with self._lock:
            entry = self._entries.get(template)
            if entry is not None:
                self._entries.move_to_end(template)
                return entry[0]

        # Outside the lock: computing reads include files.
        variables = compute(template)
        weight = entry_weight(template, variables)
        if weight <= self._max_entry_bytes:
            with self._lock:
                if template not in self._entries:
                    self._entries[template] = (variables, weight)
                    self._bytes_held += weight
                    while self._bytes_held > self._max_bytes:
                        _, (_, evicted_weight) = self._entries.popitem(last=False)
                        self._bytes_held -= evicted_weight

        return variables

    @property
    def bytes_held(self) -> int:
        with self._lock:
            return self._bytes_held

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bytes_held = 0
