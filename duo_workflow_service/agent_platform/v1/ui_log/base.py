from abc import ABC, abstractmethod
from enum import StrEnum, auto
from typing import Any, Callable, NamedTuple, Protocol, Self

from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator

from duo_workflow_service.agent_platform.v1.state import FlowStateKeys
from duo_workflow_service.entities import UiChatLog

__all__ = [
    "LogLevels",
    "BaseUILogEvents",
    "UILogCallback",
    "BaseUILogWriter",
    "UIHistory",
]


class LogLevels(StrEnum):
    SUCCESS = auto()
    ERROR = auto()
    WARNING = auto()


class BaseUILogEvents(StrEnum):
    """Base class for UI log event enumerations.

    This class provides a base for defining event enumerations for UI logging.
    It enforces naming conventions for enum values and keys.

    Enum values must:
    - Start with 'on_' prefix
    - Have keys that are uppercase versions of the values

    Example:
        class MyUIEvents(BaseUILogEvents):
            ON_START = "on_start"  # Correct naming
            ON_END = "on_end"      # Correct naming

            # The following would raise errors:
            # START = "start"      # Missing 'on_' prefix
            # Start = "on_start"   # Key not uppercase version of value
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        for member in cls:
            if not member.value.startswith("on_"):
                raise ValueError(
                    f"All enum values must start with 'on_', but got: {member.value}"
                )

            # Validate that key is uppercase version of value
            expected_key = member.value.upper()
            if member.name != expected_key:
                raise ValueError(
                    f"Enum key '{member.name}' should be '{expected_key}' "
                    f"(uppercase of value '{member.value}')"
                )

    @staticmethod
    def _generate_next_value_(
        name: str, start: int, count: int, last_values: list[str]
    ) -> str:
        return name.lower()


class _UILogEntry(NamedTuple):
    record: UiChatLog
    event: BaseUILogEvents


class UILogCallback(Protocol):
    def __call__(self, log_entry: _UILogEntry) -> None: ...


class BaseUILogWriter[E: BaseUILogEvents](ABC):
    """Abstract base class for UI log writers.

    This class provides a foundation for implementing UI log writers that can create logs at
    different severity levels (success, error, warning) and associate them with specific events.

    Subclasses must:
    1. Implement the 'events_type' property to return the specific BaseUILogEvents enum type
    2. Implement the '_log_success', '_log_error', and '_log_warning' methods to create UiChatLog objects

    The class automatically provides methods named after each log level that can be called
    to create and log messages.

    Args:
        log_callback: Callback function that receives log entries (_UILogEntry objects)

    Example:
        ```python
        # Define a custom events enum
        class MyUIEvents(BaseUILogEvents):
            ON_START = "on_start"  # Define events with 'on_' prefix
            ON_END = "on_end"      # Keys must be uppercase version of values

        # Create a custom writer class
        class MyUILogWriter(BaseUILogWriter[MyUIEvents]):
            @property
            def events_type(self) -> type[MyUIEvents]:
                return MyUIEvents

            # Implement log methods for each severity level
            def _log_success(self, message: str, **kwargs) -> UiChatLog:
                return UiChatLog(
                    message_type="text",
                    content=message,
                    timestamp="2023-01-01T12:00:00"
                )

            def _log_error(self, message: str, **kwargs) -> UiChatLog:
                return UiChatLog(
                    message_type="text",
                    content=f"ERROR: {message}",
                    timestamp="2023-01-01T12:00:00"
                )

            def _log_warning(self, message: str, **kwargs) -> UiChatLog:
                return UiChatLog(
                    message_type="text",
                    content=f"WARNING: {message}",
                    timestamp="2023-01-01T12:00:00"
                )

        # Usage:
        # Create a callback function that will receive log entries
        def my_log_handler(log_entry):
            print(f"Event: {log_entry.event}, Message: {log_entry.record.content}")

        # Instantiate the writer
        log_writer = MyUILogWriter(my_log_handler)

        # Create logs for different events
        log_writer.success("Process started successfully", event=MyUIEvents.ON_START)
        log_writer.error("Process failed", event=MyUIEvents.ON_END)
        ```
    """

    def __init__(self, log_callback: UILogCallback):
        self._log_callback = log_callback
        self._levels = {
            LogLevels.SUCCESS: self._log_success,
            LogLevels.ERROR: self._log_error,
            LogLevels.WARNING: self._log_warning,
        }

    @property
    @abstractmethod
    def events_type(self) -> type[E]:
        raise NotImplementedError

    def _log_success(self, *args, **kwargs) -> UiChatLog:
        raise NotImplementedError

    def _log_error(self, *args, **kwargs) -> UiChatLog:
        raise NotImplementedError

    def _log_warning(self, *args, **kwargs) -> UiChatLog:
        raise NotImplementedError

    def _log(self, level: LogLevels, *args, **kwargs) -> None:
        fn = self._levels.get(level, None)
        if not fn:
            raise KeyError(f"No log function registered for the '{level.value} level'")

        event: E | None = kwargs.pop("event", None)
        if not event:
            raise ValueError(
                "Missing required keyword argument: 'event' cannot be None or empty"
            )

        if event not in self.events_type:
            raise TypeError(
                f"Expected 'event' to be an instance of {self.events_type}, got {type(event).__name__} instead"
            )

        record: UiChatLog = fn(*args, **kwargs)
        self._log_callback(_UILogEntry(record=record, event=event))

    def __getattr__(self, level: str) -> Callable[..., None]:
        if level in self._levels:
            return lambda *args, **kwargs: self._log(LogLevels(level), *args, **kwargs)

        raise AttributeError(
            f"'{self.__class__.__name__}' has no log level method '{level}'"
        )


class UIHistory[W: BaseUILogWriter, E: BaseUILogEvents](BaseModel):
    """A model for tracking UI log history.

    Maintains a history of UI log entries and provides access to a log writer.
    Only logs events that are included in the events list.

    Attributes:
        writer_class: The BaseUILogWriter subclass to use for creating logs
        events: A list of event enum values to include in the history

    Example:
        class MyEvents(BaseUILogEvents):
            ON_START = "on_start"
            ON_END = "on_end"

        class MyWriter(BaseUILogWriter[MyEvents]):
            @property
            def events_type(self) -> type[MyEvents]:
                return MyEvents

            def _create_success_log(self, message: str, **kwargs) -> UiChatLog:
                return UiChatLog(
                    message_type="text",
                    content=message,
                    timestamp="2023-01-01T12:00:00"
                )

        # Create a history tracking specific events
        history = UIHistory(writer_class=MyWriter, events=[MyEvents.ON_START])

        # Use the log writer to create logs
        history.log.success("Starting process", event=MyEvents.ON_START)

        # The following will raise an error as the ON_END event was not enabled
        history.log.success("Starting process", event=MyEvents.ON_END)

        # Access the state updates containing all logs and clear UIHistory
        state = history.pop_state_updates()
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    writer_class: type[W] | Callable[[UILogCallback], W]
    events: list[E]

    _logs: list[_UILogEntry] = PrivateAttr(default_factory=list)
    _writer: W = PrivateAttr()

    @model_validator(mode="after")
    def validate_ui_history_model(self) -> Self:
        self._writer = self.writer_class(self._add_log_to_history)

        is_valid_type = all(
            isinstance(e, self._writer.events_type) for e in self.events
        )
        if not is_valid_type:
            raise TypeError(
                f"All items in 'events' must be instances of {self._writer.events_type.__name__}"
            )

        return self

    def _add_log_to_history(self, log_entry: _UILogEntry) -> None:
        """Callback function for writers."""
        self._logs.append(log_entry)

    @property
    def log(self) -> W:
        return self._writer

    def pop_state_updates(self) -> dict[str, Any]:
        # Log only specified events
        logs = [log.record for log in self._logs if log.event in self.events]
        self._logs.clear()

        return {FlowStateKeys.UI_CHAT_LOG: logs}
