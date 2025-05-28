import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from duo_workflow_service.entities.state import SearchAndReplaceConfig


class CustomEncoder(json.JSONEncoder):
    """Custom JSON encoder class that extends json.JSONEncoder to handle langchain object types."""

    def default(self, o: Any) -> Any:
        """Overrides the default method to provide custom encoding for specific types.

        Args:
            o: The object to encode.

        Returns:
            JSON-serializable representation of the object.
        """
        if isinstance(
            o,
            (
                SystemMessage,
                HumanMessage,
                AIMessage,
                ToolMessage,
                SearchAndReplaceConfig,
            ),
        ):
            data = o.model_dump()
            data.update({"type": o.__class__.__name__})
            return data
        return super().default(o)
