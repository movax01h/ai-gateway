from typing import Dict, List

import structlog
from langchain_community.adapters.openai import convert_message_to_dict
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


class ApproximateTokenCounter:
    AGENT_TOKEN_MAP: Dict[str, int] = {
        "context_builder": 4735,
        "planner": 823,
        "executor": 5650,
        "replacement_agent": 1000,
    }

    def __init__(self, agent_name: str):
        self.tool_tokens = self.AGENT_TOKEN_MAP.get(agent_name, 0)
        self._logger = structlog.stdlib.get_logger("approximate_token_counter")

    def count_string_content(self, content: str) -> int:
        estimated_tokens = len(content) // 4

        return int(round(estimated_tokens * 1.5))

    def count_tokens_in_list(self, content_list: list) -> int:
        result = 0
        for item in content_list:
            if isinstance(item, dict):
                result += self.count_tokens_in_dict(item)
            elif isinstance(item, str):
                result += self.count_string_content(item)
            else:
                self._logger.debug(
                    f"Unexpected type {type(item)} in list item",
                    item=item,
                )

        return result

    def count_tokens_in_dict(self, content: dict) -> int:
        result = 0
        for key, value in content.items():
            if isinstance(value, str):
                result += self.count_string_content(value)
            elif isinstance(value, list):
                result += self.count_tokens_in_list(value)
            elif isinstance(value, dict):
                result += self.count_tokens_in_dict(value)

        return result

    def count_tokens(self, prompt: List[BaseMessage]) -> int:
        result = 0
        for message in prompt:
            if isinstance(
                message, (SystemMessage, HumanMessage, AIMessage, ToolMessage)
            ):
                try:
                    message_dict = convert_message_to_dict(message)
                except TypeError as e:
                    self._logger.debug(f"Could not convert message to dictionary: {e}")
                    message_dict = {}
                token = self.count_tokens_in_dict(message_dict)
                result += token

        return result + self.tool_tokens
