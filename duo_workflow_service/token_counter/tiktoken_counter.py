from typing import Dict, List

import structlog
import tiktoken
from langchain_community.adapters.openai import convert_message_to_dict
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


class TikTokenCounter:
    AGENT_TOKEN_MAP: Dict[str, int] = {
        "context_builder": 4735,
        "planner": 823,
        "executor": 5650,
        "replacement_agent": 1000,
        "Chat Agent": 2500,
    }

    def __init__(self, agent_name: str, model: str = "gpt-4o"):
        self.tool_tokens = self.AGENT_TOKEN_MAP.get(agent_name, 0)
        self._logger = structlog.stdlib.get_logger("tiktoken_counter")
        self._encoding = tiktoken.encoding_for_model(model)

    def count_string_content(
        self, content: str, n_samples: int = 30, sample_size: int = 300
    ) -> int:
        threshold = n_samples * sample_size

        if len(content) <= threshold:
            return len(self._encoding.encode(content))

        # Divide file into n_samples equal cells, sample from center of each
        cell_size = len(content) // n_samples
        offset = (cell_size - sample_size) // 2

        densities = []
        for i in range(n_samples):
            start_pos = i * cell_size + offset
            sample = content[start_pos : start_pos + sample_size]
            tokens = len(self._encoding.encode(sample))
            densities.append(tokens / len(sample))

        # Use trimmed mean
        trim = int(n_samples * 0.1)
        trimmed_densities = densities[trim:-trim]
        if not trimmed_densities:
            avg_density = sum(densities) / len(densities)
        else:
            avg_density = sum(trimmed_densities) / len(trimmed_densities)

        # Add 10% safety buffer to prevent undercounts
        return int(len(content) * avg_density * 1.10)

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

    def count_tokens(
        self, prompt: List[BaseMessage], include_tool_tokens: bool = True
    ) -> int:
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
        if include_tool_tokens:
            result += self.tool_tokens
        return result
