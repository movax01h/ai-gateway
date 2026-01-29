from typing import Dict

from langsmith.evaluation import EvaluationResult
from langsmith.schemas import Example, Run
from structlog import get_logger

from eval.routing.common import CHAT

logger = get_logger("eval.routing.validator")


def execute_routing(inputs: Dict):
    llm = CHAT.bind_tools(inputs["tools"])
    return llm.invoke(inputs["messages"])


def is_correct(run: Run, example: Example) -> EvaluationResult:
    expected_tool_name = (example.outputs or {})["tool"]["name"]
    expected_tool_input = (example.outputs or {})["tool"]["args"] or {}
    tool_calls = (run.outputs or {}).get("tool_calls")

    if not tool_calls:
        logger.debug(f"tool_calls is empty: {tool_calls}")
        return EvaluationResult(key="is_correct", score=0.0)

    for tool_call in tool_calls:
        # Check tool name
        if tool_call["name"] != expected_tool_name:
            logger.debug(
                f"Tool name is not correct: {tool_call["name"]} vs expected: {expected_tool_name}"
            )
            return EvaluationResult(key="is_correct", score=0.0)

        # Check tool input
        if tool_call["args"] != expected_tool_input:
            logger.debug(
                f"Tool input is not correct: {tool_call["args"]} vs expected: {expected_tool_input}"
            )
            return EvaluationResult(key="is_correct", score=0.0)

    return EvaluationResult(key="is_correct", score=1.0)
