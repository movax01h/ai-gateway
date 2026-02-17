"""Shared test helpers for agent tests.

Provides a fluent interface for testing agent responses:

    result = await ask_agent(agent, state, "How many open issues?")

    result.assert_called_tool("run_glql_query")
    result.assert_tool_call_count("run_glql_query", 1)
    await result.assert_llm_validates(["Response mentions the count"])
"""

from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.messages import AIMessage

from agent_tests.llm_validator import validate_with_llm

DEFAULT_VALIDATION_MODEL = "claude-haiku-4-5-20251001"


@dataclass
class ToolCall:
    """Represents a single tool call."""

    name: str
    args: dict[str, Any]
    id: str


@dataclass
class AgentResult:
    """Wrapper around agent response with assertion helpers."""

    ai_message: AIMessage
    state: dict[str, Any]
    tool_calls: list[ToolCall] = field(default_factory=list)
    validation_model: str = ""

    @property
    def tool_names(self) -> list[str]:
        """List of tool names that were called."""
        return [tc.name for tc in self.tool_calls]

    @property
    def content(self) -> str:
        """The text content of the response."""
        if isinstance(self.ai_message.content, str):
            return self.ai_message.content
        if isinstance(self.ai_message.content, list):
            text_parts = [
                part.get("text", "")
                for part in self.ai_message.content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            return " ".join(text_parts)
        return ""

    def assert_called_tool(self, tool_name: str) -> "AgentResult":
        """Assert that a specific tool was called."""
        assert tool_name in self.tool_names, (
            f"Expected tool '{tool_name}' to be called. "
            f"Tools called: {self.tool_names or 'none'}"
        )
        return self

    def assert_not_called_tool(self, tool_name: str) -> "AgentResult":
        """Assert that a specific tool was NOT called."""
        assert tool_name not in self.tool_names, (
            f"Expected tool '{tool_name}' NOT to be called. "
            f"Tools called: {self.tool_names}"
        )
        return self

    def assert_tool_call_count(
        self,
        tool_name: str,
        expected: int | dict[str, int],
    ) -> "AgentResult":
        """Assert the number of times a tool was called.

        Args:
            tool_name: Name of the tool
            expected: Either exact count or {"min": n} / {"max": n} / {"min": n, "max": m}
        """
        actual = sum(1 for tc in self.tool_calls if tc.name == tool_name)

        if isinstance(expected, int):
            assert actual == expected, (
                f"Expected '{tool_name}' to be called {expected} times, "
                f"but was called {actual} times"
            )
        else:
            min_count = expected.get("min")
            max_count = expected.get("max")
            if min_count is not None:
                assert actual >= min_count, (
                    f"Expected '{tool_name}' to be called at least {min_count} times, "
                    f"but was called {actual} times"
                )
            if max_count is not None:
                assert actual <= max_count, (
                    f"Expected '{tool_name}' to be called at most {max_count} times, "
                    f"but was called {actual} times"
                )
        return self

    def assert_has_tool_calls(self) -> "AgentResult":
        """Assert that the agent made at least one tool call."""
        assert self.tool_calls, (
            f"Agent must call at least one tool. Response: {self.content[:500]}"
        )
        return self

    async def assert_llm_validates(self, criteria: list[str]) -> "AgentResult":
        """Assert response passes LLM-as-judge validation.

        Args:
            criteria: List of criteria to validate (plain English statements)
        """
        model = self.validation_model or DEFAULT_VALIDATION_MODEL
        result = await validate_with_llm(self.content, criteria, model=model)
        if not result.all_passed:
            failed = [r for r in result.results if not r.passed]
            failures = "\n".join(f"  - {r.criterion}: {r.explanation}" for r in failed)
            error_detail = f"\n\nError: {result.error}" if result.error else ""
            raise AssertionError(
                f"LLM validation failed:\n{failures}{error_detail}"
                f"\n\nResponse: {self.content[:1000]}"
            )
        return self


async def ask_agent(
    agent: Any,
    initial_state_factory: Callable[[str], dict],
    question: str,
    max_turns: int = 5,
    validation_model: str = "",
) -> AgentResult:
    """Ask the agent a question and run until final response.

    Runs the agent loop, executing tool calls and continuing until the agent
    produces a final text response without tool calls, or max_turns is reached.

    Args:
        agent: The ChatAgent instance
        initial_state_factory: Factory function that creates initial state from a question
        question: The question to ask
        max_turns: Maximum number of agent turns to prevent infinite loops

    Returns:
        AgentResult with assertion helpers (includes all tool calls from all turns)
    """
    from langchain_core.messages import (  # pylint: disable=import-outside-toplevel
        ToolMessage,
    )

    agent_name = agent.name
    state = initial_state_factory(question, agent_name=agent_name)
    all_tool_calls: list[ToolCall] = []

    result = None
    ai_message = None

    tools_by_name = {tool.name: tool for tool in agent.prompt_adapter._tools}

    for _ in range(max_turns):
        result = await agent.run(state)
        ai_message = result["conversation_history"][agent_name][-1]

        if ai_message.tool_calls:
            for tc in ai_message.tool_calls:
                all_tool_calls.append(
                    ToolCall(
                        name=tc["name"],
                        args=tc.get("args", {}),
                        id=tc.get("id") or "",
                    )
                )

            tool_results = []
            for tc in ai_message.tool_calls:
                tool_name = tc["name"]
                tool = tools_by_name.get(tool_name)
                if tool:
                    try:
                        tool_result = await tool.ainvoke(tc.get("args", {}))
                        tool_results.append(
                            ToolMessage(
                                content=str(tool_result),
                                tool_call_id=tc.get("id") or "",
                            )
                        )
                    except Exception as e:
                        tool_results.append(
                            ToolMessage(
                                content=f"Error: {e}",
                                tool_call_id=tc.get("id") or "",
                            )
                        )
                else:
                    tool_results.append(
                        ToolMessage(
                            content="Tool not found",
                            tool_call_id=tc.get("id") or "",
                        )
                    )

            current_history = state["conversation_history"][agent_name]
            current_history.append(ai_message)
            current_history.extend(tool_results)
            state = {
                **state,
                "conversation_history": {agent_name: current_history},
            }
        else:
            break

    if ai_message is None or result is None:
        raise RuntimeError("Agent did not produce any response")

    if ai_message.tool_calls:
        import warnings  # pylint: disable=import-outside-toplevel

        warnings.warn(
            f"Agent did not produce a final text response within {max_turns} turns. "
            f"Last action was a tool call to: {[tc['name'] for tc in ai_message.tool_calls]}",
            stacklevel=2,
        )

    agent_result = AgentResult(
        ai_message=ai_message,
        state=result,
        validation_model=validation_model or DEFAULT_VALIDATION_MODEL,
    )
    agent_result.tool_calls = all_tool_calls
    return agent_result
