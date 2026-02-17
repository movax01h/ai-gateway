"""LLM-as-judge validator for semantic validation of agent responses.

Model is configurable via --validation-model pytest CLI option.
"""

import asyncio
import re
from dataclasses import dataclass

from langchain_anthropic import ChatAnthropic


@dataclass
class LLMValidationResult:
    """Result of validating a single criterion."""

    criterion: str
    passed: bool
    explanation: str | None = None


@dataclass
class LLMValidationSummary:
    """Summary of all validation results."""

    all_passed: bool
    results: list[LLMValidationResult]
    error: str | None = None


def _build_validation_prompt(response: str, criteria: list[str]) -> str:
    """Build a validation prompt for the LLM judge."""
    criteria_list = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(criteria))

    return f"""You are a strict test validator. Your job is to evaluate if an agent's response meets specific criteria.

IMPORTANT: Be strict. Only mark PASS if the criterion is CLEARLY and UNAMBIGUOUSLY met.

=== AGENT RESPONSE TO EVALUATE ===
{response}
=== END OF RESPONSE ===

=== CRITERIA TO CHECK ===
{criteria_list}
=== END OF CRITERIA ===

For each criterion, respond with EXACTLY this format (one line per criterion, no extra text):
CRITERION_1: PASS - reason
CRITERION_2: FAIL - reason
(continue for all criteria)

Remember: Be strict. When in doubt, mark FAIL."""


def _parse_validation_response(
    llm_response: str, criteria: list[str]
) -> LLMValidationSummary:
    """Parse the LLM's validation response into structured results."""
    results: list[LLMValidationResult] = []

    for i, criterion in enumerate(criteria):
        pattern = rf"CRITERION_{i + 1}:\s*(PASS|FAIL)\s*-?\s*(.*)"
        match = re.search(pattern, llm_response, re.IGNORECASE)

        if match:
            results.append(
                LLMValidationResult(
                    criterion=criterion,
                    passed=match.group(1).upper() == "PASS",
                    explanation=match.group(2).strip() if match.group(2) else None,
                )
            )
        else:
            results.append(
                LLMValidationResult(
                    criterion=criterion,
                    passed=False,
                    explanation="Could not parse LLM response for this criterion",
                )
            )

    return LLMValidationSummary(
        all_passed=all(r.passed for r in results),
        results=results,
    )


async def validate_with_llm(
    response: str,
    criteria: list[str],
    model: str = "claude-haiku-4-5-20251001",
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> LLMValidationSummary:
    """Validate a response against multiple criteria using LLM-as-judge.

    Args:
        response: The agent response to validate.
        criteria: List of criteria to check (plain English statements).
        model: The Anthropic model to use for validation.
        max_retries: Number of retries on transient API errors.
        retry_delay: Seconds to wait between retries.

    Returns:
        LLMValidationSummary with pass/fail for each criterion.
    """
    llm = ChatAnthropic(model=model, temperature=0.0)  # type: ignore[call-arg]
    prompt = _build_validation_prompt(response, criteria)
    last_error = None

    for attempt in range(max_retries):
        try:
            result = await llm.ainvoke(prompt)

            content = result.content
            if isinstance(content, list):
                content = " ".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )

            return _parse_validation_response(content, criteria)
        except Exception as e:  # pylint: disable=broad-exception-caught
            last_error = e
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)

    return LLMValidationSummary(
        all_passed=False,
        results=[
            LLMValidationResult(
                criterion=c,
                passed=False,
                explanation="LLM validation threw an error",
            )
            for c in criteria
        ],
        error=str(last_error),
    )
