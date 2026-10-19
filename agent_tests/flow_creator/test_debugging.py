"""Debugging test: the agent must name the rule a broken flow violates.

The last acceptance criterion of
https://gitlab.com/gitlab-org/gitlab/-/work_items/604709 is that the agent
"correctly identifies rule violations by name when given a broken YAML". Its own
prompt requires this: "say 'this violates Rule 3 - the router after the
HumanInputComponent is missing the modify route' rather than describing it
generically."

The flow below is deliberately correct in every other respect - project_id is
threaded, the prompts declare unit_primitives, placeholder: history, matching
Jinja2 placeholders and stopping instructions, and sends_response_to points at a
component that has already run - so that the only rule broken is Rule 3, and a
response naming any other rule is genuinely wrong rather than differently right.
"""

import pytest

from agent_tests.flow_creator.helpers import run_conversation

BROKEN_FLOW_MISSING_MODIFY_ROUTE = """\
version: "v1"
environment: ambient
components:
  - name: "reviewer"
    type: AgentComponent
    prompt_id: "reviewer_prompt"
    inputs:
      - from: "context:goal"
        as: "goal"
      - from: "context:project_id"
        as: "project_id"
    toolset:
      - get_merge_request
      - list_merge_request_diffs
    ui_log_events:
      - "on_agent_final_answer"
  - name: "review_gate"
    type: HumanInputComponent
    interaction_type: "approval"
    sends_response_to: "reviewer"
    message_template: |
      Please review this code review summary before it is posted.
    ui_log_events:
      - "on_user_input_prompt"
      - "on_user_response"
  - name: "publisher"
    type: AgentComponent
    prompt_id: "publisher_prompt"
    inputs:
      - from: "context:reviewer.final_answer"
        as: "review_summary"
      - from: "context:project_id"
        as: "project_id"
    toolset:
      - create_merge_request_note
    ui_log_events:
      - "on_agent_final_answer"
routers:
  - from: "reviewer"
    to: "review_gate"
  - from: "review_gate"
    condition:
      input: "context:review_gate.approval"
      routes:
        "approve": "publisher"
        "reject": "end"
        "default_route": "end"
  - from: "publisher"
    to: "end"
flow:
  entry_point: "reviewer"
prompts:
  - name: Reviewer
    prompt_id: "reviewer_prompt"
    unit_primitives:
      - duo_agent_platform
    prompt_template:
      system: |
        You review merge requests and write a concise review summary.
        When you have written the summary, your final answer is the summary
        itself. No further steps are needed after that.
      user: |
        Project ID: {{project_id}}
        Task: {{goal}}
      placeholder: history
  - name: Publisher
    prompt_id: "publisher_prompt"
    unit_primitives:
      - duo_agent_platform
    prompt_template:
      system: |
        You post a review summary as a merge request comment.
        When the comment is posted, your final answer is the comment URL.
        No further steps are needed after that.
      user: |
        Project ID: {{project_id}}
        Summary to post: {{review_summary}}
      placeholder: history
"""

DEBUGGING_PROMPT = (
    "This flow is not working. When I click 'Request changes' on the approval "
    "gate, the session ends instead of revising the summary. Here is the YAML - "
    "what is wrong with it?\n\n"
    f"```yaml\n{BROKEN_FLOW_MISSING_MODIFY_ROUTE}```"
)


@pytest.mark.asyncio
async def test_names_the_violated_rule_for_a_broken_hitl_gate(
    flow_registry_agent,
    initial_state,
    validation_model,
):
    """The agent must name Rule 3 and the missing ``modify`` route."""
    conversation = await run_conversation(
        flow_registry_agent,
        initial_state,
        [DEBUGGING_PROMPT],
        validation_model=validation_model,
    )

    await conversation.result.assert_llm_validates(
        [
            "The response explicitly names Rule 3 as the rule that is violated, "
            "using the rule number (for example 'this violates Rule 3'). A "
            "response that describes the problem without naming a numbered rule "
            "does not satisfy this criterion.",
            "The response identifies that the conditional router after the "
            "HumanInputComponent is missing the 'modify' route, and that 'modify' "
            "should route back to the agent that produced the output being "
            "reviewed.",
        ]
    )
