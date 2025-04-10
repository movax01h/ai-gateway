import os
from datetime import datetime, timezone
from enum import StrEnum
from itertools import chain
from typing import Any, Union

import yaml
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from duo_workflow_service.agents import Agent, RunToolNode, ToolsExecutor
from duo_workflow_service.agents.handover import HandoverAgent
from duo_workflow_service.components import ToolsRegistry
from duo_workflow_service.entities import (
    MAX_CONTEXT_TOKENS,
    MessageTypeEnum,
    SearchAndReplaceConfig,
    SearchAndReplaceWorkflowState,
    ToolStatus,
    UiChatLog,
    WorkflowStatusEnum,
)
from duo_workflow_service.entities.state import Plan
from duo_workflow_service.llm_factory import new_chat_client
from duo_workflow_service.token_counter.approximate_token_counter import (
    ApproximateTokenCounter,
)
from duo_workflow_service.tracking.errors import log_exception
from duo_workflow_service.workflows.abstract_workflow import AbstractWorkflow
from duo_workflow_service.workflows.search_and_replace.prompts import (
    SEARCH_AND_REPLACE_FILE_USER_MESSAGE,
    SEARCH_AND_REPLACE_SYSTEM_MESSAGE,
    SEARCH_AND_REPLACE_USER_GUIDELINES,
)

# Constants
AGENT_NAME = "replacement_agent"
CONFIG_PATH = ".duo_workflow/search_and_replace_config.yml"
DEBUG = os.getenv("DEBUG")
MAX_MESSAGE_LENGTH = 200
MAX_TOKENS_TO_SAMPLE = 4096
RECURSION_LIMIT = 50000


# RunToolNode input builders & output parsers
def _scan_directory_tree_input_parser(state: SearchAndReplaceWorkflowState):
    if not state["config"]:
        raise RuntimeError(
            "Failed to load config, ensure that %s file is present", CONFIG_PATH
        )

    input_args = []
    # It appears thaT git ls-files does not support globs without
    # altering bash config, therefore additional tool call is needed to
    # combine search for both subdirectories as well as scan root directory
    for file_type in state["config"].file_types:
        # search in scan root directory
        input_args.append(
            {
                "directory": "N/A",
                "name_pattern": f"{state['directory']}/{file_type}",
            }
        )
        # search in subdirectories
        input_args.append(
            {
                "directory": "N/A",
                "name_pattern": f"{state['directory']}/**/{file_type}",
            }
        )
    return input_args


def _scan_directory_tree_output_parser(outputs, state: SearchAndReplaceWorkflowState):
    return {
        "pending_files": [
            file_path
            for file_path in chain(
                *[content.strip().split("\n") for content in outputs]
            )
            if len(file_path) > 0 and not file_path.isspace()
        ]
    }


def _detect_affected_components_input_parser(state: SearchAndReplaceWorkflowState):
    if not state["config"]:
        raise RuntimeError(
            "Failed to load config, ensure that %s file is present", CONFIG_PATH
        )

    return [
        {
            "pattern": replacement_rule.element,
            "search_directory": state["pending_files"][0],
            "flags": ["-n"],
        }
        for replacement_rule in state["config"].replacement_rules
    ]


def _build_affected_components_messages(
    components: str, state: SearchAndReplaceWorkflowState
) -> list[Union[SystemMessage, HumanMessage]]:
    if not state["config"]:
        raise RuntimeError(
            "Failed to load config, ensure that %s file is present", CONFIG_PATH
        )

    system_prompt = SEARCH_AND_REPLACE_SYSTEM_MESSAGE.format(
        domain_speciality=state["config"].domain_speciality,
        assignment_description=state["config"].assignment_description,
    )

    template = """
        <guideline element='{element}'>
            {guideline}
        </guideline>
    """

    guidelines = "\n".join(
        [
            template.format(
                guideline=replacement_rule.rules, element=replacement_rule.element
            )
            for replacement_rule in state["config"].replacement_rules
        ]
    )

    human_prompt = SEARCH_AND_REPLACE_USER_GUIDELINES.format(
        reviewable_components=components,
        guidelines=guidelines,
        elements=", ".join(
            [rule.element for rule in state["config"].replacement_rules]
        ),
    )
    return [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]


def _backwards_compatible_ui_message(message: str):
    return AIMessage(
        content=[
            {
                "id": "toolu_01XXHoxWjj3c6izbUJdJEK9d",
                "name": "handover_tool",
                "type": "tool_use",
                "input": {"summary": message, "description": message},
            }
        ]
    )


def _log_ai_message(state: SearchAndReplaceWorkflowState):
    hisotry = state["conversation_history"]
    messages = hisotry.get(AGENT_NAME, [])
    if len(messages) == 0:
        return {"conversation_history": hisotry}

    last_msg = hisotry[AGENT_NAME][-1].content
    if not isinstance(last_msg, str):
        last_msg = (  # type: ignore
            last_msg[0].get("text")  # type :ignore
            if isinstance(last_msg[0], dict)  # type: ignore
            else last_msg[0]  # type :ignore
        )  # type: ignore

    if not isinstance(last_msg, str):
        return {
            "conversation_history": {
                AGENT_NAME: hisotry[AGENT_NAME],
            }
        }

    return {
        "ui_chat_log": [
            UiChatLog(
                message_type=MessageTypeEnum.AGENT,
                content=last_msg,
                timestamp=datetime.now(timezone.utc).isoformat(),
                status=ToolStatus.SUCCESS,
                correlation_id=None,
                tool_info=None,
            )
        ],
        "conversation_history": {
            AGENT_NAME: hisotry[AGENT_NAME],
            "planner": [_backwards_compatible_ui_message(last_msg)],
        },
    }


def _detect_affected_components_output_parser(
    outputs, state: SearchAndReplaceWorkflowState
):
    components = ""

    for grep_result in outputs:
        if (
            "No matches found for pattern" in grep_result
            or "Error running git grep" in grep_result
        ):
            # Skip empty results
            continue
        components += f"\n{grep_result}"

    logs = [
        UiChatLog(
            message_type=MessageTypeEnum.TOOL,
            content=f"Scanned file: {state['pending_files'][0]}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=ToolStatus.SUCCESS,
            correlation_id=None,
            tool_info=None,
        )
    ]

    if DEBUG:
        components_cnt = len(components.split("\n"))
        logs.append(
            UiChatLog(
                message_type=MessageTypeEnum.TOOL,
                content=f"Detected {components_cnt} components that require accessibility review.",
                timestamp=datetime.now(timezone.utc).isoformat(),
                status=ToolStatus.SUCCESS,
                correlation_id=None,
                tool_info=None,
            )
        )

    if len(components) == 0:
        state["pending_files"].pop(0)
        return {
            "pending_files": state["pending_files"],
            "ui_chat_log": logs,
            "conversation_history": {},
        }

    return {
        "ui_chat_log": logs,
        "conversation_history": {
            AGENT_NAME: _build_affected_components_messages(components, state)
        },
    }


def _append_affected_file(
    file_contents: list[str], state: SearchAndReplaceWorkflowState
):
    logs: list[UiChatLog] = []
    current_file = state["pending_files"][0]

    if not state["config"]:
        raise RuntimeError(
            "Failed to load config, ensure that %s file is present", CONFIG_PATH
        )

    human_prompt = SEARCH_AND_REPLACE_FILE_USER_MESSAGE.format(
        file_path=current_file,
        file_content=file_contents[0],
        elements=", ".join(
            [rule.element for rule in state["config"].replacement_rules]
        ),
    )

    state["pending_files"].pop(0)
    messages = [
        *state["conversation_history"][AGENT_NAME],
        HumanMessage(content=human_prompt),
    ]

    if ApproximateTokenCounter(AGENT_NAME).count_tokens(messages) > MAX_CONTEXT_TOKENS:
        messages = []
        logs.append(
            UiChatLog(
                message_type=MessageTypeEnum.TOOL,
                content=f"File too large, skipping {current_file}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                status=ToolStatus.FAILURE,
                correlation_id=None,
                tool_info=None,
            )
        )
    else:
        logs.append(
            UiChatLog(
                message_type=MessageTypeEnum.TOOL,
                content=f"Loaded {current_file}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                status=ToolStatus.SUCCESS,
                correlation_id=None,
                tool_info=None,
            )
        )

    return {
        "conversation_history": {
            AGENT_NAME: messages,
            "planner": [_backwards_compatible_ui_message(f"Processing {current_file}")],
        },
        "pending_files": state["pending_files"],
        "ui_chat_log": logs,
    }


# ROUTERS
class Routes(StrEnum):
    CONTINUE = "continue"
    END = "end"
    SKIP = "skip"


def _pending_files_present(state: SearchAndReplaceWorkflowState) -> str:
    if len(state["pending_files"]) > 0:
        return Routes.CONTINUE

    return Routes.END


def _prompt_present(state: SearchAndReplaceWorkflowState) -> str:
    if (
        AGENT_NAME in state["conversation_history"]
        and len(state["conversation_history"][AGENT_NAME]) > 0
    ):
        return Routes.CONTINUE

    if len(state["pending_files"]) > 0:
        return Routes.SKIP

    return Routes.END


def _patches_present(state: SearchAndReplaceWorkflowState) -> str:
    if state["status"] == WorkflowStatusEnum.CANCELLED:
        return Routes.END

    if AGENT_NAME in state["conversation_history"]:
        last_message = state["conversation_history"][AGENT_NAME][-1]
        if len(getattr(last_message, "tool_calls", [])) > 0:
            return Routes.CONTINUE

    if len(state["pending_files"]) > 0:
        return Routes.SKIP

    return Routes.END


class Workflow(AbstractWorkflow):
    async def _handle_workflow_failure(
        self, error: BaseException, compiled_graph: Any, graph_config: Any
    ):
        log_exception(error, extra={"workflow_id": self._workflow_id})

    def _recursion_limit(self):
        return RECURSION_LIMIT

    def _compile(
        self,
        goal: str,
        tools_registry: ToolsRegistry,
        checkpointer: BaseCheckpointSaver,
    ):
        graph = StateGraph(SearchAndReplaceWorkflowState)

        # Setup workflow graph
        graph.set_entry_point("load_config")
        # Load config is a temporary workaround for a lack of UI
        graph.add_node(
            "load_config",
            RunToolNode[SearchAndReplaceWorkflowState](
                tool=tools_registry.get("read_file"),  # type: ignore
                input_parser=lambda _: [{"file_path": CONFIG_PATH}],
                output_parser=lambda content, _: {
                    "config": SearchAndReplaceConfig(**yaml.safe_load(content[0]))
                },
            ).run,
        )

        graph.add_edge("load_config", "scan_directory_tree")

        # this is search part of a pipeline
        graph.add_node(
            "scan_directory_tree",
            RunToolNode[SearchAndReplaceWorkflowState](
                tool=tools_registry.get("find_files"),  # type: ignore
                input_parser=_scan_directory_tree_input_parser,
                output_parser=_scan_directory_tree_output_parser,
            ).run,
        )
        graph.add_conditional_edges(
            "scan_directory_tree",
            _pending_files_present,
            {
                "end": "complete",
                "continue": "detect_affected_components",
            },
        )

        graph.add_node(
            "detect_affected_components",
            RunToolNode[SearchAndReplaceWorkflowState](
                tool=tools_registry.get("grep_files"),  # type: ignore
                input_parser=_detect_affected_components_input_parser,
                output_parser=_detect_affected_components_output_parser,
            ).run,
        )
        graph.add_conditional_edges(
            "detect_affected_components",
            _prompt_present,
            {
                "skip": "detect_affected_components",
                "continue": "append_affected_file",
                "end": "complete",
            },
        )
        # this is replace part of a pipeline
        graph.add_node(
            "append_affected_file",
            RunToolNode(
                tool=tools_registry.get("read_file"),  # type: ignore
                input_parser=lambda state: [{"file_path": state["pending_files"][0]}],
                output_parser=_append_affected_file,
            ).run,
        )
        graph.add_conditional_edges(
            "append_affected_file",
            _prompt_present,
            {
                "skip": "detect_affected_components",
                "continue": "request_patch",
                "end": "complete",
            },
        )

        accessibility_tools = [
            "edit_file",
        ]
        agent = Agent(
            goal="N/A",  # "Not used, Agent always gets prepared messages from previous steps",
            system_prompt="N/A",
            name=AGENT_NAME,
            model=new_chat_client(max_tokens=MAX_TOKENS_TO_SAMPLE),
            tools=tools_registry.get_batch(accessibility_tools),
            http_client=self._http_client,
            workflow_id=self._workflow_id,
        )
        graph.add_node("request_patch", agent.run)
        graph.add_edge("request_patch", "log_agent_response")
        graph.add_node("log_agent_response", _log_ai_message)

        graph.add_conditional_edges(
            "log_agent_response",
            _patches_present,
            {
                "skip": "detect_affected_components",
                "continue": "apply_patch",
                "end": "complete",
            },
        )
        apply_patch = ToolsExecutor(
            tools_agent_name=AGENT_NAME,
            agent_tools=tools_registry.get_handlers(accessibility_tools),
            workflow_id=self._workflow_id,
        ).run

        graph.add_node("apply_patch", apply_patch)
        graph.add_conditional_edges(
            "apply_patch",
            _pending_files_present,
            {
                "continue": "detect_affected_components",
                "end": "complete",
            },
        )
        graph.add_node(
            "complete",
            HandoverAgent(
                new_status=WorkflowStatusEnum.COMPLETED,
                handover_from=AGENT_NAME,
                include_conversation_history=True,
            ).run,
        )
        graph.add_edge("complete", END)

        return graph.compile(checkpointer=checkpointer)

    def get_workflow_state(self, goal: str) -> SearchAndReplaceWorkflowState:
        target_dir = goal
        initial_ui_chat_log = UiChatLog(
            message_type=MessageTypeEnum.TOOL,
            content=f"Starting UI accessibility workflow from directory: {target_dir}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=ToolStatus.SUCCESS,
            correlation_id=None,
            tool_info=None,
        )

        return SearchAndReplaceWorkflowState(
            config=None,
            directory=target_dir,
            pending_files=[],
            status=WorkflowStatusEnum.NOT_STARTED,
            ui_chat_log=[initial_ui_chat_log],
            conversation_history={},
            plan=Plan(steps=[]),
            handover=[],
        )

    def log_workflow_elements(self, element):
        self.log.info("###############################")
        if "ui_chat_log" in element:
            for log in element["ui_chat_log"]:
                self.log.info(
                    f"%s: %{'' if DEBUG else f'.{MAX_MESSAGE_LENGTH}'}s",
                    log["message_type"],
                    log["content"],
                )
