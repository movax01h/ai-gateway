"""Test module for search and replace workflow components."""

from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from duo_workflow_service.components.tools_registry import (
    _AGENT_PRIVILEGES,
    ToolsRegistry,
)
from duo_workflow_service.entities import (
    MessageTypeEnum,
    ToolStatus,
    WorkflowStatusEnum,
)
from duo_workflow_service.entities.state import MAX_CONTEXT_TOKENS, ReplacementRule
from duo_workflow_service.internal_events.event_enum import CategoryEnum
from duo_workflow_service.workflows.search_and_replace.prompts import (
    SEARCH_AND_REPLACE_FILE_USER_MESSAGE,
)
from duo_workflow_service.workflows.search_and_replace.workflow import (
    Routes,
    SearchAndReplaceConfig,
    Workflow,
    _append_affected_file,
    _build_affected_components_messages,
    _detect_affected_components_input_parser,
    _detect_affected_components_output_parser,
    _patches_present,
    _pending_files_present,
    _prompt_present,
    _scan_directory_tree_input_parser,
    _scan_directory_tree_output_parser,
)


@pytest.fixture
def tools_registry_with_all_privileges(tool_metadata):
    return ToolsRegistry(
        tool_metadata=tool_metadata,
        enabled_tools=list(_AGENT_PRIVILEGES.keys()),
        preapproved_tools=list(_AGENT_PRIVILEGES.keys()),
    )


@pytest.fixture
def mock_config():
    """Create a mock search and replace configuration."""
    return SearchAndReplaceConfig(
        file_types=["*.vue"],
        domain_speciality="accessibility expert",
        assignment_description="accessibility issues",
        replacement_rules=[
            ReplacementRule(element="gl-icon", rules="Add aria-label"),
            ReplacementRule(element="gl-avatar", rules="Add alt text"),
        ],
    )


@pytest.fixture
def mock_state_with_file_types():
    """Create a mock state with multiple file types for testing directory scanning."""
    return {
        "config": SearchAndReplaceConfig(
            file_types=["*.py", "*.rb"],
            domain_speciality="code expert",
            assignment_description="code issues",
            replacement_rules=[],
        ),
        "directory": "/test/repo",
        "pending_files": ["test_file.py"],
        "conversation_history": {},
        "status": WorkflowStatusEnum.NOT_STARTED,
        "ui_chat_log": [],
        "plan": {},
    }


@pytest.fixture
def mock_state(mock_config):
    """Create a mock workflow state."""
    return {
        "config": mock_config,
        "directory": "/test/path",
        "pending_files": ["test_file.vue"],
        "conversation_history": {},
        "status": WorkflowStatusEnum.NOT_STARTED,
        "ui_chat_log": [],
        "plan": {},
    }


def test_scan_directory_tree_input_parser_missing_config(mock_state_with_file_types):
    """Test scan directory tree input parser raises error when config is missing."""
    mock_state_with_file_types["config"] = None

    with pytest.raises(RuntimeError) as excinfo:
        _scan_directory_tree_input_parser(mock_state_with_file_types)

    assert "Failed to load config" in str(excinfo.value)


def test_scan_directory_tree_input_parser_with_multiple_file_types(
    mock_state_with_file_types,
):
    """Test scan directory tree input parser generates correct patterns for both root and subdirectories."""
    result = _scan_directory_tree_input_parser(mock_state_with_file_types)

    assert len(result) == 4  # 2 file types * 2 patterns (root and subdirs)

    # Check patterns for *.py files
    assert {
        "directory": "N/A",
        "name_pattern": "/test/repo/*.py",
    } in result
    assert {
        "directory": "N/A",
        "name_pattern": "/test/repo/**/*.py",
    } in result

    # Check patterns for *.rb files
    assert {
        "directory": "N/A",
        "name_pattern": "/test/repo/*.rb",
    } in result
    assert {
        "directory": "N/A",
        "name_pattern": "/test/repo/**/*.rb",
    } in result


def test_detect_affected_components_input_parser(mock_state):
    """Test scan file input parser function."""
    result = _detect_affected_components_input_parser(mock_state)

    assert len(result) == 2
    assert result[0] == {
        "pattern": "gl-icon",
        "search_directory": "test_file.vue",
        "flags": ["-n"],
    }
    assert result[1] == {
        "pattern": "gl-avatar",
        "search_directory": "test_file.vue",
        "flags": ["-n"],
    }


def test_build_affected_components_messages(mock_state):
    """Test building scan messages."""
    components = "test components"
    messages = _build_affected_components_messages(components, mock_state)

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert "accessibility expert" in messages[0].content
    assert "accessibility issues" in messages[0].content
    assert "test components" in messages[1].content
    assert "gl-icon" in messages[1].content
    assert "gl-avatar" in messages[1].content


def test_scan_directory_tree_output_parser_empty_results(mock_state_with_file_types):
    """Test scan directory tree output parser handles empty results correctly."""
    outputs = [
        "",  # Empty result
        "\n",  # Just newline
        "   \n",  # Whitespace and newline
    ]

    result = _scan_directory_tree_output_parser(outputs, mock_state_with_file_types)

    # Should return empty list of pending files when no files are found
    assert len(result["pending_files"]) == 0


def test_scan_directory_tree_output_parser_with_results(mock_state_with_file_types):
    """Test scan directory tree output parser correctly combines multiple file lists."""
    outputs = [
        "file1.py\nfile2.py\n",  # Results from first pattern
        "subdir/file3.py\n",  # Results from second pattern
        "file1.rb\n",  # Results from third pattern
        "subdir/file2.rb\n",  # Results from fourth pattern
    ]

    result = _scan_directory_tree_output_parser(outputs, mock_state_with_file_types)

    assert len(result["pending_files"]) == 5
    assert "file1.py" in result["pending_files"]
    assert "file2.py" in result["pending_files"]
    assert "subdir/file3.py" in result["pending_files"]
    assert "file1.rb" in result["pending_files"]
    assert "subdir/file2.rb" in result["pending_files"]


def test_detect_affected_components_output_parser_with_results(mock_state):
    """Test scan file output parser with grep results."""
    outputs = ["line 1: <gl-icon>", "line 2: <gl-avatar>"]
    result = _detect_affected_components_output_parser(outputs, mock_state)

    assert "ui_chat_log" in result
    assert len(result["ui_chat_log"]) > 0
    assert result["ui_chat_log"][0]["message_type"] == MessageTypeEnum.TOOL
    assert result["ui_chat_log"][0]["status"] == ToolStatus.SUCCESS
    assert "conversation_history" in result
    assert "replacement_agent" in result["conversation_history"]


def test_detect_affected_components_output_parser_no_results(mock_state):
    """Test scan file output parser with no grep results."""
    outputs = ["No matches found for pattern"]
    result = _detect_affected_components_output_parser(outputs, mock_state)

    assert "ui_chat_log" in result
    assert len(result["ui_chat_log"]) > 0
    assert "pending_files" in result
    assert len(result["pending_files"]) == 0
    assert "conversation_history" in result
    assert result["conversation_history"] == {}


def test_pending_files_present(mock_state):
    """Test pending files present router."""
    assert _pending_files_present(mock_state) == Routes.CONTINUE

    mock_state["pending_files"] = []
    assert _pending_files_present(mock_state) == Routes.END


def test_prompt_present(mock_state):
    """Test fixable entities present router."""
    # No conversation history
    assert _prompt_present(mock_state) == Routes.SKIP

    # With conversation history but empty
    mock_state["conversation_history"] = {"replacement_agent": []}
    assert _prompt_present(mock_state) == Routes.SKIP

    # With conversation history
    mock_state["conversation_history"] = {
        "replacement_agent": [SystemMessage(content="test")]
    }
    assert _prompt_present(mock_state) == Routes.CONTINUE

    # No pending files and no conversation
    mock_state["pending_files"] = []
    mock_state["conversation_history"] = {}
    assert _prompt_present(mock_state) == Routes.END


@patch(
    "duo_workflow_service.workflows.search_and_replace.workflow.ApproximateTokenCounter"
)
def test_append_affected_file_too_large_content(mock_token_counter, mock_state):
    """Test append_affected_file functionality."""
    mock_state["conversation_history"] = {"replacement_agent": []}
    mock_token_counter.return_value.count_tokens.return_value = MAX_CONTEXT_TOKENS + 1

    # Mock file content that would exceed MAX_CONTEXT_TOKENS
    large_file_content = "A" * 10000  # Large content
    current_file = mock_state["pending_files"][0]

    result = _append_affected_file([large_file_content], mock_state)
    message_content = SEARCH_AND_REPLACE_FILE_USER_MESSAGE.format(
        file_content=large_file_content,
        elements="gl-icon, gl-avatar",
        file_path=current_file,
    )
    mock_token_counter.assert_called_once_with("replacement_agent")
    mock_token_counter.return_value.count_tokens.assert_called_once_with(
        [HumanMessage(content=message_content)]
    )
    assert "File too large" in result["ui_chat_log"][0]["content"]
    assert result["conversation_history"].get("replacement_agent") == []


@patch(
    "duo_workflow_service.workflows.search_and_replace.workflow.ApproximateTokenCounter"
)
def test_append_affected_file(mock_token_counter, mock_state):
    """Test append_affected_file functionality."""
    mock_state["conversation_history"] = {
        "replacement_agent": [SystemMessage(content="test")]
    }
    current_file = mock_state["pending_files"][0]
    mock_token_counter.return_value.count_tokens.return_value = MAX_CONTEXT_TOKENS - 1

    file_content = "Small file content"

    result = _append_affected_file([file_content], mock_state)
    message_content = SEARCH_AND_REPLACE_FILE_USER_MESSAGE.format(
        file_content=file_content, elements="gl-icon, gl-avatar", file_path=current_file
    )
    mock_token_counter.assert_called_once_with("replacement_agent")
    mock_token_counter.return_value.count_tokens.assert_called_once_with(
        [SystemMessage(content="test"), HumanMessage(content=message_content)]
    )
    assert "Loaded test_file.vue" == result["ui_chat_log"][0]["content"]
    assert "conversation_history" in result
    assert "replacement_agent" in result["conversation_history"]
    assert result["conversation_history"]["replacement_agent"] == [
        SystemMessage(content="test"),
        HumanMessage(content=message_content),
    ]


def test_patches_present(mock_state):
    """Test patches present router."""
    # No conversation history
    assert _patches_present(mock_state) == Routes.SKIP

    # With tool calls
    mock_state["conversation_history"] = {
        "replacement_agent": [
            AIMessage(
                content="test", tool_calls=[{"id": "123", "name": "test", "args": {}}]
            )
        ]
    }
    assert _patches_present(mock_state) == Routes.CONTINUE

    # Without tool calls
    mock_state["conversation_history"] = {
        "replacement_agent": [AIMessage(content="test")]
    }
    assert _patches_present(mock_state) == Routes.SKIP

    # No pending files
    mock_state["pending_files"] = []
    assert _patches_present(mock_state) == Routes.END


@pytest.fixture
def mock_tools_registry():
    """Create a mock tools registry."""
    registry = Mock()
    registry.get = Mock(return_value=Mock(name="test_tool"))
    registry.get_batch = Mock(return_value=[Mock(name="test_tool")])
    registry.get_handlers = Mock(return_value=[Mock(name="test_tool")])
    return registry


@pytest.fixture
def mock_checkpointer():
    """Create a mock checkpointer."""
    return Mock()


@pytest.mark.asyncio
@patch("duo_workflow_service.workflows.search_and_replace.workflow.create_chat_model")
@patch("duo_workflow_service.workflows.search_and_replace.workflow.Agent")
async def test_workflow_compilation(
    mock_agent, mock_new_chat_client, mock_tools_registry, mock_checkpointer
):
    """Test workflow compilation process."""
    workflow = Workflow(
        workflow_id="test_id",
        workflow_metadata={},
        workflow_type=CategoryEnum.WORKFLOW_SEARCH_AND_REPLACE,
    )

    # Compile the workflow graph
    compiled_graph = workflow._compile(
        goal="/test/path",
        tools_registry=mock_tools_registry,
        checkpointer=mock_checkpointer,
    )

    assert compiled_graph is not None
    mock_agent.assert_called_with(
        goal="N/A",
        system_prompt="N/A",
        name="replacement_agent",
        toolset=mock_tools_registry.toolset.return_value,
        model=mock_new_chat_client.return_value,
        workflow_id="test_id",
        http_client=workflow._http_client,
        workflow_type=CategoryEnum.WORKFLOW_SEARCH_AND_REPLACE.value,
    )
    mock_tools_registry.get.assert_called()  # Should call get() for tools
    mock_tools_registry.toolset.assert_called()


@pytest.mark.asyncio
async def test_workflow_initialization():
    """Test workflow initialization and state setup."""
    workflow = Workflow(
        workflow_id="test_id",
        workflow_metadata={},
        workflow_type=CategoryEnum.WORKFLOW_SEARCH_AND_REPLACE,
    )
    initial_state = workflow.get_workflow_state("/test/path")

    assert initial_state["directory"] == "/test/path"
    assert initial_state["status"] == WorkflowStatusEnum.NOT_STARTED
    assert len(initial_state["pending_files"]) == 0
    assert len(initial_state["ui_chat_log"]) == 1
    assert initial_state["conversation_history"] == {}
    assert initial_state["plan"] == {"steps": []}


@pytest.mark.asyncio
@patch("duo_workflow_service.workflows.search_and_replace.workflow.create_chat_model")
async def test_accessibility_tools(
    tools_registry_with_all_privileges, mock_checkpointer
):
    """Test that all tools used by the accessibility agent are available in the tools registry."""
    workflow = Workflow(
        workflow_id="test_id",
        workflow_metadata={},
        workflow_type=CategoryEnum.WORKFLOW_SEARCH_AND_REPLACE,
    )
    captured_tool_names = []

    # The accessibility agent is initialized with tools via `tools=tools_registry.get_batch(accessibility_tools),`
    with patch.object(
        tools_registry_with_all_privileges,
        "get_batch",
        side_effect=lambda tool_names: captured_tool_names.extend(tool_names),
    ):
        workflow._compile(
            goal="/test/path",
            tools_registry=tools_registry_with_all_privileges,
            checkpointer=mock_checkpointer,
        )

    missing_tools = []
    for tool_name in captured_tool_names:
        if tools_registry_with_all_privileges.get(tool_name) is None:
            missing_tools.append(tool_name)

    assert (
        not missing_tools
    ), f"The following tools are missing from the tools registry: {missing_tools}"


@pytest.mark.asyncio
@patch("duo_workflow_service.workflows.search_and_replace.workflow.create_chat_model")
async def test_non_accessibility_tools(
    tools_registry_with_all_privileges, mock_checkpointer
):
    """Test that all other tools used in the search and replace workflow are available in the tools registry."""
    workflow = Workflow(
        workflow_id="test_id",
        workflow_metadata={},
        workflow_type=CategoryEnum.WORKFLOW_SEARCH_AND_REPLACE,
    )

    captured_tool_names = []

    # A few nodes use RunToolNode. For each, a single tool is specified. E.g., `tool=tools_registry.get("read_file")`
    with patch.object(
        tools_registry_with_all_privileges,
        "get",
        side_effect=lambda tool_name: captured_tool_names.append(tool_name),
    ):
        workflow._compile(
            goal="/test/path",
            tools_registry=tools_registry_with_all_privileges,
            checkpointer=mock_checkpointer,
        )

    # Remove duplicates (since "read_file" is used twice)
    unique_tool_names = list(set(captured_tool_names))

    missing_tools = []
    for tool_name in unique_tool_names:
        if tools_registry_with_all_privileges.get(tool_name) is None:
            missing_tools.append(tool_name)

    assert (
        not missing_tools
    ), f"The following tools are missing from the tools registry: {missing_tools}"
