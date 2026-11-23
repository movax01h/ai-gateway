from pathlib import Path
from unittest.mock import patch

import yaml
from gitlab_cloud_connector import GitLabUnitPrimitive
from langchain_core.messages import HumanMessage

from ai_gateway.model_selection.models import ModelClassProvider
from ai_gateway.prompts.config.base import PromptConfig
from duo_workflow_service.agents.prompt_adapter import ChatAgentPromptTemplate
from duo_workflow_service.entities.state import ChatWorkflowState, WorkflowStatusEnum
from duo_workflow_service.gitlab.gitlab_api import Project
from duo_workflow_service.gitlab.gitlab_service_context import GitLabServiceContext

REPO_ROOT = Path(__file__).parents[5]
ORBIT_AGENT_CONFIG = Path(__file__).parents[1] / "config/1.0.0.yml"
CLAUDE_CHAT_PROMPT = (
    REPO_ROOT / "ai_gateway/prompts/definitions/chat/agent/claude/1.0.1.yml"
)


@patch("duo_workflow_service.agents.prompt_adapter.get_model_metadata")
def test_orbit_agent_prompt_renders_through_the_chat_template(
    mock_get_model_metadata, project_mock: Project
):
    mock_get_model_metadata.return_value = None
    config = yaml.safe_load(ORBIT_AGENT_CONFIG.read_text())
    override = config["prompts"][0]["prompt_template"]["system"]
    prompt_config = PromptConfig(
        name="orbit_agent",
        unit_primitive=GitLabUnitPrimitive.DUO_AGENT_PLATFORM,
        prompt_template=yaml.safe_load(CLAUDE_CHAT_PROMPT.read_text())[
            "prompt_template"
        ],
    )
    state = ChatWorkflowState(
        plan={"steps": []},
        status=WorkflowStatusEnum.EXECUTION,
        conversation_history={"orbit_agent": [HumanMessage(content="Hello")]},
        ui_chat_log=[],
        last_human_input=None,
        goal="Test goal",
        project=project_mock,
        namespace=None,
        approval=None,
        preapproved_tools=None,
        denied_tools=None,
    )

    template = ChatAgentPromptTemplate(ModelClassProvider.LITE_LLM, prompt_config)

    with patch.object(
        GitLabServiceContext, "get_current_instance_info", return_value=None
    ):
        result = template.invoke(
            state, agent_name="orbit_agent", system_template_override=override
        )

    content = str(result.to_messages()[0].content)
    assert "<orbit>" in content
    assert "<orbit_agent_rules>" in content
