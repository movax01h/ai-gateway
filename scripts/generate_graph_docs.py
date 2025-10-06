import importlib
import logging
import os
import sys
from unittest.mock import MagicMock

import structlog
from gitlab_cloud_connector import CloudConnectorUser
from langgraph.checkpoint.memory import MemorySaver

from ai_gateway.config import Config
from ai_gateway.container import ContainerApplication
from duo_workflow_service.components import ToolsRegistry
from duo_workflow_service.server import CONTAINER_APPLICATION_PACKAGES
from lib.internal_events.event_enum import CategoryEnum

HEADER_TEXT = """
# Duo Workflow Service Graphs

These diagrams show the LangGraph structure of each Workflow in the duo_workflow_service. Do not manually edit
this file, instead update it by running `make duo-workflow-docs`.
"""


def main():
    # Setup variables so we can see the full graphs:
    # pylint: disable=direct-environment-variable-reference
    os.environ["FEATURE_GOAL_DISAMBIGUATION"] = "true"
    os.environ["WORKFLOW_INTERRUPT"] = "true"

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.ERROR),
    )
    os.environ["ANTHROPIC_API_KEY"] = "real_key_not_required_for_graph"
    # pylint: enable=direct-environment-variable-reference

    container_application = ContainerApplication()
    container_application.config.from_dict(Config().model_dump())
    container_application.wire(packages=CONTAINER_APPLICATION_PACKAGES)

    output_file_path = sys.argv[1]
    with open(output_file_path, "w") as output_file:
        output_file.write(HEADER_TEXT)
        for graph_name in CategoryEnum:
            if graph_name in [
                "software_development",
                "chat",
                "convert_to_gitlab_ci",
                "issue_to_merge_request",
            ]:
                # Dynamically import Workflow class. Equivalent to import statements in this format:
                #     from duo_workflow_service.workflows.chat import Workflow
                workflow_module = importlib.import_module(
                    f"duo_workflow_service.workflows.{graph_name}"
                )
                Workflow = getattr(  # pylint: disable=invalid-name
                    workflow_module, "Workflow"
                )

                tools_reg = MagicMock(spec=ToolsRegistry)
                wrk = Workflow(
                    "",
                    {"git_branch": "test-branch"},
                    workflow_type=graph_name,
                    user=CloudConnectorUser(True, is_debug=True),
                )
                wrk._project = {
                    "id": "",
                    "name": "",
                    "http_url_to_repo": "",
                    "web_url": "http://gitlab.com/project_name",
                    "default_branch": "main",
                }
                goal = ""
                if graph_name == "issue_to_merge_request":
                    goal = "http://gitlab.com/project_name/-/issues/1"
                graph = wrk._compile(goal, tools_reg, MemorySaver())

                diagram = graph.get_graph().draw_mermaid()
                diagram = diagram.replace("\t", "    ")

                output_file.write(f"\n## Graph: `{graph_name}`\n\n")
                output_file.write("```mermaid\n" + diagram + "```\n")


if __name__ == "__main__":
    main()
