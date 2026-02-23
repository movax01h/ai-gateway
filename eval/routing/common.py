import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from joblib import Memory
from langchain_anthropic.chat_models import ChatAnthropic
from langchain_core.utils.function_calling import convert_to_openai_tool
from langsmith import Client
from structlog import get_logger

from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from eval.routing.schema import ToolRoutingEvaluation

CHAT = ChatAnthropic(
    model_name="claude-sonnet-4-20250514",
    temperature=0,
    betas=[],
    timeout=30,
    stop=None,
)

ls_client = Client()

# Create cache directory
memory = Memory(location=os.path.join(tempfile.gettempdir(), "eval/cache"), verbose=0)

logger = get_logger("eval.routing.common")


def get_all_subclasses(cls):
    all_subclasses = set()

    for subclass in cls.__subclasses__():
        all_subclasses.add(subclass)
        all_subclasses.update(get_all_subclasses(subclass))

    return all_subclasses


def get_all_ops_tools(tool_class: type = DuoBaseTool) -> List[Dict[str, Any]]:
    tool_specs = []

    for subclass in get_all_subclasses(tool_class):
        # Check if the subclass has _execute method
        if hasattr(subclass, "_execute"):
            method = getattr(subclass, "_execute")

            # Check if it's not abstract
            if not getattr(method, "__isabstractmethod__", False):
                tool_specs.append(convert_to_openai_tool(subclass()))

    return sorted(tool_specs, key=lambda x: x["function"]["name"])


def load_yamls(base_dir: str) -> List[Tuple[str, Dict[str, Any]]]:
    yaml_base_path = Path(base_dir)
    results = []
    yaml_files = list(yaml_base_path.rglob("*.yaml")) + list(
        yaml_base_path.rglob("*.yml")
    )

    for yaml_file in yaml_files:
        try:
            relative_path = str(yaml_file.relative_to(yaml_base_path))
            with open(yaml_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                results.append((relative_path, data))
        except Exception:
            logger.error(f"Failed to load yaml: {relative_path}!")
            raise
    return results


def load_routing_configs(
    base_dir: str = "config/routing",
) -> List[ToolRoutingEvaluation]:
    configs = []
    for yaml_file, data in load_yamls(base_dir):
        try:
            tool_evaluation = ToolRoutingEvaluation(**data)
            configs.append(tool_evaluation)
            logger.debug(f"Successfully loaded routing config: {yaml_file}")
        except Exception as e:
            logger.error(f"Failed to load routing config: {yaml_file}!: {str(e)}")
            raise e
    return sorted(configs, key=lambda x: x.tool_name)
