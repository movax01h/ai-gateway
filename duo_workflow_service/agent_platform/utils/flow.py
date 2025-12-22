from pathlib import Path


def parse_workflow_definition(value: str) -> tuple[str, str]:
    flow_version = Path(value).name
    flow_config_path = str(Path(value).parent)

    if flow_config_path == ".":
        raise ValueError(
            f"Unsupported workflow_definition value: '{flow_config_path}'."
            " Expected format: '<flow_name>/<flow_registry_version>' (e.g., 'my_flow/v1')"
        )

    return flow_version, flow_config_path
