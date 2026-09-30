from duo_workflow_service.agent_platform.v1.flows.flow_config import FlowConfig

__all__ = ["normalize_engine_owned_config"]


def normalize_engine_owned_config(config: FlowConfig) -> FlowConfig:
    """Complete an engine-owned config before it reaches the engine.

    The config completes itself through ``to_config``: a chat-partial config fills in its entry point and its route to
    ``end``, which the environment lets authors omit. Component fields pass through untouched: the defaults the chat
    surface guarantees are ``AgentComponent``'s own under the ``chat-partial`` environment, filled when the builder
    constructs it.

    Args:
        config: A chat-partial config.

    Returns:
        A complete ``FlowConfig``. The input is not mutated.
    """
    return config.to_config()
