"""The one place that answers which delivery path a flow version takes.

A version opts in by declaring its prompts with `template`: the config is the signal, not the flow's name and not
a runtime table. Everything that needs to know asks here: the flow that hands a run its inbox, the registry that
picks an engine, and the validator. So the rule lives once and the runtime and the validator can't drift apart.

Opting in is limited for now, and all three limits are conditions in `opts_into_chat_engine`.

Chat-partial only: `ChatFlow` is what puts a turn's message in the run's inbox, and no other engine delivers one
yet. A `template` prompt anywhere else would render its `system` block and never a message.

Flows in this repository only: the validator, the docs and the Rails prompt builder still assume the old format, so
a flow written in the AI Catalog can't use it yet. Lifting this one is deleting the condition.

Prompts declared inline only: a prompt referenced by `prompt_id` and `prompt_version` is loaded later, so its format
can't be read here. Lifting this one means resolving the referenced prompt in this check.
"""

from typing import Optional

from ai_gateway.prompts.config.base import is_legacy_prompt
from duo_workflow_service.agent_platform.constants import CHAT_PARTIAL_ENVIRONMENT
from duo_workflow_service.agent_platform.v1.flows.flow_config import BaseFlowConfig

__all__ = ["opts_into_chat_engine"]


def opts_into_chat_engine(config: Optional[BaseFlowConfig]) -> bool:
    """Return whether this flow version opts into the chat engine, refusing it if it may not.

    Asking and checking are one call on purpose: a caller that could ask without the check would be a way to run a
    version that isn't allowed to, which is what this exists to prevent.

    Args:
        config: The flow config, or None before a flow has fetched one.

    Returns:
        True when the version's prompts declare `template`, so its messages reach components as inbound messages
        rendered through the prompt's `message` block.

    Raises:
        ValueError: When the version mixes the two formats, declares `template` where nothing renders it, or opts
            in with a prompt this check can't read. A config that fails here never runs, in the runtime and in the
            validator alike.
    """
    prompts = list(config.prompts or []) if config else []
    legacy = [prompt for prompt in prompts if is_legacy_prompt(prompt)]
    if config is None or len(legacy) == len(prompts):
        return False

    if legacy:
        raise ValueError(
            "A flow version declares all its prompts with 'template' or none of them. "
            f"These still use 'prompt_template': {', '.join(sorted(p.prompt_id for p in legacy))}"
        )

    if config.environment != CHAT_PARTIAL_ENVIRONMENT:
        raise ValueError(
            f"A flow version declares 'template' only in the '{CHAT_PARTIAL_ENVIRONMENT}' environment, which renders "
            f"it. This one declares '{config.environment}'."
        )

    # Only a config from this repository's registry carries an id: the registry loader sets it, and an inline or
    # fetched config can't. Until the validator, the docs and Rails's prompt builder handle the new format, a flow
    # written in the AI Catalog that declares `template` is refused when it is saved, rather than running on an
    # engine that isn't ready for it.
    if not config.config_id:
        raise ValueError(
            "'template' is not available for flows defined outside the AI Gateway repository yet. Declare the "
            "prompts with 'prompt_template'."
        )

    # A repository prompt's format can't be read from the config, so a version that opts in declares its prompts
    # inline. Lifting this needs the registry to resolve the referenced version here.
    if repository_prompts := config.components_with_repository_prompt:
        raise ValueError(
            "A flow version that uses 'template' declares its prompts inline. These components reference a "
            f"repository prompt: {', '.join(sorted(repository_prompts))}"
        )

    return True
