from enum import StrEnum

__all__ = ["BillingEvent"]


class BillingEvent(StrEnum):
    DAP_FLOW_ON_COMPLETION = "duo_agent_platform_workflow_completion"
    AIGW_PROXY_USE = "ai_gateway_proxy_use"
    CODE_SUGGESTIONS_CODE_COMPLETIONS = "code_completions"
    CODE_SUGGESTIONS_CODE_GENERATIONS = "code_generations"
