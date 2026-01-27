from contextvars import ContextVar

X_GITLAB_SELF_HOSTED_DAP_BILLING_ENABLED = "x-gitlab-self-hosted-dap-billing-enabled"

current_self_hosted_dap_billing_enabled: ContextVar[bool] = ContextVar(
    "current_self_hosted_dap_billing_enabled", default=False
)


def set_self_hosted_dap_billing_enabled(value: str | None) -> None:
    current_self_hosted_dap_billing_enabled.set(value == "true")
