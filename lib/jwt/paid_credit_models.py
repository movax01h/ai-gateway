from collections.abc import Mapping
from typing import Any

PAID_CREDIT_MODELS_ALLOWED_CLAIM = "gitlab_paid_credit_models_allowed"


def paid_credit_models_allowed_from_claims_extra(
    extra: Mapping[str, Any] | None,
) -> bool | None:
    """Return the signed gitlab_paid_credit_models_allowed claim, or None when it is absent.

    CustomersDot sets it on self-managed instance tokens. GitLab.com Rails sets it only on Duo Workflow Service tokens;
    its other tokens have no claim and are not gated. A present value that is not a boolean is read as False, so a
    malformed claim never widens access.
    """
    if not extra or PAID_CREDIT_MODELS_ALLOWED_CLAIM not in extra:
        return None

    value = extra[PAID_CREDIT_MODELS_ALLOWED_CLAIM]
    return value if isinstance(value, bool) else False
