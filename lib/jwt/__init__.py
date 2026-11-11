# JWT claim parsing shared between ai_gateway and duo_workflow_service

from lib.jwt.feature_enablement import (
    FEATURE_ENABLEMENT_TYPE_CLAIM,
    MAX_FEATURE_ENABLEMENT_TYPE_LENGTH,
    parse_feature_enablement_type,
    resolve_feature_enablement_type,
)
from lib.jwt.instance_id import instance_uid_claim, trusted_instance_id
from lib.jwt.root_namespace import (
    GITLAB_ROOT_NAMESPACE_ID_CLAIM,
    MISSING_SAAS_ROOT_NAMESPACE_CLAIM_ERROR,
    lacks_saas_root_namespace_claim,
    parse_root_namespace_id,
    root_namespace_id_from_claims_extra,
    root_namespace_id_from_header,
)

__all__ = [
    "FEATURE_ENABLEMENT_TYPE_CLAIM",
    "GITLAB_ROOT_NAMESPACE_ID_CLAIM",
    "MAX_FEATURE_ENABLEMENT_TYPE_LENGTH",
    "MISSING_SAAS_ROOT_NAMESPACE_CLAIM_ERROR",
    "instance_uid_claim",
    "lacks_saas_root_namespace_claim",
    "parse_feature_enablement_type",
    "parse_root_namespace_id",
    "resolve_feature_enablement_type",
    "root_namespace_id_from_claims_extra",
    "root_namespace_id_from_header",
    "trusted_instance_id",
]
