# JWT claim parsing shared between ai_gateway and duo_workflow_service

from lib.jwt.root_namespace import (
    GITLAB_ROOT_NAMESPACE_ID_CLAIM,
    parse_root_namespace_id,
    root_namespace_id_from_claims_extra,
    root_namespace_id_from_header,
)

__all__ = [
    "GITLAB_ROOT_NAMESPACE_ID_CLAIM",
    "parse_root_namespace_id",
    "root_namespace_id_from_claims_extra",
    "root_namespace_id_from_header",
]
