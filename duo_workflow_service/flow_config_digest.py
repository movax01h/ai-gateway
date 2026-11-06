"""The digest that binds a workflow token to one flow config.

GenerateToken puts the digest of the ``flow_config`` it receives into the token. ExecuteWorkflow compares it with the
inline ``flowConfig``, or with the definition a fetched flow gets from GitLab.
"""

import hashlib
import json
from typing import Any

from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct

__all__ = ["definition_digest", "flow_config_digest"]


def flow_config_digest(flow_config: Struct) -> str:
    """Return the SHA-256 of the config as canonical JSON, with sorted keys and no spaces."""
    canonical_json = json.dumps(
        MessageToDict(flow_config), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical_json.encode()).hexdigest()


def definition_digest(definition: dict[str, Any]) -> str:
    """Return the digest of a definition parsed from JSON, as if it had come as a Struct.

    A Struct keeps every number as a double, so ``180`` becomes ``180.0``. The token's digest was computed on the
    Struct that GenerateToken received, so the definition goes through a Struct first to give the same JSON.
    """
    struct = Struct()
    struct.update(definition)
    return flow_config_digest(struct)
