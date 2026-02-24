from typing import Any, FrozenSet, Tuple, override

import ormsgpack
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer, _msgpack_enc
from langgraph.types import Interrupt


class CheckpointValidationError(ValueError):
    """Raised when checkpoint data format is invalid."""

    pass


class CheckpointSerializer(JsonPlusSerializer):
    """Serializer with strict data format validation for checkpoint consistency.

    Enforces msgpack-only serialization to ensure reliable checkpoint restoration across different environments and
    Python versions.
    """

    _SUPPORTED_MODULES: FrozenSet[str] = frozenset(
        [
            "langgraph.types",
        ]
    )

    def __init__(self, *args, **kwargs):
        kwargs["pickle_fallback"] = False
        kwargs["__unpack_ext_hook__"] = self._validate_extension_format

        super().__init__(*args, **kwargs)

    def _validate_extension_format(self, code: int, data: bytes) -> Any:
        """Validate and reconstruct msgpack extension data.

        Ensures extensions reference known module paths to maintain compatibility across checkpoint versions.
        """
        try:
            tup = ormsgpack.unpackb(
                data,
                ext_hook=self._validate_extension_format,
                option=ormsgpack.OPT_NON_STR_KEYS,
            )
        except CheckpointValidationError:
            raise
        except Exception:
            raise CheckpointValidationError("Malformed checkpoint data")

        if not isinstance(tup, (list, tuple)) or len(tup) < 3:
            raise CheckpointValidationError("Invalid extension structure")

        module_name = tup[0]
        class_name = tup[1]
        args_data = tup[2]

        if not isinstance(module_name, str) or not isinstance(class_name, str):
            raise CheckpointValidationError("Invalid module or class name type")

        if module_name not in self._SUPPORTED_MODULES:
            raise CheckpointValidationError(
                f"Module '{module_name}' not supported. "
                f"Checkpoint may be from incompatible version."
            )

        if module_name == "langgraph.types" and class_name == "Interrupt" and code == 2:
            return Interrupt(**args_data)

        return None

    @override
    def dumps_typed(self, obj: Any) -> Tuple[str, bytes]:
        """Serialize object using msgpack format."""
        try:
            return "msgpack", _msgpack_enc(obj)
        except Exception:
            raise CheckpointValidationError("Cannot serialize checkpoint")

    @override
    def loads_typed(self, data: Tuple[str, bytes]) -> Any:
        """Deserialize checkpoint data with format validation."""
        data_type, data_bytes = data

        if data_type == "msgpack":
            try:
                return ormsgpack.unpackb(
                    data_bytes,
                    ext_hook=self._validate_extension_format,
                    option=ormsgpack.OPT_NON_STR_KEYS,
                )
            except CheckpointValidationError:
                raise
            except Exception:
                raise CheckpointValidationError("Failed to restore checkpoint")

        raise CheckpointValidationError(
            f"Unsupported format '{data_type}'. Only msgpack format is supported."
        )
