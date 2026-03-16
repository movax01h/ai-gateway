from typing import Annotated, Optional

from pydantic import BaseModel, ConfigDict, StringConstraints


class Telemetry(BaseModel):
    # Opt out protected namespace "model_" (https://github.com/pydantic/pydantic/issues/6322).
    model_config = ConfigDict(protected_namespaces=())

    # TODO: Once the header telemetry format is removed, we can unmark these as optional
    model_engine: Optional[Annotated[str, StringConstraints(max_length=50)]] = None
    model_name: Optional[Annotated[str, StringConstraints(max_length=50)]] = None
    lang: Optional[Annotated[str, StringConstraints(max_length=50)]] = None
    requests: int
    accepts: int
    errors: int
