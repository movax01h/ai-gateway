from datetime import date
from typing import Optional

from pydantic import BaseModel, ConfigDict


class DevConfig(BaseModel):
    """Configuration for developer-only models."""

    model_config = ConfigDict(extra="forbid")

    selectable_models: list[str]


class DeprecationInfo(BaseModel):
    """Information about model deprecation."""

    deprecation_date: date
    removal_version: str


class FeatureDeprecatedModel(DeprecationInfo):
    """Deprecation of a model for one specific feature setting.

    Unlike DeprecationInfo on a model definition (global deprecation), this marks a model as being phased out for a
    single feature's selectable_models while it remains fully supported elsewhere.
    """

    identifier: str


class DefaultModelEntry(BaseModel):
    """A default model entry with an optional traffic-split weight.

    When ``weight`` is provided for at least one entry in a ``default_models``
    list, **all** entries must carry a weight (validated by
    ``UnitPrimitiveConfig``).  Weights are passed directly to
    ``random.choices`` as the ``weights`` argument, so they do not need to sum
    to any particular value — only their relative magnitudes matter.
    """

    model_config = ConfigDict(extra="forbid")

    identifier: str
    weight: Optional[float] = None
