from datetime import date

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
