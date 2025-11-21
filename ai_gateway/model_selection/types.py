from datetime import date

from pydantic import BaseModel, ConfigDict


class DevConfig(BaseModel):
    """Configuration for developer-only models."""

    model_config = ConfigDict(extra="forbid")

    selectable_models: list[str]
    group_ids: list[int]


class DeprecationInfo(BaseModel):
    """Information about model deprecation."""

    deprecation_date: date
    removal_version: str
