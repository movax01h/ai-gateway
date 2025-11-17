from datetime import date

from pydantic import BaseModel


class DeprecationInfo(BaseModel):
    """Information about model deprecation."""

    deprecation_date: date
    removal_version: str
