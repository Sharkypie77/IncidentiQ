"""Shared utility functions for the IncidentIQ environment."""

from __future__ import annotations

from datetime import datetime


def iso_timestamp(dt: datetime) -> str:
    """Format a datetime as an ISO-8601 string with millisecond precision."""
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"
