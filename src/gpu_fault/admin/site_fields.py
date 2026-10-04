"""Primitive site.yaml field parsers shared by the site schema modules."""

from __future__ import annotations

import re
from typing import Mapping, cast

IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")


class SiteConfigError(ValueError):
    pass


def mapping(
    value: object,
    path: str,
    *,
    allowed: set[str],
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SiteConfigError(f"{path} must be a mapping")
    normalized = cast(Mapping[str, object], value)
    unknown = sorted(set(normalized) - allowed)
    if unknown:
        raise SiteConfigError(f"{path} contains unknown fields: {', '.join(unknown)}")
    return normalized


def required_text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SiteConfigError(f"{path} must be a non-empty string")
    normalized = value.strip()
    if normalized.startswith("REPLACE_"):
        raise SiteConfigError(f"{path} is still a placeholder")
    return normalized


def optional_text(value: object, path: str) -> str | None:
    if value is None:
        return None
    return required_text(value, path)
