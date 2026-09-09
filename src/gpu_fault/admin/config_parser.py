"""The generic reader and writer behind the admin-config schema.

``config`` declares the schema once, as frozen dataclasses that own every
default. This module walks them: the persisted record is the dataclass fields
in snake_case (``snake_section``), the editable YAML ``spec`` is the same tree
in camelCase (``camel_case``, ``camel_section``), and one ``read_section``
reads either spelling on top of a base value. A field is therefore declared
once and the two spellings cannot drift apart.

``boolean_field`` is also the site-config reader's rule for a YAML switch, so
the text a site-config reader raises is the text an admin-config reader raises.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from typing import Any, Callable, TypeVar, cast


class AdminConfigError(ValueError):
    pass


class AdminConfigParseError(ValueError):
    pass


_SectionT = TypeVar("_SectionT")


def camel_case(name: str) -> str:
    """``max_active_region`` -> ``maxActiveRegion``: the YAML spelling of a field."""

    head, *rest = name.split("_")
    return head + "".join(part.capitalize() for part in rest)


def boolean_field(
    value: object,
    path: str,
    *,
    default: bool,
    error: type[ValueError] = AdminConfigParseError,
) -> bool:
    """Validate an already-parsed YAML boolean; ``None`` means ``default``.

    Each caller keeps its own exception family by passing ``error``.
    """

    if value is None:
        return default
    if not isinstance(value, bool):
        raise error(f"{path} must be a boolean")
    return value


def mapping_field(
    value: object,
    path: str,
    *,
    allowed: set[str],
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise AdminConfigError(f"{path} must be a mapping")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise AdminConfigError(f"{path} contains unknown fields: {', '.join(unknown)}")
    return value


def integer_field(value: object, path: str, *, default: int) -> int:
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool):
        raise AdminConfigError(f"{path} must be an integer")
    return value


def number_field(value: object, path: str, *, default: float) -> float:
    if value is None:
        return default
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise AdminConfigError(f"{path} must be a number")
    return float(value)


def snake_section(
    section: object,
    *,
    constants: Mapping[type, Mapping[str, int]],
) -> dict[str, object]:
    """The persisted form of a section: its fields, plus the constants a class
    still records for digest stability although they are no longer fields."""

    result: dict[str, object] = {}
    for field in fields(cast(Any, section)):
        value = getattr(section, field.name)
        result[field.name] = (
            snake_section(value, constants=constants) if is_dataclass(value) else value
        )
    result.update(constants.get(type(section), {}))
    return result


def camel_section(section: object) -> dict[str, object]:
    """The editable YAML form of a section: its fields in camelCase, no constants."""

    result: dict[str, object] = {}
    for field in fields(cast(Any, section)):
        value = getattr(section, field.name)
        result[camel_case(field.name)] = (
            camel_section(value) if is_dataclass(value) else value
        )
    return result


def read_section(
    base: _SectionT,
    value: object,
    *,
    camel: bool,
    path: str,
    constants: Mapping[type, Mapping[str, int]],
) -> _SectionT:
    """One mapping laid over ``base``; an absent key keeps the base value.

    ``camel`` selects the YAML spelling, the persisted record is snake_case.
    The field type comes from the base value, so a section, a switch, a number
    and an integer each get the one reader the whole admin config uses. A
    constant of the class may be spelled at its fixed value only.
    """

    section: Any = base
    spell: Callable[[str], str] = camel_case if camel else (lambda name: name)
    fixed = constants.get(type(section), {})
    names = {spell(field.name): field.name for field in fields(section)}
    data = mapping_field(
        value or {},
        path,
        allowed=set(names) | {spell(name) for name in fixed},
    )
    for name, constant in fixed.items():
        raw = data.get(spell(name))
        if raw is not None and (isinstance(raw, bool) or raw != constant):
            raise AdminConfigError(f"{path}.{spell(name)} is fixed at {constant}")
    values: dict[str, object] = {}
    for key, name in names.items():
        current = getattr(section, name)
        raw = data.get(key)
        field_path = f"{path}.{key}"
        if is_dataclass(current):
            values[name] = read_section(
                current, raw, camel=camel, path=field_path, constants=constants
            )
        elif isinstance(current, bool):
            values[name] = boolean_field(
                raw,
                field_path,
                default=current,
                error=AdminConfigError,
            )
        elif isinstance(current, float):
            values[name] = number_field(raw, field_path, default=current)
        else:
            values[name] = integer_field(raw, field_path, default=current)
    return cast(_SectionT, replace(section, **values))
