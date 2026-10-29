"""The generic reader and writer behind the admin-config schema.

``config`` declares the schema once, as frozen dataclasses that own every
default. This module walks them: the persisted record is the dataclass fields
in snake_case (``snake_section``), the editable YAML ``spec`` is the same tree
in camelCase (``camel_case``, ``camel_section``), and one ``read_section``
reads either spelling on top of a base value. A field is therefore declared
once and the two spellings cannot drift apart.

``boolean_field`` is also the site-config reader's rule for a YAML switch, so
the text a site-config reader raises is the text an admin-config reader raises.

A field a release retired stays a field of its section, listed in the class's
``RETIRED_FIELDS`` (and ``OMIT_WHEN_DEFAULT`` with a ``None`` default): a
persisted record keeps the value it was written with, so every digest over it
still verifies, while the YAML ``spec`` never shows it and a ``spec`` that
still spells it is accepted, ignored and reported (``retired_spec_fields``).
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
    """``managed_node_count`` -> ``managedNodeCount``: the YAML spelling of a field."""

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
        if lifted_default(section, field.name, value):
            continue
        result[field.name] = (
            snake_section(value, constants=constants) if is_dataclass(value) else value
        )
    result.update(constants.get(type(section), {}))
    return result


def lifted_default(section: object, name: str, value: object) -> bool:
    """Whether a field added after sites went live is left out of the record.

    A section lists such fields in ``OMIT_WHEN_DEFAULT``; while the value is the
    field's default the persisted and YAML forms omit it, so every existing
    ``desired.json`` keeps its digest and older records read back unchanged
    (``read_section`` fills the default). A non-default value is recorded.
    """

    omit: frozenset[str] = getattr(type(section), "OMIT_WHEN_DEFAULT", frozenset())
    if name not in omit:
        return False
    default = next(f.default for f in fields(cast(Any, section)) if f.name == name)
    return value == default


def retired_fields(section: object) -> frozenset[str]:
    """The fields of ``section`` a release retired (see the module docstring)."""

    return getattr(type(section), "RETIRED_FIELDS", frozenset())


def retired_spec_fields(section: object, value: object, *, path: str) -> list[str]:
    """The retired fields a camelCase ``spec`` laid over ``section`` still spells.

    ``read_section`` ignores them; the caller says so to the administrator.
    """

    if not isinstance(value, Mapping):
        return []
    found: list[str] = []
    retired = retired_fields(section)
    for field in fields(cast(Any, section)):
        key = camel_case(field.name)
        if key not in value:
            continue
        current = getattr(section, field.name)
        if field.name in retired:
            found.append(f"{path}.{key}")
        elif is_dataclass(current):
            found.extend(retired_spec_fields(current, value[key], path=f"{path}.{key}"))
    return found


def camel_section(section: object) -> dict[str, object]:
    """The editable YAML form of a section: its fields in camelCase, no
    constants, no retired fields and no section left without a field."""

    result: dict[str, object] = {}
    for field in fields(cast(Any, section)):
        if field.name in retired_fields(section):
            continue
        if lifted_default(section, field.name, getattr(section, field.name)):
            continue
        value = getattr(section, field.name)
        if not is_dataclass(value):
            result[camel_case(field.name)] = value
        elif nested := camel_section(value):
            result[camel_case(field.name)] = nested
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
    constant of the class may be spelled at its fixed value only. A retired
    field is read from a record as recorded and ignored in a ``spec``: the base
    value stays.
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
    retired = retired_fields(section)
    for key, name in names.items():
        current = getattr(section, name)
        raw = data.get(key)
        field_path = f"{path}.{key}"
        if name in retired:
            values[name] = (
                current
                if camel or raw is None
                else integer_field(raw, field_path, default=current)
            )
        elif is_dataclass(current):
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
