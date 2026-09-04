from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Annotated, Iterable, Protocol, Self

from pydantic import AfterValidator, BeforeValidator, model_validator

from gpu_fault.digests import SHA256_PATTERN
from gpu_fault.models import StrictModel

UNIT_PATTERN = re.compile(
    r"^gpu-fault-[a-z0-9](?:[a-z0-9-]*[a-z0-9])?"
    r"\.(?:service|timer)$"
)
MAX_INSTALLED_UNITS = 128
DEFAULT_UNIT_INVENTORY = Path("/opt/gpu-fault/installed-units.txt")
DEFAULT_SYSTEMD_DIRECTORY = Path("/etc/systemd/system")


def normalize_installed_units(
    value: Iterable[str],
) -> list[str]:
    if isinstance(value, (str, bytes)):
        raise ValueError("installed units must be a string list")
    units = sorted({str(item).strip() for item in value if str(item).strip()})
    if len(units) > MAX_INSTALLED_UNITS:
        raise ValueError("too many installed systemd units")
    invalid = [unit for unit in units if not UNIT_PATTERN.fullmatch(unit)]
    if invalid:
        raise ValueError("invalid installed systemd unit: " + invalid[0])
    return units


def validate_inventory_digest(value: str) -> str:
    normalized = value.strip().lower()
    if not SHA256_PATTERN.fullmatch(normalized):
        raise ValueError("installed unit digest must be SHA-256")
    return normalized


InstalledUnits = Annotated[
    list[str],
    BeforeValidator(normalize_installed_units),
]
InventoryDigest = Annotated[
    str,
    AfterValidator(validate_inventory_digest),
]


def installed_units_digest(units: Iterable[str]) -> str:
    normalized = normalize_installed_units(units)
    return hashlib.sha256(
        json.dumps(
            normalized,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


class InstalledUnitReport(StrictModel):
    digest: InventoryDigest
    units: InstalledUnits | None = None

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_units_digest(self) -> Self:
        if self.units is not None and installed_units_digest(self.units) != self.digest:
            raise ValueError("installed unit inventory digest mismatch")
        return self


class InstalledUnitInventory(StrictModel):
    digest: InventoryDigest
    units: InstalledUnits

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_units_digest(self) -> Self:
        if installed_units_digest(self.units) != self.digest:
            raise ValueError("installed unit inventory digest mismatch")
        return self


class AgentInventoryRecord(Protocol):
    installed_unit_inventory: InstalledUnitInventory | None


def installed_unit_report(
    units: Iterable[str],
    *,
    include_units: bool,
) -> InstalledUnitReport:
    normalized = normalize_installed_units(units)
    return InstalledUnitReport(
        digest=installed_units_digest(normalized),
        units=normalized if include_units else None,
    )


def resolve_installed_unit_inventory(
    reported: InstalledUnitReport | None,
    existing: InstalledUnitInventory | None,
) -> InstalledUnitInventory | None:
    if reported is None:
        return existing
    if reported.units is not None:
        return InstalledUnitInventory(
            digest=reported.digest,
            units=reported.units,
        )
    if existing is None or existing.digest != reported.digest:
        raise ValueError(
            "full installed unit inventory is required when digest changes"
        )
    return existing


def resolve_agent_inventory(
    reported: InstalledUnitReport | None,
    existing_record: AgentInventoryRecord | None,
) -> tuple[InstalledUnitInventory | None, bool]:
    existing = (
        getattr(
            existing_record,
            "installed_unit_inventory",
            None,
        )
        if existing_record is not None
        else None
    )
    resolved = resolve_installed_unit_inventory(
        reported,
        existing,
    )
    return (
        resolved,
        existing_record is not None and existing != resolved,
    )


def read_installed_systemd_units(
    inventory_path: Path = DEFAULT_UNIT_INVENTORY,
    systemd_directory: Path = DEFAULT_SYSTEMD_DIRECTORY,
) -> list[str]:
    if inventory_path.is_file():
        return normalize_installed_units(
            inventory_path.read_text(encoding="utf-8").splitlines()
        )
    discovered = [
        path.name
        for pattern in (
            "gpu-fault-*.service",
            "gpu-fault-*.timer",
        )
        for path in systemd_directory.glob(pattern)
        if path.is_file()
    ]
    return normalize_installed_units(discovered)
