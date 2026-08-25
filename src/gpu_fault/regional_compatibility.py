from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


LEGACY_REGIONAL_EXECUTOR_PROTOCOL_VERSION = 1
CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION = 2


@dataclass(frozen=True)
class RegionalExecutorCompatibilityPolicy:
    required_version: int
    compatible_versions: frozenset[int]

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, str]
    ) -> RegionalExecutorCompatibilityPolicy:
        required = int(
            values.get(
                "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_PROTOCOL_VERSION",
                str(CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION),
            )
        )
        compatible = frozenset(
            int(value.strip())
            for value in values.get(
                "GPU_FAULT_COMPATIBLE_REGIONAL_EXECUTOR_PROTOCOL_VERSIONS",
                str(LEGACY_REGIONAL_EXECUTOR_PROTOCOL_VERSION),
            ).split(",")
            if value.strip()
        ) - {required}
        if required < 1 or any(value < 1 for value in compatible):
            raise ValueError("regional executor protocol versions must be positive")
        return cls(
            required_version=required,
            compatible_versions=compatible,
        )

    def rejection_reason(self, actual: int) -> str | None:
        accepted = {self.required_version, *self.compatible_versions}
        if actual in accepted:
            return None
        expected = (
            str(self.required_version)
            if len(accepted) == 1
            else "one of " + ", ".join(str(value) for value in sorted(accepted))
        )
        return (
            "regional executor protocol version mismatch: "
            f"expected {expected}, got {actual}"
        )
