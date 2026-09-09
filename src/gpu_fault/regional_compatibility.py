from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

LEGACY_REGIONAL_EXECUTOR_PROTOCOL_VERSION = 1
# Version 3 understands a compound command (``RemoteActionCommand.batched_steps``)
# and reports per-step progress on ``POST .../{command_id}/progress``. The claim
# response gained a field an older executor's ``extra="forbid"`` model would
# refuse, so this is a version bump rather than the additive-request-field path
# ``wait_seconds`` took (design §8.1): the control plane only mints compound
# commands once every accepted executor version is at least this one, and the
# claim never hands a compound command to an executor that advertised less.
CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION = 3
REMOTE_STEP_BATCHING_PROTOCOL_VERSION = 3


@dataclass(frozen=True)
class RegionalExecutorCompatibilityPolicy:
    required_version: int
    compatible_versions: frozenset[int]
    required_artifact_sha256: str | None = None
    compatible_artifact_sha256s: frozenset[str] = frozenset()
    required_compatibility_digest: str | None = None
    compatible_compatibility_digests: frozenset[str] = frozenset()

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
        artifact = (
            values.get(
                "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_ARTIFACT_SHA256",
                "",
            )
            .strip()
            .lower()
            or None
        )
        compatible_artifacts = frozenset(
            value.strip().lower()
            for value in values.get(
                "GPU_FAULT_COMPATIBLE_REGIONAL_EXECUTOR_ARTIFACT_SHA256S",
                "",
            ).split(",")
            if value.strip()
        ) - ({artifact} if artifact else set())
        if artifact and not re.fullmatch(r"[0-9a-f]{64}", artifact):
            raise ValueError("required regional executor artifact must be SHA-256")
        if any(
            not re.fullmatch(r"[0-9a-f]{64}", value) for value in compatible_artifacts
        ):
            raise ValueError("compatible regional executor artifacts must be SHA-256")
        compatibility_digest = (
            values.get("GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_COMPATIBILITY_DIGEST")
            or artifact
            or ""
        ).strip().lower() or None
        compatible_compatibility = frozenset(
            value.strip().lower()
            for value in values.get(
                "GPU_FAULT_COMPATIBLE_REGIONAL_EXECUTOR_COMPATIBILITY_DIGESTS",
                "",
            ).split(",")
            if value.strip()
        ) - ({compatibility_digest} if compatibility_digest else set())
        if compatibility_digest and not re.fullmatch(
            r"[0-9a-f]{64}", compatibility_digest
        ):
            raise ValueError(
                "required regional executor compatibility digest must be SHA-256"
            )
        if any(
            not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in compatible_compatibility
        ):
            raise ValueError(
                "compatible regional executor compatibility digests must be SHA-256"
            )
        return cls(
            required_version=required,
            compatible_versions=compatible,
            required_artifact_sha256=artifact,
            compatible_artifact_sha256s=compatible_artifacts,
            required_compatibility_digest=compatibility_digest,
            compatible_compatibility_digests=compatible_compatibility,
        )

    @property
    def minimum_accepted_version(self) -> int:
        """The oldest executor protocol this control plane still admits.

        The required and compatible pins are the control plane's declared
        knowledge of what is deployed (a rolling upgrade widens them, the
        follow-up deploy narrows them again), so a feature that needs every
        executor to understand a new response field gates on this value rather
        than on the version of whichever executor happens to claim next.
        """

        return min({self.required_version, *self.compatible_versions})

    def rejection_reason(
        self,
        actual: int,
        artifact_sha256: str | None = None,
        compatibility_digest: str | None = None,
    ) -> str | None:
        accepted = {self.required_version, *self.compatible_versions}
        if actual not in accepted:
            expected = (
                str(self.required_version)
                if len(accepted) == 1
                else "one of " + ", ".join(str(value) for value in sorted(accepted))
            )
            return (
                "regional executor protocol version mismatch: "
                f"expected {expected}, got {actual}"
            )
        accepted_artifacts = {
            *self.compatible_artifact_sha256s,
            *(
                {self.required_artifact_sha256}
                if self.required_artifact_sha256
                else set()
            ),
        }
        if accepted_artifacts:
            normalized = (artifact_sha256 or "").lower()
        else:
            normalized = ""
        if accepted_artifacts and normalized not in accepted_artifacts:
            expected = (
                next(iter(accepted_artifacts))
                if len(accepted_artifacts) == 1
                else "one of " + ", ".join(sorted(accepted_artifacts))
            )
            return (
                "regional executor artifact mismatch: "
                f"expected {expected}, got {normalized or 'MISSING'}"
            )
        accepted_compatibility = {
            *self.compatible_compatibility_digests,
            *(
                {self.required_compatibility_digest}
                if self.required_compatibility_digest
                else set()
            ),
        }
        if not accepted_compatibility:
            return None
        actual_compatibility = (compatibility_digest or artifact_sha256 or "").lower()
        if actual_compatibility in accepted_compatibility:
            return None
        expected = (
            next(iter(accepted_compatibility))
            if len(accepted_compatibility) == 1
            else "one of " + ", ".join(sorted(accepted_compatibility))
        )
        return (
            "regional executor compatibility digest mismatch: "
            f"expected {expected}, got {actual_compatibility or 'MISSING'}"
        )
