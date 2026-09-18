"""Track and restore only the resources held by a Collector acceptance case."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from scripts.e2e.regional.collector_acceptance_fixture import CollectorAcceptanceFixture
from scripts.e2e.regional.host_probe_fixture import HostProbeFixture
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from scripts.e2e.regional.collector_recovery_safety import (
    require_bound_refresh,
    require_settled_recovery,
)


@dataclass
class CaseCleanup:
    """What a case took hold of, so ``execute_case`` can let go of it on every path.

    Handlers register a state the moment the store shows a workflow for their
    injection, and an annotation the moment they write it; the success path
    restores through the same registry (``restore``/``remove_annotation``) and
    marks the item done. Whatever is still registered when the case ends --
    because an assertion raised, a wait timed out, or a handler forgot -- is
    restored best-effort by ``finish`` and its failures are recorded as
    cleanup errors, which fail the case.
    """

    incident_states: list[tuple[CollectorAcceptanceFixture, dict[str, Any]]] = field(
        default_factory=list
    )
    annotations: list[tuple[CollectorAcceptanceFixture, str]] = field(
        default_factory=list
    )
    restore_workflows: list[list[dict[str, Any]]] = field(default_factory=list)
    seed_markers: list[tuple[CollectorAcceptanceFixture, str]] = field(
        default_factory=list
    )
    quiesce_hosts: list[tuple[CollectorAcceptanceFixture, HostProbeFixture]] = field(
        default_factory=list
    )
    state_readers: list[
        tuple[CollectorAcceptanceFixture, Callable[[], dict[str, Any]]]
    ] = field(default_factory=list)

    def register_refresh(
        self,
        fixture: CollectorAcceptanceFixture,
        read: Callable[[], dict[str, Any]],
    ) -> None:
        """Keep a bound recovery lookup even if evidence polling later raises."""
        self.state_readers.append((fixture, read))

    def register_seed(
        self,
        fixture: CollectorAcceptanceFixture,
        marker: str,
        *,
        quiesce_host: HostProbeFixture | None = None,
    ) -> None:
        if not marker:
            raise RegionalFixtureError("collector seed identity is empty")
        if (fixture, marker) not in self.seed_markers:
            self.seed_markers.append((fixture, marker))
        if (
            quiesce_host is not None
            and (fixture, quiesce_host) not in self.quiesce_hosts
        ):
            self.quiesce_hosts.append((fixture, quiesce_host))

    def register_state(
        self,
        fixture: CollectorAcceptanceFixture,
        state: dict[str, Any],
    ) -> None:
        self.incident_states.append((fixture, state))

    def register_annotation(
        self,
        fixture: CollectorAcceptanceFixture,
        annotation: str,
    ) -> None:
        self.annotations.append((fixture, annotation))

    def restore(
        self,
        fixture: CollectorAcceptanceFixture,
        state: dict[str, Any],
        *,
        profile_version: str,
        reason: str,
    ) -> list[dict[str, Any]]:
        marker = state.get("seed_marker")
        if isinstance(marker, str) and marker:
            current = fixture.store_snapshot(marker)
            require_bound_refresh(state, current, marker)
            state = current
        require_settled_recovery(state)
        # Host references never authorize clearing reset_issued or restoring
        # services. The product owns quiesce compensation and its failsafe.
        for owner, host in self.quiesce_hosts:
            if owner is not fixture:
                continue
            observed = host.execute("snapshot", timeout=180)
            if (
                not isinstance(observed.get("quiesce_states"), list)
                or observed["quiesce_states"]
            ):
                raise RegionalFixtureError(
                    "product quiesce restoration is unproven; operator hold retained"
                )
        result = fixture.restore_incidents(
            state,
            profile_version=profile_version,
            reason=reason,
        )
        # A validated restore that returned proved the node holds no ownership
        # or quarantine taint at all (restore_incidents raises otherwise), so
        # every state registered for this node is released, not only ``state``.
        self.incident_states = [
            item for item in self.incident_states if item[0] is not fixture
        ]
        self.seed_markers = [
            item
            for item in self.seed_markers
            if item != (fixture, state.get("seed_marker"))
        ]
        self.restore_workflows.append(result)
        return result

    def remove_annotation(
        self,
        fixture: CollectorAcceptanceFixture,
        annotation: str,
    ) -> None:
        fixture.regional.kubectl(
            "gpu",
            "annotate",
            "node",
            fixture.node,
            f"{annotation}-",
        )
        self.annotations = [
            item for item in self.annotations if item != (fixture, annotation)
        ]

    def finish(self, *, profile_version: str, reason: str) -> dict[str, Any]:
        """Release everything still registered; never raises, always reports."""

        errors: list[str] = []
        restored: list[list[dict[str, Any]]] = []
        for fixture, read in self.state_readers:
            try:
                self.register_state(fixture, read())
            except Exception as exc:
                errors.append(
                    f"{fixture.node}: recovery lookup unresolved: "
                    f"{type(exc).__name__}: {exc}"
                )
        for fixture, annotation in list(self.annotations):
            try:
                self.remove_annotation(fixture, annotation)
            except Exception as exc:
                errors.append(
                    f"{fixture.node}: annotation {annotation} removal failed: "
                    f"{type(exc).__name__}: {exc}"
                )
        for fixture, marker in list(self.seed_markers):
            try:
                state = fixture.store_snapshot(marker)
                if not any(
                    state.get(key)
                    for key in ("events", "fabric_events", "evidence", "incidents")
                ):
                    raise RegionalFixtureError("collector seed receipt is unresolved")
                self.register_state(fixture, state)
            except Exception as exc:
                errors.append(
                    f"{fixture.node}: seed {marker} cleanup lookup failed: "
                    f"{type(exc).__name__}: {exc}"
                )
        for fixture, state in list(reversed(self.incident_states)):
            if not any(
                owner is fixture and registered is state
                for owner, registered in self.incident_states
            ):
                continue
            try:
                restored.append(
                    self.restore(
                        fixture,
                        state,
                        profile_version=profile_version,
                        reason=reason,
                    )
                )
            except Exception as exc:
                errors.append(
                    f"{fixture.node}: validated restore failed: "
                    f"{type(exc).__name__}: {exc}"
                )
        return {"restore_workflows": restored, "errors": errors}
