"""The capacity cleanup walk attributes records by incident owner or own cluster."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("psycopg")

from scripts.perf import regional_capacity_data as data

CLUSTERS = ["perf-cap-000"]


class Cursor:
    """Answers the walk's four query shapes from canned rows."""

    def __init__(self, records: list[tuple[str, str, dict]], incidents: dict[str, str]):
        self.records = records
        self.incidents = incidents
        self._rows: list[Any] = []

    def execute(self, query: Any, params: Any = None) -> None:
        text = query if isinstance(query, str) else query.as_string(None)
        if "kind='incident' AND key=%s" in text:
            owner = self.incidents.get(params[0])
            self._rows = [(owner,)] if owner is not None else []
        elif (
            "payload->>'cluster_id'=ANY" in text
            and "kind <> 'regional_cluster'" in text
        ):
            self._rows = list(self.records)
        else:
            self._rows = list(self.records)

    def fetchall(self) -> list[Any]:
        return self._rows

    def fetchone(self) -> Any:
        return self._rows[0] if self._rows else None


def test_a_record_owned_by_its_own_cluster_survives_a_derived_incident_id() -> None:
    notification = {
        "cluster_id": "perf-cap-000",
        "incident_id": "collector-silent-integrated-node-c000-w00",
    }
    cursor = Cursor([("notification", "n-1", notification)], incidents={})
    records = data.scope_records(cursor, CLUSTERS)
    assert ("notification", "n-1") in records, "own cluster field proves ownership"


def test_a_foreign_or_unattributable_incident_still_refuses() -> None:
    foreign = {"cluster_id": "perf-cap-000", "incident_id": "inc-foreign"}
    with pytest.raises(RuntimeError, match="unknown or foreign"):
        data.scope_records(
            Cursor(
                [("notification", "n-2", foreign)], incidents={"inc-foreign": "prod-1"}
            ),
            CLUSTERS,
        )
    orphan = {"incident_id": "inc-missing"}
    with pytest.raises(RuntimeError, match="unknown or foreign"):
        data.scope_records(Cursor([("marker", "m-1", orphan)], incidents={}), CLUSTERS)


def test_only_this_runs_drill_notifications_count_as_late_residue() -> None:
    run_id = "integrated20260920T221955"
    drill = {
        "drill_id": "perf-capacity",
        "cluster_name": "perf-cap-000",
        "incident_id": f"collector-silent-integrated-node-{run_id}-c000-w00",
    }
    assert data.drill_residue_only({("notification", "n-1"): drill}, run_id), (
        "a drill notification carrying the run id is sweepable"
    )
    other_run = {
        **drill,
        "incident_id": "collector-silent-integrated-node-other-c000-w00",
    }
    assert not data.drill_residue_only({("notification", "n-2"): other_run}, run_id), (
        "another run's notification is not ours"
    )
    real = {"cluster_id": "perf-cap-000", "incident_id": f"inc-{run_id}"}
    assert not data.drill_residue_only({("notification", "n-3"): real}, run_id), (
        "a notification without the drill marker is not residue"
    )
    workflow = {**drill}
    assert not data.drill_residue_only({("workflow", "w-1"): workflow}, run_id), (
        "only notifications qualify"
    )
    assert not data.drill_residue_only({}, run_id), "nothing is not residue"
    satellites = {
        ("notification", "n-1"): drill,
        ("notification_delivery", "n-1"): {"notification_id": "n-1", "status": "DONE"},
        ("notification_result", "n-1"): {"notification_id": "n-1", "status": "SENT"},
    }
    assert data.drill_residue_only(satellites, run_id), (
        "the delivery row and the result of a drill notification go with it"
    )
    foreign_satellite = {
        **satellites,
        ("notification_result", "n-9"): {"notification_id": "n-9"},
    }
    assert not data.drill_residue_only(foreign_satellite, run_id), (
        "a satellite of some other notification is not residue"
    )


def test_a_deregistered_run_reports_no_registration_instead_of_refusing() -> None:
    from gpu_fault.regional import (
        RegionalRegistryRevision,
        regional_registry_content_sha256,
    )

    content = regional_registry_content_sha256([])
    revision = RegionalRegistryRevision.model_validate(
        {
            "generation": 3,
            "registrations": [],
            "content_sha256": content,
            "reason": "test",
        }
    ).model_dump(mode="json")

    class RegistryCursor:
        def __init__(self) -> None:
            self._rows: list[Any] = []

        def execute(self, query: Any, params: Any = None) -> None:
            if "kind='regional_registry_head'" in query:
                self._rows = [({"generation": 3, "content_sha256": content},)]
            elif "kind='regional_registry_revision'" in query:
                self._rows = [(revision,)]
            else:
                self._rows = []

        def fetchone(self) -> Any:
            return self._rows[0] if self._rows else None

    assert data.require_registry_scope(RegistryCursor(), "run-1", CLUSTERS) is False, (
        "no registration left means the late-residue rule decides, not a refusal"
    )


def test_record_bound_scales_with_the_run_and_bulk_batches_skip_cas_kinds() -> None:
    assert data.record_bound(["perf-cap-000"]) == data.MAX_RECORDS, (
        "small runs keep the floor"
    )
    assert data.record_bound([f"perf-cap-{i:03d}" for i in range(50)]) == 200_000, (
        "50 clusters get 4000 records each"
    )
    records = {
        ("workflow", "w-1"): {},
        ("remote_command", "c-1"): {},
        ("incident", "i-1"): {},
        **{("telemetry_metric_latest", f"t-{i}"): {} for i in range(5001)},
        ("collector_status", "s-1"): {},
    }
    batches = data.bulk_delete_batches(records)
    kinds = [kind for kind, _ in batches]
    assert (
        "workflow" not in kinds
        and "incident" not in kinds
        and "remote_command" not in kinds
    ), "control-state kinds stay on the CAS path"
    assert [
        len(keys) for kind, keys in batches if kind == "telemetry_metric_latest"
    ] == [5000, 1], "hot-state keys are deleted in bounded batches"
    assert ("collector_status", ["s-1"]) in batches, "every other kind is bulk-deleted"
