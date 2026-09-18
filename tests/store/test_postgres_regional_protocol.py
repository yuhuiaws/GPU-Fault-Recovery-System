"""Regional probe SQL against a fresh, exclusively owned loopback database."""

from __future__ import annotations

import io
import json
import os
import sys
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator
from uuid import uuid4

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.models import FaultIncident, IncidentState, WorkflowStatus
from gpu_fault.regional import RegionalClusterRegistration, RegionalRegistryRevision
from gpu_fault.state_table_migrate import backfill_state_table, set_state_table_mode
from gpu_fault.store import NotFoundError, PostgresStore
from scripts.e2e.regional import audit_raw_evidence_periodic_cleanup as evidence
from scripts.e2e.regional import net_command_fixture as net
from scripts.e2e.regional import run_cmd018_open_sibling_hold as cmd018
from scripts.e2e.regional import run_net003_result_retry as net003
from scripts.e2e.regional import run_preempt037_dispatcher_liveness as liveness
from scripts.e2e.regional import seeded_command_fixture as seeded
from scripts.perf import regional_capacity_data as capacity_data
from tests._builders import workflow_request

POSTGRES_URL = os.getenv("GPU_FAULT_TEST_POSTGRES_URL", "")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="GPU_FAULT_TEST_POSTGRES_URL is not configured"
)


@pytest.fixture(params=["legacy", "dual", "dedicated"])
def database(request: pytest.FixtureRequest) -> Iterator[tuple[str, PostgresStore]]:
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import conninfo_to_dict, make_conninfo

    settings = conninfo_to_dict(POSTGRES_URL)
    if settings.get("host") != "127.0.0.1":
        pytest.fail(
            "protocol SQL tests require an explicitly allocated loopback database"
        )
    name = "gpu_fault_protocol_" + uuid4().hex
    url = make_conninfo(POSTGRES_URL, dbname=name)
    with psycopg.connect(POSTGRES_URL, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        try:
            store = PostgresStore(url, initialize_schema=True)
            store.close()
            try:
                with psycopg.connect(url, autocommit=True) as connection:
                    if request.param != "legacy":
                        for kind in ("workflow", "remote_command"):
                            set_state_table_mode(
                                connection, kind, "dual", expected_mode="legacy"
                            )
                            if request.param == "dedicated":
                                backfill_state_table(connection, kind)
                                set_state_table_mode(
                                    connection,
                                    kind,
                                    "dedicated",
                                    expected_mode="dual",
                                    confirm_dedicated=True,
                                )
                store = PostgresStore(url, initialize_schema=False)
                yield url, store
            finally:
                store.close()
        finally:
            admin.execute(
                sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name))
            )


@pytest.fixture
def transport(
    database: tuple[str, PostgresStore], monkeypatch: pytest.MonkeyPatch
) -> Any:
    url, store = database

    def run(source: str, *arguments: str) -> dict[str, Any]:
        with monkeypatch.context() as patch:
            patch.setenv("GPU_FAULT_STORE_URL", url)
            patch.setattr(sys, "argv", ["protocol-sql-probe", *arguments])
            patch.setattr(
                ApplicationContext,
                "from_environment",
                lambda: SimpleNamespace(store=store),
            )
            output = io.StringIO()
            with redirect_stdout(output):
                exec(
                    compile(source, "<protocol-sql-probe>", "exec"),
                    {"__name__": "__main__"},
                )
            return json.loads(output.getvalue().splitlines()[-1])

    monkeypatch.setattr(seeded, "cpu_python", run)
    monkeypatch.setattr(net, "cpu_python", run)
    return run


def seed(run_id: str) -> dict[str, Any]:
    return seeded.seed_command(
        run_id,
        owner="protocol-test-owner",
        operation="TRIGGER_HEALTH_SNAPSHOT",
        node_ids=["synthetic-node"],
    )


def test_seeded_purge_preserves_another_run(
    database: tuple[str, PostgresStore], transport: Any
) -> None:
    _, store = database
    foreign = seed("foreign-run")
    owned = seed("owned-run")
    result = seeded.purge_seed(owned)
    assert result["remaining"] == [] and result["remaining_links"] == 0
    assert (
        store.get_remote_command(foreign["command_id"]).workflow_request_id
        == foreign["workflow_id"]
    )
    with pytest.raises(NotFoundError):
        store.get_remote_command(owned["command_id"])
    with pytest.raises(NotFoundError):
        store.get_incident(owned["incident_id"])


def test_net003_notification_purge_uses_authoritative_state(
    database: tuple[str, PostgresStore], transport: Any
) -> None:
    _, store = database
    foreign = seed("foreign-net003")
    owned = net003.seed_command("owned-net003")
    result = net003.purge_seed(owned)
    assert result["remaining"] == [] and result["remaining_links"] == 0
    assert (
        store.get_remote_command(foreign["command_id"]).cluster_id
        == seeded.SYNTHETIC_CLUSTER_ID
    )
    with pytest.raises(NotFoundError):
        store.get_workflow(owned["workflow_id"])


def test_cmd018_purge_discovers_commands_after_an_ack_loss(
    database: tuple[str, PostgresStore],
    transport: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, store = database
    foreign = seed("foreign-cmd018")
    owned_ids: list[str] = []

    def run_case(
        probe: Any,
        case_dir: Path,
        run_id: str,
        attempt: int,
        deadline: datetime,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        state["seed"] = seed(run_id)
        first = store.get_remote_command(state["seed"]["command_id"])
        second = first.model_copy(
            update={
                "command_id": first.command_id + "-ack-loss",
                "idempotency_key": first.idempotency_key + "/ack-loss",
            }
        )
        store.ensure_remote_command(second)
        owned_ids.extend((first.command_id, second.command_id))
        state["command_ids"] = []
        return {"verdict": "PASS"}

    def cleanup(*args: Any, state: dict[str, Any], purge: Any) -> None:
        purged = purge(state["seed"])
        assert purged["remaining_commands"] == 0

    monkeypatch.setattr(cmd018, "_run_case", run_case)
    monkeypatch.setattr(seeded, "cleanup", cleanup)
    assert (
        cmd018.run_case(tmp_path, 1, datetime.now(timezone.utc) + timedelta(hours=1))
        == 0
    )
    for command_id in owned_ids:
        with pytest.raises(NotFoundError):
            store.get_remote_command(command_id)
    assert (
        store.get_remote_command(foreign["command_id"]).command_id
        == foreign["command_id"]
    )


def test_dispatcher_census_sees_active_work_beyond_the_latest_500(
    database: tuple[str, PostgresStore], transport: Any
) -> None:
    _, store = database
    store.save_workflow(
        workflow_request("old-active", "incident", status=WorkflowStatus.RUNNING)
    )
    for index in range(501):
        store.save_workflow(
            workflow_request(
                f"new-terminal-{index:04d}", "incident", status=WorkflowStatus.SUCCEEDED
            )
        )
    report = transport(liveness.WORKFLOW_STATUSES_PROBE)
    assert report["status_counts"] == {"RUNNING": 1, "SUCCEEDED": 501}
    assert liveness.verdicts.workflow_census_errors(report), (
        "test_dispatcher_census_sees_active_work_beyond_the_latest_500: expected liveness.verdicts.workflow_census_errors(report)"
    )


@pytest.mark.parametrize("commit_ack_loss", [False, True])
def test_evidence_cleanup_recovers_real_transaction_abort_or_commit_ack_loss(
    database: tuple[str, PostgresStore], commit_ack_loss: bool
) -> None:
    import psycopg

    url, _ = database
    identity = evidence.audit_identity("protocol-evidence-" + uuid4().hex[:12])
    with psycopg.connect(url) as connection:
        if commit_ack_loss:

            class LostCommitAck:
                def __init__(self) -> None:
                    self.lost = False

                def cursor(self) -> Any:
                    return connection.cursor()

                def rollback(self) -> None:
                    connection.rollback()

                def commit(self) -> None:
                    connection.commit()
                    if not self.lost:
                        self.lost = True
                        raise TimeoutError("commit acknowledgement lost")

            report = evidence.audit(LostCommitAck(), identity["run_id"])
            assert report["verdict"] == "FAIL"
            assert report["error"] == "audit failed: TimeoutError"
            assert report["residual_rows"] == 0
            return
        now = datetime.now(timezone.utc)
        incident = FaultIncident(
            incident_id=identity["incident_id"],
            event_id="protocol-evidence-event",
            event_type="AUDIT_EVIDENCE_PIN",
            cluster_id=identity["cluster_id"],
            node_ids=[identity["pinned_node"]],
            policy_version="audit",
            policy_source="audit",
            state=IncidentState.ESCALATED,
            created_at=now,
            updated_at=now,
        ).model_dump(mode="json")
        connection.execute(
            "INSERT INTO gpu_fault_objects(kind,key,payload) VALUES ('incident',%s,%s::jsonb)",
            (identity["incident_id"], json.dumps(incident)),
        )
        connection.commit()
        with pytest.raises(psycopg.errors.DivisionByZero):
            connection.execute("SELECT 1/0")
        result = evidence.cleanup_rows(connection, identity)
        assert result == {"residual_rows": 0}
        assert connection.execute(
            "SELECT count(*) FROM gpu_fault_objects WHERE key=%s",
            (identity["incident_id"],),
        ).fetchone() == (0,)


def capacity_registration(cluster_id: str, run_id: str) -> RegionalClusterRegistration:
    return RegionalClusterRegistration(
        cluster_id=cluster_id,
        region="us-west-2",
        hyperpod_cluster_name=cluster_id,
        eks_cluster_arn=f"arn:aws:eks:us-west-2:000000000000:cluster/{cluster_id}",
        token_sha256="a" * 64,
        agent_endpoint_allowed_cidrs=["127.0.0.1/32"],
        synthetic=True,
        synthetic_run_id=run_id,
        synthetic_expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )


def publish_capacity_registry(
    store: PostgresStore, registrations: list[RegionalClusterRegistration]
) -> None:
    store.publish_regional_registry_revision(
        RegionalRegistryRevision.build(
            generation=1,
            registrations=registrations,
            previous_generation=None,
            required_member_ids=[],
            reason="capacity helper SQL test",
        ),
        expected_generation=0,
    )


def capacity_pair(
    store: PostgresStore,
    suffix: str,
    cluster_id: str,
    *,
    status: WorkflowStatus = WorkflowStatus.SUCCEEDED,
) -> tuple[str, str]:
    incident_id, workflow_id = (
        f"capacity-incident-{suffix}",
        f"capacity-workflow-{suffix}",
    )
    incident = FaultIncident(
        incident_id=incident_id,
        event_id=f"capacity-event-{suffix}",
        event_type="CAPACITY_HELPER_TEST",
        cluster_id=cluster_id,
        node_ids=[f"capacity-node-{suffix}"],
        policy_version="test",
        policy_source="test",
        state=IncidentState.RECOVERED,
        workflow_request_id=workflow_id,
    )
    store.save_incident_and_workflow(
        incident, workflow_request(workflow_id, incident_id, status=status)
    )
    return incident_id, workflow_id


def test_capacity_cleanup_preserves_another_run_and_removes_linked_records(
    database: tuple[str, PostgresStore],
) -> None:
    import psycopg

    url, store = database
    publish_capacity_registry(
        store,
        [
            capacity_registration("perf-cap-000", "run-a"),
            capacity_registration("perf-cap-001", "run-b"),
        ],
    )
    with psycopg.connect(url) as connection:
        assert (
            capacity_data.inspect_or_cleanup(
                connection, run_id="run-a", cluster_ids=["perf-cap-000"], cleanup=False
            )["total"]
            == 0
        )
    owned_incident, owned_workflow = capacity_pair(store, "owned", "perf-cap-000")
    foreign_incident, foreign_workflow = capacity_pair(store, "foreign", "perf-cap-001")
    rows = [
        (
            "notification",
            "capacity-notice",
            {"notification_id": "capacity-notice", "cluster_name": "perf-cap-000"},
        ),
        (
            "notification_delivery",
            "capacity-delivery",
            {"notification_id": "capacity-notice"},
        ),
        (
            "notification_result",
            "capacity-result",
            {"notification_id": "capacity-notice"},
        ),
        (
            "marker",
            "capacity-marker",
            {
                "incident_id": owned_incident,
                "scope": {"node_ids": ["capacity-node-owned"]},
            },
        ),
        (
            "xid_policy_decision",
            "capacity-decision",
            {"event_id": "capacity-event-owned"},
        ),
    ]
    with psycopg.connect(url, autocommit=True) as connection:
        for kind, key, payload in rows:
            connection.execute(
                "INSERT INTO gpu_fault_objects(kind,key,payload) VALUES (%s,%s,%s::jsonb)",
                (kind, key, json.dumps(payload)),
            )
        connection.execute(
            "INSERT INTO gpu_fault_links(kind,key,value) VALUES ('notification_by_dedup_key','capacity-dedup','capacity-notice')"
        )
        before = capacity_data.inspect_or_cleanup(
            connection, run_id="run-a", cluster_ids=["perf-cap-000"], cleanup=False
        )
        assert before["records"] == 7
        result = capacity_data.inspect_or_cleanup(
            connection, run_id="run-a", cluster_ids=["perf-cap-000"], cleanup=True
        )
        assert result["total"] == 0 and result["deleted_records"] == 7
        assert connection.execute(
            "SELECT count(*) FROM gpu_fault_links WHERE key IN ('capacity-event-owned','capacity-dedup')"
        ).fetchone() == (0,)
    assert store.get_workflow(foreign_workflow).incident_id == foreign_incident
    with pytest.raises(NotFoundError):
        store.get_workflow(owned_workflow)
    assert store.get_regional_registry_head().generation == 1
    assert store.get_regional_cluster("perf-cap-000").synthetic_run_id == "run-a"


@pytest.mark.parametrize("registry", ["absent", "foreign", "partial"])
def test_capacity_cleanup_refuses_missing_or_foreign_registry_before_deleting_old_rows(
    database: tuple[str, PostgresStore], registry: str
) -> None:
    import psycopg

    url, store = database
    registrations = (
        []
        if registry == "absent"
        else [
            capacity_registration(
                "perf-cap-000", "run-b" if registry == "foreign" else "run-a"
            )
        ]
    )
    publish_capacity_registry(store, registrations)
    _, workflow_id = capacity_pair(store, "old", "perf-cap-000")
    targets = (
        ["perf-cap-000", "perf-cap-001"] if registry == "partial" else ["perf-cap-000"]
    )
    with psycopg.connect(url) as connection:
        with pytest.raises(RuntimeError, match="complete current run-owned registry"):
            capacity_data.inspect_or_cleanup(
                connection, run_id="run-a", cluster_ids=targets, cleanup=True
            )
    assert store.get_workflow(workflow_id).status is WorkflowStatus.SUCCEEDED
    assert store.get_incident_by_event("capacity-event-old") is not None


def test_capacity_cleanup_refuses_inflight_work_without_removing_its_links(
    database: tuple[str, PostgresStore],
) -> None:
    import psycopg

    url, store = database
    publish_capacity_registry(store, [capacity_registration("perf-cap-000", "run-a")])
    _, workflow_id = capacity_pair(
        store, "running", "perf-cap-000", status=WorkflowStatus.RUNNING
    )
    with psycopg.connect(url) as connection:
        with pytest.raises(RuntimeError, match="still nonterminal"):
            capacity_data.inspect_or_cleanup(
                connection, run_id="run-a", cluster_ids=["perf-cap-000"], cleanup=True
            )
    assert store.get_workflow(workflow_id).status is WorkflowStatus.RUNNING
    assert store.get_incident_by_event("capacity-event-running") is not None


def test_capacity_cleanup_refuses_a_foreign_successor(
    database: tuple[str, PostgresStore],
) -> None:
    import psycopg

    url, store = database
    publish_capacity_registry(
        store,
        [
            capacity_registration("perf-cap-000", "run-a"),
            capacity_registration("perf-cap-001", "run-b"),
        ],
    )
    _, owned = capacity_pair(store, "owned-parent", "perf-cap-000")
    _, foreign = capacity_pair(store, "foreign-child", "perf-cap-001")
    workflow = store.get_workflow(foreign).model_copy(
        update={"predecessor_workflow_id": owned}
    )
    store.save_workflow(workflow)
    with psycopg.connect(url) as connection:
        with pytest.raises(RuntimeError, match="ownership"):
            capacity_data.inspect_or_cleanup(
                connection, run_id="run-a", cluster_ids=["perf-cap-000"], cleanup=True
            )
    assert store.get_workflow(owned).status is WorkflowStatus.SUCCEEDED
    assert store.get_workflow(foreign).predecessor_workflow_id == owned
