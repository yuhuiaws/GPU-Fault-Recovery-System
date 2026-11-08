"""The S74 event-driven cordon lift of GF-GENERAL-LIVE-013
``post_repair_isolation_lifted`` (``general_live_cordon_lift``).

Nothing reaches a cluster: ``tests/regional/_cordon_lift_fakes.FakeNodeServer``
plays the Node and the product's writes on it. The tests pin the trigger (the
provocation, the decisive re-lift inside T_Q, never a third write), every stop,
the cordon-only JSON patches, the restore, the latency record, the executor's
claim lines and the overreach check the grading re-runs on the write log.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_cordon_lift as lift
from scripts.e2e.regional import general_live_efa_isolation_lifted as variant
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tests.regional._cordon_lift_fakes import (
    BOOT,
    FENCING_KEY,
    HANDOFF,
    INCIDENT_KEY,
    NODE,
    SOURCE_INCIDENT,
    SOURCE_WORKFLOW,
    SUCCESSOR,
    FakeApiError,
    FakeClock,
    FakeNodeServer,
    handoff,
    mark,
    quarantine,
    reassert,
)
from tests.regional._general_live_fakes import FakeRegional, make_context, make_settings

SPEC = {item.variant: item for item in variant.VARIANTS}["post_repair_isolation_lifted"]


def plan(**overrides: Any) -> lift.LiftPlan:
    values = {
        "node": NODE,
        "source_workflow_id": SOURCE_WORKFLOW,
        "source_incident_id": SOURCE_INCIDENT,
        "baseline_boot": BOOT,
    }
    values.update(overrides)
    return lift.LiftPlan(**values)


def run(
    server: FakeNodeServer, *steps: Callable[[FakeNodeServer], Any], window: float = 600
) -> tuple[lift.CordonLifter, dict[str, Any]]:
    server.script.extend(steps)
    lifter = lift.CordonLifter(
        server, monotonic=server.clock.monotonic, wall=server.clock.wall
    )
    record = lifter.run(plan(), window_seconds=window)
    json.dumps(record)  # the evidence is plain JSON
    return lifter, record


def lifts_of(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in record["writes"] if item["kind"] == "lift"]


# --------------------------------------------------------------------------- #
# identities and patch shapes
# --------------------------------------------------------------------------- #
def test_the_successor_and_hand_off_are_named_from_the_source_workflow() -> None:
    assert lift.successor_incident_id(SOURCE_WORKFLOW) == SUCCESSOR
    assert lift.successor_workflow_id(SOURCE_WORKFLOW) == (
        "workflow-reboot-after-" + SOURCE_WORKFLOW
    )
    assert lift.handoff_incident_id(SOURCE_WORKFLOW) == HANDOFF
    assert plan().successor_incident == SUCCESSOR
    assert plan().handoff_incident == HANDOFF


def test_a_lift_tests_ownership_and_the_cordon_then_replaces_only_the_cordon() -> None:
    ops = lift.lift_ops(SUCCESSOR, "7")
    assert ops == [
        {
            "op": "test",
            "path": "/metadata/annotations/gpu-fault.io~1incident-id",
            "value": SUCCESSOR,
        },
        {
            "op": "test",
            "path": "/metadata/annotations/gpu-fault.io~1fencing-token",
            "value": "7",
        },
        {"op": "test", "path": "/spec/unschedulable", "value": True},
        {"op": "replace", "path": "/spec/unschedulable", "value": False},
    ]
    writes = [item for item in ops if item["op"] != "test"]
    assert [item["path"] for item in writes] == ["/spec/unschedulable"], (
        "the taint and the annotations are never written"
    )
    assert lift.restore_ops(SUCCESSOR, "7")[-1] == {
        "op": "add",
        "path": "/spec/unschedulable",
        "value": True,
    }


def test_node_state_reads_the_ownership_cordon_boot_and_server_time() -> None:
    state = lift.node_state(FakeNodeServer(FakeClock()).read())
    assert state == lift.NodeState(
        resource_version="100",
        incident=SOURCE_INCIDENT,
        fencing="1",
        unschedulable=True,
        boot_id=BOOT,
        managed_at="2026-11-08T03:00:00Z",
    )
    assert lift.node_state({}).unschedulable is False


# --------------------------------------------------------------------------- #
# the trigger
# --------------------------------------------------------------------------- #
def test_provocation_then_decisive_re_lift_then_hand_over() -> None:
    server = FakeNodeServer(FakeClock())
    lifter, record = run(server, mark, quarantine, handoff)
    assert record["stopped"] == "handed_over", record["stopped"]
    lifts = lifts_of(record)
    assert [item["role"] for item in lifts] == ["provocation", "decisive"]
    assert all(item["status"] == "applied" for item in lifts), lifts
    assert record["roles"] == {
        "provocation": lifts[0]["seq"],
        "decisive": lifts[1]["seq"],
    }
    assert record["decisive_seq"] == lifts[1]["seq"]
    # QUARANTINE's re-cordon happened between the two lifts and was undone.
    cordons = [item["unschedulable"] for item in record["events"]]
    assert cordons == [True, True, False, True, False, True], record["events"]
    assert server.patches == [lift.lift_ops(SUCCESSOR, "1")] * 2
    assert lift.overreach_errors(record, successor=SUCCESSOR) == []
    assert lifter.decider is not None and lifter.decider.fencing == "1"


def test_a_provocation_that_landed_after_quarantine_is_the_decisive_lift() -> None:
    server = FakeNodeServer(FakeClock())
    _lifter, record = run(server, mark, handoff)
    assert record["stopped"] == "handed_over"
    assert [item["role"] for item in lifts_of(record)] == ["provocation"]
    assert record["roles"]["decisive"] is None
    assert lift.decisive_lift(record) is record["lifts"][0]


def test_a_re_cordon_after_t_q_is_the_reboot_s_reassert_and_is_never_fought() -> None:
    server = FakeNodeServer(FakeClock())
    _lifter, record = run(server, mark, reassert)
    assert record["stopped"] == "reasserted"
    assert len(lifts_of(record)) == 1, "no write after the product re-cordoned"
    assert server.cordoned, "the product's re-cordon stands"


def test_never_a_third_lift_a_re_cordon_after_the_decisive_lift_stops() -> None:
    server = FakeNodeServer(FakeClock())
    _lifter, record = run(server, mark, quarantine, quarantine, reassert)
    assert record["stopped"] == "reasserted"
    assert len(lifts_of(record)) == lift.MAX_LIFTS == 2
    assert len(server.patches) == 2


def test_t_q_is_measured_from_the_provocation_s_ack() -> None:
    def slow_quarantine(server: FakeNodeServer) -> None:
        server.clock.advance(lift.T_Q_SECONDS + 0.5)
        server.product_write(unschedulable=True)

    server = FakeNodeServer(FakeClock())
    _lifter, record = run(server, mark, slow_quarantine)
    assert record["stopped"] == "reasserted"
    assert len(lifts_of(record)) == 1


def test_a_plan_capped_at_one_lift_takes_no_decisive_re_lift() -> None:
    server = FakeNodeServer(FakeClock())
    server.script.extend([mark, quarantine])
    lifter = lift.CordonLifter(
        server, monotonic=server.clock.monotonic, wall=server.clock.wall
    )
    record = lifter.run(plan(max_lifts=1), window_seconds=60)
    assert record["stopped"] == "reasserted"
    assert len(lifts_of(record)) == 1


def test_the_source_incident_is_never_lifted_and_the_window_ends_the_hold() -> None:
    server = FakeNodeServer(FakeClock())
    _lifter, record = run(server, window=130)
    assert record["stopped"] == "window"
    assert server.patches == []
    assert "restore_at_stop" in record
    assert record["restore_at_stop"]["reason"] == "no cordon lift was applied"


@pytest.mark.parametrize(
    ("owner", "stopped"),
    [
        ("inc-replace-after-" + SOURCE_WORKFLOW, "foreign_incident"),
        ("inc-support-after-" + SOURCE_WORKFLOW, "foreign_incident"),
        ("inc-someone-else", "foreign_incident"),
    ],
)
def test_another_incident_stops_the_hold_without_a_write(
    owner: str, stopped: str
) -> None:
    def foreign(server: FakeNodeServer) -> None:
        server.product_write(incident=owner, unschedulable=True)

    server = FakeNodeServer(FakeClock())
    _lifter, record = run(server, foreign)
    assert record["stopped"] == stopped
    assert server.patches == []


def test_a_foreign_incident_after_the_successor_stops_the_hold() -> None:
    def foreign(server: FakeNodeServer) -> None:
        server.product_write(incident="inc-someone-else", unschedulable=True)

    server = FakeNodeServer(FakeClock())
    _lifter, record = run(server, mark, foreign)
    assert record["stopped"] == "foreign_incident"
    assert len(server.patches) == 1


def test_a_boot_change_stops_the_hold() -> None:
    def rebooted(server: FakeNodeServer) -> None:
        server.product_write(boot="boot-after")

    server = FakeNodeServer(FakeClock())
    _lifter, record = run(server, mark, rebooted)
    assert record["stopped"] == "rebooted"
    assert len(server.patches) == 1


def test_a_released_node_stops_the_hold() -> None:
    def released(server: FakeNodeServer) -> None:
        server.product_write(drop_incident=True, unschedulable=False)

    server = FakeNodeServer(FakeClock())
    _lifter, record = run(server, mark, released)
    assert record["stopped"] == "released"


def test_a_new_fencing_generation_of_the_successor_stops_the_hold() -> None:
    def regenerated(server: FakeNodeServer) -> None:
        server.product_write(fencing="2", unschedulable=True)

    server = FakeNodeServer(FakeClock())
    _lifter, record = run(server, mark, regenerated)
    assert record["stopped"] == "foreign_generation"
    assert len(server.patches) == 1


def test_an_unfenced_successor_is_not_lifted() -> None:
    def unfenced(server: FakeNodeServer) -> None:
        server.document["metadata"]["annotations"].pop(FENCING_KEY)
        server.product_write(incident=SUCCESSOR, unschedulable=True)

    server = FakeNodeServer(FakeClock())
    _lifter, record = run(server, unfenced)
    assert record["stopped"] == "unfenced"
    assert server.patches == []


def test_a_successor_already_holding_the_node_at_arming_is_not_touched() -> None:
    server = FakeNodeServer(FakeClock(), incident=SUCCESSOR)
    _lifter, record = run(server, quarantine)
    assert record["stopped"] == "armed_late"
    assert server.patches == []
    assert record["events"][0]["type"] == "ARM"


def test_a_deleted_node_stops_the_hold_without_a_decision() -> None:
    def deleted(server: FakeNodeServer) -> None:
        # The last state of a deleted Node would be lift-eligible: no decision.
        gone = copy.deepcopy(server.document)
        gone["metadata"]["annotations"][INCIDENT_KEY] = SUCCESSOR
        gone["metadata"]["resourceVersion"] = "999"
        server.queue.append({"type": "DELETED", "object": gone})

    server = FakeNodeServer(FakeClock())
    _lifter, record = run(server, deleted)
    assert record["stopped"] == "node_deleted"
    assert server.patches == []
    assert record["events"][-1]["type"] == "DELETED"


# --------------------------------------------------------------------------- #
# preconditions, errors and the watch
# --------------------------------------------------------------------------- #
def test_a_failed_test_writes_nothing_and_the_next_event_decides() -> None:
    def moved(server: FakeNodeServer) -> None:
        server.document["spec"].pop("unschedulable")  # someone uncordoned it

    server = FakeNodeServer(FakeClock())
    server.before_patch = moved
    _lifter, record = run(server, mark, quarantine, handoff)
    statuses = [item["status"] for item in lifts_of(record)]
    assert statuses[0] == "precondition_failed", statuses
    assert record["writes"][0]["error"].startswith("FakeApiError(422)"), record[
        "writes"
    ]
    assert record["stopped"] == "handed_over"


def test_repeated_precondition_failures_end_the_hold() -> None:
    server = FakeNodeServer(FakeClock())
    server.patch_errors = [FakeApiError(422)] * 3

    def cordon_again(server: FakeNodeServer) -> None:
        server.product_write(unschedulable=True, fencing="1")

    def nudge(server: FakeNodeServer) -> None:
        server.document["metadata"]["annotations"]["x"] = str(server.version)
        server.commit()

    _lifter, record = run(server, mark, nudge, cordon_again, nudge)
    assert record["stopped"] == "precondition_failures"
    assert [item["status"] for item in lifts_of(record)] == ["precondition_failed"] * 3
    assert record["restore_at_stop"]["reason"] == "no cordon lift was applied"


def test_an_unknown_write_outcome_counts_against_the_cap() -> None:
    server = FakeNodeServer(FakeClock())
    server.patch_errors = [TimeoutError("read timed out")]
    _lifter, record = run(server, mark, quarantine, handoff)
    statuses = [item["status"] for item in lifts_of(record)]
    assert statuses[0] == "unknown"
    assert record["writes"][0]["error"] == "TimeoutError"
    assert lift.overreach_errors(record, successor=SUCCESSOR) == []


def test_a_410_relists_and_watches_on_from_the_new_version() -> None:
    def gone(server: FakeNodeServer) -> None:
        server.queue.append({"type": "ERROR", "object": {"code": 410}})

    def bookmark(server: FakeNodeServer) -> None:
        server.queue.append(
            {"type": "BOOKMARK", "object": {"metadata": {"resourceVersion": "555"}}}
        )

    server = FakeNodeServer(FakeClock())
    _lifter, record = run(server, bookmark, gone, mark, quarantine, handoff)
    assert record["relists"] == 1
    assert record["stopped"] == "handed_over"
    assert any(item["type"] == "RELIST" for item in record["events"]), record["events"]
    assert server.watches[0][0] == "100"


def test_a_watch_error_event_other_than_410_is_resynced() -> None:
    def broken(server: FakeNodeServer) -> None:
        server.queue.append({"type": "ERROR", "object": {"code": 500}})

    server = FakeNodeServer(FakeClock())
    _lifter, record = run(server, broken, mark, quarantine, handoff)
    assert record["errors"] == ["RegionalFixtureError"]
    assert record["stopped"] == "handed_over"


def test_a_watch_that_keeps_failing_ends_the_hold_and_restores() -> None:
    server = FakeNodeServer(FakeClock())
    server.script.append(mark)
    server.script.append(lambda s: "close")
    lifter = lift.CordonLifter(
        server, monotonic=server.clock.monotonic, wall=server.clock.wall
    )

    original_watch = server.watch
    calls = {"n": 0}

    def watch(version: str, seconds: int) -> Iterator[dict[str, Any]]:
        calls["n"] += 1
        if calls["n"] > 1:
            raise ConnectionError("stream reset")
        return original_watch(version, seconds)

    original_read = server.read
    reads = {"n": 0}

    def read() -> dict[str, Any]:
        # The arming read answers; the resync after the first chunk does not.
        reads["n"] += 1
        if reads["n"] == 2:
            raise ConnectionError("x")
        return original_read()

    server.watch = watch
    server.read = read
    record = lifter.run(plan(), window_seconds=600)
    assert record["stopped"] == "watch_failed"
    assert record["errors"] == ["ConnectionError"] * lift.MAX_WATCH_FAILURES
    assert record["restore_at_stop"]["restored"] is True, record["restore_at_stop"]
    assert server.cordoned, "the cordon was put back"
    assert record["writes"][-1]["kind"] == "restore"


def test_an_unreadable_node_at_arming_is_a_harness_failure() -> None:
    server = FakeNodeServer(FakeClock())
    server.read_errors = [FakeApiError(503, "unavailable")]
    lifter = lift.CordonLifter(
        server, monotonic=server.clock.monotonic, wall=server.clock.wall
    )
    with pytest.raises(RegionalFixtureError) as excinfo:
        lifter.run(plan(), window_seconds=60)
    assert "FakeApiError(503) unavailable" in str(excinfo.value)
    assert server.patches == []


def test_latencies_are_recorded_for_every_write() -> None:
    server = FakeNodeServer(FakeClock())
    _lifter, record = run(server, mark, quarantine, handoff)
    lifts = lifts_of(record)
    assert record["latency"]["patch_ack_ms"] == [20.0, 20.0]
    assert [item["patch_ack_ms"] for item in lifts] == [20.0, 20.0]
    assert [item["watch_delivery_ms"] for item in lifts] == [0.0, 0.0]
    assert [item["echo_after_send_ms"] for item in lifts] == [20.0, 20.0]
    assert record["latency"]["watch_delivery_ms"] == [0.0, 0.0]
    assert record["latency"]["reaction_ms"] == lifts[1]["reaction_ms"] == 20.0
    assert lifts[0]["sent_at"] < lifts[0]["acked_at"]


def test_the_event_log_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lift, "MAX_EVENTS", 3)

    def noise(server: FakeNodeServer) -> None:
        for index in range(5):
            server.document["metadata"]["annotations"]["x"] = str(index)
            server.commit()

    server = FakeNodeServer(FakeClock())
    _lifter, record = run(server, noise, window=70)
    assert len(record["events"]) == 3
    assert record["events_dropped"] >= 3


# --------------------------------------------------------------------------- #
# the restore
# --------------------------------------------------------------------------- #
def _lifted_then(
    *steps: Callable[[FakeNodeServer], Any],
) -> tuple[FakeNodeServer, lift.CordonLifter]:
    server = FakeNodeServer(FakeClock())
    lifter, _record = run(server, mark, *steps)
    return server, lifter


def test_restore_re_cordons_a_node_left_lifted_while_the_successor_holds_it() -> None:
    server = FakeNodeServer(FakeClock())
    server.script.extend([mark])
    lifter = lift.CordonLifter(
        server, monotonic=server.clock.monotonic, wall=server.clock.wall
    )
    record = lifter.run(plan(), window_seconds=130)
    assert record["stopped"] == "window"
    assert record["restore_at_stop"] == {
        "restored": True,
        "resource_version_after": server.document["metadata"]["resourceVersion"],
    }
    assert server.cordoned, "the restore at stop re-cordoned the node"
    assert server.patches[-1] == lift.restore_ops(SUCCESSOR, "1")
    assert lifter.restore() == {"skipped": True, "reason": "the node is cordoned"}
    assert lift.overreach_errors(record, successor=SUCCESSOR) == []


def test_restore_leaves_a_node_another_incident_took_over() -> None:
    server, lifter = _lifted_then(quarantine, handoff)
    assert lifter.restore() == {
        "skipped": True,
        "reason": f"the node is held by {HANDOFF!r}, not the successor",
    }


def test_restore_is_a_no_op_without_a_lift_or_without_the_node() -> None:
    lifter = lift.CordonLifter(FakeNodeServer(FakeClock()))
    assert lifter.restore()["reason"] == "no cordon lift was applied"
    server, lifted = _lifted_then(reassert)
    server.read_errors = [FakeApiError(404)]
    assert lifted.restore() == {"skipped": True, "reason": "the node is absent"}


def test_restore_that_cannot_read_or_land_fails_the_cleanup() -> None:
    server, lifter = _lifted_then(reassert)
    server.document["spec"].pop("unschedulable")
    server.read_errors = [FakeApiError(500, "boom")]
    with pytest.raises(RegionalFixtureError, match="cannot read"):
        lifter.restore()
    server.patch_errors = [FakeApiError(500, "boom")]
    with pytest.raises(RegionalFixtureError, match="did not land"):
        lifter.restore()


def test_restore_verifies_an_unknown_or_raced_write() -> None:
    server, lifter = _lifted_then(reassert)
    server.document["spec"].pop("unschedulable")

    def cordoned_meanwhile(server: FakeNodeServer) -> None:
        server.document["spec"]["unschedulable"] = True

    server.patch_errors = [TimeoutError()]
    server.before_patch = cordoned_meanwhile
    assert lifter.restore() == {"restored": True, "verified": "the node reads cordoned"}

    server.document["spec"].pop("unschedulable")

    def taken(server: FakeNodeServer) -> None:
        server.document["metadata"]["annotations"][INCIDENT_KEY] = HANDOFF

    server.before_patch = taken
    assert lifter.restore() == {
        "skipped": True,
        "reason": "the node moved during the restore",
    }
    # The patch fails its test and the re-read finds the node gone.
    server.document["metadata"]["annotations"][INCIDENT_KEY] = SUCCESSOR
    server.patch_errors = [FakeApiError(422)]
    server.before_patch = None
    original_read = server.read
    reads = {"n": 0}

    def read() -> dict[str, Any]:
        reads["n"] += 1
        if reads["n"] == 1:
            return original_read()
        raise FakeApiError(404)

    server.read = read
    assert lifter.restore() == {"skipped": True, "reason": "the node is absent"}


def test_close_is_idempotent() -> None:
    server = FakeNodeServer(FakeClock())
    lifter = lift.CordonLifter(server)
    assert lifter.close() == {"closed": True}
    assert lifter.close() == {"closed": True}
    assert server.closed == 1


# --------------------------------------------------------------------------- #
# the client
# --------------------------------------------------------------------------- #
class FakeResponse:
    def __init__(self, data: Any = b"{}", chunks: list[Any] | None = None) -> None:
        self.data = data
        self.chunks = chunks or []
        self.closed = False
        self.released = False

    def stream(self, amt: Any = None, decode_content: Any = None) -> Iterator[Any]:
        yield from self.chunks

    def close(self) -> None:
        self.closed = True

    def release_conn(self) -> None:
        self.released = True


class FakeCore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.watch_response = FakeResponse(
            chunks=[
                b'{"type": "MODIFIED", "object": {"metadata": {"resourceVersion": "2"}}}\n{"ty',
                'pe": "BOOKMARK", "object": {}}\n\n',
                b'[1]\n{"type": "MODIFIED", "object": {}}',
            ]
        )

    def read_node(self, *args: Any, **kwargs: Any) -> FakeResponse:
        self.calls.append(("read_node", args, kwargs))
        return FakeResponse(b'{"metadata": {"resourceVersion": "1"}}')

    def list_node(self, *args: Any, **kwargs: Any) -> FakeResponse:
        self.calls.append(("list_node", args, kwargs))
        return self.watch_response

    def patch_node(self, *args: Any, **kwargs: Any) -> FakeResponse:
        self.calls.append(("patch_node", args, kwargs))
        return FakeResponse('{"metadata": {"resourceVersion": "3"}}')


class FakeApiClient:
    def __init__(self, *, verify_ssl: bool = True, host: str = "https://api") -> None:
        self.configuration = type("C", (), {"verify_ssl": verify_ssl, "host": host})()
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_node_client_reads_watches_and_patches_over_one_client() -> None:
    core = FakeCore()
    api = FakeApiClient()
    client = lift.NodeClient(api, core, NODE)
    assert client.read()["metadata"]["resourceVersion"] == "1"
    events = list(client.watch("1", 30))
    assert [item.get("type") for item in events] == ["MODIFIED", "BOOKMARK", "MODIFIED"]
    assert core.watch_response.closed and core.watch_response.released
    ops = lift.lift_ops(SUCCESSOR, "1")
    assert client.patch(ops)["metadata"]["resourceVersion"] == "3"
    client.close()
    assert api.closed, "close() closes the one ApiClient"
    names = [name for name, _args, _kwargs in core.calls]
    assert names == ["read_node", "list_node", "patch_node"]
    _name, args, kwargs = core.calls[1]
    assert kwargs["field_selector"] == f"metadata.name={NODE}"
    assert kwargs["watch"] is True and kwargs["resource_version"] == "1"
    assert kwargs["timeout_seconds"] == 30 and kwargs["allow_watch_bookmarks"] is True
    assert kwargs["_request_timeout"] == (5, 45)
    _name, args, kwargs = core.calls[2]
    assert args == (NODE, ops), "a list body: application/json-patch+json"
    assert kwargs["_preload_content"] is False


def test_node_client_refuses_a_non_object_answer() -> None:
    core = FakeCore()
    core.read_node = lambda *a, **k: FakeResponse(b"[]")
    with pytest.raises(RegionalFixtureError, match="non-object"):
        lift.NodeClient(FakeApiClient(), core, NODE).read()


def test_from_kubeconfig_requires_verified_tls_and_never_persists_the_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kubernetes.client
    import kubernetes.config

    seen: dict[str, Any] = {}
    clients = [
        FakeApiClient(verify_ssl=False),
        FakeApiClient(host="http://api"),
        FakeApiClient(),
    ]

    def new_client_from_config(**kwargs: Any) -> FakeApiClient:
        seen.update(kwargs)
        return clients.pop(0)

    monkeypatch.setattr(
        kubernetes.config, "new_client_from_config", new_client_from_config
    )
    monkeypatch.setattr(kubernetes.client, "CoreV1Api", lambda api: ("core", api))
    for _ in range(2):
        with pytest.raises(RegionalFixtureError, match="verified TLS"):
            lift.NodeClient.from_kubeconfig(Path("/k"), "ctx", NODE)
    built = lift.NodeClient.from_kubeconfig(Path("/k"), "ctx", NODE)
    assert seen == {"config_file": "/k", "context": "ctx", "persist_config": False}
    assert built.core[0] == "core" and built.node == NODE


def test_describe_names_type_status_and_reason_only() -> None:
    error = FakeApiError(422, "Unprocessable Entity")
    error.body = "secret-looking body"
    assert lift.describe(error) == "FakeApiError(422) Unprocessable Entity"
    assert lift.describe(ValueError("anything")) == "ValueError"


# --------------------------------------------------------------------------- #
# the context wiring
# --------------------------------------------------------------------------- #
class ClaimsRegional(FakeRegional):
    def __init__(self, logs: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.logs = logs

    def ready_pods(self, plane: str, app: str) -> list[dict[str, Any]]:
        assert (plane, app) == ("gpu", lift.EXECUTOR_APP), (plane, app)
        return [{"name": name} for name in self.logs]

    def kubectl(self, plane: str, *arguments: str, **kwargs: Any) -> str:
        super().kubectl(plane, *arguments, **kwargs)
        value = self.logs[arguments[1]]
        if isinstance(value, Exception):
            raise value
        return str(value)


CLAIMED = (
    "2026-11-08T03:00:01.123456789Z INFO regional cluster executor claimed "
    "command=remote-abc cluster=hp-cluster operation=RESTART_NODE "
    "owner=gpu-fault-hyperpod-adapter nodes={node} workflow={workflow} step=3 "
    "generation=1"
)
FINISHED = (
    "2026-11-08T03:00:02.5Z INFO regional cluster executor finished "
    "command={command} operation=RESTART_NODE nodes={node} status=FAILED"
)


def _claim_logs(workflow: str) -> str:
    return "\n".join(
        [
            "2026-11-08T03:00:00Z INFO regional cluster executor heartbeat: x",
            CLAIMED.format(node=NODE, workflow=workflow),
            CLAIMED.format(node="other-node", workflow=workflow),
            CLAIMED.format(node=NODE, workflow="workflow-other").replace(
                "remote-abc", "remote-other"
            ),
            FINISHED.format(command="remote-abc", node=NODE),
            FINISHED.format(command="remote-zzz", node=NODE),
            "garbage",
        ]
    )


def test_the_executor_claim_lines_are_read_as_fields_only() -> None:
    workflow = lift.successor_workflow_id(SOURCE_WORKFLOW)
    regional = ClaimsRegional(
        {"exec-a": _claim_logs(workflow), "exec-b": RegionalFixtureError("gone")}
    )
    report = lift.executor_claims(
        regional, workflow_id=workflow, node=NODE, since="2026-11-08T02:59:00Z"
    )
    assert report["pods"] == ["exec-a"]
    assert report["claims"] == [
        {
            "pod": "exec-a",
            "at": "2026-11-08T03:00:01.123456789Z",
            "command": "remote-abc",
            "step": "3",
            "generation": "1",
        }
    ]
    assert report["finishes"] == [
        {
            "pod": "exec-a",
            "at": "2026-11-08T03:00:02.5Z",
            "command": "remote-abc",
            "status": "FAILED",
        }
    ]
    assert report["errors"] == ["exec-b: RegionalFixtureError"]
    call = regional.kubectl_calls[0]
    assert call[:5] == ("gpu", "logs", "exec-a", "-c", "executor")
    assert "--timestamps" in call and "--since-time=2026-11-08T02:59:00Z" in call
    assert "INFO" not in json.dumps(report), "no raw log line is kept"


def test_executor_claims_survive_a_failed_pod_listing() -> None:
    report = lift.executor_claims(
        FakeRegional(), workflow_id="w", node=NODE, since="2026-11-08T02:59:00Z"
    )
    assert report["errors"] == ["executor Pods: AttributeError"]
    assert report["claims"] == []


def test_open_and_hold_register_close_then_restore_before_the_first_lift(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    server = FakeNodeServer(clock)
    server.script.extend([mark, quarantine, handoff])
    workflow = lift.successor_workflow_id(SOURCE_WORKFLOW)
    regional = ClaimsRegional({"exec-a": _claim_logs(workflow)})
    ctx = make_context(
        SPEC,
        tmp_path=tmp_path,
        regional=regional,
        probes={},
        settings=make_settings(SPEC, node=NODE),
    )
    seen: dict[str, Any] = {}

    def factory(kubeconfig: Path, context: str, node: str) -> FakeNodeServer:
        seen.update(kubeconfig=kubeconfig, context=context, node=node)
        return server

    lifter = lift.open_cordon_lifter(ctx, factory=factory)
    assert seen["node"] == NODE and seen["context"] == "ctx"
    assert ctx.cleanup.pending() == [lift.CLIENT_CLEANUP]
    lifter.monotonic = clock.monotonic
    lifter.wall = clock.wall
    pending_at_first_write: list[list[str]] = []
    original_patch = server.patch

    def patch(ops: list[dict[str, Any]]) -> dict[str, Any]:
        pending_at_first_write.append(ctx.cleanup.pending())
        return original_patch(ops)

    server.patch = patch
    record = lift.hold_cordon_lifted(ctx, lifter, plan(), window=600)
    assert pending_at_first_write[0] == [lift.CLIENT_CLEANUP, lift.RESTORE_CLEANUP]
    assert record["stopped"] == "handed_over"
    assert record["executor_claims"]["claims"][0]["command"] == "remote-abc"
    assert record["executor_claims"]["since"] == "2026-11-08T02:59:00Z"
    saved = json.loads((ctx.case_dir / f"{lift.HOLD_EVIDENCE}.json").read_text())
    assert saved["schema"] == lift.SCHEMA and len(saved["lifts"]) == 2
    results = ctx.cleanup.run()
    assert [item["name"] for item in results] == [
        lift.RESTORE_CLEANUP,
        lift.CLIENT_CLEANUP,
    ]
    assert results[0]["result"]["skipped"] is True
    assert server.closed == 1


def test_the_hold_window_ends_before_the_action_window(tmp_path: Path) -> None:
    clock = FakeClock()
    server = FakeNodeServer(clock)
    ctx = make_context(
        SPEC,
        tmp_path=tmp_path,
        regional=ClaimsRegional({}),
        probes={},
        settings=make_settings(SPEC, node=NODE),
        window_minutes=3,
    )
    lifter = lift.CordonLifter(server, monotonic=clock.monotonic, wall=clock.wall)
    record = lift.hold_cordon_lifted(ctx, lifter, plan(), window=600)
    assert record["window_seconds"] < 180 - lift.ACTION_MARGIN_SECONDS + 1
    assert record["stopped"] == "window"


# --------------------------------------------------------------------------- #
# the write log the grading re-checks
# --------------------------------------------------------------------------- #
def _record() -> dict[str, Any]:
    server = FakeNodeServer(FakeClock())
    _lifter, record = run(server, mark, quarantine, handoff)
    return record


def test_overreach_is_empty_for_a_clean_hold_and_for_a_pre_v2_record() -> None:
    assert lift.overreach_errors(_record(), successor=SUCCESSOR) == []
    assert (
        lift.overreach_errors({"lifts": [{"patch": {}}] * 5}, successor=SUCCESSOR) == []
    )


def test_overreach_names_every_broken_rule() -> None:
    record = _record()
    taint = copy.deepcopy(record["writes"][0])
    taint["ops"] = [{"op": "replace", "path": "/spec/taints", "value": []}]
    foreign = copy.deepcopy(record["writes"][1])
    foreign["incident"] = "inc-someone-else"
    late = copy.deepcopy(record["writes"][1])
    late["seq"] = record["stop_seq"] + 5
    record["writes"] = [taint, foreign, late]
    errors = lift.overreach_errors(record, successor=SUCCESSOR)
    assert any("not a cordon-only lift patch" in item for item in errors), errors
    assert any("was decided under 'inc-someone-else'" in item for item in errors), (
        errors
    )
    assert any("after the hold stopped" in item for item in errors), errors
    assert any("3 cordon lifts were applied" in item for item in errors), errors
    other = lift.overreach_errors(_record(), successor="inc-reboot-after-workflow-x")
    assert other and "not the successor" in other[0], other


def test_rv_position_places_a_read_against_the_decisive_lift() -> None:
    record = _record()
    decisive = lift.decisive_lift(record)
    assert decisive is not None
    before = [
        e["resource_version"] for e in record["events"] if e["seq"] < decisive["seq"]
    ]
    after = [
        e["resource_version"] for e in record["events"] if e["seq"] > decisive["seq"]
    ]
    assert lift.rv_position(record, before[-1]) == "before"
    assert lift.rv_position(record, after[-1]) == "after"
    assert lift.rv_position(record, "nope") is None
    assert lift.rv_position({"lifts": [{"seq": 1}]}, before[-1]) is None
    assert lift.decisive_lift({"lifts": [{"seq": 1}]}) is None


def test_the_decider_keeps_its_first_stop_and_decides_nothing_after_it() -> None:
    decider = lift.LiftDecider(plan())
    assert decider.stop("handed_over") == lift.Decision("stop", "handed_over")
    decider.stop("window")
    assert decider.stopped == "handed_over"
    state = lift.node_state(FakeNodeServer(FakeClock(), incident=SUCCESSOR).read())
    assert decider.observe(state, 0.0) == lift.Decision("none", "stopped")


def test_a_watch_stream_that_ends_on_a_newline_has_no_tail() -> None:
    core = FakeCore()
    core.watch_response = FakeResponse(chunks=[b'{"type": "MODIFIED", "object": {}}\n'])
    events = list(lift.NodeClient(FakeApiClient(), core, NODE).watch("1", 5))
    assert [item["type"] for item in events] == ["MODIFIED"]


def test_a_failed_resync_is_recorded_and_the_hold_goes_on() -> None:
    server = FakeNodeServer(FakeClock())
    server.script.extend([mark, lambda s: "close"])
    original_read = server.read
    reads = {"n": 0}

    def read() -> dict[str, Any]:
        reads["n"] += 1
        if reads["n"] in (2, 3):
            raise ConnectionError("x")
        return original_read()

    server.read = read
    server.script.extend([quarantine, handoff])
    lifter = lift.CordonLifter(
        server, monotonic=server.clock.monotonic, wall=server.clock.wall
    )
    record = lifter.run(plan(), window_seconds=600)
    assert record["errors"][:2] == ["ConnectionError", "ConnectionError"]
    assert record["stopped"] == "handed_over"


def test_a_restore_at_stop_that_fails_is_recorded_for_the_cleanup_to_retry() -> None:
    server = FakeNodeServer(FakeClock())
    server.script.append(mark)
    original_read = server.read
    reads = {"n": 0}

    def read() -> dict[str, Any]:
        reads["n"] += 1
        if reads["n"] > 2:
            raise FakeApiError(500, "down")
        return original_read()

    server.read = read
    lifter = lift.CordonLifter(
        server, monotonic=server.clock.monotonic, wall=server.clock.wall
    )
    record = lifter.run(plan(), window_seconds=70)
    assert record["stopped"] == "window"
    assert record["restore_at_stop"] == {"error": "RegionalFixtureError"}
    assert not server.cordoned, "left for the cleanup restore"


def test_rv_position_ignores_a_malformed_event() -> None:
    record = {
        "schema": lift.SCHEMA,
        "lifts": [{"seq": 3}],
        "events": [{"resource_version": "5", "seq": "x"}],
    }
    assert lift.rv_position(record, "5") is None


def test_an_event_written_before_the_provocation_landed_is_stale() -> None:
    """A third writer cordons-and-labels the node between MARK's event and the
    provocation: its event arrives before the provocation's echo, so it is not
    QUARANTINE's re-cordon and no decisive lift is spent on it."""

    def other_writer(server: FakeNodeServer) -> None:
        server.document["metadata"]["annotations"]["health"] = "ok"
        server.commit()

    server = FakeNodeServer(FakeClock())
    server.before_patch = other_writer
    _lifter, record = run(server, mark, quarantine, handoff)
    assert record["stopped"] == "handed_over", record["stopped"]
    assert [item["role"] for item in lifts_of(record)] == ["provocation", "decisive"]
    assert record["stale_events"] == 1


def test_the_controller_taint_after_quarantine_is_not_a_reassert() -> None:
    """The node lifecycle controller adds its unschedulable taint to QUARANTINE's
    re-cordon before the decisive lift lands: stale, the hold runs on."""

    def controller_taint(server: FakeNodeServer) -> None:
        server.document["spec"]["taints"].append(
            {"key": "node.kubernetes.io/unschedulable", "effect": "NoSchedule"}
        )
        server.commit()

    def quarantine_then_controller(server: FakeNodeServer) -> None:
        quarantine(server)
        server.before_patch = controller_taint

    server = FakeNodeServer(FakeClock())
    _lifter, record = run(server, mark, quarantine_then_controller, handoff)
    assert record["stopped"] == "handed_over", record["stopped"]
    assert len(lifts_of(record)) == 2
    assert record["stale_events"] == 1
