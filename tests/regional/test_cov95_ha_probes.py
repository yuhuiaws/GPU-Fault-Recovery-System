from __future__ import annotations

import io
import json
import runpy
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.error import HTTPError, URLError

import pytest

from gpu_fault.cluster_executor import ClusterExecutorError
from gpu_fault.collectors import CollectorError
from gpu_fault.models import (
    FaultIncident,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
)
from gpu_fault.regional import RemoteActionCommand, RemoteCommandStatus
from gpu_fault.store import InMemoryStore
from scripts.e2e.regional.probes import (
    ha001_probe,
    ha005_probe,
    ha006_executor,
    ha010_probe,
)
from tests.regional._cov95_ha001_harness import Clock


def relocate(
    module: Any, patch: pytest.MonkeyPatch, root: Path, names: tuple[str, ...]
) -> None:
    patch.setattr(module, "STATE", root)
    for name in names:
        patch.setattr(module, name, root / getattr(module, name).name)


def claimable_command() -> tuple[InMemoryStore, RemoteActionCommand]:
    step = WorkflowStepSpec(
        operation=WorkflowOperation.FREEZE_EVIDENCE,
        execution_owner="gpu-fault-ha005-noop",
    )
    incident = FaultIncident(
        incident_id="unit-incident",
        event_id="unit-event",
        event_type="UNIT",
        cluster_id="unit-cluster",
        node_ids=[],
        policy_version="unit",
        policy_source="UNIT",
    )
    workflow = WorkflowRequest(
        request_id="unit-workflow",
        incident_id=incident.incident_id,
        status=WorkflowStatus.BLOCKED,
        fencing_token=1,
        official_steps=[step],
    )
    command = RemoteActionCommand(
        command_id="unit-command",
        cluster_id=incident.cluster_id,
        incident_id=incident.incident_id,
        workflow_request_id=workflow.request_id,
        step_index=0,
        fencing_token=1,
        idempotency_key="unit-workflow/0",
        step=step,
        incident=incident,
        workflow=workflow,
    )
    store = InMemoryStore()
    store.save_incident(incident)
    store.save_workflow(workflow)
    store.ensure_remote_command(command)
    return store, command


@pytest.mark.parametrize(
    "failure", ["claim-http", "health-shape", "health-error", "slow"]
)
def test_ha001_probe_records_failure_recovery_and_bounded_cycles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    probe = ha001_probe
    relocate(probe, monkeypatch, tmp_path, ("READY", "STATS", "STOP", "LEDGER"))
    for name in (
        "GPU_FAULT_CONTROL_PLANE_URL",
        "GPU_FAULT_CLUSTER_ID",
        "GPU_FAULT_CONTROL_PLANE_TOKEN",
        "GPU_FAULT_CONTROL_PLANE_CA_FILE",
        "GPU_FAULT_EXECUTOR_ARTIFACT_SHA256",
        "GPU_FAULT_EXECUTOR_COMPATIBILITY_DIGEST",
    ):
        monkeypatch.setenv(name, "REPLACE_WITH_UNIT_INPUT")
    clock = Clock()
    iteration = 0

    def claim() -> int:
        nonlocal iteration
        iteration += 1
        if failure == "slow":
            clock.now += 3
        if failure == "claim-http" and iteration <= 2:
            raise ClusterExecutorError("unit", status_code=503)
        return 0

    def health(path: str) -> dict | list:
        assert path == "/healthz"
        if iteration == 3:
            probe.STOP.touch()
        if iteration <= 2:
            if failure == "health-shape":
                return []
            if failure == "health-error":
                raise OSError("unit health unavailable")
        return {"status": "ok"}

    client = SimpleNamespace(cluster_id="unit-cluster", _get=health)
    executor = SimpleNamespace(
        executor_id="unit-executor",
        run_once=claim,
        claimed_total=0,
        reported_failures=0,
        unexpected_failures=0,
        lease_renewal_failures=0,
    )
    monkeypatch.setattr(probe, "RegionalExecutorClient", lambda *a, **kw: client)
    monkeypatch.setattr(probe, "ClusterActionExecutor", lambda *a, **kw: executor)
    monkeypatch.setattr(
        probe,
        "time",
        SimpleNamespace(
            monotonic=clock.monotonic, sleep=clock.sleep, time=lambda: 1000 + clock.now
        ),
    )
    probe.main()
    result = json.loads(probe.STATS.read_text())
    assert iteration == 3
    assert result["current_failure_window_seconds"] == 0
    if failure == "slow":
        assert clock.now == 9
        assert result["max_failure_window_seconds"] == 0
    else:
        assert result["max_failure_window_seconds"] == 4
        assert sum(result["error_counts"].values()) == 2
    assert result["counters"]["claim_success"] == (1 if failure == "claim-http" else 3)


def test_ha001_adapter_support_and_ledger_replay_use_real_step_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(ha001_probe, "LEDGER", tmp_path / "ledger.json")
    step = WorkflowStepSpec(
        operation=WorkflowOperation.FREEZE_EVIDENCE, execution_owner=ha001_probe.OWNER
    )
    adapter = ha001_probe.SimulatedAdapter()
    assert adapter.supports(step) is True
    assert (
        adapter.supports(step.model_copy(update={"execution_owner": "foreign"}))
        is False
    )
    context = SimpleNamespace(step=step, idempotency_key="unit-key")
    first, replay = adapter.execute(context), adapter.execute(context)
    assert first.details["cached"] is False
    assert replay.details["cached"] is True
    assert json.loads(ha001_probe.LEDGER.read_text())["physical_count"] == 1


@pytest.mark.parametrize(
    "mode",
    [
        "success",
        "claim-http",
        "claim-error",
        "post-http",
        "post-unbuffered",
        "no-receipt",
    ],
)
def test_ha005_probe_claim_complete_buffer_and_replay_lifecycle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str
) -> None:
    probe = ha005_probe
    relocate(probe, monkeypatch, tmp_path, ("READY", "STATS", "STOP", "OUTBOX"))
    registrations = tmp_path / "registrations.json"
    registrations.write_text(
        json.dumps([{"cluster_id": "unit-cluster", "token": "REPLACE_WITH_UNIT_TOKEN"}])
    )
    monkeypatch.setattr(probe, "Path", lambda _: registrations)
    for name in (
        "RUN_ID",
        "CONTROL_PLANE_URL",
        "EXECUTOR_ARTIFACT_SHA256",
        "EXECUTOR_COMPATIBILITY_DIGEST",
    ):
        monkeypatch.setenv(name, "unit")
    monkeypatch.setenv("COMMAND_DELAY_SECONDS", "0.5")
    clock = Clock()
    store, command = claimable_command()
    completions, posts, replayed = [], [], []

    def claim(owner: str, **kwargs: Any) -> list[RemoteActionCommand]:
        if mode == "claim-http":
            raise ClusterExecutorError("unit", status_code=503)
        if mode == "claim-error":
            raise RuntimeError("unit claim unavailable")
        return store.claim_remote_commands(
            "unit-cluster",
            owner,
            limit=1,
            lease_seconds=60,
            execution_owners=set(kwargs["execution_owners"]),
        )

    def complete(item: RemoteActionCommand, result: Any) -> None:
        completions.append(result)
        assert result.status is RemoteCommandStatus.SUCCEEDED
        assert result.details["simulated"] is True
        store.complete_remote_command(item.cluster_id, item.command_id, result)

    class Sink:
        def __init__(self, *args, evidence, **kwargs):
            self.evidence = evidence

        def post(self, path: str, payload: dict) -> dict:
            posts.append(payload)
            assert path == probe.HOST_PATH
            assert payload["cluster_id"] == "unit-cluster"
            assert payload["workload_state"] == "IDLE"
            if mode in {"post-http", "post-unbuffered"}:
                raise CollectorError(
                    "unit",
                    status_code=503 if mode == "post-http" else None,
                    buffered=mode == "post-http",
                    replayable=mode == "post-http",
                )
            result = (
                {}
                if mode == "no-receipt"
                else {"accepted": True, "processor_request_id": f"request-{len(posts)}"}
            )
            self.evidence.admissions.append(
                {
                    "batch_id": payload["batch_id"],
                    "accepted": result.get("accepted"),
                    "replay": False,
                    "spooled": False,
                    "coalesced": False,
                    "request_id": result.get("processor_request_id"),
                }
            )
            return result

        def wait_for_outbox_replay(self, seconds: float) -> bool:
            replayed.append(seconds)
            probe.OUTBOX.write_text("")
            return True

    def sleep(seconds: float) -> None:
        clock.sleep(seconds)
        if clock.now >= 6:
            probe.STOP.touch()

    probe.OUTBOX.write_text('\n{"replayable":true}\n{"replayable":false}\n')
    assert probe.outbox_status() == {"records": 2, "replayable": 1}
    monkeypatch.setattr(
        probe,
        "RegionalExecutorClient",
        lambda *a, **kw: SimpleNamespace(claim=claim, complete=complete),
    )
    monkeypatch.setattr(probe, "ObservedSink", Sink)
    monkeypatch.setattr(
        probe, "time", SimpleNamespace(monotonic=clock.monotonic, sleep=sleep)
    )
    probe.main()
    report = json.loads(probe.STATS.read_text())
    assert report["stopped"] is True
    assert replayed == [10]
    assert report["outbox"] == {"records": 0, "replayable": 0}
    assert len(posts) == 2
    if mode.startswith("claim"):
        assert completions == []
        assert report["counters"]["claim_failures"] >= 2
    else:
        assert len(completions) == 1
        assert (
            store.get_remote_command(command.command_id).status
            is RemoteCommandStatus.SUCCEEDED
        )
    if mode.startswith("post"):
        assert report["counters"]["event_failures"] == 2
        assert report["counters"].get("event_buffered", 0) == (
            2 if mode == "post-http" else 0
        )
    else:
        assert report["counters"]["event_accepted"] == 2
        assert len(report["accepted_request_ids"]) == (0 if mode == "no-receipt" else 2)


@pytest.mark.parametrize("has_claim", [False, True])
def test_ha006_probe_wires_executor_identity_and_records_a_bounded_sample(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, has_claim: bool
) -> None:
    probe = ha006_executor
    relocate(
        probe, monkeypatch, tmp_path, ("READY", "EXECUTOR_STATE", "WINNER", "PHYSICAL")
    )
    registrations = tmp_path / "registrations.json"
    registrations.write_text(
        json.dumps([{"cluster_id": "unit-cluster", "token": "REPLACE_WITH_UNIT_TOKEN"}])
    )
    monkeypatch.setattr(probe, "Path", lambda _: registrations)
    for name in (
        "RUN_ID",
        "CONTROL_PLANE_URL",
        "EXECUTOR_ARTIFACT_SHA256",
        "EXECUTOR_COMPATIBILITY_DIGEST",
    ):
        monkeypatch.setenv(name, "unit")
    monkeypatch.setenv("WINNER_SLEEP_SECONDS", "0")
    monkeypatch.setattr(
        probe, "socket", SimpleNamespace(gethostname=lambda: "unit-executor")
    )
    store = InMemoryStore()
    executor_calls = []
    adapter_refs = []

    class Recorded(BaseException):
        pass

    def sleep(seconds: float) -> None:
        if seconds == 0.5:
            raise Recorded

    class Thread:
        def __init__(self, *, target: Any, args: tuple, daemon: bool) -> None:
            self.target, self.args = target, args

        def start(self) -> None:
            with pytest.raises(Recorded):
                self.target(*self.args)

    def executor(client: Any, adapters: list, **kwargs: Any) -> Any:
        adapter_refs.extend(adapters)
        assert kwargs["allowed_namespaces"] == {"default"}
        return SimpleNamespace(
            executor_id=kwargs["executor_id"],
            claimed_total=int(has_claim),
            reported_failures=0,
            unexpected_failures=0,
            lease_renewal_failures=0,
            last_successful_claim_at=datetime.now(timezone.utc) if has_claim else None,
            run=lambda: executor_calls.append("run"),
        )

    monkeypatch.setattr(probe, "Thread", Thread)
    monkeypatch.setattr(probe, "time", SimpleNamespace(sleep=sleep, time=lambda: 1000))
    monkeypatch.setattr(
        probe, "RegionalExecutorClient", lambda *a, **kw: SimpleNamespace()
    )
    monkeypatch.setattr(probe, "RegionalFleetRegistry", lambda _: store)
    monkeypatch.setattr(probe, "ClusterActionExecutor", executor)
    probe.main()
    report = json.loads(probe.EXECUTOR_STATE.read_text())
    assert report["claimed_total"] == int(has_claim)
    assert (report["last_successful_claim_at"] is not None) is has_claim
    assert executor_calls == ["run"]
    step = WorkflowStepSpec(
        operation=WorkflowOperation.FREEZE_EVIDENCE, execution_owner=probe.OWNER
    )
    assert adapter_refs[0].supports(step) is True
    assert (
        adapter_refs[0].supports(step.model_copy(update={"execution_owner": "other"}))
        is False
    )


@pytest.mark.parametrize(
    "mode", ["json", "empty", "scalar", "invalid", "http-error", "transport"]
)
def test_ha010_fetch_preserves_http_refusal_and_unknown_transport(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    body = {
        "json": b'{"status":"ok"}',
        "empty": b"",
        "scalar": b"[]",
        "invalid": b"invalid",
    }

    def urlopen(url: str, **kwargs: Any) -> Any:
        if mode == "http-error":
            raise HTTPError(url, 503, "unit", {}, io.BytesIO(b'{"status":"unhealthy"}'))
        if mode == "transport":
            raise URLError("unit unavailable")
        return Response(body[mode])

    class Response:
        status = 200

        def __init__(self, payload: bytes) -> None:
            self.payload = payload

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def read(self) -> bytes:
            return self.payload

    monkeypatch.setattr(ha010_probe, "urlopen", urlopen)
    status, payload, error = ha010_probe.fetch("http://unit.invalid/healthz")
    if mode == "transport":
        assert status is payload is None
        assert error is not None
    else:
        assert status == (503 if mode == "http-error" else 200)
        assert error is None
        assert (payload is not None) is (mode in {"json", "http-error"})


@pytest.mark.parametrize("slow", [False, True])
def test_ha010_sampler_cadence_is_bounded_even_when_fetches_overrun_interval(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], slow: bool
) -> None:
    clock = Clock()

    def fetch(url: str) -> tuple[int, dict, None]:
        if slow:
            clock.now += 0.3
        return (
            200,
            {
                "status": "ok",
                "regional_registry": {"ready": True, "secret_drift": False},
            },
            None,
        )

    monkeypatch.setattr(ha010_probe, "fetch", fetch)
    monkeypatch.setattr(
        ha010_probe,
        "time",
        SimpleNamespace(
            monotonic=clock.monotonic, sleep=clock.sleep, time=lambda: 1000 + clock.now
        ),
    )
    assert ha010_probe.main(["unit", "8080", "2", *(["0.1"] if slow else [])]) == 0
    result = json.loads(capsys.readouterr().out)
    assert len(result["samples"]) >= 2
    assert clock.now < 3


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
@pytest.mark.parametrize("position", [2, 3], ids=["duration", "interval"])
def test_ha010_sampler_rejects_nonfinite_durations_before_sampling(
    monkeypatch: pytest.MonkeyPatch, value: str, position: int
) -> None:
    def forbidden(*args: Any) -> Any:
        raise AssertionError("invalid timing must not start a sampler")

    monkeypatch.setattr(ha010_probe, "sample", forbidden)
    arguments = ["unit", "8080", "1", "1"]
    arguments[position] = value
    with pytest.raises(SystemExit, match="finite|positive"):
        ha010_probe.main(arguments)


def test_ha010_standalone_invalid_arguments_do_not_contact_any_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["probe", "0", "1", "1"])
    with pytest.raises(SystemExit, match="positive"):
        runpy.run_path(ha010_probe.__file__, run_name="__main__")
