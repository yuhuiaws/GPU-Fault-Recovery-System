from __future__ import annotations

import json
import os
import time
from threading import Event, Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from gpu_fault.store import InMemoryStore
from scripts.e2e.regional import ha011_contracts as contracts
from scripts.e2e.regional.probes import ha011_probe as probe
from scripts.e2e.regional.probes import ha011_workers as workers
from scripts.e2e.regional.probes.ha011_processes import WorkerProcess
from tests.regional._cov95_ha011_runtime import Capture, MemoryWorkers, owned_idle_child
from tests.regional._cov95_ha011_runtime import isolated_fixture as isolated_fixture
from tests.regional._cov95_ha011_support import INTENT, POD_UID, RUN_ID, Clock
from tests.regional._cov95_ha011_support import (
    blocked_external_transports as blocked_external_transports,
)


def test_arm_is_atomic_one_shot_and_uid_bound(isolated) -> None:
    receipt = probe.arm(POD_UID, RUN_ID, INTENT)
    assert receipt == probe.wait_for_arm(), (
        "the barrier must preserve the acknowledged immutable arm identity"
    )
    with pytest.raises(FileExistsError):
        probe.arm(POD_UID, RUN_ID, INTENT)
    with pytest.raises(contracts.ProofError):
        probe.arm("foreign-pod", RUN_ID, INTENT)
    assert len(list(isolated.iterdir())) == 2, (
        "temporary arm files must be removed after success and collision"
    )


def test_no_arm_means_no_database_or_worker_activity(
    isolated, monkeypatch, capsys
) -> None:
    calls = []
    monkeypatch.setattr("psycopg.connect", lambda *_a, **_k: calls.append("SQL"))
    monkeypatch.setattr(probe, "PostgresStore", lambda *_a, **_k: calls.append("Store"))
    monkeypatch.setattr(probe, "time", Clock(step=61))
    assert probe.main() == 1, (
        "an unarmed Pod must stop without starting the acceptance workload"
    )
    assert calls == [], "database transports must not be called before arm"
    assert json.loads(capsys.readouterr().out)["error_type"] == "ProofError", (
        "unarmed failure must be redacted and explicit"
    )


@pytest.mark.parametrize(
    "change",
    [
        "pod",
        "run",
        "namespace",
        "boundary",
        "service-account",
        "store-env",
        "pg-env",
        "aws-env",
        "kube-env",
        "proxy",
        "proxy-uppercase",
    ],
)
def test_probe_refuses_inherited_or_foreign_boundaries(isolated, change: str) -> None:
    if change == "pod":
        os.environ["POD_UID"] = ""
    elif change == "run":
        os.environ["HA011_ISOLATION_ID"] = "unbound"
    elif change == "namespace":
        os.environ["POD_NAMESPACE"] = "business"
    elif change == "boundary":
        os.environ["HA011_BOUNDARY"] = "local-fake"
    elif change == "service-account":
        probe.SERVICE_ACCOUNT.write_text("public-fake-token", encoding="utf-8")
    elif change == "store-env":
        os.environ["GPU_FAULT_STORE_URL_FILE"] = "/production"
    elif change == "pg-env":
        os.environ["PGHOST"] = "production.invalid"
    elif change == "aws-env":
        os.environ["AWS_PROFILE"] = "foreign-profile"
    elif change == "kube-env":
        os.environ["KUBECONFIG"] = "/foreign/kubeconfig"
    elif change == "proxy-uppercase":
        os.environ["HTTPS_PROXY"] = "http://proxy.invalid"
    else:
        os.environ["https_proxy"] = "http://proxy.invalid"
    with pytest.raises(contracts.ProofError):
        probe.isolated_identity()


@pytest.mark.parametrize(
    "name,value",
    [
        ("KUBERNETES_SERVICE_HOST", "10.96.0.1"),
        ("KUBERNETES_SERVICE_PORT", "443"),
        ("KUBERNETES_SERVICE_PORT_HTTPS", "443"),
        ("KUBERNETES_PORT", "tcp://10.96.0.1:443"),
        ("KUBERNETES_PORT_443_TCP", "tcp://10.96.0.1:443"),
        ("KUBERNETES_PORT_443_TCP_ADDR", "10.96.0.1"),
        ("KUBERNETES_PORT_443_TCP_PORT", "443"),
        ("KUBERNETES_PORT_443_TCP_PROTO", "tcp"),
    ],
)
def test_kubelet_service_discovery_is_not_inherited_authority(
    isolated, name: str, value: str
) -> None:
    os.environ[name] = value
    assert probe.isolated_identity() == (RUN_ID, POD_UID)


@pytest.mark.parametrize(
    "name",
    [
        "KUBERNETES_SERVICE_ACCOUNT_TOKEN",
        "KUBERNETES_SERVICE_HOST_OVERRIDE",
        "KUBERNETES_PORT_443_TCP_TOKEN",
        "KUBE_CONFIG",
    ],
)
def test_service_discovery_exception_does_not_allow_other_kubernetes_inputs(
    isolated, name: str
) -> None:
    os.environ[name] = "unapproved-value"
    with pytest.raises(contracts.ProofError):
        probe.isolated_identity()


def test_prearm_failure_names_the_stage_without_echoing_configuration(
    isolated, capsys
) -> None:
    os.environ["GPU_FAULT_STORE_URL"] = "sensitive-input-not-for-output"
    assert probe.main() == 1
    output = capsys.readouterr().out
    assert json.loads(output) == {
        "case_id": contracts.CASE_ID,
        "verdict": "FAIL",
        "stage": "identity",
        "error_type": "ProofError",
    }
    assert "sensitive-input" not in output


@pytest.mark.parametrize(
    "change", ["not-armed", "pod", "run", "intent", "extra", "list", "symlink", "large"]
)
def test_barrier_requires_bounded_owned_receipt(isolated, change: str) -> None:
    receipt = {
        "armed": True,
        "pod_uid": POD_UID,
        "isolation_id": RUN_ID,
        "intent_sha256": INTENT,
    }
    if change == "not-armed":
        receipt["armed"] = 1
    elif change == "pod":
        receipt["pod_uid"] = "old-pod"
    elif change == "run":
        receipt["isolation_id"] = "b" * 32
    elif change == "intent":
        receipt["intent_sha256"] = None
    elif change == "extra":
        receipt["other"] = "unapproved"
    elif change == "list":
        receipt = []
    content = json.dumps(receipt)
    if change == "large":
        content += " " * 5000
    if change == "symlink":
        target = isolated / "other"
        target.write_text(content, encoding="utf-8")
        probe.ARM_FILE.symlink_to(target)
    else:
        probe.ARM_FILE.write_text(content, encoding="utf-8")
    with pytest.raises(contracts.ProofError):
        probe.wait_for_arm()


@pytest.mark.parametrize("role", contracts.ROLES)
def test_orchestration_fences_old_completion_while_replacement_is_live(
    role: str,
) -> None:
    store = InMemoryStore()
    factory = MemoryWorkers(store)
    result = probe.exercise_role(store, role, RUN_ID, factory=factory)
    assert (
        result["replacement_live_before_late"] and result["replacement_live_after_late"]
    ), "the stale callback must be attempted before current work has drained"
    assert (
        result["late_completion_refused"] is True and result["completed_count"] == 4
    ), "the owned backlog must drain through current-owner completions only"
    assert all(not worker.alive() for worker in factory.workers), (
        "all fake owned lifetimes must close"
    )


@pytest.mark.parametrize("role", contracts.ROLES)
@pytest.mark.parametrize(
    "failure", ["not-busy", "duplicate-completion", "duplicate-claim"]
)
def test_orchestration_never_passes_missing_live_owner_or_duplicate_drain(
    role: str, failure: str
) -> None:
    store = InMemoryStore()
    factory = MemoryWorkers(store, fail=failure)
    with pytest.raises(contracts.ProofError):
        probe.exercise_role(store, role, RUN_ID, factory=factory)
    assert all(not worker.alive() for worker in factory.workers), (
        "failed proof must stop all owned workers"
    )


def test_unfenced_spool_transport_is_rejected_before_replacement_finishes() -> None:
    class UnfencedStore(InMemoryStore):
        def complete_telemetry_spool(self, items):
            if items[0].request_id.endswith("-0") and items[0].attempts == 1:
                return 1
            return super().complete_telemetry_spool(items)

    store = UnfencedStore()
    factory = MemoryWorkers(store)
    with pytest.raises(contracts.ProofError, match="not fenced"):
        probe.exercise_role(store, "spool", RUN_ID, factory=factory)
    assert not factory.workers[-1].release.is_set(), (
        "the unsafe callback must be discovered while replacement is still blocked"
    )


@pytest.mark.parametrize("role", contracts.ROLES)
def test_actual_public_worker_loop_replays_only_owned_local_work(
    isolated, role: str
) -> None:
    store = InMemoryStore()
    request = probe.sample(RUN_ID, role, 0)
    probe.admit(store, role, [request])
    capture, stop, release = Capture(), Event(), Event()
    failures = []

    def consume():
        try:
            workers.run_worker(store, role, request.request_id, capture, stop, release)
        except BaseException as exc:
            failures.append(type(exc).__name__)

    thread = Thread(target=consume)
    thread.start()
    try:
        capture.wait("ready")
        claim = capture.wait("claim")
        busy = capture.wait("busy")
        assert busy["cpu_seconds"] >= contracts.MIN_CPU_SECONDS, (
            "the real owned replay must consume bounded CPU time"
        )
        assert probe.live_claim(store, role, claim["item"]), (
            "real worker must still own live work while blocked"
        )
        release.set()
        capture.wait("complete")
        assert probe.depth(store, role) == 0, (
            "the production worker loop must commit the owned sample"
        )
    finally:
        stop.set()
        release.set()
        thread.join(timeout=8)
    assert not thread.is_alive() and not failures, (
        "the local worker and its listener must stop cleanly"
    )


def test_replay_endpoint_refuses_foreign_token_without_work() -> None:
    capture = Capture()
    replay = workers.Replay(RUN_ID + "-processor-0", Event(), workers.Emitter(capture))
    with workers.http_server(replay, "public-fake-replay-secret") as server:
        serving = Thread(target=server.serve_forever)
        serving.start()
        try:
            request = Request(
                f"http://127.0.0.1:{server.server_port}/owned",
                data=b"{}",
                method="POST",
            )
            with pytest.raises(HTTPError) as failure:
                urlopen(request, timeout=2)
            assert failure.value.code == 403, (
                "the owned loopback replay endpoint must still require its own token"
            )
            assert capture.events == [], (
                "unauthorized replay must never reach the busy handler"
            )
        finally:
            server.shutdown()
            serving.join(timeout=3)


def test_replay_rejects_foreign_identity_and_bounds_a_stuck_handler(
    monkeypatch,
) -> None:
    replay = workers.Replay(RUN_ID + "-spool-0", Event(), workers.Emitter(Capture()))
    with pytest.raises(contracts.ProofError, match="outside"):
        replay.execute("foreign-0")
    clock = Clock(step=61)
    clock.process_time = lambda: 0.1
    monkeypatch.setattr(workers, "time", clock)
    with pytest.raises(contracts.ProofError, match="expired"):
        replay.execute(RUN_ID + "-spool-0")


@pytest.mark.parametrize("crash", [False, True])
def test_owned_spawned_child_can_stop_or_crash_without_targeting_external_pid(
    crash: bool,
) -> None:
    process = WorkerProcess(
        owned_idle_child, role="processor", request_id="owned-local"
    )
    try:
        ready = process.wait("ready", time.monotonic() + 10)
        assert ready["pid"] == process.process.pid, (
            "observations must match the child actually created by the supervisor"
        )
        if crash:
            assert process.crash() == -9, (
                "the owned child crash must be confirmed by wait status"
            )
        else:
            assert process.finish(), "normal owned child shutdown must drain"
        assert not process.alive(), "no spawned process may remain after the test"
    finally:
        process.close()
