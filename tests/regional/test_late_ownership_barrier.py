from __future__ import annotations

import multiprocessing
import os
import socket
import time
from datetime import datetime, timedelta, timezone

import pytest

from scripts.e2e.regional import late_ownership_barrier as barrier
from scripts.e2e.regional.late_ownership_contract import RecheckPermit, StopReceipt
from tests.regional._late_ownership_support import evidence, mutation_for


def callback_worker(endpoint, unused, parent_identity, proof, results):
    unused.close()
    channel = barrier.PeerChannel(
        endpoint, parent_identity, expires=time.monotonic() + 10
    )
    rendezvous = barrier.BeforeActionRendezvous(proof.scope, channel)
    stop = proof.stop.model_copy(
        update={"producer": barrier.process_identity(os.getpid())}
    )
    try:
        permit = rendezvous.arrive(stop)
        results.send(("recheck", permit.model_dump_json()))
        try:
            rendezvous.arrive(stop)
        except barrier.BoundaryDenied as exc:
            results.send(("late-refused", str(exc)))
    except (barrier.BoundaryDenied, ValueError) as exc:
        results.send(("denied", str(exc), rendezvous.state))
    finally:
        rendezvous.revoke()
        results.close()


@pytest.fixture
def parked_callback():
    context = multiprocessing.get_context("fork")
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    for endpoint in (parent, child):
        endpoint.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
    receive, send = context.Pipe(duplex=False)
    proof = evidence()
    owner = barrier.process_identity(os.getpid())
    worker = context.Process(
        target=callback_worker, args=(child, parent, owner, proof, send)
    )
    worker.start()
    child.close()
    send.close()
    channel = barrier.PeerChannel(
        parent, barrier.process_identity(worker.pid), expires=time.monotonic() + 10
    )
    controller = barrier.BoundaryController(proof.scope, channel)
    try:
        stop = controller.wait_for_stop()
        yield proof, worker, receive, controller, stop
    finally:
        channel.close()
        worker.join(10)
        if worker.is_alive():
            worker.terminate()
            worker.join(10)
        receive.close()
        assert not worker.is_alive(), "owned callback was not reaped"
        worker.close()


def test_actual_callback_cannot_run_ahead_of_mutation_acknowledgement(parked_callback):
    proof, worker, result, controller, stop = parked_callback
    assert worker.is_alive(), (
        "the callback worker must remain parked before mutation acknowledgement"
    )
    assert not result.poll(0), "callback escaped before the explicit release"
    mutation = mutation_for(proof.scope, stop)
    permit = controller.release(mutation)
    assert result.poll(10), "callback did not consume its causal permit"
    phase, raw = result.recv()
    assert phase == "recheck"
    assert RecheckPermit.model_validate_json(raw) == permit
    assert result.poll(10), "late callback verdict was not reported"
    phase, error = result.recv()
    assert phase == "late-refused"
    assert error == "late or duplicate boundary callback"
    assert permit.instruction == "RECHECK_ONLY"
    assert permit.mutation_sha256 == mutation.digest()
    with pytest.raises(barrier.BoundaryDenied, match="unreleased"):
        controller.release(mutation)
    with pytest.raises(barrier.BoundaryDenied, match="consumed"):
        controller.wait_for_stop()


def test_parent_connection_loss_while_parked_permanently_denies_recheck(
    parked_callback,
):
    _proof, _worker, result, controller, _stop = parked_callback
    controller.channel.close()
    assert result.poll(10), "callback did not observe parent connection loss"
    phase, error, state = result.recv()
    assert phase == "denied"
    assert error == "boundary peer disconnected"
    assert state == "REVOKED"


@pytest.mark.parametrize("field", ["scope_sha256", "boundary_id", "stop_sha256"])
def test_stale_or_wrong_uid_permit_cannot_release_the_owned_callback(
    parked_callback, field
):
    proof, _worker, result, controller, stop = parked_callback
    permit = RecheckPermit(
        scope_sha256=proof.scope.digest(),
        boundary_id=stop.boundary_id,
        stop_sha256=stop.digest(),
        mutation_sha256="d" * 64,
    ).model_copy(update={field: "0" * 64})
    controller.channel.send(permit)
    assert result.poll(10), "callback did not reject the mismatched permit"
    phase, error, state = result.recv()
    assert (phase, error, state) == (
        "denied",
        "recheck permit does not match this STOP",
        "REVOKED",
    )


@pytest.mark.parametrize(
    "field", ["scope_sha256", "stop_sha256", "executor_uid", "producer", "sequence"]
)
def test_controller_will_not_release_stale_mutation_or_replaced_owner(
    parked_callback, field
):
    proof, _worker, _result, controller, stop = parked_callback
    mutation = mutation_for(proof.scope, stop)
    replacement = {
        "scope_sha256": "0" * 64,
        "stop_sha256": "0" * 64,
        "executor_uid": "wrong-executor",
        "producer": proof.witness_starts[0].producer,
        "sequence": stop.sequence,
    }[field]
    with pytest.raises(barrier.BoundaryDenied, match="acknowledge"):
        controller.release(mutation.model_copy(update={field: replacement}))


@pytest.mark.parametrize(
    "field,value",
    [
        ("scope_sha256", "0" * 64),
        ("executor_uid", "wrong"),
        ("participants", ()),
        ("absent_pod_uids", ("pod-original-a",)),
        ("absent_pod_uids", ("pod-original-a", "pod-original-b", "pod-original-b")),
        ("empty_client_node_uids", ("node-uid-a",)),
        ("empty_client_node_uids", ("node-uid-a", "node-uid-b", "node-uid-b")),
    ],
)
def test_stop_requires_exact_physical_absence_for_every_original_participant(
    field, value
):
    proof = evidence()
    with pytest.raises(barrier.BoundaryDenied, match="all participants"):
        barrier.check_stop(proof.scope, proof.stop.model_copy(update={field: value}))


def test_stop_uid_and_owner_are_not_names_only():
    proof = evidence()
    for field in ("uid", "owner_uid"):
        stop = proof.stop.model_copy(
            update={
                "workload": proof.stop.workload.model_copy(update={field: "replaced"})
            }
        )
        with pytest.raises(barrier.BoundaryDenied, match="original owner"):
            barrier.check_stop(proof.scope, stop)
    stop = proof.stop.model_copy(
        update={"participants": (*proof.stop.participants, proof.stop.participants[0])}
    )
    with pytest.raises(barrier.BoundaryDenied, match="all participants"):
        barrier.check_stop(proof.scope, stop)


def test_process_identity_reads_only_uid_starttime_and_boot(tmp_path):
    root = tmp_path / "proc"
    directory = root / "123"
    directory.mkdir(parents=True)
    (root / "sys/kernel/random").mkdir(parents=True)
    (root / "sys/kernel/random/boot_id").write_text("boot-local\n", encoding="ascii")
    (directory / "stat").write_text(
        "123 (space ) name) S " + "0 " * 18 + "4321\n", encoding="ascii"
    )
    identity = barrier.process_identity(123, proc=root)
    assert identity.pid == 123
    assert identity.start_ticks == 4321
    assert identity.uid == os.geteuid()
    assert identity.boot_id == "boot-local"
    (directory / "stat").write_text(
        "123 (gone) Z " + "0 " * 18 + "4321\n", encoding="ascii"
    )
    with pytest.raises(barrier.BoundaryDenied, match="unavailable"):
        barrier.process_identity(123, proc=root)


@pytest.mark.parametrize(
    "contents", ["", "123 (x) R broken", "123 (x) R " + "0 " * 18 + "bad"]
)
def test_unknown_process_identity_is_a_refusal(tmp_path, contents):
    directory = tmp_path / "123"
    directory.mkdir()
    (directory / "stat").write_text(contents, encoding="ascii")
    with pytest.raises(barrier.BoundaryDenied, match="unavailable"):
        barrier.process_identity(123, proc=tmp_path)
    with pytest.raises(barrier.BoundaryDenied, match="unavailable"):
        barrier.process_identity(999, proc=tmp_path)


@pytest.mark.parametrize("deadline", [float("nan"), float("inf"), -1.0])
def test_channel_rejects_unbounded_or_expired_deadline(deadline):
    first, second = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    try:
        with pytest.raises(barrier.BoundaryDenied, match="deadline"):
            barrier.PeerChannel(
                first, barrier.process_identity(os.getpid()), expires=deadline
            )
        assert first.fileno() == -1
    finally:
        first.close()
        second.close()


def test_channel_rejects_stream_socket_before_waiting():
    first, second = socket.socketpair()
    try:
        with pytest.raises(barrier.BoundaryDenied, match="sequenced"):
            barrier.PeerChannel(
                first,
                barrier.process_identity(os.getpid()),
                expires=time.monotonic() + 10,
            )
    finally:
        first.close()
        second.close()


def test_pidfd_rejects_replaced_peer_and_socket_wait_has_a_real_deadline(monkeypatch):
    first, second = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    owner = barrier.process_identity(os.getpid())
    channel = barrier.PeerChannel(first, owner, expires=time.monotonic() + 10)
    try:
        monkeypatch.setattr(
            barrier,
            "process_identity",
            lambda pid: owner.model_copy(update={"start_ticks": owner.start_ticks + 1}),
        )
        with pytest.raises(barrier.BoundaryDenied, match="replaced"):
            channel.check()
        monkeypatch.undo()
        channel.expires = time.monotonic() + 0.01
        with pytest.raises(barrier.BoundaryDenied, match="deadline"):
            channel.receive(StopReceipt)
    finally:
        channel.close()
        second.close()
    with pytest.raises(barrier.BoundaryDenied, match="closed"):
        channel.check()


def peer_that_exits(endpoint, unused, connection):
    unused.close()
    connection.send("ready")
    connection.recv()
    endpoint.close()


def test_real_peer_process_exit_invalidates_the_pidfd_without_a_sleep():
    context = multiprocessing.get_context("fork")
    first, second = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    parent, child = context.Pipe()
    worker = context.Process(target=peer_that_exits, args=(second, first, child))
    worker.start()
    second.close()
    child.close()
    channel = None
    try:
        assert parent.poll(10), (
            "the owned peer must announce readiness before channel construction"
        )
        assert parent.recv() == "ready"
        channel = barrier.PeerChannel(
            first, barrier.process_identity(worker.pid), expires=time.monotonic() + 10
        )
        parent.send("exit")
        worker.join(10)
        assert not worker.is_alive(), "owned peer did not exit"
        with pytest.raises(barrier.BoundaryDenied, match="lost"):
            channel.check()
    finally:
        if channel is not None:
            channel.close()
        first.close()
        parent.close()
        if worker.is_alive():
            worker.terminate()
            worker.join(10)
        worker.close()


@pytest.mark.parametrize(
    "payload", [b"{invalid}", b"x" * (barrier.MAX_PACKET_BYTES + 1)]
)
def test_wire_refuses_malformed_or_truncated_receipt(payload):
    first, second = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    owner = barrier.process_identity(os.getpid())
    with barrier.PeerChannel(first, owner, expires=time.monotonic() + 10) as channel:
        try:
            second.send(payload)
            with pytest.raises(barrier.BoundaryDenied, match="invalid|truncated"):
                channel.receive(StopReceipt)
        finally:
            second.close()


def test_kernel_credentials_reject_a_different_sender_process(parked_callback):
    proof, _worker, result, controller, stop = parked_callback
    # A descriptor leaked to another process is not a capability to act as
    # the original parent; SCM_CREDENTIALS comes from the kernel sender.
    context = multiprocessing.get_context("fork")
    permit = RecheckPermit(
        scope_sha256=proof.scope.digest(),
        boundary_id=stop.boundary_id,
        stop_sha256=stop.digest(),
        mutation_sha256="d" * 64,
    )
    sender = context.Process(
        target=controller.channel.endpoint.send,
        args=(permit.model_dump_json().encode(),),
    )
    sender.start()
    sender.join(10)
    assert not sender.is_alive(), (
        "the untrusted sender process must exit before cleanup"
    )
    sender.close()
    assert result.poll(10), (
        "the callback must report rejection of the unexpected sender"
    )
    phase, error, state = result.recv()
    assert (phase, error, state) == (
        "denied",
        "boundary message belongs to another process",
        "REVOKED",
    )


def test_revoked_and_out_of_window_callback_never_gets_to_send(monkeypatch):
    proof = evidence()
    first, second = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    channel = barrier.PeerChannel(
        first, barrier.process_identity(os.getpid()), expires=time.monotonic() + 10
    )
    binding = proof.scope.model_copy(
        update={"maintenance_end": datetime.now(timezone.utc) - timedelta(seconds=1)}
    )
    rendezvous = barrier.BeforeActionRendezvous(binding, channel)
    try:
        with pytest.raises(barrier.BoundaryDenied, match="not active"):
            rendezvous.check()
        with pytest.raises(ValueError, match="outside"):
            rendezvous.arrive(proof.stop)
        assert rendezvous.state == "REVOKED"
        assert second.recv(1) == b"", "expired callback must close, not emit STOP"
    finally:
        channel.close()
        second.close()
