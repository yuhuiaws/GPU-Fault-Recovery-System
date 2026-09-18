from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts.e2e.regional import late_ownership_live as live
from scripts.e2e.regional.late_ownership_barrier import BoundaryDenied
from scripts.e2e.regional.run_late_ownership_acceptance import run_case
from tests.regional._late_ownership_support import evidence
from tests.regional.test_late_ownership_stream import Socket


class LocalBoundaryIO(live.LiveBoundaryIO):
    evidence_mode = "LOCAL_TEST"


def peer_factories(proof, run, channels, witnesses, calls, get_io):
    scope = proof.scope

    class GpuSocket(Socket):
        sequence = 0
        quiesced = False
        terminal = False
        callback = None
        fail_at = None

        def write_stdin(self, text):
            super().write_stdin(text)
            data = json.loads(text)
            kind = data.get("kind", "initialize")
            calls.append("gpu-" + kind)
            if kind == self.fail_at:
                raise OSError("GPU stream disconnected")
            if kind == "continue-boundary":
                self.quiesced = True
            if kind == "quiesce":
                self.quiesced = False
            if kind == "abort":
                self.open = False
                return
            responses = {
                "initialize": (
                    "ready",
                    {"producer": proof.stop.producer.model_dump(mode="json")},
                ),
                "calibrate": ("calibrated", {"nodes": list(run.settings.nodes)}),
                "begin": (
                    "contained",
                    {"stop_idempotency_key": proof.stop.stop_command_id},
                ),
                "continue-boundary": ("stop", proof.stop.model_dump(mode="json")),
                "observe-mutation": (
                    "mutation",
                    proof.mutation.model_dump(mode="json"),
                ),
                "recheck": ("decision", proof.decision.model_dump(mode="json")),
                "quiesce": ("services-restored", {"node_commands_terminal": True}),
                "restore-scheduling": (
                    "actions-drained",
                    {"workflow": get_io().workflow.model_dump(mode="json")},
                ),
                "confirm-terminal": (
                    "quiescence",
                    proof.quiescence.model_dump(mode="json"),
                ),
                "revoke": ("revoked", {"gate_revoked": True}),
                "finish": ("finished", {"gate_revoked": True}),
            }
            response, payload = responses[kind]
            if self.callback:
                payload = self.callback(kind, payload)
            self.sequence += 1
            self.chunks.append(
                (
                    json.dumps(
                        {
                            "kind": response,
                            "payload": payload,
                            "sequence": self.sequence,
                            "scope_sha256": scope.digest(),
                        }
                    )
                    + "\n",
                    "",
                )
            )

    class Witness:
        bad_phase = False
        client_defect = None
        cleanup_error = False
        peer_replaced = False

        def __init__(self, regional, settings, binding, node, directory):
            self.node = node
            self.index = binding.nodes.index(node)
            self.created = False
            self.finished = False
            witnesses[node.name] = self

        def create(self):
            self.created = True
            calls.append("witness-create:" + self.node.name)

        def request(self, action, payload):
            calls.append(f"witness-{action}:{self.node.name}")
            assert self.created, (
                "witness requests require acknowledged owned Pod creation"
            )
            assert not (channels[0].quiesced and self.index == 0), (
                "live witness RPC depended on a stopped kubelet"
            )
            if action == "status":
                return {"phase": "UNKNOWN" if self.bad_phase else "ATTACHED"}
            if action == "calibrated":
                return {
                    "receipt": proof.witness_starts[self.index].model_dump(mode="json")
                }
            if action == "clients":
                data = {
                    "scope_sha256": scope.digest(),
                    "node": self.node.model_dump(mode="json"),
                    "pod_uids": payload["pod_uids"],
                    "source": "nvidia-compute-apps+proc-cgroup/v1",
                    "observations": [{"physical_pid": 321}]
                    if payload["pod_uids"] == [proof.mutation.live_sibling_pod_uid]
                    else [],
                }
                if self.client_defect == "stale":
                    data["scope_sha256"] = "0" * 64
                elif self.client_defect == "source-active":
                    data["observations"] = [{"physical_pid": 321}]
                elif self.client_defect == "late-absent":
                    data["observations"] = []
                return {"observation": data}
            if action == "finish":
                self.finished = True
                return {
                    "receipt": proof.witness_ends[self.index].model_dump(mode="json")
                }
            assert action == "exit" and self.finished
            return {"exited": not self.peer_replaced}

        def cleanup(self, host):
            calls.append("witness-cleanup:" + self.node.name)
            assert host is run.probes[self.node.name]
            if self.cleanup_error:
                raise BoundaryDenied("witness cleanup failed")

    return GpuSocket, Witness


@pytest.fixture
def assembly(tmp_path, monkeypatch):
    def create(scenario="ownership-drift"):
        proof = evidence(scenario)
        scope = proof.scope
        calls, channels, witnesses = [], [], {}
        identity = {
            "name": "local-executor",
            "uid": scope.executor_uid,
            "container_id": "container",
        }
        baselines = {
            node.name: {
                "boot_id": node.boot_id,
                "gpu_inventory": [{"uuid": f"GPU-{index}", "pci_bdf": f"bdf-{index}"}],
                "gpu_fault_timers": [],
                "quiesce_states": [],
                "services": {"kubelet": {"ActiveState": "active"}},
            }
            for index, node in enumerate(scope.nodes)
        }

        class Regional:
            bad_identity = False
            terminal_failure = False

            def cpu_python(self, program, payload, **kwargs):
                request = json.loads(payload)
                calls.append("cpu-" + request["action"])
                assert "def hold_workflow(" in program
                if request["action"] == "hold":
                    return {"workflow": request["workflow"]}
                if self.terminal_failure:
                    raise OSError("CPU terminal ACK lost")
                return {
                    "workflow_id": scope.workflow_id,
                    "fencing_token": scope.fencing_token,
                    "status": "SUPERSEDED",
                }

            def kubectl(self, side, *args, **kwargs):
                assert side == "gpu"
                if args[0] == "wait":
                    calls.append("node-ready:" + args[2])
                    return ""
                assert args[:2] == ("get", "namespace")
                return json.dumps(
                    {
                        "metadata": {
                            "uid": "replaced"
                            if self.bad_identity
                            else scope.namespace_uid
                        }
                    }
                )

            def node_snapshot(self, name):
                return {
                    "uid": next(node.uid for node in scope.nodes if node.name == name)
                }

            def verify_runtime_identity(self, *args, **kwargs):
                calls.append("runtime-pin")

        regional = Regional()

        class Host:
            residual = False
            changed = False

            def __init__(self, node):
                self.node = node
                self.pod = "host-" + node
                self.settings = SimpleNamespace(node=node)

            def _check_target(self):
                calls.append("host-check:" + self.node)

            def execute(self, *args, **kwargs):
                calls.append("host-snapshot:" + self.node)
                value = deepcopy(baselines[self.node])
                if self.changed:
                    value["boot_id"] = "replaced"
                return value

            def cleanup(self):
                calls.append("host-cleanup:" + self.node)
                return {"pod": self.residual}

        def deleted():
            calls.append("workload-delete")

        run = SimpleNamespace(
            settings=SimpleNamespace(
                nodes=tuple(node.name for node in scope.nodes),
                job_id="owned-job",
                regional=SimpleNamespace(gpu_kubeconfig="/dev/null"),
            ),
            regional=regional,
            run_id=scope.run_id,
            case_dir=tmp_path / scenario,
            bdf={node.name: f"bdf-{index}" for index, node in enumerate(scope.nodes)},
            baselines=baselines,
            preflight={"runtime_identity": "pinned"},
            probes={node.name: Host(node.name) for node in scope.nodes},
            workload=SimpleNamespace(delete=deleted),
            workload_submitted=True,
            prewarm=SimpleNamespace(
                cleanup=lambda: calls.append("prewarm-cleanup") or {}
            ),
        )
        source = {
            "spec": {
                "pytorchReplicaSpecs": {
                    "Worker": {
                        "template": {
                            "spec": {"containers": [{"image": "approved-image"}]}
                        }
                    }
                }
            }
        }

        class Mutation:
            def __init__(self, *args):
                self.journal_path = None

            def acknowledge_stop(self, stop):
                assert stop == proof.stop
                calls.append("acknowledge-stop")

            def change_owner(self):
                calls.append("mutate-owner")

            def late_sibling(self):
                calls.append("late-cuda-pod")
                return proof.mutation.live_sibling_pod_uid

            def cleanup(self):
                calls.append("mutation-cleanup")

        GpuSocket, Witness = peer_factories(
            proof, run, channels, witnesses, calls, lambda: io
        )

        def opened(**kwargs):
            assert kwargs["pod"] == identity["name"]
            channel = GpuSocket()
            channels.append(channel)
            return channel

        monkeypatch.setattr(live, "NodeWitness", Witness)
        monkeypatch.setattr(live, "OwnedMutation", Mutation)
        monkeypatch.setattr(live, "open_executor_stream", opened)
        monkeypatch.setattr(live, "executor_identity", lambda regional: identity)
        monkeypatch.setattr(live, "probe_program", lambda role: ("pass", "a" * 64))
        io = LocalBoundaryIO(run, scope, source, identity)
        return SimpleNamespace(
            io=io,
            scope=scope,
            proof=proof,
            calls=calls,
            witnesses=witnesses,
            run=run,
            channels=channels,
            witness_type=Witness,
        )

    return create


@pytest.mark.parametrize(
    "scenario", ["unchanged-owner", "ownership-drift", "late-sibling"]
)
def test_live_assembly_drives_actual_channel_order_and_never_calls_stopped_kubelet(
    assembly, scenario
):
    owned = assembly(scenario)
    result = run_case(owned.scope, owned.io)
    assert result.verdict == "PASS", result.errors
    assert (
        result.evidence_mode == "LOCAL_TEST"
        and not result.summary()["promotes_ordinary_case"]
    )
    calls = owned.calls
    assert (
        calls.index("gpu-begin")
        < calls.index("witness-clients:node-a")
        < calls.index("gpu-continue-boundary")
    )
    assert (
        calls.index("gpu-continue-boundary")
        < calls.index("acknowledge-stop")
        < calls.index("gpu-observe-mutation")
        < calls.index("gpu-recheck")
    )
    assert calls.index("host-snapshot:node-a") < calls.index("gpu-restore-scheduling")
    assert calls.index("cpu-finish") < calls.index("witness-finish:node-a")
    assert (
        calls.index("gpu-finish")
        < calls.index("witness-cleanup:node-a")
        < calls.index("host-cleanup:node-a")
    )
    assert owned.channels[0].close_count >= 1 and not owned.channels[0].is_open()
    assert owned.run.workload_submitted is False


@pytest.mark.parametrize("defect", ["stale", "source-active", "late-absent"])
def test_physical_client_absence_or_presence_cannot_come_from_unbound_counter(
    assembly, defect
):
    owned = assembly("late-sibling")
    owned.witness_type.client_defect = defect
    result = run_case(owned.scope, owned.io)
    assert result.verdict == "FAIL"
    assert "gpu-recheck" not in owned.calls
    assert "workload-delete" not in owned.calls, (
        "cleanup cannot race a parked/unconfirmed physical command"
    )
    assert result.cleanup is None


@pytest.mark.parametrize(
    "defect", ["terminal", "services", "witness", "prewarm", "host", "exit"]
)
def test_assembly_cleanup_failures_never_produce_pass(assembly, defect):
    owned = assembly()
    if defect == "terminal":
        owned.run.regional.terminal_failure = True
    elif defect == "services":
        owned.run.probes["node-a"].changed = True
    elif defect == "witness":
        owned.witness_type.cleanup_error = True
    elif defect == "prewarm":
        owned.run.prewarm.cleanup = lambda: {"residual": True}
    elif defect == "host":
        owned.run.probes["node-a"].residual = True
    else:
        owned.witness_type.peer_replaced = True
    result = run_case(owned.scope, owned.io)
    assert result.verdict == "FAIL" and result.cleanup is None
    assert not owned.channels[0].is_open(), (
        "failed physical cleanup must close its Executor channel"
    )
    if defect in {"terminal", "services"}:
        assert "workload-delete" not in owned.calls


def test_replaced_namespace_blocks_all_mutation_at_preflight(assembly):
    owned = assembly()
    owned.run.regional.bad_identity = True
    result = run_case(owned.scope, owned.io)
    assert result.verdict == "FAIL" and owned.channels == []
    assert owned.calls == []


def test_partial_witness_arm_cleans_only_setup_resources(assembly):
    owned = assembly()
    owned.witness_type.bad_phase = True
    result = run_case(owned.scope, owned.io)
    assert result.verdict == "FAIL" and result.cleanup is not None
    assert "gpu-begin" not in owned.calls
    assert "cpu-finish" in owned.calls and "witness-cleanup:node-a" in owned.calls
    assert not owned.channels[0].is_open(), (
        "partial witness setup must release its owned channel"
    )


def test_parent_connection_loss_at_native_checkpoint_does_not_authorize_or_clean(
    assembly,
):
    owned = assembly()
    starts = owned.io.arm_witnesses(owned.scope)
    owned.channels[0].fail_at = "continue-boundary"
    with pytest.raises(OSError):
        owned.io.stop_at_boundary(owned.scope, starts)
    with pytest.raises(BoundaryDenied):
        owned.io.revoke(owned.scope)
    with pytest.raises(BoundaryDenied):
        owned.io.cleanup(owned.scope)
    assert "gpu-recheck" not in owned.calls and "workload-delete" not in owned.calls
    assert not owned.channels[0].is_open(), (
        "parent loss must not leave a usable recheck channel"
    )


def test_scope_mismatch_or_unarmed_executor_cannot_be_used(assembly):
    owned = assembly()
    with pytest.raises(BoundaryDenied, match="scope"):
        owned.io.preflight(owned.scope.model_copy(update={"executor_uid": "other"}))
    with pytest.raises(BoundaryDenied, match="armed"):
        owned.io._gpu()
    owned.io.revoke(owned.scope)
    assert owned.io.revoked, "revocation must also close an unarmed experiment"


@pytest.mark.parametrize("defect", ["contained", "stop", "terminal-actions", "revoke"])
def test_missing_or_stale_product_acknowledgements_fail_the_assembly(assembly, defect):
    owned = assembly()
    starts = owned.io.arm_witnesses(owned.scope)

    def callback(kind, payload):
        payload = deepcopy(payload)
        if defect == "contained" and kind == "begin":
            payload.clear()
        if defect == "stop" and kind == "continue-boundary":
            payload["stop_command_id"] = "stale-stop"
        if defect == "terminal-actions" and kind == "quiesce":
            payload["node_commands_terminal"] = False
        if defect == "revoke" and kind == "revoke":
            payload["gate_revoked"] = False
        return payload

    owned.channels[0].callback = callback
    try:
        if defect in {"contained", "stop"}:
            with pytest.raises(BoundaryDenied):
                owned.io.stop_at_boundary(owned.scope, starts)
        elif defect == "terminal-actions":
            with pytest.raises(BoundaryDenied):
                owned.io.quiesce(owned.scope, owned.proof.decision)
        else:
            owned.io.node_actions_started = True
            owned.io.quiet = True
            with pytest.raises(BoundaryDenied, match="acknowledgement"):
                owned.io.revoke(owned.scope)
    finally:
        owned.channels[0].close()


@pytest.mark.parametrize(
    "defect", ["scope", "producer", "witness", "participants", "acknowledgement"]
)
def test_stop_transition_is_acknowledged_only_after_native_receipt_validation(
    assembly, monkeypatch, defect
):
    owned = assembly()
    starts = owned.io.arm_witnesses(owned.scope)

    def callback(kind, payload):
        value = deepcopy(payload)
        if kind != "continue-boundary":
            return value
        if defect == "scope":
            value["scope_sha256"] = "f" * 64
        elif defect == "producer":
            value["producer"]["pid"] += 1
        elif defect == "witness":
            value["witness_start_sha256"][0] = "f" * 64
        elif defect == "participants":
            value["participants"] = []
        return value

    def failed_ack(stop):
        owned.calls.append("acknowledge-stop-refused")
        raise BoundaryDenied(
            "STOP transition differs from the original declared source"
        )

    owned.channels[0].callback = callback
    if defect == "acknowledgement":
        monkeypatch.setattr(owned.io.mutation, "acknowledge_stop", failed_ack)
    try:
        with pytest.raises(BoundaryDenied):
            owned.io.stop_at_boundary(owned.scope, starts)
        assert "acknowledge-stop" not in owned.calls
        assert ("acknowledge-stop-refused" in owned.calls) == (
            defect == "acknowledgement"
        )
        assert not (owned.io.directory / "queued-stop.json").exists(), (
            "an invalid or unacknowledged STOP cannot become the accepted boundary journal"
        )
        assert (
            "gpu-observe-mutation" not in owned.calls
            and "gpu-recheck" not in owned.calls
        )
    finally:
        owned.channels[0].close()
