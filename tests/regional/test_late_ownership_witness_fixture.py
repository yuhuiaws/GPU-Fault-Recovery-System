from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts.e2e.regional import late_ownership_witness_fixture as fixture
from scripts.e2e.regional.host_probe_fixture import HostProbeSettings
from scripts.e2e.regional.late_ownership_barrier import BoundaryDenied
from tests.regional._late_ownership_support import evidence


@pytest.fixture
def witness(tmp_path, monkeypatch):
    proof = evidence()
    binding = proof.scope
    config = tmp_path / "owned-config"
    config.write_text("{}")
    script = tmp_path / "original.py"
    script.write_text("pass")
    settings = HostProbeSettings(
        kubeconfig=config,
        context=binding.context,
        namespace=binding.workload.namespace,
        node=binding.nodes[0].name,
        image="approved-image",
        case_id=binding.case_id,
        run_id=binding.run_id,
        probe_script=script,
        state_directory=tmp_path / "original",
    )
    calls = []
    producer = proof.witness_starts[0].producer.model_dump(mode="json")
    reply = {
        "scope_sha256": binding.digest(),
        "peer": producer,
        "producer": producer,
        "phase": "ATTACHED",
    }

    def kubectl(*args, **kwargs):
        calls.append((args, kwargs))
        return json.dumps(reply)

    monkeypatch.setattr(fixture, "probe_program", lambda role: ("pass\n", "a" * 64))
    owned = fixture.NodeWitness(
        SimpleNamespace(kubectl=kubectl),
        settings,
        binding,
        binding.nodes[0],
        tmp_path / "new",
    )
    checks = []
    monkeypatch.setattr(owned.probe, "_check_target", lambda: checks.append("uid"))
    return SimpleNamespace(
        witness=owned,
        calls=calls,
        reply=reply,
        checks=checks,
        scope=binding,
        proof=proof,
        settings=settings,
        directory=tmp_path / "new",
    )


def test_witness_pod_main_not_exec_session_survives_kubelet_quiesce(
    witness, monkeypatch
):
    owned = witness.witness
    manifests = owned.probe.manifests()
    config, pod = manifests
    spec = pod["spec"]
    assert spec["hostPID"] is True and spec["hostNetwork"] is True
    assert spec["nodeName"] == witness.scope.nodes[0].name
    assert spec["automountServiceAccountToken"] is False
    assert spec["restartPolicy"] == "Never"
    assert spec["containers"][0]["securityContext"]["privileged"] is True
    container = spec["containers"][0]
    assert container["args"][0].startswith("exec chroot /host "), (
        "the witness must run as the Pod's host-root main process"
    )
    assert "-I -u -c" in container["args"][0]
    assert container["env"] == [
        {
            "name": "LATE_OWNERSHIP_WITNESS_POD_UID",
            "valueFrom": {"fieldRef": {"fieldPath": "metadata.uid"}},
        }
    ]
    assert container["readinessProbe"]["exec"]["command"][1] == "-S"
    assert config["data"]["probe.py"] == owned.probe.settings.probe_script.read_text()
    created = []
    monkeypatch.setattr(owned.probe, "create", lambda: created.append(True))
    owned.create()
    assert created == [True]
    assert owned.probe.settings.probe_script.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        fixture.NodeWitness(
            owned.regional,
            witness.settings,
            witness.scope,
            owned.node,
            witness.directory,
        )


def test_witness_rpc_pins_source_uid_and_same_process_incarnation(witness):
    owned = witness.witness
    actual = owned.request("status", {})
    assert actual["peer"] == witness.reply["peer"]
    assert witness.checks == ["uid", "uid"]
    args, kwargs = witness.calls[0]
    assert args[:4] == ("gpu", "exec", "-i", owned.probe.pod)
    assert args[-1] == fixture.stdin_loader("pass\n")
    assert kwargs["input_text"].startswith("pass\n"), (
        "witness RPC must send the pinned source before its request"
    )
    request = json.loads(kwargs["input_text"][len("pass\n") :])
    assert request["scope"] == witness.scope.model_dump(mode="json")
    assert request["node"] == owned.node.name and request["action"] == "status"
    assert request["payload"] == {}
    witness.reply["receipt"] = witness.proof.witness_starts[0].model_dump(mode="json")
    assert owned.request("calibrated", {})["receipt"] == witness.reply["receipt"]


@pytest.mark.parametrize(
    "defect",
    [
        "source",
        "scope",
        "peer",
        "replaced",
        "boot",
        "producer",
        "receipt",
        "observation",
    ],
)
def test_witness_drift_refuses_before_any_receipt_can_be_accepted(
    witness, monkeypatch, defect
):
    owned = witness.witness
    if defect == "source":
        monkeypatch.setattr(
            fixture, "probe_program", lambda role: ("changed", "b" * 64)
        )
    elif defect == "scope":
        witness.reply["scope_sha256"] = "0" * 64
    elif defect == "peer":
        witness.reply["peer"] = None
    elif defect == "replaced":
        owned.peer = witness.reply["peer"] | {"pid": 999}
    elif defect == "boot":
        witness.reply["peer"] = witness.reply["peer"] | {"boot_id": "new"}
    elif defect == "producer":
        witness.reply["producer"] = witness.reply["producer"] | {"pid": 999}
    else:
        witness.reply[defect] = {"producer": witness.reply["producer"] | {"pid": 999}}
    with pytest.raises(BoundaryDenied):
        owned.request("status", {})
    if defect == "source":
        assert witness.calls == [], "changed source must not be dispatched"


def install_state(owned, *, uid="owned-pod", started=True):
    owned.probe.state_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    owned.probe.state_path.write_text(
        json.dumps({"resources": {"pod": {"uid": uid, "create_started": started}}})
    )


def test_cleanup_removes_witness_before_using_original_owned_host_channel(
    witness, monkeypatch
):
    owned = witness.witness
    install_state(owned)
    calls = []
    monkeypatch.setattr(
        owned.probe, "cleanup", lambda: calls.append("pod-cleaned") or {}
    )
    transport = SimpleNamespace(
        pod="original-host", _check_target=lambda: calls.append("transport-uid")
    )
    witness.reply.clear()
    witness.reply.update(scope_sha256=witness.scope.digest(), mailbox_absent=True)
    owned.cleanup(transport)
    assert calls == ["pod-cleaned", "transport-uid", "transport-uid"]
    args, kwargs = witness.calls[0]
    assert args[3] == "original-host"
    request = json.loads(kwargs["input_text"][len("pass\n") :])
    assert request["action"] == "cleanup-mailbox"
    assert request["payload"] == {"pod_uid": "owned-pod"}


@pytest.mark.parametrize(
    "defect",
    ["residual", "wrong-scope", "mailbox", "unknown-uid", "never-started", "no-state"],
)
def test_cleanup_refuses_unknown_ownership_and_unconfirmed_absence(
    witness, monkeypatch, defect
):
    owned = witness.witness
    install_state(
        owned,
        uid=None if defect in {"unknown-uid", "never-started"} else "owned-pod",
        started=defect != "never-started",
    )
    if defect == "no-state":
        owned.probe.state_path.unlink()
    monkeypatch.setattr(owned.probe, "cleanup", lambda: {"pod": defect == "residual"})
    transport = SimpleNamespace(pod="original-host", _check_target=lambda: None)
    witness.reply.clear()
    witness.reply.update(
        scope_sha256="0" * 64 if defect == "wrong-scope" else witness.scope.digest(),
        mailbox_absent=defect != "mailbox",
    )
    if defect in {"never-started", "no-state"}:
        owned.cleanup(transport)
    else:
        with pytest.raises(BoundaryDenied):
            owned.cleanup(transport)
    if defect in {"residual", "unknown-uid", "never-started", "no-state"}:
        assert witness.calls == []


def test_unacknowledged_resource_without_state_cannot_be_cleaned_by_name(
    witness, monkeypatch
):
    monkeypatch.setattr(witness.witness.probe, "cleanup", lambda: {"pod": True})
    with pytest.raises(BoundaryDenied, match="unacknowledged"):
        witness.witness.cleanup(SimpleNamespace())
    assert witness.calls == []
