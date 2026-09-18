"""Warm-audit membership and durable custody through public, offline boundaries."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.admin import execution
from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from scripts.e2e.regional import acceptance_supervision
from scripts.e2e.regional import audit_warm_spare_guardrails as audit
from tools import pytest_result_identity

ROOT = Path(__file__).resolve().parents[2]
CASE = "GF-REGIONAL-DESTR-005"
IDENTITY = {
    "release_id": "unit-release",
    "cluster_id": "unit-cluster",
    "eks_cluster_arn": "arn:aws:eks:us-west-2:111122223333:cluster/unit",
    "registry_generation": "1",
}


def forbidden(*args, **kwargs):
    raise AssertionError("warm peer regression reached an unfaked external operation")


def block_external(patch):
    for name in ("run", "Popen", "call", "check_call", "check_output"):
        patch.setattr(subprocess, name, forbidden)
    for name in ("system", "kill", "killpg"):
        patch.setattr(os, name, forbidden)
    for name in ("connect", "connect_ex", "bind", "sendto"):
        patch.setattr(socket.socket, name, forbidden)
    patch.setattr(socket, "create_connection", forbidden)
    patch.setattr(socket, "getaddrinfo", forbidden)
    for name in ("client", "resource", "Session"):
        patch.setattr(f"boto3.{name}", forbidden)
    patch.setattr("psycopg.connect", forbidden)
    patch.setattr(audit, "command", forbidden)
    patch.setattr(audit, "kubectl", forbidden)
    patch.setattr(execution, "run_command", forbidden)


def raw_gpu(name, *, allocatable="8", quarantine=False):
    return {
        "metadata": {
            "name": name,
            "uid": f"uid-{name}",
            "annotations": (
                {audit.OWNERSHIP_ANNOTATIONS[0]: f"incident-{name}"}
                if quarantine
                else {}
            ),
        },
        "spec": {
            "unschedulable": quarantine,
            "taints": (
                [
                    {
                        "key": audit.QUARANTINE_TAINT,
                        "value": "held",
                        "effect": "NoSchedule",
                    }
                ]
                if quarantine
                else []
            ),
        },
        "status": {
            "capacity": {"nvidia.com/gpu": "8"},
            "allocatable": {"nvidia.com/gpu": allocatable},
            "conditions": [{"type": "Ready", "status": "True"}],
        },
    }


def snapshot_document(patch, document):
    block_external(patch)

    def kubectl(_config, _context, *arguments, **kwargs):
        assert arguments == ("get", "nodes", "-o", "json"), (
            "node regression must issue only its mocked inventory read"
        )
        return json.dumps(document)

    patch.setattr(audit, "kubectl", kubectl)
    return audit.node_snapshot()


def snapshot(patch, rows):
    return snapshot_document(
        patch, {"apiVersion": "v1", "kind": "NodeList", "items": rows}
    )


def test_healthy_gpu_control_remains_admissible(monkeypatch):
    rows = snapshot(monkeypatch, [raw_gpu("gpu-a")])
    assert [item["name"] for item in rows] == ["gpu-a"], (
        "healthy control did not produce the expected node identity"
    )
    assert audit.node_preflight_errors(rows) == [], (
        "the existing healthy baseline policy unexpectedly refused the control"
    )


@pytest.mark.parametrize("quantity", ["0", 0])
def test_zero_allocatable_quarantined_gpu_is_not_hidden_by_a_healthy_neighbor(
    monkeypatch, quantity
):
    rows = snapshot(
        monkeypatch,
        [raw_gpu("gpu-a"), raw_gpu("gpu-b", allocatable=quantity, quarantine=True)],
    )
    names = {item["name"] for item in rows}
    errors = audit.node_preflight_errors(rows)
    assert names == {"gpu-a", "gpu-b"}, (
        f"capacity-8 quarantined GPU B vanished: names={sorted(names)}, errors={errors}"
    )
    assert any("gpu-b" in error for error in errors), (
        "pre-existing ownership on zero-allocatable GPU B did not refuse admission"
    )


def test_malformed_gpu_quantity_cannot_disappear_beside_a_healthy_neighbor(monkeypatch):
    try:
        rows = snapshot(
            monkeypatch, [raw_gpu("gpu-a"), raw_gpu("gpu-b", allocatable="unreadable")]
        )
    except (ValueError, RuntimeError):
        return
    errors = audit.node_preflight_errors(rows)
    assert errors, (
        "unreadable GPU B was removed while its healthy neighbor was admitted: "
        f"names={[item['name'] for item in rows]}, errors={errors}"
    )


@pytest.mark.parametrize("field", ["capacity", "allocatable"])
@pytest.mark.parametrize("quantity", [None, "", "-1", -1, 1.5, True, {}, "8.0"])
def test_each_declared_gpu_quantity_must_be_a_nonnegative_integer(
    monkeypatch, field, quantity
):
    malformed = raw_gpu("gpu-b")
    malformed["status"][field]["nvidia.com/gpu"] = quantity
    with pytest.raises(RuntimeError, match="GPU resource quantity"):
        snapshot(monkeypatch, [raw_gpu("gpu-a"), malformed])


def test_known_gpu_without_allocatable_declaration_is_not_treated_as_cpu(monkeypatch):
    incomplete = raw_gpu("gpu-b")
    incomplete["status"].pop("allocatable")
    with pytest.raises(RuntimeError, match="allocatable resource declaration"):
        snapshot(monkeypatch, [raw_gpu("gpu-a"), incomplete])


def test_zero_gpu_declarations_preserve_identity_without_requiring_ready_or_schedulable(
    monkeypatch,
):
    gpu = raw_gpu("gpu-b", allocatable="0")
    gpu["status"]["capacity"]["nvidia.com/gpu"] = "0"
    gpu["status"]["conditions"][0]["status"] = "False"
    gpu["spec"] = {
        "unschedulable": True,
        "taints": [{"key": "business.example/reserved", "effect": "NoSchedule"}],
    }
    rows = snapshot(monkeypatch, [gpu])
    assert [row["name"] for row in rows] == ["gpu-b"], (
        "zero-valued GPU declarations lost node membership"
    )
    assert audit.node_preflight_errors(rows) == [], (
        "membership validation introduced a readiness or business-taint policy"
    )
    assert audit.node_state_drift(rows, rows) == [], (
        "an unchanged zero-allocatable GPU is not comparable"
    )


def test_zero_allocatable_node_remains_in_postflight_drift_comparison(monkeypatch):
    before = snapshot(
        monkeypatch, [raw_gpu("gpu-a"), raw_gpu("gpu-b", allocatable="0")]
    )
    replacement = raw_gpu("gpu-b", allocatable="0")
    replacement["metadata"]["uid"] = "replacement-gpu-b"
    after = snapshot(monkeypatch, [raw_gpu("gpu-a"), replacement])
    errors = audit.node_state_drift(before, after)
    assert any("gpu-b changed uid" in error for error in errors), (
        "zero-allocatable GPU replacement disappeared from postflight evidence"
    )


@pytest.mark.parametrize(
    "document",
    [
        None,
        [],
        {},
        {"items": None},
        {"items": [None]},
        {"items": [], "metadata": None},
        {"items": [], "metadata": {"continue": "unread-page"}},
    ],
)
def test_incomplete_node_inventory_is_not_usable_evidence(monkeypatch, document):
    with pytest.raises(RuntimeError, match="node inventory"):
        snapshot_document(monkeypatch, document)


@pytest.mark.parametrize("field", ["status", "capacity", "allocatable"])
def test_malformed_resource_maps_cannot_hide_a_gpu_node(monkeypatch, field):
    malformed = raw_gpu("gpu-b")
    if field == "status":
        malformed[field] = []
    else:
        malformed["status"][field] = []
    with pytest.raises(RuntimeError, match="node (resource status|status[.])"):
        snapshot(monkeypatch, [raw_gpu("gpu-a"), malformed])


def projected_gpu():
    return {
        "name": "gpu-a",
        "uid": "uid-gpu-a",
        "ready": "True",
        "gpu_allocatable": 8,
        "unschedulable": False,
        "taints": [],
        "ownership_annotations": {key: None for key in audit.OWNERSHIP_ANNOTATIONS},
    }


def child_main(mode, run_dir):
    if mode not in {"seed", "retry", "loss", "config-error"}:
        raise ValueError("unknown warm peer child mode")
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    marker = run_dir / "command-supervision-lost.json"
    case_path = run_dir / "cases" / CASE / f"{CASE}.json"
    calls = []
    with pytest.MonkeyPatch.context() as patch:
        block_external(patch)
        if mode == "seed":
            acceptance_supervision.bind_command_supervision(run_dir)
            acceptance_supervision.record_supervision_loss()
            audit.write_json(
                case_path, {"case_id": CASE, "verdict": "PASS", **IDENTITY}
            )
            document = json.loads(marker.read_text())
            print(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "mode": mode,
                        "marker_status": document["status"],
                        "case_verdict": json.loads(case_path.read_text())["verdict"],
                    }
                )
            )
            return 0
        if mode == "config-error":
            audit.write_json(
                case_path, {"case_id": CASE, "verdict": "PASS", **IDENTITY}
            )
        marker_before = marker.is_file()
        shared_gate_refuses = False
        try:
            acceptance_supervision.require_supervision_clear(run_dir)
        except RuntimeError:
            shared_gate_refuses = True
        patch.setattr(
            audit, "os", SimpleNamespace(umask=lambda _: 0o077, getenv=os.getenv)
        )

        def configure(*_args):
            if mode == "config-error":
                calls.append("configure-refused")
                raise RuntimeError("synthetic missing audit configuration")

        patch.setattr(audit, "configure", configure)
        patch.setattr(audit, "predecessor_path", lambda *_args: (None, None))
        patch.setattr(pytest_result_identity, "source_identity", lambda _root: "a" * 64)
        patch.setattr(
            sys, "argv", ["warm-peer-test", "--case", CASE, "--run-dir", str(run_dir)]
        )

        def identity():
            calls.append("identity")
            return dict(IDENTITY)

        patch.setattr(audit, "audit_identity", identity)
        if mode == "loss":
            patch.setattr(audit, "node_snapshot", lambda: [projected_gpu()])

            def lost(*_args, **_kwargs):
                calls.append("pytest-supervision-lost")
                raise ProcessSupervisionLost("synthetic loss of owned pytest custody")

            patch.setattr(execution, "run_command", lost)
        else:

            def complete(*_args, **_kwargs):
                calls.append("audit-work")
                return 0

            patch.setattr(audit, "run_audit", complete)
        returned = None
        error_type = None
        try:
            returned = audit.main()
        except BaseException as error:
            error_type = type(error).__name__
        print(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "mode": mode,
                    "marker_before": marker_before,
                    "shared_gate_refuses": shared_gate_refuses,
                    "marker_after": marker.is_file(),
                    "marker_status": (
                        json.loads(marker.read_text()).get("status")
                        if marker.is_file()
                        else None
                    ),
                    "returncode": returned,
                    "error_type": error_type,
                    "calls": calls,
                    "case": json.loads(case_path.read_text())
                    if case_path.is_file()
                    else None,
                },
                sort_keys=True,
            )
        )
        return 0


def run_child(mode, run_dir):
    environment = {
        "HOME": "/tmp",
        "PATH": os.environ.get("PATH", os.defpath),
        "PYTHONPATH": "src:.",
        "PYTHONDONTWRITEBYTECODE": "1",
        "AWS_CONFIG_FILE": "/dev/null",
        "AWS_SHARED_CREDENTIALS_FILE": "/dev/null",
        "AWS_EC2_METADATA_DISABLED": "true",
        "KUBECONFIG": "/dev/null",
    }
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--peer-child",
            mode,
            str(run_dir),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert completed.returncode == 0, (
        f"owned peer child failed before producing observations: {completed.stderr}"
    )
    result = json.loads(completed.stdout)
    assert result["pid"] != os.getpid(), "peer check did not use an owned fresh process"
    return result


def test_actual_loss_recorder_is_bound_by_the_warm_audit_entrypoint(tmp_path):
    observed = run_child("loss", tmp_path)
    assert observed["error_type"] == "ProcessSupervisionLost", (
        f"the audit did not propagate the intended custody failure: {observed}"
    )
    assert observed["calls"] == ["identity", "pytest-supervision-lost"], (
        f"failure did not originate from the actual focused-pytest boundary: {observed}"
    )
    assert observed["case"]["verdict"] == "FAIL", "failing audit left a passing result"
    assert (
        observed["marker_after"] and observed["marker_status"] == "RECOVERY_REQUIRED"
    ), f"actual recorder was called without a durable run binding: {observed}"


def test_fresh_retry_honors_real_supervision_marker_and_invalidates_prior_pass(
    tmp_path,
):
    seeded = run_child("seed", tmp_path)
    assert (
        seeded["marker_status"] == "RECOVERY_REQUIRED"
        and seeded["case_verdict"] == "PASS"
    ), "actual binder/recorder did not create the persisted precondition"
    observed = run_child("retry", tmp_path)
    assert seeded["pid"] != observed["pid"], "retry reused the previous process"
    assert observed["marker_before"] and observed["shared_gate_refuses"], (
        "the real persistent marker was not visible to the fresh process"
    )
    assert observed["error_type"] is not None and observed["calls"] == [], (
        f"a fresh audit bypassed the persisted supervision refusal: {observed}"
    )
    assert (
        observed["case"]["case_id"] == CASE and observed["case"]["verdict"] != "PASS"
    ), "binding refusal preserved an earlier PASS or changed the case identity"


def test_configuration_refusal_replaces_prior_pass_before_any_external_read(tmp_path):
    observed = run_child("config-error", tmp_path)
    assert observed["error_type"] == "RuntimeError", "configuration refusal disappeared"
    assert observed["calls"] == ["configure-refused"], (
        "invalid configuration reached live identity or audit work"
    )
    assert (
        observed["case"]["case_id"] == CASE and observed["case"]["verdict"] == "FAIL"
    ), "configuration refusal retained the old PASS or changed case identity"


def test_clean_fresh_process_control_reaches_only_mocked_audit_work(tmp_path):
    observed = run_child("retry", tmp_path)
    assert observed["returncode"] == 0 and observed["error_type"] is None, (
        f"clean-run control did not reach the harmless audit stub: {observed}"
    )
    assert (
        observed["calls"] == ["identity", "audit-work"]
        and not observed["marker_before"]
    ), "control did not exercise the same entrypoint without a supervision marker"


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--peer-child":
        raise SystemExit("this file is only a pytest module or owned peer child")
    raise SystemExit(child_main(sys.argv[2], Path(sys.argv[3])))
