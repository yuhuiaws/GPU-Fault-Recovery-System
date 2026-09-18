"""Additional private cursor ownership, file races and runner failure paths."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.channel_registry import COLLECTOR_HEALTH_PATH, FABRIC_MANAGER_PATH
from scripts.e2e.regional import run_collect022_fm_cursor_recovery as runner
from scripts.e2e.regional.probes import collect022_fm_cursor_probe as probe
from tests.regional import test_collect022_fm_cursor_recovery as cursor_support
from tests.regional._cov95_collect_net import no_external_effects  # noqa: F401
from tests.regional.test_collect022_fm_cursor_recovery import (
    NONCE,
    RUNTIME,
    SERVICE,
    action,
    execute,
    prepared_run,
    report,
)

private_environment = cursor_support.private_environment


def test_directory_open_creates_only_the_leaf_and_cleanup_handles_absence(
    private_environment: Any, tmp_path: Path, monkeypatch: Any
) -> None:
    with pytest.raises(probe.ProbeError, match="absolute"):
        probe.directory_fd(Path("relative"))
    private_environment.base.rmdir()
    assert action(private_environment, "cleanup")["private_root"] is False
    action(private_environment, "init")
    assert private_environment.base.stat().st_mode & 0o777 == 0o700
    action(private_environment, "cleanup")
    private_environment.base.chmod(0o777)
    with pytest.raises(probe.ProbeError, match="permissions"):
        action(private_environment, "init")
    monkeypatch.setattr(probe, "PRIVATE_BASE", tmp_path / "missing-parent" / "leaf")
    with pytest.raises(FileNotFoundError):
        action(private_environment, "init")
    assert not (tmp_path / "missing-parent").exists(), (
        "do not create unapproved parent directories"
    )


@pytest.mark.parametrize(
    "problem", ["unexpected", "root-mode", "owner-schema", "owner-list"]
)
def test_private_ownership_uncertainty_preserves_files(
    private_environment: Any, problem: str
) -> None:
    action(private_environment, "init")
    root = private_environment.root
    if problem == "unexpected":
        (root / "foreign").write_text("preserve")
    elif problem == "root-mode":
        root.chmod(0o755)
    else:
        owner_path = root / "owner.json"
        owner = json.loads(owner_path.read_text())
        owner["schema_version"] = True
        owner_path.write_text(json.dumps([] if problem == "owner-list" else owner))
    with pytest.raises(probe.ProbeError):
        action(private_environment, "cleanup")
    assert (root / "active.log").exists(), "unproven ownership cannot authorize removal"


def test_private_root_detects_inode_replacement_while_open(
    private_environment: Any,
) -> None:
    action(private_environment, "init")
    with probe.owned_root(NONCE, "cluster-a", "node-a") as root:
        private_environment.root.rename(private_environment.base / "retained")
        private_environment.root.mkdir(mode=0o700)
        with pytest.raises(probe.ProbeError, match="identity changed"):
            root.validate()


@pytest.mark.parametrize(
    "failure", ["read-race", "write-limit", "append-limit", "short-write", "missing"]
)
def test_private_file_io_enforces_size_and_complete_write(
    private_environment: Any, monkeypatch: Any, failure: str
) -> None:
    action(private_environment, "init")
    with probe.owned_root(NONCE, "cluster-a", "node-a") as root:
        system = SimpleNamespace(**vars(os))
        monkeypatch.setattr(probe, "os", system)
        if failure == "read-race":
            system.read = lambda fd, count: b"x" * (probe.MAX_FILE_BYTES + 1)
        if failure == "short-write":
            system.write = lambda fd, data: len(data) - 1
        with pytest.raises((probe.ProbeError, FileNotFoundError)):
            if failure == "read-race":
                root.read("active.log")
            elif failure == "write-limit":
                root.write("active.log", b"x" * (probe.MAX_FILE_BYTES + 1))
            elif failure == "append-limit":
                root.write("active.log", b"x" * probe.MAX_FILE_BYTES, append=True)
            elif failure == "missing":
                root.read("rotated.log")
            else:
                root.write("active.log", b"new-content")
    assert (private_environment.root / "owner.json").exists(), (
        "failed I/O must retain owner proof"
    )


@pytest.mark.parametrize("contents", [b"[]", b'{"files":[]}'])
def test_cursor_nonobject_state_is_explicitly_invalid(
    private_environment: Any, contents: bytes
) -> None:
    action(private_environment, "init")
    with probe.owned_root(NONCE, "cluster-a", "node-a") as root:
        root.write("cursor.json", contents)
        assert root.snapshot()["cursor"]["valid"] is False


def test_recording_sink_rejects_external_or_excess_deliveries() -> None:
    sink = probe.RecordingSink()
    assert sink.post(COLLECTOR_HEALTH_PATH, {}) == {"status": "recorded-locally"}
    for number in range(16):
        sink.post(
            FABRIC_MANAGER_PATH, {"record_id": str(number), "unrelated": "not-exported"}
        )
    assert sink.health_summaries == 1
    assert len(sink.events) == 16
    assert "unrelated" not in sink.events[0], "sink exports only audited event fields"
    for path in (FABRIC_MANAGER_PATH, "/other"):
        with pytest.raises(probe.ProbeError, match="unexpected delivery"):
            sink.post(path, {})
    with pytest.raises(probe.ProbeError, match="external command"):
        probe.refuse_external_command("fake")


@pytest.mark.parametrize(
    "change",
    [{"cluster_id": ""}, {"node_id": ""}, {"expires_at": None}, {"expires_at": True}],
)
def test_action_rejects_missing_scope_and_deadline(
    private_environment: Any, change: Any
) -> None:
    with pytest.raises(probe.ProbeError):
        action(private_environment, "init", **change)
    assert not private_environment.root.exists(), (
        "invalid admission cannot create files"
    )


def test_action_rejects_unknown_phase_malformed_progress_and_empty_boot(
    private_environment: Any,
) -> None:
    with pytest.raises(probe.ProbeError, match="action"):
        action(private_environment, "unknown")
    private_environment.boot.write_text("")
    with pytest.raises(probe.ProbeError, match="boot identity"):
        action(private_environment, "init")
    private_environment.boot.write_text("boot-a")
    action(private_environment, "init")
    with pytest.raises(probe.ProbeError, match="not allowed"):
        action(private_environment, "step", "other")
    progress = private_environment.root / "progress.json"
    progress.write_text(json.dumps({"next_step": True, "inflight": None}))
    with pytest.raises(probe.ProbeError, match="malformed"):
        action(private_environment, "step", "baseline")


def test_rotation_refuses_existing_destination(private_environment: Any) -> None:
    action(private_environment, "init")
    with probe.owned_root(NONCE, "cluster-a", "node-a") as root:
        root.write("rotated.log", b"preserve", exclusive=True)
        with pytest.raises(probe.ProbeError, match="already exists"):
            probe.mutate_private(root, "rotate")
        assert root.read("rotated.log") == b"preserve"


@pytest.mark.parametrize("identity", ["service", "runtime"])
def test_mid_collection_identity_change_leaves_unresolved_phase(
    private_environment: Any, monkeypatch: Any, identity: str
) -> None:
    action(private_environment, "init")
    values = iter(
        [dict(SERVICE if identity == "service" else RUNTIME), {"changed": "identity"}]
    )
    monkeypatch.setattr(
        probe,
        f"{'service' if identity == 'service' else 'deployed'}_identity",
        lambda: next(values),
    )
    with pytest.raises(probe.ProbeError, match="changed during collection"):
        action(private_environment, "step", "baseline")
    progress = json.loads((private_environment.root / "progress.json").read_text())
    assert progress["inflight"] == "baseline"
    assert progress["next_step"] == 0


@pytest.mark.parametrize("bad", ["returncode", "load", "active", "pid", "invocation"])
def test_service_identity_refuses_incomplete_process_state(
    monkeypatch: Any, bad: str
) -> None:
    fields = dict(SERVICE)
    fields[
        {
            "load": "LoadState",
            "active": "ActiveState",
            "pid": "MainPID",
            "invocation": "InvocationID",
        }.get(bad, "ignored")
    ] = ""
    monkeypatch.setattr(
        probe.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=int(bad == "returncode"),
            stdout="\n".join(f"{k}={v}" for k, v in fields.items()),
        ),
    )
    with pytest.raises(probe.ProbeError, match="unproven"):
        probe.service_identity()


def test_runtime_and_process_identity_read_only_local_fixture_files(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(probe, "NODE_PREFIX", tmp_path)
    monkeypatch.setattr(probe, "sys", SimpleNamespace(prefix=str(tmp_path)))
    result = probe.deployed_identity()
    assert result["python_prefix"] == str(tmp_path)
    assert len(result["collector_module_sha256"]) == 64
    stat = tmp_path / "stat"
    stat.write_text("123 (reader process) " + " ".join(["7"] * 20))
    monkeypatch.setattr(
        probe, "Path", lambda value: stat if value == "/proc/self/stat" else Path(value)
    )
    monkeypatch.setattr(
        probe, "os", SimpleNamespace(**{**vars(os), "getpid": lambda: 123})
    )
    assert probe.process_identity() == "123:7"


@pytest.mark.parametrize("valid", [False, True])
def test_probe_main_emits_bound_cleanup_or_validation_error(
    private_environment: Any, monkeypatch: Any, capsys: Any, valid: bool
) -> None:
    monkeypatch.setattr(
        probe.sys,
        "argv",
        [
            "cursor-probe",
            "cleanup",
            "--nonce",
            NONCE if valid else "invalid",
            "--cluster-id",
            "cluster-a",
            "--node-id",
            "node-a",
        ],
    )
    assert probe.main() == (0 if valid else 1)
    output = json.loads(capsys.readouterr().out)
    assert (
        output.get("private_root") is False
        if valid
        else output["error"] == "ProbeError"
    )


@pytest.mark.parametrize(
    "failure",
    [
        "create",
        "init-proof",
        "private-cleanup",
        "private-residual",
        "host-cleanup",
        "node-drift",
        "cpu-drift",
        "final-read",
    ],
)
def test_runner_records_cleanup_failures_without_losing_nonce_ownership(
    private_environment: Any, tmp_path: Path, monkeypatch: Any, failure: str
) -> None:
    prepared = prepared_run(private_environment, tmp_path, monkeypatch)
    original_host = runner.HostProbeFixture

    class Host(original_host):
        def create(self) -> None:
            if failure == "create":
                raise RuntimeError("create ACK unavailable")
            super().create()

        def execute(self, command: str, *args: str, **kwargs: Any) -> Any:
            value = super().execute(command, *args, **kwargs)
            if command == "init" and failure == "init-proof":
                value["initialized"] = False
            if command == "cleanup":
                if failure == "private-cleanup":
                    raise RuntimeError("cleanup ACK unavailable")
                if failure == "private-residual":
                    return {"private_root": None}
            return value

        def cleanup(self) -> Any:
            value = super().cleanup()
            if failure == "host-cleanup":
                raise RuntimeError("host cleanup unconfirmed")
            return value

    monkeypatch.setattr(runner, "HostProbeFixture", Host)
    if failure == "node-drift":
        prepared.regional.node_snapshot = lambda _: {"uid": "changed"}
    if failure == "cpu-drift":
        prepared.regional.cpu_blast_snapshot = lambda: {"changed": True}
    if failure == "final-read":

        def failed_read(_node: str) -> Any:
            raise RuntimeError("read unavailable")

        prepared.regional.node_snapshot = failed_read
    assert execute(prepared) == 1
    result = report(prepared)
    assert result["verdict"] == "FAIL"
    assert result.get("error") or result.get("cleanup_errors"), (
        "failure needs an auditable reason"
    )
    assert prepared.hosts[0].cleaned, "HostProbe cleanup must run after every failure"
    intent = json.loads(
        (prepared.case_dir / "private-cursor-intent-a1.json").read_text()
    )
    assert intent["state"] == (
        "CLOSED" if failure in {"create", "init-proof"} else "CLEANUP_REQUIRED"
    )


@pytest.mark.parametrize(
    "problem", ["identity", "unclean", "agent", "generation", "workload"]
)
def test_preflight_refuses_each_missing_node_or_agent_premise(
    tmp_path: Path, monkeypatch: Any, problem: str
) -> None:
    node = {
        "uid": "node-a",
        "boot_id": "boot-a",
        "ready": "True",
        "unschedulable": False,
        "taints": [],
        "ownership_annotations": {},
    }
    agent = {"lifecycle_state": "ACTIVE", "generation": 1}
    if problem == "identity":
        del node["uid"]
    if problem == "unclean":
        node["taints"] = ["foreign"]
    if problem == "agent":
        agent["lifecycle_state"] = "INACTIVE"
    if problem == "generation":
        agent["generation"] = True
    regional = SimpleNamespace(
        evidence_identity=lambda: {
            "release_id": "release-a",
            "cluster_id": "cluster-a",
        },
        node_snapshot=lambda _: node,
        store_snapshot=lambda **k: {"agent": agent},
        business_workloads=lambda _: ["workload"] if problem == "workload" else [],
        cpu_blast_snapshot=lambda: {},
    )
    monkeypatch.setattr(runner, "RegionalLiveFixture", lambda _: regional)
    monkeypatch.setattr(runner, "predecessor_evidence", lambda *a, **k: {"valid": True})
    settings = runner.Settings(object(), "node-a", "image", tmp_path / "predecessor")
    result = runner.read_only_preflight(settings, tmp_path)
    assert len(result["errors"]) == 1


def test_runner_preflight_failure_stops_before_host_creation(
    private_environment: Any, tmp_path: Path, monkeypatch: Any
) -> None:
    prepared = prepared_run(private_environment, tmp_path, monkeypatch)
    prepared.preflight["errors"] = ["not ready"]
    with pytest.raises(runner.RegionalFixtureError, match="not ready"):
        execute(prepared)
    assert prepared.hosts == []


def test_configuration_and_main_delegate_to_guarded_case(
    tmp_path: Path, monkeypatch: Any
) -> None:
    settings = SimpleNamespace(environment=lambda: {"BASE": "value"})
    monkeypatch.setattr(runner, "settings_from_arguments", lambda _: settings)
    args = runner.parser().parse_args(
        [
            "--run-dir",
            str(tmp_path),
            "--node",
            "node-a",
            "--host-probe-image",
            "probe@sha256:" + "a" * 64,
            "--predecessor-evidence",
            str(tmp_path / "previous.json"),
        ]
    )
    configured = runner.configure(args)
    assert configured.environment()["GPU_FAULT_COLLECT022_NODE"] == "node-a"
    assert configured.predecessor_path == tmp_path / "previous.json"
    args.host_probe_image = "mutable:latest"
    with pytest.raises(runner.RegionalFixtureError, match="immutable"):
        runner.configure(args)
    cases = []
    monkeypatch.setattr(
        runner, "run_standard_case", lambda case: cases.append(case) or 7
    )
    assert runner.main() == 7
    assert cases[0].case_id == runner.CASE_ID
    assert cases[0].execute_case is runner.execute_case


@pytest.mark.parametrize(
    "mutation", ["no-events", "loss", "corrupt", "rotation", "bad-truncation"]
)
def test_phase_verdict_refuses_missing_receipts_and_invalid_cursor_transitions(
    private_environment: Any, mutation: str
) -> None:
    initial = action(private_environment, "init")
    previous = []
    target = {
        "no-events": "append",
        "loss": "loss",
        "corrupt": "corrupt",
        "rotation": "rotate",
        "bad-truncation": "truncate",
    }[mutation]
    for step in probe.STEPS:
        receipt = action(private_environment, "step", step)
        if step == target:
            break
        previous.append(receipt)
    value = deepcopy(receipt)
    if mutation == "no-events":
        value["events"] = None
    elif mutation == "loss":
        value["before"]["cursor"]["present"] = True
    elif mutation == "corrupt":
        value["before"]["cursor"]["valid"] = True
    elif mutation == "rotation":
        value["after"]["logs"]["active.log"]["inode"] = previous[-1]["after"]["logs"][
            "active.log"
        ]["inode"]
    else:
        value["truncation"] = ["invalid"]
    assert runner.phase_errors(
        value, step=target, initial=initial, previous=previous
    ), f"invalid {mutation} transition must not pass"
