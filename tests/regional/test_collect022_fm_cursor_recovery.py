from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from scripts.e2e.regional import run_collect022_fm_cursor_recovery as runner
from scripts.e2e.regional.probes import collect022_fm_cursor_probe as probe

NONCE = "1234567890abcdef1234567890abcdef"
RUNTIME = {
    "python_prefix": "/opt/gpu-fault/releases/test/venv",
    "package_version": "test-version",
    "collector_module_sha256": "a" * 64,
}
SERVICE = {
    "LoadState": "loaded",
    "ActiveState": "active",
    "MainPID": "123",
    "InvocationID": "service-a",
}
EXPECTED = [
    ("baseline", []),
    ("append", ["A"]),
    ("restart", []),
    ("loss", []),
    ("after-loss", ["B"]),
    ("corrupt", []),
    ("after-corrupt", ["C"]),
    ("rotate", ["R", "N"]),
    ("rotated-restart", []),
    ("truncate", ["T"]),
    ("truncated-restart", []),
]


@pytest.fixture
def private_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> SimpleNamespace:
    from gpu_fault.collectors.sinks import HttpEventSink

    base = tmp_path / "private"
    base.mkdir(mode=0o700)
    boot = tmp_path / "boot-id"
    boot.write_text("boot-a")
    monkeypatch.setattr(probe, "PRIVATE_BASE", base)
    monkeypatch.setattr(probe, "BOOT_FILE", boot)
    monkeypatch.setattr(probe, "deployed_identity", lambda: dict(RUNTIME))
    monkeypatch.setattr(probe, "service_identity", lambda: dict(SERVICE))
    processes = iter(f"300:{index}" for index in range(100))
    monkeypatch.setattr(probe, "process_identity", lambda: next(processes))
    external = Mock(
        side_effect=AssertionError("no external process is allowed in this test")
    )
    monkeypatch.setattr(probe, "refuse_external_command", external)
    monkeypatch.setattr(
        HttpEventSink,
        "__init__",
        Mock(
            side_effect=AssertionError(
                "private cursor reader must not construct an HTTP sink"
            )
        ),
    )
    return SimpleNamespace(
        base=base,
        root=base / f"c022-{NONCE}",
        boot=boot,
        expires_at=datetime.now(timezone.utc).timestamp() + 900,
        external=external,
    )


def action(
    environment: SimpleNamespace, mode: str, step: str | None = None, **overrides: Any
) -> dict[str, Any]:
    args = {
        "nonce": NONCE,
        "cluster_id": "cluster-a",
        "node_id": "node-a",
        "expires_at": environment.expires_at,
        "step": step,
        **overrides,
    }
    return probe.run_action(mode, **args)


def test_deployed_reader_private_missing_corrupt_rotation_and_restart_protocol(
    private_environment: SimpleNamespace, tmp_path: Path
) -> None:
    outside = tmp_path / "existing-fabric-manager.log"
    outside.write_text("existing collector data must remain untouched")
    initial = action(private_environment, "init")
    receipts = []
    delivered = []
    for step, tags in EXPECTED:
        value = action(private_environment, "step", step)
        messages = sorted(
            event["message"].rsplit("marker=", 1)[1] for event in value["events"]
        )
        assert messages == sorted(f"c022-{NONCE}-{tag}" for tag in tags)
        assert (
            runner.phase_errors(value, step=step, initial=initial, previous=receipts)
            == []
        )
        delivered.extend(event["record_id"] for event in value["events"])
        receipts.append(value)
    assert len(delivered) == len(set(delivered)) == 6
    assert (private_environment.root.stat().st_mode & 0o777) == 0o700
    assert all(
        (path.stat().st_mode & 0o777) == 0o600
        for path in private_environment.root.iterdir()
    ), "COLLECT-022 private log, cursor and ownership files must all remain mode 0600"
    assert action(private_environment, "cleanup") == {
        "private_root": False,
        "creation_unresolved": False,
    }
    assert action(private_environment, "cleanup") == {
        "private_root": False,
        "creation_unresolved": False,
    }
    assert not private_environment.root.exists(), (
        f"COLLECT-022 cleanup left private directory {private_environment.root}"
    )
    assert outside.read_text() == "existing collector data must remain untouched"
    private_environment.external.assert_not_called()


def test_lost_ack_reuses_receipt_without_appending_or_collecting_twice(
    private_environment: SimpleNamespace,
) -> None:
    action(private_environment, "init")
    action(private_environment, "step", "baseline")
    original = action(private_environment, "step", "append")
    contents = (private_environment.root / "active.log").read_bytes()
    assert action(private_environment, "step", "append") == original
    assert (private_environment.root / "active.log").read_bytes() == contents


@pytest.fixture
def before_truncation(private_environment: SimpleNamespace) -> SimpleNamespace:
    initial = action(private_environment, "init")
    receipts = []
    for step, _ in EXPECTED:
        if step == "truncate":
            break
        receipts.append(action(private_environment, "step", step))
    return SimpleNamespace(
        environment=private_environment,
        initial=initial,
        receipts=receipts,
        log=private_environment.root / probe.TRUNCATED_LOG,
        cursor=private_environment.root / "cursor.json",
    )


def test_private_truncation_delivers_once_and_advances_the_same_inode_cursor(
    before_truncation: SimpleNamespace,
) -> None:
    prepared = before_truncation
    environment = prepared.environment
    preserved = environment.root / "rotated.log"
    preserved_bytes = preserved.read_bytes()
    before = prepared.receipts[-1]["after"]
    original = before["logs"][probe.TRUNCATED_LOG]
    original_cursor = before["cursor"]["files"][str(prepared.log)]
    receipt = action(environment, "step", "truncate")
    zero = receipt["truncation"]["truncated"]
    after = receipt["after"]
    cursor = after["cursor"]["files"][str(prepared.log)]
    assert receipt["truncation"]["before"] == before
    assert zero["logs"][probe.TRUNCATED_LOG] == {**original, "size": 0}
    assert zero["cursor"] == before["cursor"]
    assert 0 < after["logs"][probe.TRUNCATED_LOG]["size"] < original["size"]
    assert cursor == {
        "device": original["device"],
        "inode": original["inode"],
        "offset": prepared.log.stat().st_size,
        "generation": original_cursor["generation"] + 1,
    }
    assert [
        event["message"].rsplit("marker=", 1)[1] for event in receipt["events"]
    ] == [f"c022-{NONCE}-T"]
    assert receipt["stats"]["delivered"] == 1
    assert receipt["events"][0]["fields"]["offset"] == "0"
    emitted = receipt["truncation"]["reused_event"]
    assert emitted in next(
        item["events"] for item in prepared.receipts if item["step"] == "rotate"
    )
    assert emitted["fields"]["offset"] == "0"
    assert emitted["fields"]["inode"] == receipt["events"][0]["fields"]["inode"]
    assert emitted["record_id"] != receipt["events"][0]["record_id"]
    assert emitted["evidence_ref"] != receipt["events"][0]["evidence_ref"]
    assert (
        runner.phase_errors(
            receipt,
            step="truncate",
            initial=prepared.initial,
            previous=prepared.receipts,
        )
        == []
    )
    assert action(environment, "step", "truncate") == receipt
    restarted = action(environment, "step", "truncated-restart")
    assert restarted["events"] == []
    assert restarted["after"]["cursor"] == after["cursor"]
    assert preserved.read_bytes() == preserved_bytes
    assert action(environment, "cleanup")["private_root"] is False
    environment.external.assert_not_called()


@pytest.mark.parametrize("change", ["missing", "corrupt", "short-log", "wrong-offset"])
def test_private_truncation_refuses_unproven_checkpoint_before_writing(
    before_truncation: SimpleNamespace, change: str
) -> None:
    prepared = before_truncation
    if change == "missing":
        prepared.cursor.unlink()
    elif change == "corrupt":
        prepared.cursor.write_text('{"files":')
    elif change == "short-log":
        prepared.log.write_text("short\n")
    else:
        checkpoint = json.loads(prepared.cursor.read_text())
        checkpoint["files"][str(prepared.log)]["offset"] = 0
        prepared.cursor.write_text(json.dumps(checkpoint))
    before = prepared.log.read_bytes()
    with pytest.raises(probe.ProbeError, match="truncation"):
        action(prepared.environment, "step", "truncate")
    assert prepared.log.read_bytes() == before
    assert action(prepared.environment, "cleanup")["private_root"] is False


@pytest.mark.parametrize(
    "change", ["missing-proof", "not-smaller", "inode", "cursor", "duplicate", "held"]
)
def test_private_truncation_verdict_rejects_missing_or_inconsistent_proof(
    before_truncation: SimpleNamespace, change: str
) -> None:
    prepared = before_truncation
    receipt = action(prepared.environment, "step", "truncate")
    path = str(prepared.log)
    if change == "missing-proof":
        del receipt["truncation"]
    elif change == "not-smaller":
        receipt["before"]["logs"][probe.TRUNCATED_LOG]["size"] = receipt["before"][
            "cursor"
        ]["files"][path]["offset"]
    elif change == "inode":
        receipt["truncation"]["truncated"]["logs"][probe.TRUNCATED_LOG]["inode"] += 1
    elif change == "cursor":
        receipt["before"]["cursor"]["files"][path]["offset"] = 0
    elif change == "duplicate":
        receipt["events"].append(deepcopy(receipt["events"][0]))
    else:
        receipt["after"]["cursor"]["files"][path]["offset"] = receipt["before"][
            "cursor"
        ]["files"][path]["offset"]
    assert runner.phase_errors(
        receipt, step="truncate", initial=prepared.initial, previous=prepared.receipts
    ), f"COLLECT-022 accepted inconsistent truncation proof: {change}"


@pytest.mark.parametrize(
    "change",
    [
        "missing-reused-event",
        "unemitted-event",
        "old-generation",
        "new-generation",
        "cursor-generation",
        "same-record-id",
        "wrong-record-id",
        "same-evidence-ref",
        "wrong-evidence-ref",
        "other-cursor",
        "noncanonical-offset",
    ],
)
def test_private_truncation_verdict_binds_emitted_record_generation_and_reference(
    before_truncation: SimpleNamespace, change: str
) -> None:
    prepared = before_truncation
    receipt = deepcopy(action(prepared.environment, "step", "truncate"))
    path = str(prepared.log)
    event = receipt["events"][0]
    reused = receipt["truncation"]["reused_event"]
    if change == "missing-reused-event":
        del receipt["truncation"]["reused_event"]
    elif change == "unemitted-event":
        reused["record_id"] = "fm-file-" + "f" * 64
    elif change == "old-generation":
        reused["fields"]["generation"] = str(int(reused["fields"]["generation"]) + 1)
    elif change == "new-generation":
        event["fields"]["generation"] = str(int(event["fields"]["generation"]) + 1)
    elif change == "cursor-generation":
        receipt["after"]["cursor"]["files"][path]["generation"] += 1
    elif change == "same-record-id":
        event["record_id"] = reused["record_id"]
    elif change == "wrong-record-id":
        event["record_id"] = "fm-file-" + "f" * 64
    elif change == "same-evidence-ref":
        event["evidence_ref"] = reused["evidence_ref"]
    elif change == "wrong-evidence-ref":
        event["evidence_ref"] += ":different-generation"
    elif change == "other-cursor":
        receipt["after"]["cursor"]["files"][
            str(prepared.environment.root / "rotated.log")
        ]["generation"] += 1
    else:
        event["fields"]["offset"] = "00"
    assert runner.phase_errors(
        receipt, step="truncate", initial=prepared.initial, previous=prepared.receipts
    ), f"COLLECT-022 accepted a forged emitted-offset proof: {change}"
    assert action(prepared.environment, "cleanup")["private_root"] is False


def test_private_truncation_never_follows_an_outside_symlink(
    before_truncation: SimpleNamespace, tmp_path: Path
) -> None:
    prepared = before_truncation
    outside = tmp_path / "preserved-log"
    outside.write_text("preserve outside content")
    prepared.log.unlink()
    prepared.log.symlink_to(outside)
    with pytest.raises(probe.ProbeError, match="unsafe"):
        action(prepared.environment, "step", "truncate")
    with pytest.raises(probe.ProbeError, match="unsafe"):
        action(prepared.environment, "cleanup")
    assert outside.read_text() == "preserve outside content"


def test_interrupted_phase_is_not_repeated_and_private_cleanup_still_works(
    private_environment: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    action(private_environment, "init")
    monkeypatch.setattr(
        probe, "collect_private", Mock(side_effect=RuntimeError("interrupted read"))
    )
    with pytest.raises(RuntimeError, match="interrupted read"):
        action(private_environment, "step", "baseline")
    with pytest.raises(probe.ProbeError, match="unresolved"):
        action(private_environment, "step", "baseline")
    assert action(private_environment, "cleanup")["private_root"] is False


@pytest.mark.parametrize("expires_at", [float("nan"), float("inf"), 0.0, -1.0])
def test_bad_deadline_refuses_before_private_directory_creation(
    private_environment: SimpleNamespace, expires_at: float
) -> None:
    with pytest.raises(probe.ProbeError, match="deadline"):
        action(private_environment, "init", expires_at=expires_at)
    assert not private_environment.root.exists(), (
        f"invalid deadline {expires_at!r} created a private cursor directory"
    )


def test_deadline_is_rechecked_after_runtime_and_service_reads(
    private_environment: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [private_environment.expires_at - 10]
    monkeypatch.setattr(probe, "time", SimpleNamespace(time=lambda: clock[0]))

    def service() -> dict[str, str]:
        clock[0] += 20
        return dict(SERVICE)

    monkeypatch.setattr(probe, "service_identity", service)
    with pytest.raises(probe.ProbeError, match="deadline"):
        action(private_environment, "init")
    assert not private_environment.root.exists(), (
        "runtime/service reads exhausted the deadline but initialization still wrote files"
    )


@pytest.mark.parametrize("nonce", ["", "../other", NONCE.upper(), "a" * 31])
def test_private_paths_have_no_caller_chosen_location(
    private_environment: SimpleNamespace, nonce: str
) -> None:
    with pytest.raises(probe.ProbeError, match="nonce"):
        action(private_environment, "init", nonce=nonce)
    assert list(private_environment.base.iterdir()) == []


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo"])
def test_unsafe_private_file_never_reads_or_modifies_an_outside_target(
    private_environment: SimpleNamespace, tmp_path: Path, kind: str
) -> None:
    action(private_environment, "init")
    outside = tmp_path / "outside"
    outside.write_text("preserve")
    outside.chmod(0o600)
    log = private_environment.root / "active.log"
    log.unlink()
    if kind == "symlink":
        log.symlink_to(outside)
    elif kind == "hardlink":
        os.link(outside, log)
    else:
        os.mkfifo(log, mode=0o600)
    with pytest.raises(probe.ProbeError, match="unsafe"):
        action(private_environment, "step", "baseline")
    with pytest.raises(probe.ProbeError, match="unsafe"):
        action(private_environment, "cleanup")
    assert outside.read_text() == "preserve"


def test_root_symlink_is_not_followed_during_creation_or_cleanup(
    private_environment: SimpleNamespace, tmp_path: Path
) -> None:
    outside = tmp_path / "outside-directory"
    outside.mkdir()
    preserved = outside / "owned-by-someone-else"
    preserved.write_text("preserve")
    private_environment.root.symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError):
        action(private_environment, "init")
    with pytest.raises(OSError):
        action(private_environment, "cleanup")
    assert preserved.read_text() == "preserve"


def test_unproved_existing_directory_is_not_adopted_or_deleted(
    private_environment: SimpleNamespace,
) -> None:
    private_environment.root.mkdir(mode=0o700)
    with pytest.raises(FileExistsError):
        action(private_environment, "init")
    with pytest.raises(FileNotFoundError):
        action(private_environment, "cleanup")
    assert private_environment.root.exists(), (
        "cleanup removed an existing directory without a matching owner document"
    )


@pytest.mark.parametrize(
    "field,value", [("nonce", "a" * 32), ("cluster_id", "other"), ("node_id", "other")]
)
def test_owner_document_mismatch_retains_private_files(
    private_environment: SimpleNamespace, field: str, value: str
) -> None:
    action(private_environment, "init")
    path = private_environment.root / "owner.json"
    owner = json.loads(path.read_text())
    owner[field] = value
    path.write_text(json.dumps(owner))
    with pytest.raises(probe.ProbeError, match="ownership"):
        action(private_environment, "cleanup")
    assert (private_environment.root / "active.log").exists(), (
        f"ownership mismatch {field}={value!r} did not preserve the private log"
    )


def test_replacement_directory_cannot_reuse_the_original_owner_document(
    private_environment: SimpleNamespace,
) -> None:
    action(private_environment, "init")
    owner = (private_environment.root / "owner.json").read_bytes()
    private_environment.root.rename(private_environment.base / "retained-original")
    private_environment.root.mkdir(mode=0o700)
    path = private_environment.root / "owner.json"
    path.write_bytes(owner)
    path.chmod(0o600)
    with pytest.raises(probe.ProbeError, match="ownership"):
        action(private_environment, "cleanup")


def test_active_owner_lock_prevents_cleanup(
    private_environment: SimpleNamespace,
) -> None:
    action(private_environment, "init")
    with probe.owned_root(NONCE, "cluster-a", "node-a"):
        with pytest.raises(BlockingIOError):
            action(private_environment, "cleanup")
    assert action(private_environment, "cleanup")["private_root"] is False


@pytest.mark.parametrize("changed", ["runtime", "service", "boot"])
def test_identity_drift_refuses_before_advancing_a_phase(
    private_environment: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    action(private_environment, "init")
    if changed == "runtime":
        monkeypatch.setattr(
            probe, "deployed_identity", lambda: {**RUNTIME, "package_version": "other"}
        )
    elif changed == "service":
        monkeypatch.setattr(
            probe, "service_identity", lambda: {**SERVICE, "InvocationID": "other"}
        )
    else:
        private_environment.boot.write_text("other-boot")
    before = (private_environment.root / "progress.json").read_bytes()
    with pytest.raises(probe.ProbeError, match="changed"):
        action(private_environment, "step", "baseline")
    assert (private_environment.root / "progress.json").read_bytes() == before
    assert action(private_environment, "cleanup")["private_root"] is False


def test_phase_order_cannot_skip_the_missing_cursor_assertion(
    private_environment: SimpleNamespace,
) -> None:
    action(private_environment, "init")
    before = (private_environment.root / "active.log").read_bytes()
    with pytest.raises(probe.ProbeError, match="strict order"):
        action(private_environment, "step", "after-loss")
    assert (private_environment.root / "active.log").read_bytes() == before


def test_cursor_with_foreign_path_is_refused_before_the_reader_runs(
    private_environment: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    action(private_environment, "init")
    path = private_environment.root / "cursor.json"
    path.write_text(
        json.dumps({"files": {"/var/log/fabricmanager.log": {"offset": 0}}})
    )
    path.chmod(0o600)
    collect = Mock(side_effect=AssertionError("reader must not start"))
    monkeypatch.setattr(probe, "collect_private", collect)
    with pytest.raises(probe.ProbeError, match="outside"):
        action(private_environment, "step", "baseline")
    collect.assert_not_called()


@pytest.mark.parametrize(
    "mutation", ["marker", "cursor", "sink", "process", "schema", "stats"]
)
def test_verdict_rejects_forged_or_incomplete_local_evidence(
    private_environment: SimpleNamespace, mutation: str
) -> None:
    initial = action(private_environment, "init")
    baseline = action(private_environment, "step", "baseline")
    receipt = action(private_environment, "step", "append")
    if mutation == "marker":
        receipt["events"][0]["message"] = f"marker=c022-{NONCE}-H"
    elif mutation == "cursor":
        receipt["after"]["cursor"]["files"] = {}
    elif mutation == "sink":
        receipt["sink"] = "http"
    elif mutation == "process":
        receipt["process_identity"] = baseline["process_identity"]
    elif mutation == "schema":
        receipt["schema_version"] = True
    else:
        receipt["stats"]["delivered"] = True
    assert runner.phase_errors(
        receipt, step="append", initial=initial, previous=[baseline]
    ), f"COLLECT-022 accepted forged append evidence: {mutation}"


def test_live_service_probe_is_read_only_and_projects_no_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def read(command: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append((command, kwargs["timeout"]))
        return SimpleNamespace(
            returncode=0,
            stdout="LoadState=loaded\nActiveState=active\nMainPID=123\nInvocationID=service-a\n",
        )

    monkeypatch.setattr(probe, "subprocess", SimpleNamespace(run=read))
    assert probe.service_identity() == SERVICE
    assert calls == [
        (
            [
                "systemctl",
                "show",
                "gpu-fault-fabric-manager-collector.service",
                "--property=LoadState",
                "--property=ActiveState",
                "--property=MainPID",
                "--property=InvocationID",
            ],
            10,
        )
    ]


def test_probe_refuses_a_non_deployed_python_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(probe, "NODE_PREFIX", tmp_path / "not-a-node-runtime")
    with pytest.raises(probe.ProbeError, match="deployed Node Runtime"):
        probe.deployed_identity()


class Abort(BaseException):
    pass


def prepared_run(
    environment: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_after_init: bool = False,
    abort: bool = False,
    bad_phase: str = "",
    residuals: dict[str, Any] | None = None,
) -> SimpleNamespace:
    kubeconfig = tmp_path / "unit-kubeconfig"
    kubeconfig.touch()
    settings = runner.Settings(
        regional=SimpleNamespace(
            gpu_kubeconfig=kubeconfig,
            gpu_context="unit-context",
            namespace="unit-namespace",
            cluster_id="cluster-a",
        ),
        node="node-a",
        host_probe_image="unit-probe@sha256:" + "a" * 64,
        predecessor_path=tmp_path / "predecessor.json",
    )
    node = {
        "uid": "node-uid",
        "boot_id": "boot-a",
        "ready": "True",
        "unschedulable": False,
        "taints": [],
        "ownership_annotations": {},
    }
    preflight = {
        "release_id": "release-a",
        "cluster_id": "cluster-a",
        "node": node,
        "agent_generation": 1,
        "errors": [],
        "cpu_blast": {},
        "predecessor": {
            "valid": True,
            "case_id": runner.PREDECESSOR_CASE_ID,
            "verdict": "PASS",
        },
    }
    regional = SimpleNamespace(
        evidence_identity=lambda: {
            "release_id": "release-a",
            "cluster_id": "cluster-a",
        },
        node_snapshot=lambda _: deepcopy(node),
        cpu_blast_snapshot=lambda: {},
    )
    case_dir = tmp_path / "cases" / runner.CASE_ID
    case_dir.mkdir(parents=True)
    (case_dir / "plan.json").write_text(
        json.dumps({"details": runner.plan_details(settings, preflight)})
    )
    hosts = []

    class Host:
        def __init__(self, settings: Any) -> None:
            self.settings = settings
            self.calls: list[str] = []
            self.cleaned = False
            self.nonce = ""
            hosts.append(self)

        def create(self) -> None:
            self.calls.append("create")

        def execute(self, command: str, *args: str, timeout: int) -> dict[str, Any]:
            arguments = probe.parser().parse_args([command, *args])
            self.calls.append(
                command + (":" + arguments.step if arguments.step else "")
            )
            self.nonce = arguments.nonce
            intent = json.loads(
                (case_dir / "private-cursor-intent-a1.json").read_text()
            )
            assert intent["nonce"] == arguments.nonce
            assert timeout == 90
            value = probe.run_action(
                command,
                nonce=arguments.nonce,
                cluster_id=arguments.cluster_id,
                node_id=arguments.node_id,
                expires_at=arguments.expires_at,
                step=arguments.step,
            )
            if command == "init" and fail_after_init:
                if abort:
                    raise Abort("lost init ACK")
                raise RuntimeError("lost init ACK")
            if arguments.step == bad_phase:
                value["events"].append(
                    {"message": "historical marker replay", "record_id": "wrong-record"}
                )
            return value

        def cleanup(self) -> dict[str, Any]:
            self.cleaned = True
            return {
                "pod": False,
                "configmap": False,
                "host_script": False,
                "creation_unresolved": False,
                **(residuals or {}),
            }

    monkeypatch.setattr(runner, "RegionalLiveFixture", lambda _: regional)
    monkeypatch.setattr(runner, "HostProbeFixture", Host)
    monkeypatch.setattr(
        runner, "read_only_preflight", lambda *args: deepcopy(preflight)
    )
    return SimpleNamespace(
        settings=settings,
        regional=regional,
        case_dir=case_dir,
        run_dir=tmp_path,
        preflight=preflight,
        hosts=hosts,
        environment=environment,
    )


def execute(prepared: SimpleNamespace) -> int:
    return runner.execute_case(
        prepared.settings,
        prepared.run_dir,
        1,
        datetime.now(timezone.utc) + timedelta(minutes=15),
    )


def report(prepared: SimpleNamespace) -> dict[str, Any]:
    return dict(json.loads((prepared.case_dir / f"{runner.CASE_ID}.json").read_text()))


def test_guarded_runner_completes_private_protocol_and_cleans_both_ownership_layers(
    private_environment: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = prepared_run(private_environment, tmp_path, monkeypatch)
    assert execute(prepared) == 0
    host = prepared.hosts[0]
    assert host.settings.state_directory == prepared.case_dir / "host-probes"
    assert host.calls == [
        "create",
        "init",
        *(f"step:{name}" for name, _ in EXPECTED),
        "cleanup",
    ]
    assert host.cleaned, (
        f"successful protocol omitted HostProbe cleanup: {host.calls!r}"
    )
    assert list(private_environment.base.iterdir()) == []
    result = report(prepared)
    assert result["verdict"] == "PASS"
    assert result["private_residuals"] == {
        "private_root": False,
        "creation_unresolved": False,
    }
    assert len(result["phases"]) == 11
    assert (
        json.loads((prepared.case_dir / "private-cursor-intent-a1.json").read_text())[
            "state"
        ]
        == "CLOSED"
    )


@pytest.mark.parametrize("abort", [False, True])
def test_runner_owns_nonce_before_lost_ack_and_always_cleans(
    private_environment: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    abort: bool,
) -> None:
    prepared = prepared_run(
        private_environment, tmp_path, monkeypatch, fail_after_init=True, abort=abort
    )
    if abort:
        with pytest.raises(Abort):
            execute(prepared)
    else:
        assert execute(prepared) == 1
    assert prepared.hosts[0].calls == ["create", "init", "cleanup"]
    assert prepared.hosts[0].cleaned, (
        f"lost init ACK (abort={abort}) omitted HostProbe cleanup"
    )
    assert list(private_environment.base.iterdir()) == []
    assert report(prepared)["verdict"] == "FAIL"


def test_bad_phase_stops_before_corruption_or_rotation(
    private_environment: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = prepared_run(
        private_environment, tmp_path, monkeypatch, bad_phase="after-loss"
    )
    assert execute(prepared) == 1
    assert "step:corrupt" not in prepared.hosts[0].calls
    assert "step:rotate" not in prepared.hosts[0].calls
    assert prepared.hosts[0].calls[-1] == "cleanup"
    assert report(prepared)["verdict"] == "FAIL"


def test_bad_truncation_stops_before_restart_and_cleans_private_files(
    private_environment: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = prepared_run(
        private_environment, tmp_path, monkeypatch, bad_phase="truncate"
    )
    assert execute(prepared) == 1
    assert "step:truncated-restart" not in prepared.hosts[0].calls
    assert prepared.hosts[0].calls[-1] == "cleanup"
    assert prepared.hosts[0].cleaned, (
        "rejected truncation evidence did not trigger HostProbe cleanup"
    )
    assert list(private_environment.base.iterdir()) == []
    assert report(prepared)["verdict"] == "FAIL"


@pytest.mark.parametrize(
    "key,value", [("host_script", True), ("creation_unresolved", True), ("pod", None)]
)
def test_any_new_or_unknown_hostprobe_residual_fails_the_case(
    private_environment: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    value: Any,
) -> None:
    prepared = prepared_run(
        private_environment, tmp_path, monkeypatch, residuals={key: value}
    )
    assert execute(prepared) == 1
    assert report(prepared)["cleanup_errors"] == [
        "HostProbe cleanup has unresolved residuals"
    ]


def test_plan_node_drift_refuses_before_probe_creation(
    private_environment: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = prepared_run(private_environment, tmp_path, monkeypatch)
    path = prepared.case_dir / "plan.json"
    plan = json.loads(path.read_text())
    plan["details"]["preflight_identity"]["node_uid"] = "old-node"
    path.write_text(json.dumps(plan))
    with pytest.raises(runner.RegionalFixtureError, match="plan identity"):
        execute(prepared)
    assert prepared.hosts == []


def test_existing_attempt_intent_is_not_overwritten(
    private_environment: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = prepared_run(private_environment, tmp_path, monkeypatch)
    path = prepared.case_dir / "private-cursor-intent-a1.json"
    path.write_text('{"existing":"proof"}')
    with pytest.raises(FileExistsError):
        execute(prepared)
    assert path.read_text() == '{"existing":"proof"}'
    assert all(not host.calls for host in prepared.hosts), (
        f"existing attempt intent did not block host actions: "
        f"{[host.calls for host in prepared.hosts]!r}"
    )


def test_parser_defaults_are_manual_plan_only_and_use_predecessor005(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "settings_from_arguments", lambda _: object())
    arguments = runner.parser().parse_args(
        [
            "--run-dir",
            str(tmp_path),
            "--node",
            "node-a",
            "--host-probe-image",
            "probe@sha256:" + "a" * 64,
        ]
    )
    assert arguments.execute is False
    assert arguments.attempt == 1
    assert runner.CONFIRMATION == "COLLECT022_PRIVATE_FM_CURSOR_RECOVERY"
    settings = runner.configure(arguments)
    assert settings.predecessor_path == (
        tmp_path / "cases/GF-REGIONAL-COLLECT-005/GF-REGIONAL-COLLECT-005.json"
    )


def test_preflight_refuses_invalidated_historical_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node = {
        "uid": "uid",
        "boot_id": "boot",
        "ready": "True",
        "unschedulable": False,
        "taints": [],
        "ownership_annotations": {},
    }
    regional = SimpleNamespace(
        evidence_identity=lambda: {
            "release_id": "release-a",
            "cluster_id": "cluster-a",
        },
        node_snapshot=lambda _: node,
        store_snapshot=lambda **kwargs: {
            "agent": {"lifecycle_state": "ACTIVE", "generation": 1}
        },
        business_workloads=lambda _: [],
        cpu_blast_snapshot=lambda: {},
    )
    seen = {}

    def predecessor(path: Path, case_id: str, **identity: Any) -> dict[str, Any]:
        seen.update(case_id=case_id, **identity)
        return {"valid": False, "verdict": "NOT_RUN"}

    monkeypatch.setattr(runner, "RegionalLiveFixture", lambda _: regional)
    monkeypatch.setattr(runner, "predecessor_evidence", predecessor)
    settings = runner.Settings(
        object(), "node-a", "image", tmp_path / "old-evidence.json"
    )
    result = runner.read_only_preflight(settings, tmp_path)
    assert result["errors"] == [
        "COLLECT-005 predecessor is not current bound PASS evidence"
    ]
    assert seen == {
        "case_id": "GF-REGIONAL-COLLECT-005",
        "release_id": "release-a",
        "cluster_id": "cluster-a",
    }
