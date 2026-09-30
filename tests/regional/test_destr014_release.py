"""DESTR-014 ``--release``: the documented operator handling of a held attempt,
driven from the case's own records through the admin CLI and the fixtures, in
order, idempotently, and refused on an identity mismatch."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import destr014_release as release
from scripts.e2e.regional import run_destr014_branch_exhaustion as destr014
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError

FAULT = "hyperpod-i-fault"
SIBLING = "hyperpod-i-sibling"
INCIDENT = "inc-two-node-replace"
FOLLOW_UP = "inc-support-after-workflow"
IDENTITY = {
    "release_id": "release-a",
    "fault_node_uid": "uid-fault",
    "fault_node_boot_id": "boot-fault-0",
    "sibling_node_uid": "uid-sibling",
    "sibling_node_boot_id": "boot-sibling-0",
    "runtime_profile_version": "profile-v1",
}


def write_case(case_dir: Path, *, holder_armed: bool = True) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "plan.json").write_text(
        json.dumps({"details": {"preflight_identity": IDENTITY}})
    )
    (case_dir / release.JOURNAL_FILE).write_text(
        json.dumps(
            {
                "scope": {"run_id": "destr014-run-a3", "node": SIBLING},
                "phase": "RECOVERY_REQUIRED",
                "run": {
                    "incident_id": INCIDENT,
                    "follow_up_incident_id": FOLLOW_UP,
                    "holder_armed": holder_armed,
                    "preflight": {
                        "store": {"profile": {"profile_version": "profile-v1"}}
                    },
                },
            }
        )
    )
    (case_dir / f"{release.CASE_ID}.json").write_text(
        json.dumps({"verdict": "BLOCKED", "incident_id": INCIDENT})
    )
    (case_dir / "executor-env-window.json").write_text("{}")
    (case_dir / "control-plane-env-window.json").write_text("{}")


class Regional:
    def __init__(
        self, *, release_id: str = "release-a", sibling_boot: str = "boot-sibling-1"
    ):
        self.live_release = release_id
        self.boots = {FAULT: "boot-fault-0", SIBLING: sibling_boot}
        self.uids = {FAULT: "uid-fault", SIBLING: "uid-sibling"}

    def release_id(self) -> str:
        return self.live_release

    def node_snapshot(self, node: str) -> dict[str, Any]:
        return {
            "uid": self.uids[node],
            "boot_id": self.boots[node],
            "unschedulable": False,
            "taints": [],
            "ownership_annotations": {},
        }


class Probe:
    def __init__(self, log: list[str]) -> None:
        self.log = log

    def cleanup(self) -> dict[str, bool]:
        self.log.append("probe.cleanup")
        return {}

    def create(self) -> None:
        self.log.append("probe.create")

    def execute(self, *arguments: str, **_kwargs: Any) -> dict[str, Any]:
        self.log.append("probe:" + " ".join(arguments))
        return {"disarmed": True}


def settings(tmp_path: Path) -> Any:
    site = tmp_path / "state" / "site.yaml"
    site.parent.mkdir(parents=True, exist_ok=True)
    site.write_text("clusters: []\n")
    return SimpleNamespace(
        fault_node=FAULT,
        sibling_node=SIBLING,
        site_file=site,
        hyperpod_cluster="hp",
        host_probe_image="img",
    )


def context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    log: list[str],
    *,
    regional: Regional | None = None,
    residual: str | None = None,
    with_workload: bool = True,
) -> release.ReleaseContext:
    case_dir = tmp_path / "run" / "cases" / release.CASE_ID
    write_case(case_dir)

    def admin(state_dir: Path, *argv: str) -> dict[str, Any]:
        log.append("admin:" + " ".join(argv))
        assert state_dir == tmp_path / "state", state_dir
        return {"argv": list(argv), "returncode": 0}

    def restore(
        _regional: Any, _warm: Any, _settings: Any, incident_id: str, profile: str
    ):
        log.append(f"restore:{incident_id}:{profile}")
        return {"nodes": {}}

    def close(_warm: Any, _regional: Any, ids: list[str], **kwargs: Any):
        log.append("close:" + ",".join(item for item in ids if item))
        reports = {
            item: {
                "incident_id": item,
                "closed": residual is None,
                "state_before": "ESCALATED",
            }
            for item in ids
            if item
        }
        if residual:
            for report in reports.values():
                report["residual"] = residual
        return reports

    def close_env(settings_value: Any, _regional: Any, _survey: Any) -> dict[str, Any]:
        log.append(f"env-close:{settings_value.baseline.name}")
        return {"state": "CLOSED"}

    monkeypatch.setattr(release, "restore_isolated_nodes", restore)
    monkeypatch.setattr(release, "close_drill_incidents", close)
    monkeypatch.setattr(release.env_window, "close_window", close_env)
    monkeypatch.setattr(release.env_window, "survey", lambda _r: {})
    monkeypatch.setattr(release.control_window, "close_window", close_env)
    monkeypatch.setattr(release.control_window, "survey", lambda _r: {})
    workload = SimpleNamespace(delete=lambda: log.append("workload.delete"))
    return release.ReleaseContext(
        settings=settings(tmp_path),
        inputs=release.load_release_inputs(case_dir),
        regional=regional or Regional(),
        warm=SimpleNamespace(
            wait_incident_idle=lambda incident_id, **_kw: log.append(
                f"wait-idle:{incident_id}"
            )
        ),
        state_dir=tmp_path / "state",
        fault_probe=Probe(log),
        workload=workload if with_workload else None,
        admin=admin,
    )


def test_release_performs_the_documented_handling_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log: list[str] = []
    ctx = context(tmp_path, monkeypatch, log)

    report = release.release_hold(ctx)

    assert report["released"] is True, report
    assert log == [
        # 1. the node whose boot id changed (the sibling) has a known outcome now
        f"admin:submit-remediation --incident-id {INCIDENT} --disposition "
        f"confirm-node-action --node {SIBLING} --reference {release.RELEASE_REFERENCE}",
        # 2. the product's own restore releases the confirmed BLOCKED record ...
        f"admin:submit-remediation --incident-id {INCIDENT} --disposition restore "
        f"--reference {release.RELEASE_REFERENCE}",
        # ... and its workflow is waited out before anything races it
        f"wait-idle:{INCIDENT}",
        # 3. validated restore through whichever incident still owns a node
        f"restore:{INCIDENT}:profile-v1",
        # 3b. the records are settled once the incident is RECOVERED
        f"admin:workflow-reconcile --incident-id {INCIDENT} --reference {release.RELEASE_REFERENCE}",
        f"admin:workflow-reconcile --incident-id {FOLLOW_UP} --reference {release.RELEASE_REFERENCE}",
        # 4-5. the drill job, then the holder on a fresh probe Pod
        "workload.delete",
        "probe.cleanup",
        "probe.create",
        "probe:disarm-holder --run-id destr014-run-a3",
        "probe.cleanup",
        # 6. both env windows through their own records
        "env-close:executor-env-window.json",
        "env-close:control-plane-env-window.json",
        # 7. the drill incidents
        f"close:{INCIDENT},{FOLLOW_UP}",
    ], log
    history = json.loads((ctx.inputs.case_dir / release.RELEASE_RECORD).read_text())
    assert [item["released"] for item in history] == [True]
    assert history[0]["steps"]["rebooted_nodes"] == [SIBLING]


def test_release_refuses_when_the_identity_moved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log: list[str] = []
    ctx = context(tmp_path, monkeypatch, log, regional=Regional(release_id="release-b"))

    report = release.release_hold(ctx)

    assert report["released"] is False
    assert report["refused"] == "identity mismatch"
    assert any("release id changed" in item for item in report["errors"]), report
    assert log == [], "no admin verb, restore or delete runs on a foreign identity"


def test_release_without_a_rebooted_node_confirms_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log: list[str] = []
    ctx = context(
        tmp_path, monkeypatch, log, regional=Regional(sibling_boot="boot-sibling-0")
    )

    release.release_hold(ctx)

    assert not any("confirm-node-action" in item for item in log), log
    assert any(item.startswith("admin:workflow-reconcile") for item in log), log


def test_release_reports_open_incidents_and_a_failed_step_without_stopping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log: list[str] = []
    ctx = context(tmp_path, monkeypatch, log, residual="still has an open workflow")

    def admin(_state_dir: Path, *argv: str) -> dict[str, Any]:
        log.append("admin:" + argv[0])
        if argv[0] == "workflow-reconcile":
            raise RegionalFixtureError("gpu-fault-admin workflow-reconcile exited 1")
        return {}

    ctx.admin = admin
    report = release.release_hold(ctx)

    assert report["released"] is False
    assert report["residual_incidents"] == {
        INCIDENT: "still has an open workflow",
        FOLLOW_UP: "still has an open workflow",
    }
    assert any(
        item.startswith(f"workflow_reconcile:{INCIDENT}:") for item in report["errors"]
    ), report["errors"]
    assert "workload.delete" in log and "env-close:executor-env-window.json" in log, (
        "a failed product-side step does not skip the cluster-side release"
    )


def test_release_is_idempotent_on_an_already_released_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log: list[str] = []
    ctx = context(tmp_path, monkeypatch, log, with_workload=False)
    ctx.inputs.holder_armed = False
    (ctx.inputs.case_dir / "executor-env-window.json").unlink()

    report = release.release_hold(ctx)

    assert report["released"] is True, report
    assert report["steps"]["workload_delete"]["skipped"], report["steps"]
    assert report["steps"]["holder_disarm"]["skipped"], report["steps"]
    assert "env-close:executor-env-window.json" not in log
    assert "env-close:control-plane-env-window.json" in log


def test_release_inputs_need_the_attempt_records(tmp_path: Path) -> None:
    with pytest.raises(RegionalFixtureError, match="plan and journal"):
        release.load_release_inputs(tmp_path / "missing")


def test_rebooted_nodes_and_identity_read_the_plan(tmp_path: Path) -> None:
    regional = Regional()
    plan_settings = settings(tmp_path)

    assert release.identity_errors(regional, plan_settings, IDENTITY) == []
    assert release.rebooted_nodes(regional, plan_settings, IDENTITY) == [SIBLING]
    regional.uids[FAULT] = "uid-replacement"
    assert release.identity_errors(regional, plan_settings, IDENTITY) == [
        "fault node UID changed: plan uid-fault live uid-replacement"
    ]


def test_run_admin_reports_the_verb_and_its_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def run(argv: list[str], **kwargs: Any) -> Any:
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout='{"ok": true}', stderr="")

    monkeypatch.setattr(release.subprocess, "run", run)

    record = release.run_admin(
        Path("/state"), "workflow-reconcile", "--incident-id", "i"
    )

    assert record["returncode"] == 0 and record["argv"][0] == "workflow-reconcile"
    assert calls[0][-2:] == ["--state-dir", "/state"], calls
    assert calls[0][1] == "-c", (
        "the checkout's admin CLI runs in-process, not from PATH"
    )

    monkeypatch.setattr(
        release.subprocess,
        "run",
        lambda argv, **kwargs: SimpleNamespace(
            returncode=2, stdout="", stderr="refused"
        ),
    )
    with pytest.raises(RegionalFixtureError, match="exited 2: refused"):
        release.run_admin(Path("/state"), "workflow-reconcile")


# --- runner hook -------------------------------------------------------------------


def test_release_skips_the_product_refusals_it_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live a3 (2026-09-24): the fault node's confirm was refused ("no BLOCKED
    workflow ... has an unresolved node action on node ..."), and a2's restore
    was refused because the support-after incident owned the node. Both are the
    product saying "nothing to do here", so the release records a skip, keeps
    going and still ends released."""

    log: list[str] = []
    ctx = context(tmp_path, monkeypatch, log)

    def admin(_state_dir: Path, *argv: str) -> dict[str, Any]:
        log.append("admin:" + " ".join(argv[:4]))
        if "confirm-node-action" in argv:
            raise RegionalFixtureError(
                "gpu-fault-admin submit-remediation exited 2: no BLOCKED workflow "
                f"of incident {INCIDENT} has an unresolved node action on node "
                f"{SIBLING}; unresolved elsewhere: step 3 RESTART_NODE on other"
            )
        if "restore" in argv:
            raise RegionalFixtureError(
                f"gpu-fault-admin submit-remediation exited 2: no node of incident "
                f"{INCIDENT} is still isolated by it: ... close the incident with "
                "gpu-fault-admin workflow-reconcile --close-quarantined"
            )
        return {"argv": list(argv), "returncode": 0}

    ctx.admin = admin
    report = release.release_hold(ctx)

    assert report["released"] is True, report["errors"]
    steps = report["steps"]
    assert steps[f"confirm_node_action:{SIBLING}"]["skipped"].startswith(
        "no unresolved node action"
    ), steps
    assert steps["restore_disposition"]["skipped"].startswith("no node is isolated"), (
        steps
    )
    assert "restore_workflow_wait" not in steps, "nothing to wait for after a skip"
    assert not any(item.startswith("wait-idle:") for item in log), log
    assert f"restore:{INCIDENT}:profile-v1" in log, "owners still restore in step 3"


def test_release_reports_an_unexpected_restore_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log: list[str] = []
    ctx = context(tmp_path, monkeypatch, log)

    def admin(_state_dir: Path, *argv: str) -> dict[str, Any]:
        if "restore" in argv:
            raise RegionalFixtureError(
                "gpu-fault-admin submit-remediation exited 2: incident still has "
                "an open workflow workflow-1 (RUNNING)"
            )
        return {"argv": list(argv), "returncode": 0}

    ctx.admin = admin
    report = release.release_hold(ctx)

    assert report["released"] is False, report
    assert any(item.startswith("restore_disposition:") for item in report["errors"]), (
        report["errors"]
    )
    assert f"restore:{INCIDENT}:profile-v1" in log, "the owner restore still runs"


class KubeRegional:
    """A regional fixture whose GPU cluster holds one object, until deleted."""

    def __init__(self, document: dict[str, Any] | None) -> None:
        self.document = document
        self.settings = SimpleNamespace(namespace="gpu-fault-system")
        self.calls: list[tuple[str, ...]] = []

    def kubectl(self, *arguments: str, **_kwargs: Any) -> str:
        self.calls.append(arguments)
        if "get" in arguments:
            return json.dumps(self.document) if self.document is not None else ""
        if "delete" in arguments:
            self.document = None
            return ""
        raise AssertionError(arguments)


def pinned_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "pinned-workload.yaml"
    path.write_text(
        "kind: PyTorchJob\nmetadata:\n  name: gpu-fault-two-node-replace\n"
        "  namespace: gpu-fault-system\n",
        encoding="utf-8",
    )
    return path


def live_job(job_id: str) -> dict[str, Any]:
    return {
        "kind": "PyTorchJob",
        "metadata": {
            "name": "gpu-fault-two-node-replace",
            "namespace": "gpu-fault-system",
            "uid": "uid-job",
            "resourceVersion": "7",
            "labels": {
                "gpu-fault.io/job-id": job_id,
                "gpu-fault.io/attempt-id": "attempt-3",
                "gpu-fault.io/acceptance-owner": "nonce-of-the-run",
            },
        },
    }


def test_pinned_workload_deletes_the_drill_job_by_identity(tmp_path: Path) -> None:
    """Live 2026-09-24: two releases left `gpu-fault-two-node-replace` Suspended
    because the managed workload fixture only deletes what it submitted."""

    regional = KubeRegional(live_job("job-14"))
    record = release.PinnedWorkload(
        regional, pinned_manifest(tmp_path), "job-14"
    ).delete()

    assert record == {
        "kind": "pytorchjob",
        "name": "gpu-fault-two-node-replace",
        "uid": "uid-job",
        "attempt_id": "attempt-3",
        "deleted": True,
    }
    assert regional.document is None, "the live copy is gone"
    delete = next(call for call in regional.calls if "delete" in call)
    assert "/apis/kubeflow.org/v1/namespaces/gpu-fault-system/pytorchjobs/" in " ".join(
        delete
    ), delete


def test_pinned_workload_never_deletes_a_foreign_job(tmp_path: Path) -> None:
    regional = KubeRegional(live_job("someone-else"))
    with pytest.raises(RegionalFixtureError, match="foreign workload is never deleted"):
        release.PinnedWorkload(regional, pinned_manifest(tmp_path), "job-14").delete()
    assert regional.document is not None, "refused before any delete"


def test_pinned_workload_reports_an_absent_job_as_nothing_to_do(tmp_path: Path) -> None:
    regional = KubeRegional(None)
    record = release.PinnedWorkload(
        regional, pinned_manifest(tmp_path), "job-14"
    ).delete()
    assert record["deleted"] is False and record["absent"] is True, record


def runner_arguments(tmp_path: Path, *extra: str) -> list[str]:
    kube = tmp_path / "kube"
    kube.write_text("apiVersion: v1\n")
    return [
        "--run-dir",
        str(tmp_path / "run"),
        "--attempt",
        "3",
        "--cpu-kubeconfig",
        str(kube),
        "--gpu-kubeconfig",
        str(kube),
        "--gpu-context",
        "ctx",
        "--cluster-id",
        "cluster-a",
        "--region",
        "us-west-2",
        "--site-file",
        str(tmp_path / "state" / "site.yaml"),
        "--hyperpod-cluster",
        "hp",
        "--host-probe-image",
        "img",
        "--fault-node",
        FAULT,
        "--sibling-node",
        SIBLING,
        *extra,
    ]


def test_release_mode_needs_its_own_confirmation_and_no_plan_or_execute(
    tmp_path: Path,
) -> None:
    settings(tmp_path)
    arguments = destr014.parser().parse_args(runner_arguments(tmp_path, "--release"))
    assert arguments.release is True, arguments

    with pytest.raises(RegionalFixtureError, match="requires --confirm"):
        destr014.run_release(arguments)
    arguments = destr014.parser().parse_args(
        runner_arguments(
            tmp_path, "--release", "--plan", "--confirm", release.RELEASE_CONFIRMATION
        )
    )
    with pytest.raises(RegionalFixtureError, match="its own mode"):
        destr014.run_release(arguments)


def test_release_settings_do_not_need_the_device_identities(tmp_path: Path) -> None:
    settings(tmp_path)
    arguments = destr014.parser().parse_args(runner_arguments(tmp_path, "--release"))

    value = destr014.release_settings(arguments)

    assert value.fault_pci_bdf == "" and value.fault_device == ""
    assert value.fault_node == FAULT and value.sibling_node == SIBLING
    with pytest.raises(RegionalFixtureError, match="PCI BDF"):
        destr014.configure(arguments)


def test_main_routes_release_through_its_own_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        destr014.sys,
        "argv",
        [
            "runner",
            *runner_arguments(
                tmp_path, "--release", "--confirm", release.RELEASE_CONFIRMATION
            ),
        ],
    )
    monkeypatch.setattr(
        destr014, "install_site_profile", lambda: calls.append("profile")
    )
    monkeypatch.setattr(
        destr014, "install_abort_signals", lambda: calls.append("signals")
    )
    monkeypatch.setattr(
        destr014, "run_standard_case", lambda _case: calls.append("standard") or 9
    )

    def build(settings_value: Any, run_dir: Path, attempt: int) -> Any:
        calls.append(f"build:{run_dir.name}:{attempt}")
        return "context"

    monkeypatch.setattr(destr014, "build_release_context", build)
    monkeypatch.setattr(
        destr014.release,
        "release_hold",
        lambda ctx: calls.append(f"release:{ctx}") or {"released": True},
    )

    assert destr014.main() == 0
    assert calls == ["profile", "signals", "build:run:3", "release:context"], calls
