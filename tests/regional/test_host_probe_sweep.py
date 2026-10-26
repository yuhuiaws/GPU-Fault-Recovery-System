"""The dead-record sweep of ``host_probe_sweep`` and what it unblocks.

A host probe's ownership record left open by a run that never reached its
cleanup (live 2026-10-26: the 024 composites' discarded undo, 022
``recovery_evidence``'s ENOSPC in the teardown) refused the next run of the
same case with ``host probe ownership record does not match this run`` once
the probe script had changed. The sweep closes such a record when its Pod is
gone or terminal, under the fixture's own ownership proof, and the next probe
of the same identity overwrites the script copy the dead one left; a record
whose Pod still runs, a Pod that is not the record's, a record another process
holds -- all stay as they are.
"""

from __future__ import annotations

import errno
import fcntl
import json
import os
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import host_probe_fixture as hp
from scripts.e2e.regional.host_probe_sweep import (
    foreign_reason,
    sweep_dead_host_probe_records,
)
from tests.regional._host_probe_cluster import ProbeCluster
from tests.regional._host_probe_support import host_probe

CONTEXT = "gpu-context"
NAMESPACE = "gpu-fault-system"


@pytest.fixture
def cluster(monkeypatch: pytest.MonkeyPatch) -> ProbeCluster:
    api = ProbeCluster()
    monkeypatch.setattr(hp, "run_fixture_command", api.run)
    monkeypatch.setattr(hp, "POD_DELETE_WAIT_SECONDS", 0)
    monkeypatch.setattr(hp, "POD_FORCE_DELETE_WAIT_SECONDS", 0)
    monkeypatch.setattr(hp.time, "sleep", lambda _seconds: None)
    return api


def _dead_run(
    tmp_path: Path, cluster: ProbeCluster, *, phase: str = "Failed"
) -> hp.HostProbeFixture:
    """A probe whose run ended without a cleanup: record open, Pod ``phase``.

    The owner's lock is released the way a dead process releases it -- here
    through a cleanup that failed at its first step (the host-script removal)
    and so left the Pod, the ConfigMap and the script copy on the node.
    """

    probe = host_probe(tmp_path)
    probe.create()
    probe.execute("snapshot")
    cluster.cleanup_error = True
    with pytest.raises(hp.HostProbeError, match="command failed"):
        probe.cleanup()
    cluster.cleanup_error = False
    cluster.set_phase(probe.pod, phase)
    record = _record(probe)
    assert record["closed"] is False and record["script_may_exist"] is True, record
    return probe


def _record(probe: hp.HostProbeFixture) -> dict[str, Any]:
    return json.loads(probe.state_path.read_text(encoding="utf-8"))


def _sweep(tmp_path: Path, cluster: ProbeCluster, **overrides: Any) -> dict[str, Any]:
    arguments: dict[str, Any] = dict(
        kubeconfig=tmp_path / "gpu.kubeconfig",
        context=CONTEXT,
        namespace=NAMESPACE,
        run=cluster.run,
        sleep=lambda _seconds: None,
    )
    arguments.update(overrides)
    return sweep_dead_host_probe_records(tmp_path / "host-probes", **arguments)


def _changed_script(tmp_path: Path) -> hp.HostProbeFixture:
    """The same probe identity (same script name) with a changed script: the
    rerun under a later harness."""

    script = tmp_path / "v2" / "probe.py"
    script.parent.mkdir(exist_ok=True)
    script.write_text('print(\'{"ok": true, "v": 2}\')\n', encoding="utf-8")
    return host_probe(tmp_path, probe_script=script)


def test_dead_pod_is_deleted_record_closed_and_the_rerun_proceeds(
    tmp_path: Path, cluster: ProbeCluster
) -> None:
    dead = _dead_run(tmp_path, cluster)
    old_digest = _record(dead)["scope"]["script_sha256"]
    rerun = _changed_script(tmp_path)
    assert rerun.pod == dead.pod, "same identity, same Pod name"
    with pytest.raises(hp.HostProbeError, match="does not match this run"):
        rerun.create()

    report = _sweep(tmp_path, cluster)
    assert report["records"] == 1 and report["open"] == 1, report
    [swept] = report["swept"]
    assert swept == {
        "record": dead.pod,
        "pod": dead.pod,
        "pod_phase": "Failed",
        "node": "node-a",
        "deleted": [f"pod/{dead.pod}", f"configmap/{dead.configmap}"],
        "host_script_left": True,
    }
    assert cluster.pods() == [] and cluster.configmaps() == [], cluster.objects
    record = _record(dead)
    assert record["closed"] is True
    assert record["script_may_exist"] is False
    assert record[hp.STALE_HOST_SCRIPT_KEY] == old_digest
    assert record[hp.SWEPT_KEY]["pod_phase"] == "Failed"
    assert record[hp.SWEPT_KEY]["host_script_left_sha256"] == old_digest
    assert record[hp.SWEPT_KEY]["deleted"] == swept["deleted"]

    # The dead probe's copy is still on the node; the rerun's first call names
    # it as the one stale copy it may overwrite, and the rerun runs clean.
    assert cluster.host_scripts == {f"/host{dead.host_script}": old_digest}
    rerun.create()
    minted = _record(rerun)
    assert minted["closed"] is False and minted[hp.STALE_HOST_SCRIPT_KEY] == old_digest
    assert minted["owner"] != record["owner"], "a fresh record, not the dead run's"
    assert rerun.execute("snapshot") == {"ok": True}, (
        "the install step accepted the copy"
    )
    # The rerun's call (the last exec; the dead run's own calls precede it).
    call = [c for c in cluster.exec_calls() if "host-probe" in c][-1]
    index = call.index("host-probe")
    assert call[index + 3 : index + 5] == [minted["scope"]["script_sha256"], old_digest]
    assert hp.STALE_HOST_SCRIPT_KEY not in _record(rerun), "overwritten: marker dropped"
    assert cluster.host_scripts == {
        f"/host{rerun.host_script}": minted["scope"]["script_sha256"]
    }
    assert not any(rerun.cleanup().values()), "rerun clean"
    assert cluster.host_scripts == {}
    assert _record(rerun)["closed"] is True


def test_without_the_stale_digest_a_foreign_copy_is_still_rejected(
    tmp_path: Path, cluster: ProbeCluster
) -> None:
    """The tolerance is for the one digest the swept record names, nothing else."""

    probe = host_probe(tmp_path)
    cluster.host_scripts[f"/host{probe.host_script}"] = "f" * 64
    probe.create()
    with pytest.raises(hp.HostProbeError, match=r"host probe failed \(exit 1\)"):
        probe.execute("snapshot")
    [call] = [c for c in cluster.exec_calls() if "host-probe" in c]
    assert call[call.index("host-probe") + 4] == "-"


def test_a_running_pod_is_reported_and_its_record_and_refusal_stay(
    tmp_path: Path, cluster: ProbeCluster
) -> None:
    dead = _dead_run(tmp_path, cluster, phase="Running")
    report = _sweep(tmp_path, cluster)
    assert report["swept"] == [] and report["skipped"] == [], report
    assert report["live"] == [
        {
            "record": dead.pod,
            "pod": dead.pod,
            "phase": "Running",
            "node": "node-a",
            "started_at": "2026-10-26T21:26:36Z",
        }
    ]
    assert cluster.pods() == [dead.pod]
    assert _record(dead)["closed"] is False
    with pytest.raises(hp.HostProbeError, match="does not match this run"):
        _changed_script(tmp_path).create()


def test_a_pending_pod_is_live_too(tmp_path: Path, cluster: ProbeCluster) -> None:
    _dead_run(tmp_path, cluster, phase="Pending")
    report = _sweep(tmp_path, cluster)
    assert len(report["live"]) == 1 and report["swept"] == [], report


def test_an_absent_pod_with_its_configmap_left_is_swept(
    tmp_path: Path, cluster: ProbeCluster
) -> None:
    dead = _dead_run(tmp_path, cluster)
    cluster.objects.pop(("pod", dead.pod))
    report = _sweep(tmp_path, cluster)
    [swept] = report["swept"]
    assert swept["pod_phase"] == "absent"
    assert swept["deleted"] == [f"configmap/{dead.configmap}"]
    assert cluster.configmaps() == [], cluster.configmaps()
    assert _record(dead)["closed"] is True


def test_a_terminal_pod_that_is_not_the_records_is_left_alone(
    tmp_path: Path, cluster: ProbeCluster
) -> None:
    dead = _dead_run(tmp_path, cluster)
    pod = cluster.objects[("pod", dead.pod)]
    pod["metadata"]["annotations"]["gpu-fault.io/probe-owner"] = "another-run"
    report = _sweep(tmp_path, cluster)
    assert report["swept"] == [] and report["live"] == [], report
    assert report["skipped"] == [
        {
            "record": dead.pod,
            "reason": "terminal Pod is not this record's: owner annotation differs",
        }
    ]
    assert cluster.pods() == [dead.pod]
    assert _record(dead)["closed"] is False


def test_a_configmap_that_is_not_the_records_is_left_alone(
    tmp_path: Path, cluster: ProbeCluster
) -> None:
    dead = _dead_run(tmp_path, cluster)
    cluster.objects[("configmap", dead.configmap)]["metadata"]["uid"] = "other"
    report = _sweep(tmp_path, cluster)
    [skipped] = report["skipped"]
    assert (
        skipped["reason"]
        == "ConfigMap is not this record's: UID differs from the record"
    )
    assert cluster.pods() == [], "the dead Pod itself was deleted first"
    assert cluster.configmaps() == [dead.configmap]
    assert _record(dead)["closed"] is False


def test_a_pod_whose_delete_is_not_acknowledged_keeps_the_record_open(
    tmp_path: Path, cluster: ProbeCluster
) -> None:
    dead = _dead_run(tmp_path, cluster)
    cluster.stuck.add(("pod", dead.pod))
    ticks = iter(range(0, 10_000, 30))
    report = _sweep(tmp_path, cluster, clock=lambda: float(next(ticks)))
    assert report["skipped"] == [
        {"record": dead.pod, "reason": "Pod deletion was not acknowledged"}
    ]
    assert _record(dead)["closed"] is False


def test_a_record_another_process_holds_is_skipped(
    tmp_path: Path, cluster: ProbeCluster
) -> None:
    dead = _dead_run(tmp_path, cluster)
    holder = os.open(dead.state_path.with_suffix(".lock"), os.O_RDWR)
    try:
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = _sweep(tmp_path, cluster)
    finally:
        os.close(holder)
    assert report["skipped"] == [
        {"record": dead.pod, "reason": "another process owns this host probe"}
    ]
    assert cluster.pods() == [dead.pod]
    assert _record(dead)["closed"] is False


def test_records_of_another_context_closed_or_unreadable_are_not_touched(
    tmp_path: Path, cluster: ProbeCluster
) -> None:
    dead = _dead_run(tmp_path, cluster)
    directory = tmp_path / "host-probes"
    (directory / "gpu-fault-host-probe-broken.json").write_text("{", encoding="utf-8")
    closed = dict(_record(dead))
    closed["closed"] = True
    (directory / "gpu-fault-host-probe-closed.json").write_text(
        json.dumps(closed), encoding="utf-8"
    )
    report = _sweep(tmp_path, cluster, context="another-context")
    assert report["records"] == 3 and report["open"] == 1, report
    assert report["swept"] == [] and report["live"] == [], report
    assert sorted(report["skipped"], key=lambda entry: entry["reason"]) == [
        {
            "record": dead.pod,
            "reason": "record belongs to another cluster context or namespace",
        },
        {
            "record": "gpu-fault-host-probe-broken",
            "reason": "unreadable: JSONDecodeError",
        },
    ]
    assert cluster.pods() == [dead.pod]
    assert _record(dead)["closed"] is False
    assert _sweep(tmp_path / "missing", cluster)["records"] == 0


def test_foreign_reason_names_every_divergence() -> None:
    record = {
        "owner": "o",
        "node_uid": "n",
        "scope": {"case_id": "c", "run_id": "r", "script_sha256": "s", "node": "node"},
        "resources": {"pod": {"uid": "p"}, "configmap": {"uid": None}},
    }
    owned = {
        "metadata": {
            "uid": "p",
            "labels": {
                "gpu-fault.io/acceptance-case": "c",
                "gpu-fault.io/acceptance-run": "r",
            },
            "annotations": {
                "gpu-fault.io/probe-owner": "o",
                "gpu-fault.io/probe-script-sha256": "s",
                "gpu-fault.io/probe-node-uid": "n",
            },
        },
        "spec": {"nodeName": "node"},
    }
    assert foreign_reason(owned, "pod", record) == ""
    assert foreign_reason({"metadata": {}}, "pod", record) == "no UID"
    variants = {
        "case/run labels differ": ("labels", "gpu-fault.io/acceptance-run", "x"),
        "owner annotation differs": ("annotations", "gpu-fault.io/probe-owner", "x"),
        "script digest differs": (
            "annotations",
            "gpu-fault.io/probe-script-sha256",
            "x",
        ),
        "node UID differs": ("annotations", "gpu-fault.io/probe-node-uid", "x"),
    }
    for reason, (section, key, value) in variants.items():
        changed = json.loads(json.dumps(owned))
        changed["metadata"][section][key] = value
        assert foreign_reason(changed, "pod", record) == reason
    moved = json.loads(json.dumps(owned))
    moved["spec"]["nodeName"] = "elsewhere"
    assert foreign_reason(moved, "pod", record) == "node differs"
    other_uid = json.loads(json.dumps(owned))
    other_uid["metadata"]["uid"] = "q"
    assert foreign_reason(other_uid, "pod", record) == "UID differs from the record"
    # A ConfigMap carries no node; its UID is unrecorded here, so any UID is accepted.
    assert foreign_reason(other_uid, "configmap", record) == ""


# --------------------------------------------------------------------------- #
# the fixture's own part: a cleanup that cannot save its record still deletes
# --------------------------------------------------------------------------- #
def test_cleanup_out_of_disk_still_deletes_then_fails_and_the_sweep_closes_it(
    tmp_path: Path, cluster: ProbeCluster, monkeypatch: pytest.MonkeyPatch
) -> None:
    """022 recovery_evidence, 2026-10-26 14:24Z: ENOSPC on the runner host in
    the teardown. The record save fails; the Pod, ConfigMap and host script
    are deleted anyway and the failure is raised afterwards. The record stays
    open on disk, and the next start's sweep closes it (Pod absent)."""

    probe = host_probe(tmp_path)
    probe.create()
    probe.execute("snapshot")
    saved = hp.write_json_atomic

    def out_of_disk(path: Path, value: Any) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(hp, "write_json_atomic", out_of_disk)
    with pytest.raises(hp.HostProbeError) as failure:
        probe.cleanup()
    assert str(failure.value) == (
        "host probe ownership record could not be saved (No space left on "
        "device); the probe resources were deleted"
    )
    assert cluster.pods() == [] and cluster.configmaps() == [], cluster.objects
    assert cluster.host_scripts == {}
    record = _record(probe)
    assert record["closed"] is False and record["script_may_exist"] is True, record

    monkeypatch.setattr(hp, "write_json_atomic", saved)
    report = _sweep(tmp_path, cluster)
    [swept] = report["swept"]
    assert swept["pod_phase"] == "absent" and swept["deleted"] == [], swept
    assert _record(probe)["closed"] is True
    # Outside a cleanup the save failure is the caller's error, as before.
    monkeypatch.setattr(hp, "write_json_atomic", out_of_disk)
    fresh = host_probe(tmp_path, run_id="run-b")
    with pytest.raises(OSError):
        fresh.create()
