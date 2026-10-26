"""Every host probe a GF-GENERAL-LIVE variant opens is the context's to tear down.

Live 2026-10-26 (both clusters) the 024 composites PASSed and left their
environment-replay probe Pod running to its activeDeadline: the variant closed
the replay and discarded its undo (``_composite_flow``), the replay controller
had built its ``HostProbeFixture`` outside ``ctx.probes``, and the runner's
teardown listed only ``dcgm@<node>``. These tests open two probes on one node
through the real handles over an in-memory cluster and require the teardown to
leave zero Pods, zero open ownership records and a ``probe_residuals`` entry
per handle -- for every fixture that wraps a host probe, and for a handle a
later one replaced under the same key.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_fixture as fx
from scripts.e2e.regional import host_probe_fixture as hp
from scripts.e2e.regional import host_probe_sweep as sweep
from scripts.e2e.regional import run_general_live as runner
from scripts.e2e.regional.general_live_applog_fixture import PROBE_KEY as APPLOG_KEY
from scripts.e2e.regional.general_live_applog_fixture import cri_pod_logs
from scripts.e2e.regional.general_live_env_replay import PROBE_KEY as ENV_KEY
from scripts.e2e.regional.general_live_env_replay import env_replay
from scripts.e2e.regional.general_live_kernel_replay import (
    kernel_replay,
    reviewed_record,
)
from scripts.e2e.regional.general_live_registry import NATIVE, VariantSpec
from scripts.e2e.regional.general_live_source_replay import (
    HOST_UNIT,
    UNIT_HANDLES,
    host_source_replay,
)
from tests.regional._general_live_fakes import (
    FakeRegional,
    make_context,
    make_settings,
    store_script,
)
from tests.regional._host_probe_cluster import ProbeCluster

NODE = "ip-10-0-43-5.us-west-2.compute.internal"
GPU = "GPU-11111111-1111-4111-8111-111111111111"
PLAN = {
    "steps": [{"reads": 3, "values": {}}, {"reads": 2, "values": {"gpu_tlimit_c": 0.5}}]
}


def _spec(variant: str = "thermal_warning", **overrides: Any) -> VariantSpec:
    base: dict[str, Any] = dict(
        case_id="GF-GENERAL-LIVE-024",
        variant=variant,
        proof_mode=NATIVE,
        lanes=("eks",),
        risk="L",
        injection="i",
        expected="e",
        cleanup="c",
        max_minutes=10,
        execute=lambda ctx: {"verdict": "PASS", "stages": {}},
    )
    base.update(overrides)
    return VariantSpec(**base)


@pytest.fixture
def cluster(monkeypatch: pytest.MonkeyPatch) -> ProbeCluster:
    api = ProbeCluster()
    monkeypatch.setattr(hp, "run_fixture_command", api.run)
    monkeypatch.setattr(sweep, "run_fixture_command", api.run)
    return api


def _ctx(tmp_path: Path, spec: VariantSpec | None = None) -> fx.VariantContext:
    spec = spec or _spec()
    regional = FakeRegional(cpu_python=store_script({"activity": {}}))
    return make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={},
        settings=make_settings(spec, node=NODE),
    )


def _records(ctx: fx.VariantContext) -> dict[str, dict[str, Any]]:
    directory = ctx.case_dir / "host-probes"
    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.json"))
    }


def _open_env_replay(ctx: fx.VariantContext) -> None:
    """The 024 shape: the replay's probe is created and used; ``cleanup`` is
    never pushed (``deploy`` was not reached) or was discarded."""

    replay = env_replay(ctx, PLAN, GPU, 120)
    replay.controller.create()
    replay.controller.preflight()


def _open_cri_logs(ctx: fx.VariantContext) -> None:
    cri_pod_logs(ctx).preflight()
    ctx.cleanup.discard(APPLOG_KEY)


def _open_kernel_replay(ctx: fx.VariantContext) -> None:
    kernel_replay(ctx).install([reviewed_record("Xid 79")])
    ctx.cleanup.discard("kernel-replay-journal")


def _open_source_replay(ctx: fx.VariantContext) -> None:
    host_source_replay(ctx).install("meminfo", b"MemTotal: 1 kB\n")
    ctx.cleanup.discard(UNIT_HANDLES[HOST_UNIT])


OPENERS = {
    f"{ENV_KEY}@{NODE}": _open_env_replay,
    f"{APPLOG_KEY}@{NODE}": _open_cri_logs,
    f"kernel-replay-journal@{NODE}": _open_kernel_replay,
    f"{UNIT_HANDLES[HOST_UNIT]}@{NODE}": _open_source_replay,
}


def _execute(spec: VariantSpec, run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    regional = FakeRegional(cpu_python=store_script({"activity": {}}))
    monkeypatch.setattr(runner, "RegionalLiveFixture", lambda settings: regional)
    deadline = datetime.now(timezone.utc) + timedelta(hours=1)
    return runner.execute_variant(
        spec,
        make_settings(spec, node=NODE),
        run_dir,
        1,
        deadline,
        {"errors": [], "release_id": "r", "cluster_id": "c"},
    )


# --------------------------------------------------------------------------- #
# the leak
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("second", sorted(OPENERS))
def test_two_probes_on_one_node_leave_nothing_after_the_teardown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cluster: ProbeCluster, second: str
) -> None:
    """The dcgm probe and a wrapped probe on the same node, the wrapper's own
    cleanup never run: the teardown deletes both Pods and ConfigMaps, closes
    both records and names both handles with all-false residuals."""

    def variant(ctx: fx.VariantContext) -> dict[str, Any]:
        ctx.dcgm("survey", "--port", "9400")
        OPENERS[second](ctx)
        assert len(cluster.pods()) == 2, cluster.pods()
        return {"verdict": "PASS", "stages": {}}

    spec = _spec(execute=variant)
    result = _execute(spec, tmp_path, monkeypatch)
    assert result["verdict"] == "PASS", result
    assert cluster.pods() == [], cluster.pods()
    assert cluster.configmaps() == [], cluster.configmaps()
    assert cluster.host_scripts == {}, cluster.host_scripts
    case_dir = tmp_path / "cases" / spec.key
    records = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((case_dir / "host-probes").glob("*.json"))
    }
    assert len(records) == 2, sorted(records)
    assert all(record["closed"] is True for record in records.values()), records
    assert set(result["probe_residuals"]) == {f"dcgm@{NODE}", second}, result[
        "probe_residuals"
    ]
    for name, residual in result["probe_residuals"].items():
        assert all(value is False for value in residual.values()), (name, residual)
    assert result.get("errors", []) == []


def test_replacing_a_handle_cleans_it_first_and_the_teardown_lists_both(
    tmp_path: Path, cluster: ProbeCluster
) -> None:
    """A handle assigned over another under one key: the earlier probe is
    deleted at once (the two share one ownership record, so the later one
    could not even start otherwise) and both appear in the teardown's view."""

    ctx = _ctx(tmp_path)
    first = ctx.probe("node")
    first.execute("survey")
    assert len(cluster.pods()) == 1, cluster.pods()
    second = fx.ProbeHandle(
        ctx.regional,
        script=fx.NODE_PROBE,
        node=NODE,
        image=ctx.settings.host_probe_image,
        key=ctx.spec.key,
        run_id=ctx.run_id,
        case_dir=ctx.case_dir,
        deadline_seconds=600,
    )
    ctx.probes[f"node@{NODE}"] = second
    assert cluster.pods() == [], "the replaced handle's Pod is deleted at replacement"
    assert ctx.probes.replaced == {f"node@{NODE}#1": first}
    second.execute("survey")
    assert len(cluster.pods()) == 1, cluster.pods()
    listed = dict(ctx.probes.handles())
    assert listed == {f"node@{NODE}": second, f"node@{NODE}#1": first}
    residuals = {name: handle.cleanup() for name, handle in ctx.probes.handles()}
    assert cluster.pods() == [], cluster.pods()
    assert all(
        value is False for residual in residuals.values() for value in residual.values()
    ), residuals
    assert all(record["closed"] for record in _records(ctx).values()), _records(ctx)


def test_same_handle_reassigned_is_not_a_replacement(
    tmp_path: Path, cluster: ProbeCluster
) -> None:
    ctx = _ctx(tmp_path)
    handle = ctx.probe("node")
    handle.execute("survey")
    ctx.probes[f"node@{NODE}"] = handle
    assert ctx.probes.replaced == {}
    assert len(cluster.pods()) == 1, cluster.pods()
    handle.cleanup()


def test_a_wrapped_probe_cleaned_by_its_fixture_costs_the_teardown_no_api_call(
    tmp_path: Path, cluster: ProbeCluster
) -> None:
    """The LIFO cleanup of a fixture proves its probe clean; the teardown's
    second ``cleanup`` answers from that proof. A later ``execute`` re-creates
    the probe and the next cleanup is a real one again."""

    ctx = _ctx(tmp_path)
    logs = cri_pod_logs(ctx)
    logs.preflight()
    receipts = logs.cleanup()
    assert receipts["probe_residuals"] == {
        f"pod/{logs.host.fixture.pod}": False,
        f"configmap/{logs.host.fixture.configmap}": False,
        "host_script": False,
        "creation_unresolved": False,
    }
    handle = ctx.probes[f"{APPLOG_KEY}@{NODE}"]
    assert handle is logs.host
    before = len(cluster.calls)
    assert handle.cleanup() == receipts["probe_residuals"]
    assert len(cluster.calls) == before, "a proven-clean handle makes no API call"
    handle.execute("status")
    assert len(cluster.pods()) == 1, "the handle re-created its probe"
    assert handle.cleanup() == receipts["probe_residuals"]
    assert cluster.pods() == [], cluster.pods()


def test_host_probe_returns_the_registered_handle_and_keeps_keys_apart(
    tmp_path: Path, cluster: ProbeCluster
) -> None:
    ctx = _ctx(tmp_path)
    dcgm = ctx.probe("dcgm")
    assert ctx.probe("dcgm") is dcgm
    replay = env_replay(ctx, PLAN, GPU, 120)
    assert replay.controller.host is ctx.probes[f"{ENV_KEY}@{NODE}"]
    assert replay.controller.host is not dcgm
    other = ctx.host_probe("plugin", script=fx.NODE_PROBE, node="node-b")
    assert set(ctx.probes) == {f"dcgm@{NODE}", f"{ENV_KEY}@{NODE}", "plugin@node-b"}
    assert other.node == "node-b"
    assert other.fixture.settings.active_deadline_seconds == 10 * 60 + 1200
    # Distinct identities: distinct Pod names on the same node.
    assert dcgm.fixture.pod != replay.controller.host.fixture.pod


# --------------------------------------------------------------------------- #
# the sweep at variant start
# --------------------------------------------------------------------------- #
def test_a_rerun_after_a_dead_probe_sweeps_its_record_and_proceeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cluster: ProbeCluster
) -> None:
    """A probe of this case's earlier run died (Failed) with its record open:
    the rerun closes the record at its start, deletes the dead Pod and
    ConfigMap, records the sweep as evidence and runs the variant."""

    spec = _spec(
        execute=lambda ctx: {**ctx.dcgm("survey"), "verdict": "PASS", "stages": {}}
    )
    first = _execute(spec, tmp_path, monkeypatch)
    assert first["verdict"] == "PASS", first
    assert first["host_probe_sweep"]["records"] == 0
    case_dir = tmp_path / "cases" / spec.key
    # Reopen the first run's record as a dead run left it: open, the Pod
    # terminal under its activeDeadline, the ConfigMap present.
    [path] = sorted((case_dir / "host-probes").glob("*.json"))
    record = json.loads(path.read_text(encoding="utf-8"))
    record["closed"] = False
    record["script_may_exist"] = True
    path.write_text(json.dumps(record), encoding="utf-8")
    pod_name = record["resources"]["pod"]["name"]
    owner_labels = {
        "gpu-fault.io/acceptance-case": record["scope"]["case_id"],
        "gpu-fault.io/acceptance-run": record["scope"]["run_id"],
    }
    annotations = {
        "gpu-fault.io/probe-owner": record["owner"],
        "gpu-fault.io/probe-script-sha256": record["scope"]["script_sha256"],
    }
    cluster.objects[("pod", pod_name)] = {
        "metadata": {
            "name": pod_name,
            "uid": record["resources"]["pod"]["uid"],
            "labels": owner_labels,
            "annotations": {
                **annotations,
                "gpu-fault.io/probe-node-uid": record["node_uid"],
            },
        },
        "spec": {"nodeName": NODE},
        "status": {"phase": "Failed", "startTime": "2026-10-26T14:20:01Z"},
    }
    cluster.objects[("configmap", pod_name)] = {
        "metadata": {
            "name": pod_name,
            "uid": record["resources"]["configmap"]["uid"],
            "labels": owner_labels,
            "annotations": annotations,
        }
    }

    second = _execute(spec, tmp_path, monkeypatch)
    assert second["verdict"] == "PASS", second
    report = second["host_probe_sweep"]
    assert report["open"] == 1 and report["live"] == [] and report["skipped"] == [], (
        report
    )
    [swept] = report["swept"]
    assert swept["pod"] == pod_name and swept["pod_phase"] == "Failed", swept
    assert swept["deleted"] == [f"pod/{pod_name}", f"configmap/{pod_name}"]
    assert swept["host_script_left"] is True
    assert (case_dir / f"{sweep.SWEEP_EVIDENCE}.json").is_file(), "sweep evidence"
    assert cluster.pods() == [] and cluster.configmaps() == [], cluster.objects
    # The rerun's probe of the same identity adopted the swept record and
    # minted its own (a new owner), closed again by the rerun's teardown.
    closed = json.loads(path.read_text(encoding="utf-8"))
    assert closed["closed"] is True and closed["owner"] != record["owner"], closed
    assert hp.SWEPT_KEY not in closed, closed
    evidence = json.loads(
        (case_dir / f"{sweep.SWEEP_EVIDENCE}.json").read_text(encoding="utf-8")
    )
    assert evidence["swept"] == report["swept"], evidence


def test_a_live_probe_of_an_earlier_attempt_is_reported_not_swept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cluster: ProbeCluster
) -> None:
    """The record's Pod still runs: real residue. The sweep leaves the record
    and the Pod alone and reports it; the ownership refusal is unchanged."""

    spec = _spec(
        execute=lambda ctx: {**ctx.dcgm("survey"), "verdict": "PASS", "stages": {}}
    )
    ctx = _ctx(tmp_path, spec)
    handle = ctx.probe("dcgm")
    handle.execute("survey")
    # The earlier attempt's process is gone but its Pod runs on: the record
    # stays open on disk while the handle (and its lock) are released.
    monkeypatch.setattr(hp, "POD_DELETE_WAIT_SECONDS", 0)
    monkeypatch.setattr(hp, "POD_FORCE_DELETE_WAIT_SECONDS", 0)
    monkeypatch.setattr(hp.time, "sleep", lambda _seconds: None)
    cluster.stuck.add(("pod", handle.fixture.pod))
    with pytest.raises(hp.HostProbeError, match="cleanup is incomplete"):
        handle.cleanup()
    cluster.stuck.clear()
    assert cluster.pods() == [handle.fixture.pod]

    result = _execute(spec, tmp_path, monkeypatch)
    report = result["host_probe_sweep"]
    assert report["swept"] == [] and report["skipped"] == [], report
    [live] = report["live"]
    assert live["pod"] == handle.fixture.pod and live["phase"] == "Running", live
    # The rerun (its own run id, its own probe) passes; the earlier attempt's
    # running Pod and open record are left exactly as they were.
    assert result["verdict"] == "PASS", result
    assert cluster.pods() == [handle.fixture.pod], cluster.pods()
    record = json.loads(handle.fixture.state_path.read_text(encoding="utf-8"))
    assert record["closed"] is False, record
