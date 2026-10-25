"""Coverage of the GF-GENERAL-LIVE fixture, common helpers and plan/execute runner."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_common as common
from scripts.e2e.regional import general_live_fixture as fx
from scripts.e2e.regional.general_live_registry import (
    NATIVE,
    VariantSpec,
    deferred_variant,
)
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    make_context,
    make_settings,
    store_script,
)

NODE = "ip-10-0-43-5.us-west-2.compute.internal"


def _spec(variant="cpu_busy", **kw):
    base = dict(
        case_id="GF-GENERAL-LIVE-001",
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
    base.update(kw)
    return VariantSpec(**base)


# --------------------------------------------------------------------------- #
# CleanupStack and Evidence
# --------------------------------------------------------------------------- #
def test_cleanup_stack_lifo_discard_and_failure() -> None:
    order: list[str] = []
    stack = fx.CleanupStack()
    stack.push("a", lambda: order.append("a"))
    stack.push("b", lambda: order.append("b"))
    stack.push("c", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    stack.discard("a")
    assert stack.pending() == ["b", "c"]
    stack.run()
    assert order == ["b"]
    assert stack.failed and stack.failed[0]["name"] == "c"
    assert stack.pending() == []


def test_cleanup_stack_reraises_base_exception() -> None:
    stack = fx.CleanupStack()

    class Abort(BaseException):
        pass

    stack.push("x", lambda: (_ for _ in ()).throw(Abort()))
    with pytest.raises(Abort):
        stack.run()


def test_evidence_writes(tmp_path: Path) -> None:
    ev = fx.Evidence(tmp_path / "cases" / "x")
    ev.write("doc", {"a": 1})
    ev.append_oracle([{"t": 1}], source="cpu")
    assert (ev.case_dir / "doc.json").is_file(), "(ev.case_dir / 'doc.json').is_file()"
    assert (ev.case_dir / "oracle.jsonl").read_text().strip(), (
        "(ev.case_dir / 'oracle.jsonl').read_text().strip()"
    )


# --------------------------------------------------------------------------- #
# configure / node_provider / channel_errors
# --------------------------------------------------------------------------- #
def test_configure_requires_digest_image(tmp_path: Path) -> None:
    import argparse

    args = argparse.Namespace(
        case="GF-GENERAL-LIVE-001",
        variant="cpu_busy",
        approve=[],
        cpu_kubeconfig="",
        gpu_kubeconfig="",
        gpu_context="ctx",
        namespace="gpu-fault-system",
        cluster_id="c",
        region="us-west-2",
        fault_node=NODE,
        node_b="",
        host_probe_image="public.ecr.aws/x:latest",
        hyperpod_cluster="",
        site_file="",
    )
    import os

    kubeconfig = tmp_path / "kc"
    kubeconfig.write_text("apiVersion: v1\n")
    os.environ["GPU_FAULT_CONTROL_KUBECONFIG"] = str(kubeconfig)
    os.environ["KUBECONFIG"] = str(kubeconfig)
    with pytest.raises(Exception, match="immutable digest"):
        fx.configure(args)


def test_node_provider_and_channel_errors() -> None:
    assert (
        fx.node_provider({"sagemaker.amazonaws.com/compute-type": "hyperpod"}) == "hp"
    )
    assert fx.node_provider({"eks.amazonaws.com/nodegroup": "g"}) == "eks"
    assert fx.node_provider({}) is None
    now = datetime(2026, 10, 24, tzinfo=timezone.utc)
    fresh = now.isoformat()
    stale = (now - timedelta(seconds=9999)).isoformat()
    statuses = [
        {
            "collector": "HOST_TELEMETRY",
            "last_success_at": fresh,
            "last_error_at": None,
        },
        {"collector": "GPU_INVENTORY", "last_success_at": stale, "last_error_at": None},
        {"collector": "GPU_METRICS", "last_success_at": fresh, "last_error_at": fresh},
    ]
    errors = fx.channel_errors(statuses, now)
    assert any("GPU_INVENTORY" in e for e in errors) and any(
        "GPU_METRICS" in e for e in errors
    )
    assert fx.channel_errors([], now), "fx.channel_errors([], now)"  # all missing


# --------------------------------------------------------------------------- #
# read_only_preflight
# --------------------------------------------------------------------------- #
def _preflight_regional(**kw) -> FakeRegional:
    base = dict(
        node_snapshot={
            "name": NODE,
            "uid": "u",
            "boot_id": "b",
            "ready": "True",
            "unschedulable": False,
            "taints": [],
            "ownership_annotations": {},
            "spare_pool_annotations": {},
            "gpu_allocatable": "8",
        },
        node_metadata={"name": NODE, "labels": {"eks.amazonaws.com/nodegroup": "g"}},
        business=[],
        gpu_pods=[],
        cpu_python=store_script(
            {
                "preflight": {
                    "agent": {"lifecycle_state": "ACTIVE"},
                    "open_incidents": [],
                    "active_workflows": [],
                    "collector_statuses": _fresh_statuses(),
                }
            }
        ),
    )
    base.update(kw)
    return FakeRegional(**base)


def _fresh_statuses() -> list[dict[str, Any]]:
    fresh = datetime.now(timezone.utc).isoformat()
    return [
        {"collector": c, "last_success_at": fresh, "last_error_at": None}
        for c in ("HOST_TELEMETRY", "GPU_INVENTORY", "GPU_METRICS")
    ]


def test_read_only_preflight_passes() -> None:
    spec = _spec()
    settings = make_settings(spec, node=NODE)
    result = fx.read_only_preflight(settings, spec, _preflight_regional())
    assert result["errors"] == [], result["errors"]


def test_read_only_preflight_deferred_and_unauthorized() -> None:
    deferred = deferred_variant(
        "GF-GENERAL-LIVE-001",
        "invalid_delta",
        proof_mode="live-replay",
        reason="FIXTURE_NOT_READY: x",
    )
    settings = make_settings(deferred, node=NODE)
    result = fx.read_only_preflight(settings, deferred, _preflight_regional())
    assert result["planned_verdict"].startswith("NOT_RUN"), (
        "result['planned_verdict'].startswith('NOT_RUN')"
    )
    gated = _spec(variant="cpu_busy", approvals=("HOST_MEMORY_PRESSURE",))
    settings = make_settings(gated, node=NODE)
    settings = type(settings)(**{**settings.__dict__, "approvals": ()})
    result = fx.read_only_preflight(settings, gated, _preflight_regional())
    assert result["planned_verdict"] == "NOT_RUN(NOT_AUTHORIZED)"


def test_read_only_preflight_flags_every_prerequisite() -> None:
    spec = _spec()
    settings = make_settings(spec, node=NODE)
    regional = _preflight_regional(
        node_snapshot={
            "name": NODE,
            "uid": "u",
            "ready": "False",
            "unschedulable": True,
            "taints": [{"key": "gpu-fault.io/quarantined"}],
            "ownership_annotations": {"gpu-fault.io/incident-id": "i"},
            "spare_pool_annotations": {},
            "gpu_allocatable": "8",
        },
        business=[{"name": "biz"}],
        gpu_pods=[{"node": NODE, "name": "g"}],
        cpu_python=store_script(
            {
                "preflight": {
                    "agent": {"lifecycle_state": "REVOKED"},
                    "open_incidents": [{"incident_id": "i"}],
                    "active_workflows": [{"request_id": "w"}],
                    "collector_statuses": [],
                }
            }
        ),
    )
    result = fx.read_only_preflight(settings, spec, regional)
    joined = " ".join(result["errors"])
    for needle in (
        "not Ready",
        "cordoned",
        "ownership",
        "taints",
        "business workload",
        "GPU-requesting",
        "Agent is not ACTIVE",
        "open incidents",
        "active workflows",
    ):
        assert needle in joined, needle


def test_read_only_preflight_hyperpod_node_recovery() -> None:
    spec = _spec(lanes=("hp",))
    settings = make_settings(spec, node=NODE)
    settings = type(settings)(**{**settings.__dict__, "hyperpod_cluster": "hp-cluster"})
    regional = _preflight_regional(
        node_metadata={
            "name": NODE,
            "labels": {"sagemaker.amazonaws.com/compute-type": "hyperpod"},
        },
        run_output=lambda cmd: '"Automatic"',
    )
    result = fx.read_only_preflight(settings, spec, regional)
    assert any("NodeRecovery" in e for e in result["errors"]), (
        "any('NodeRecovery' in e for e in result['errors'])"
    )


# --------------------------------------------------------------------------- #
# ProbeHandle, VariantContext helpers, backing_root, require_*
# --------------------------------------------------------------------------- #
def test_probe_handle_lazy_create(monkeypatch: pytest.MonkeyPatch) -> None:
    created = {"n": 0}

    class FakeHostProbe:
        def __init__(self, settings) -> None:
            pass

        def create(self) -> None:
            created["n"] += 1

        def execute(self, *a, **k) -> dict[str, Any]:
            return {"args": a}

        def cleanup(self) -> dict[str, bool]:
            return {"pod": False}

        def residuals(self) -> dict[str, bool]:
            return {}

    monkeypatch.setattr(fx, "HostProbeFixture", FakeHostProbe)
    handle = fx.ProbeHandle(
        FakeRegional(),
        script=fx.NODE_PROBE,
        node=NODE,
        image="img",
        key="k",
        run_id="r",
        case_dir=Path("/tmp/pt-gl1/ph"),
        deadline_seconds=600,
    )
    assert handle.cleanup() == {}
    handle.execute("survey")
    assert created["n"] == 1
    handle.execute("again")
    assert created["n"] == 1  # reused


def test_variant_context_helpers(tmp_path: Path) -> None:
    spec = _spec()
    regional = FakeRegional(cpu_python=store_script({"activity": {}}))
    ctx = make_context(spec, tmp_path=tmp_path, regional=regional, probes={})
    ctx.survey = {
        "collector_env": {
            "GPU_FAULT_HOST_INTERVAL_SECONDS": "20",
            "GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS": "bad",
        }
    }
    assert ctx.host_interval == 20
    assert ctx.host_summary == 300  # bad value falls back to the default
    assert ctx.remaining_seconds() > 0
    polls = []
    ctx.hold(2, name="h", check=lambda: polls.append(1))
    assert polls, "polls"
    value = ctx.wait_until(lambda: None, timeout_seconds=1, poll_seconds=0.1, name="w")
    assert value is None


def test_backing_root_and_requires() -> None:
    survey = {
        "backing": [
            {
                "root": "/var/lib/gpu-fault-acceptance",
                "mount": {"fstype": "ext4"},
                "statvfs": {"available_bytes": 400 * 1024**3, "used_percent": 20.0},
            },
            {
                "root": "/opt/dlami/nvme/gpu-fault-acceptance",
                "mount": {"fstype": "tmpfs"},
                "statvfs": {"available_bytes": 999 * 1024**3, "used_percent": 1.0},
            },
        ]
    }
    assert fx.backing_root(survey) == "/var/lib/gpu-fault-acceptance"
    with pytest.raises(fx.VariantNotRun):
        fx.backing_root({"backing": []})
    with pytest.raises(fx.VariantNotRun):
        fx.require_tools({"tools": {"ip": True}}, "ip", "tc")
    fx.require_tools({"tools": {"ip": True}}, "ip")
    assert (
        fx.require_module({"modules": {"veth": {"loaded": True}}}, "veth", ()) is False
    )
    with pytest.raises(fx.VariantNotRun, match="not available"):
        fx.require_module(
            {"modules": {"dm_delay": {"loaded": False, "available": False}}},
            "dm_delay",
            (),
        )
    with pytest.raises(fx.VariantNotRun, match="KERNEL_MODULE_LOAD"):
        fx.require_module(
            {"modules": {"dm_delay": {"loaded": False, "available": True}}},
            "dm_delay",
            (),
        )
    assert (
        fx.require_module(
            {"modules": {"dm_delay": {"loaded": False, "available": True}}},
            "dm_delay",
            ("KERNEL_MODULE_LOAD",),
        )
        is True
    )


# --------------------------------------------------------------------------- #
# common: survey, oracle, window, restore, misc
# --------------------------------------------------------------------------- #
def _ctx_with_node(
    tmp_path: Path, node_script: dict[str, Any], activity_doc: Any = None
) -> Any:
    spec = _spec()
    regional = FakeRegional(cpu_python=store_script({"activity": activity_doc or {}}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"node@{NODE}": FakeProbe(node_script)},
    )
    return ctx


def test_run_survey_rejects_residue(tmp_path: Path) -> None:
    ctx = _ctx_with_node(tmp_path, {"survey": {"residue": {"paused_units": {"u": {}}}}})
    with pytest.raises(fx.VariantNotRun, match="residue"):
        common.run_survey(ctx)
    ctx = _ctx_with_node(
        tmp_path,
        {
            "survey": {
                "cpu_count": 8,
                "residue": {"units": [], "paused_units": {}, "paused_processes": {}},
            }
        },
    )
    assert common.run_survey(ctx)["cpu_count"] == 8


def test_oracle_lifecycle(tmp_path: Path) -> None:
    node = {
        "oracle-start": {"unit": "o"},
        "oracle-read": [
            {"records": [{"cpu_busy_percent": 99.0}], "next_offset": 1, "total": 1},
            {"records": [], "next_offset": 1, "total": 1},
        ],
        "oracle-stop": {"stopped": {}},
    }
    ctx = _ctx_with_node(tmp_path, node)
    oracle = common.Oracle(ctx, "cpu")
    oracle.start(
        60,
        cpu=True,
        diskstats=["dm-7"],
        netdev=["gfrsaaaaaaa"],
        statvfs=["/mnt"],
        snmp=True,
        loadavg=True,
        meminfo=True,
    )
    fresh = oracle.read()
    assert fresh and oracle.records
    oracle.stop()
    assert "cpu-load" not in ctx.cleanup.pending()


def test_window_open_close(tmp_path: Path) -> None:
    _ = {}
    spec = _spec()
    regional = FakeRegional(cpu_python=store_script({"activity": {}}))
    win = FakeProbe(
        {
            "open-window": {"after": {"ActiveState": "active"}},
            "close-window": {"after": {"ActiveState": "active"}},
        }
    )
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes={f"window@{NODE}": win}
    )
    common.open_window(
        ctx, "GPU_FAULT_FILESYSTEMS=/mnt/gfrs-live/x/fs", restore_seconds=600
    )
    common.close_window(ctx)
    bad = FakeProbe(
        {
            "open-window": {"after": {"ActiveState": "active"}},
            "close-window": {"after": {"ActiveState": "inactive"}},
        }
    )
    ctx2 = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes={f"window@{NODE}": bad}
    )
    common.open_window(
        ctx2, "GPU_FAULT_REQUIRED_INTERFACES=gfrsaaaaaaa", restore_seconds=600
    )
    with pytest.raises(Exception, match="not active"):
        common.close_window(ctx2)


def test_restore_incidents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSpare:
        def __init__(self, regional, hp) -> None:
            pass

        def wait_incident_idle(self, incident_id, **k):
            return {"incident": {"state": "ACTION_PENDING"}}

        def create_restore_workflow(self, **k):
            return {"workflow_request_id": "wf-restore"}

        def wait_workflow_id(self, wf, **k):
            return {
                "status": "SUCCEEDED",
                "official_steps": [
                    {"operation": "VALIDATE_GPU"},
                    {"operation": "RESTORE_SCHEDULING"},
                ],
                "step_executions": [],
            }

    monkeypatch.setattr(common, "WarmSpareLiveFixture", FakeSpare)
    spec = _spec()
    regional = FakeRegional(
        cpu_python=store_script(
            {"preflight": {"agent": {"runtime_profile_version": "eks-v1"}}}
        )
    )
    regional.node_snapshot_value = {
        "name": NODE,
        "uid": "u",
        "ready": "True",
        "unschedulable": False,
        "taints": [],
        "ownership_annotations": {},
        "spare_pool_annotations": {},
    }
    ctx = make_context(spec, tmp_path=tmp_path, regional=regional, probes={})
    ctx.survey = {"agent": {"runtime_profile_version": "eks-v1"}}
    results = common.restore_incidents(ctx, [{"incident_id": "inc-1"}], reason="r")
    assert results == [] or results[0]["status"] == "SUCCEEDED"


def test_open_incident_ids_and_unsettled() -> None:
    doc = {
        "incidents": [
            {"incident_id": "a", "state": "ACTION_PENDING"},
            {"incident_id": "b", "state": "RECOVERED"},
        ],
        "workflows": [
            {"request_id": "w", "status": "RUNNING"},
            {"request_id": "x", "status": "SUCCEEDED"},
        ],
    }
    assert common.open_incident_ids(doc) == ["a"]
    assert common.unsettled_workflows(doc) == ["w"]


# --------------------------------------------------------------------------- #
# run_general_live execute_variant
# --------------------------------------------------------------------------- #


def _patch_runner(monkeypatch, store_doc=None):
    from scripts.e2e.regional import run_general_live as runner

    def factory(regional_settings):
        return FakeRegional(cpu_python=store_script({"activity": store_doc or {}}))

    monkeypatch.setattr(runner, "RegionalLiveFixture", factory)
    return runner


def test_execute_variant_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _patch_runner(monkeypatch)

    spec = _spec(execute=lambda ctx: {"verdict": "PASS", "stages": {}})
    settings = make_settings(spec, node=NODE)
    deadline = datetime.now(timezone.utc) + timedelta(hours=1)
    regional_ok = {"errors": [], "release_id": "r", "cluster_id": "c"}
    result = runner.execute_variant(spec, settings, tmp_path, 1, deadline, regional_ok)
    assert result["verdict"] == "PASS"
    # BLOCKED when preflight has errors
    result = runner.execute_variant(
        spec,
        settings,
        tmp_path,
        1,
        deadline,
        {"errors": ["x"], "release_id": "r", "cluster_id": "c"},
    )
    assert result["verdict"] == "BLOCKED"
    # NOT_RUN when an approval is missing
    gated = _spec(variant="recovery", approvals=("HOST_MEMORY_PRESSURE",))
    settings_gated = type(settings)(
        **{**make_settings(gated, node=NODE).__dict__, "approvals": ()}
    )
    result = runner.execute_variant(
        gated, settings_gated, tmp_path, 1, deadline, regional_ok
    )
    assert result["verdict"] == "NOT_RUN"


def test_execute_variant_classifies_exceptions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _patch_runner(monkeypatch)

    def boom(ctx):
        raise fx.VariantInconclusive("flat")

    spec = _spec(execute=boom)
    settings = make_settings(spec, node=NODE)
    deadline = datetime.now(timezone.utc) + timedelta(hours=1)
    result = runner.execute_variant(
        spec,
        settings,
        tmp_path,
        1,
        deadline,
        {"errors": [], "release_id": "r", "cluster_id": "c"},
    )
    assert result["verdict"] == "INCONCLUSIVE"

    def crash(ctx):
        raise RuntimeError("bang")

    spec = _spec(execute=crash)
    result = runner.execute_variant(
        spec,
        settings,
        tmp_path,
        1,
        deadline,
        {"errors": [], "release_id": "r", "cluster_id": "c"},
    )
    assert result["verdict"] == "FAIL" and "bang" in result["error"]


def test_teardown_downgrades_on_dirty_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _patch_runner(monkeypatch)

    def dirty(ctx):
        ctx.cleanup.push(
            "leftover", lambda: (_ for _ in ()).throw(RuntimeError("stuck"))
        )
        return {"verdict": "PASS", "stages": {}}

    spec = _spec(execute=dirty)
    settings = make_settings(spec, node=NODE)
    deadline = datetime.now(timezone.utc) + timedelta(hours=1)
    result = runner.execute_variant(
        spec,
        settings,
        tmp_path,
        1,
        deadline,
        {"errors": [], "release_id": "r", "cluster_id": "c"},
    )
    assert result["verdict"] == "FAIL"
    assert any("cleanup did not prove" in e for e in result["errors"]), (
        "any('cleanup did not prove' in e for e in result['errors'])"
    )
    # The line names what did not prove clean: the failed undo and its error.
    assert any(
        e.startswith(
            "cleanup did not prove the node clean: leftover: RuntimeError: stuck"
        )
        for e in result["errors"]
    ), result["errors"]
    assert result["cleanup"] == [
        {"name": "leftover", "ok": False, "error": "RuntimeError: stuck"}
    ]


def test_execute_variant_reads_a_fixture_failure_as_inconclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A probe or fixture failure graded nothing: INCONCLUSIVE, never FAIL.

    2026-10-25 on the second EKS node ``reference_iostat`` (expected FAIL) hit
    the probe's ``dm-create`` ``TimeoutExpired`` before any evidence and the
    FAIL it was given read as its registered product gap, with an empty note.
    """

    from scripts.e2e.regional.host_probe_fixture import HostProbeError
    from scripts.e2e.regional.regional_commands import RegionalFixtureError

    runner = _patch_runner(monkeypatch)
    deadline = datetime.now(timezone.utc) + timedelta(hours=1)
    regional_ok = {"errors": [], "release_id": "r", "cluster_id": "c"}

    def probe_timed_out(ctx):
        raise HostProbeError(
            "host probe failed (exit 1; error TimeoutExpired); output withheld"
        )

    spec = _spec(
        variant="reference_iostat",
        execute=probe_timed_out,
        expected_outcome="FAIL",
        product_gap="weighted await",
    )
    settings = make_settings(spec, node=NODE)
    result = runner.execute_variant(spec, settings, tmp_path, 1, deadline, regional_ok)
    assert result["verdict"] == "INCONCLUSIVE", result
    assert result["reason"] == "INCONCLUSIVE"
    assert result["detail"].startswith("harness fixture failed: HostProbeError: "), (
        result["detail"]
    )
    assert "TimeoutExpired" in result["detail"]
    assert result["error"].startswith("HostProbeError: "), result["error"]
    assert result["expected_gap"] == "weighted await"

    def setup_failed(ctx):
        raise RegionalFixtureError("the test device setup failed at 'mkfs': ProbeError")

    spec = _spec(execute=setup_failed)
    result = runner.execute_variant(
        spec, make_settings(spec, node=NODE), tmp_path, 1, deadline, regional_ok
    )
    assert result["verdict"] == "INCONCLUSIVE", result
    assert "RegionalFixtureError" in result["detail"]


def test_a_dirty_cleanup_after_a_fixture_failure_is_a_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fixture failed after a mutation and the undo could not prove the
    node clean: the spec's cleanup-failure FAIL outranks INCONCLUSIVE."""

    from scripts.e2e.regional.host_probe_fixture import HostProbeError

    runner = _patch_runner(monkeypatch)
    deadline = datetime.now(timezone.utc) + timedelta(hours=1)

    def half_built(ctx):
        ctx.cleanup.push("dm", lambda: (_ for _ in ()).throw(RuntimeError("residue")))
        raise HostProbeError(
            "host probe failed (exit 1; error TimeoutExpired); output withheld"
        )

    spec = _spec(execute=half_built)
    result = runner.execute_variant(
        spec,
        make_settings(spec, node=NODE),
        tmp_path,
        1,
        deadline,
        {"errors": [], "release_id": "r", "cluster_id": "c"},
    )
    assert result["verdict"] == "FAIL", result
    assert result["error"].startswith("HostProbeError: "), result["error"]
    assert any(
        e.startswith("cleanup did not prove the node clean: dm: RuntimeError: residue")
        for e in result["errors"]
    ), result["errors"]
