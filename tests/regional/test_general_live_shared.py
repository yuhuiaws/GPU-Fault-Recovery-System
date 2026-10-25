"""GF-GENERAL-LIVE-005 injector/verdict contract over scripted records.

A fake test export server stands in for ``shared_export_fixture.TestExportServer``
(no cluster), a scripted shared probe answers the node side (no node), and the
product's records are scripted documents. The tests pin: the fixture order on
the LIFO stack, the oracle gates (a pause must really hang statvfs, a fill must
really reach 98 %), the diagnostic-only chain every shared-storage finding must
run, and the three expected-FAIL product gaps -- each of which flips to PASS the
moment the product records what the spec asks for.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_shared as shared
from scripts.e2e.regional.general_live_fixture import VariantInconclusive, VariantNotRun
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    incident,
    make_context,
    make_settings,
    store_script,
    workflow,
)

SPECS = {spec.variant: spec for spec in shared.VARIANTS}
NODE = "ip-10-0-43-5.us-west-2.compute.internal"
PEER = "ip-10-0-32-162.us-west-2.compute.internal"
IMAGE = "public.ecr.aws/gfrs/nfs-ganesha@sha256:" + "a" * 64
SIZE = 256 * 1024 * 1024
#: Product records lie in the future of the runner's wall clock, so every
#: "after the injection" filter keeps them.
T0 = datetime.now(timezone.utc) + timedelta(hours=1)


def at(minutes: float) -> str:
    return (T0 + timedelta(minutes=minutes)).isoformat()


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
class FakeExportServer:
    """The cluster side, remembered on the class so tests can read it back."""

    instances: list[FakeExportServer] = []

    def __init__(self, regional: Any, settings: Any, **_: Any) -> None:
        self.regional = regional
        self.settings = settings
        self.paused = False
        self.mode = "1777"
        self.deleted = False
        self.ready: dict[str, Any] = {}
        self.calls: list[str] = []
        FakeExportServer.instances.append(self)

    def create(self) -> dict[str, Any]:
        self.calls.append("create")
        return {"name": self.settings.suffix}

    def wait_ready(
        self, *, timeout_seconds: float, poll_seconds: float = 5.0
    ) -> dict[str, Any]:
        self.calls.append("wait_ready")
        self.ready = {
            "ip": f"10.0.9.{len(FakeExportServer.instances)}",
            "node": "ip-10-0-99-1.us-west-2.compute.internal",
            "uid": f"uid-{self.settings.suffix}",
            "export": "/a",
        }
        return dict(self.ready)

    def pause(self) -> dict[str, Any]:
        self.calls.append("pause")
        self.paused = True
        return {"state": "State:\tT (stopped)"}

    def resume(self) -> dict[str, Any]:
        self.calls.append("resume")
        self.paused = False
        return {"state": "State:\tS (sleeping)"}

    def set_export_mode(self, mode: str) -> dict[str, Any]:
        self.calls.append(f"mode-{mode}")
        self.mode = mode
        return {"mode": mode}

    def delete(self) -> dict[str, Any]:
        self.calls.append("delete")
        self.deleted = True
        return {"pod": "deleted", "configmap": "deleted"}

    def describe(self) -> dict[str, Any]:
        return {"suffix": self.settings.suffix, "ready": self.ready}


def servers_by_suffix() -> dict[str, FakeExportServer]:
    return {item.settings.suffix: item for item in FakeExportServer.instances}


@pytest.fixture(autouse=True)
def fake_server(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeExportServer.instances = []
    monkeypatch.setattr(shared, "TestExportServer", FakeExportServer)


def survey() -> dict[str, Any]:
    return {
        "tools": {
            name: True
            for name in ("mount.nfs4", "mount", "umount", "nsenter", "systemd-run")
        },
        "modules": {"nfsv4": {"loaded": True, "available": True}},
        "collector_env": {
            "GPU_FAULT_HOST_INTERVAL_SECONDS": "15",
            "GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS": "300",
        },
        "residue": {"units": [], "paused_units": {}, "paused_processes": {}},
    }


def shared_probe(node: str) -> FakeProbe:
    """The node side: statvfs hangs while the server a mount points at is paused."""

    def name_of(arguments: tuple[str, ...]) -> str:
        return arguments[arguments.index("--name") + 1]

    def server_of(name: str) -> FakeExportServer | None:
        return servers_by_suffix().get("a" if name == shared.PRIMARY else "b")

    def stat(arguments: tuple[str, ...]) -> dict[str, Any]:
        server = server_of(name_of(arguments))
        if server is not None and server.paused:
            return {"statvfs": {"timed_out": True, "elapsed": 5.0}, "host_mount": {}}
        return {"statvfs": {"used_percent": 2.0, "total_bytes": SIZE}, "host_mount": {}}

    def io(arguments: tuple[str, ...]) -> dict[str, Any]:
        server = server_of(name_of(arguments))
        if server is not None and server.mode == "000":
            return {"io": {"ok": False, "error": "PermissionError: 13", "errno": 13}}
        return {"io": {"ok": True, "sha256_match": True, "bytes": 4 << 20}}

    return FakeProbe(
        {
            "nfs-mount": {
                "statvfs": {"total_bytes": SIZE, "used_percent": 1.0},
                "host_mount": {"fstype": "nfs4"},
            },
            "nfs-stat": stat,
            "nfs-io": io,
            "nfs-fill": {"statvfs": {"used_percent": 99.2, "total_bytes": SIZE}},
            "nfs-unfill": {"statvfs": {"used_percent": 1.0}},
            "nfs-umount": {"host_mount": None, "unmounted": f"/mnt/x/{node}"},
            "cleanup": {"clean": True, "residue": {}},
        }
    )


def node_probe() -> FakeProbe:
    return FakeProbe({"survey": survey()})


def diagnostic(
    mount: str, metric: str, node: str = NODE, *, state: str = "RECOVERED"
) -> dict:
    inc = incident(
        f"inc-{node[-3:]}",
        metric,
        device=mount,
        state=state,
        workflow=f"wf-{node[-3:]}",
    )
    inc["node_ids"] = [node]
    inc["reasons"] = ["diagnostic inconclusive: validation failed"]
    wf = workflow(
        f"wf-{node[-3:]}",
        inc["incident_id"],
        ["FREEZE_EVIDENCE", "VALIDATE_HOST"],
        status="FAILED",
        probe_metric=metric,
        executions=[
            {
                "operation": "VALIDATE_HOST",
                "status": "FAILED",
                "details": {"host_diagnostic_failure": True},
            }
        ],
    )
    return {"incident": inc, "workflow": wf}


def sample(name: str, device: str | None, value: float) -> dict[str, Any]:
    return {"name": name, "device": device, "value": value, "labels": {}}


def batch(minutes: float, *items: dict[str, Any]) -> dict[str, Any]:
    return {
        "observed_at": at(minutes),
        "batch_id": f"host-{minutes}",
        "edge_filter_reasons": [],
        "collection_errors": [],
        "samples": list(items),
    }


def hang_activity(
    mount: str,
    *,
    node: str = NODE,
    detected: bool = True,
    other: str | None = None,
    companion: bool = True,
    node_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Baseline, the hang (two batches), the recovery readings."""

    companion_sample = (
        [sample(shared.COMPANION_METRIC, None, 40.0)] if companion else []
    )
    other_samples = [sample(shared.UNAVAILABLE, other, 0.0)] if other else []
    batches = [
        batch(
            0, sample(shared.UNAVAILABLE, mount, 0.0), sample(shared.USED, mount, 2.0)
        ),
        batch(
            2, sample(shared.UNAVAILABLE, mount, 1.0), *companion_sample, *other_samples
        ),
        batch(
            3, sample(shared.UNAVAILABLE, mount, 1.0), *companion_sample, *other_samples
        ),
        batch(
            9, sample(shared.UNAVAILABLE, mount, 0.0), sample(shared.USED, mount, 2.0)
        ),
        batch(10, sample(shared.UNAVAILABLE, mount, 0.0)),
    ]
    document: dict[str, Any] = {
        "incidents": [],
        "workflows": [],
        "batches": batches,
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
    }
    if detected:
        chain = diagnostic(mount, shared.UNAVAILABLE, node)
        if node_ids:
            chain["incident"]["node_ids"] = node_ids
        document["incidents"] = [chain["incident"]]
        document["workflows"] = [chain["workflow"]]
    return document


def quiet_activity(mount: str) -> dict[str, Any]:
    return {
        "incidents": [],
        "workflows": [],
        "batches": [
            batch(
                0,
                sample(shared.UNAVAILABLE, mount, 0.0),
                sample(shared.USED, mount, 2.0),
            )
        ],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
    }


def context(
    variant: str,
    tmp_path: Path,
    *,
    activity: Any,
    peer: str = "",
    image: str = IMAGE,
    probes: dict[str, FakeProbe] | None = None,
) -> Any:
    spec = SPECS[variant]
    regional = FakeRegional(cpu_python=store_script({"activity": activity}))
    settings = dataclasses.replace(
        make_settings(spec, node=NODE, peer=peer), export_image=image
    )
    all_probes = {f"node@{NODE}": node_probe(), f"shared@{NODE}": shared_probe(NODE)}
    if peer:
        all_probes[f"node@{peer}"] = node_probe()
        all_probes[f"shared@{peer}"] = shared_probe(peer)
    all_probes.update(probes or {})
    return make_context(
        spec, tmp_path=tmp_path, regional=regional, probes=all_probes, settings=settings
    )


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def test_every_005_variant_is_native_both_lanes_behind_kernel_module_load() -> None:
    assert set(SPECS) == {
        "stat_timeout",
        "capacity_full",
        "mount_disappears",
        "io_only_failure",
        "healthy_other_mount",
        "shared_failure_domain",
        "recovery",
    }
    for spec in SPECS.values():
        assert spec.implemented and not spec.deferred, spec.key
        assert spec.proof_mode == "native", spec.key
        assert spec.lanes == ("eks", "hp"), spec.key
        assert spec.approvals == ("KERNEL_MODULE_LOAD",), spec.key
    gaps = {
        name: spec for name, spec in SPECS.items() if spec.expected_outcome == "FAIL"
    }
    assert set(gaps) == {"mount_disappears", "io_only_failure", "shared_failure_domain"}
    for spec in gaps.values():
        assert (
            "system_metrics.py" in spec.product_gap
            or "host_health.py" in spec.product_gap
        )
    assert SPECS["shared_failure_domain"].flags == "X", "both lane locks on EKS"
    assert all(
        spec.flags == "S"
        for name, spec in SPECS.items()
        if name != "shared_failure_domain"
    ), "single-node variants hold the lane and site locks only"


def test_preflight_needs_an_immutable_export_image_and_node_b_for_the_pair() -> None:
    spec = SPECS["stat_timeout"]
    assert spec.preflight is not None
    assert (
        spec.preflight(make_settings(spec))
        and "export" in spec.preflight(make_settings(spec))[0]
    )
    settings = dataclasses.replace(make_settings(spec), export_image="repo/img:latest")
    assert spec.preflight(settings), "a tag is not an immutable digest"
    settings = dataclasses.replace(make_settings(spec), export_image=IMAGE)
    assert spec.preflight(settings) == []
    pair = SPECS["shared_failure_domain"]
    assert pair.preflight is not None
    assert any("--node-b" in item for item in pair.preflight(settings)), (
        "the pair variant must demand --node-b"
    )
    assert pair.preflight(dataclasses.replace(settings, peer_node=PEER)) == []


# --------------------------------------------------------------------------- #
# stat_timeout
# --------------------------------------------------------------------------- #
def test_stat_timeout_passes_and_cleans_up_lifo(tmp_path: Path) -> None:
    ctx = context("stat_timeout", tmp_path, activity=None)
    mount = shared.mount_path(ctx, shared.PRIMARY)
    ctx.regional.cpu_python_responder = store_script({"activity": hang_activity(mount)})
    result = SPECS["stat_timeout"].execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    server = servers_by_suffix()["a"]
    assert server.calls == ["create", "wait_ready", "pause", "resume"], server.calls
    assert server.settings.avoid_nodes == (NODE,) and server.settings.node == ""
    assert server.settings.image == IMAGE
    probe = ctx.probes[f"shared@{NODE}"]
    mounted = next(call for call in probe.calls if call[0] == "nfs-mount")
    assert mounted[mounted.index("--server") + 1] == server.ready["ip"], mounted
    assert mounted[mounted.index("--export") + 1] == "/a", mounted
    assert "--load-module" not in mounted, "nfsv4 is loaded in the survey"
    # Resume was discarded; the mount undo sits above the export delete (LIFO).
    assert ctx.cleanup.pending() == ["export-a", f"mount-shared@{NODE}"], (
        ctx.cleanup.pending()
    )
    ctx.cleanup.run()
    assert not ctx.cleanup.failed, ctx.cleanup.failed
    assert probe.calls[-1][:3] == ("cleanup", "--kind", "nfs-shared"), probe.calls[-1]
    assert server.deleted, "the export Pod is deleted last"


def test_stat_timeout_fails_when_the_product_misses_the_hang(tmp_path: Path) -> None:
    ctx = context("stat_timeout", tmp_path, activity=None)
    mount = shared.mount_path(ctx, shared.PRIMARY)
    ctx.regional.cpu_python_responder = store_script(
        {"activity": hang_activity(mount, detected=False)}
    )
    result = SPECS["stat_timeout"].execute(ctx)
    assert result["verdict"] == "FAIL", result
    assert any(
        "no shared_filesystem_unavailable" in error for error in result["errors"]
    ), result["errors"]
    assert servers_by_suffix()["a"].paused is False, "resumed before the verdict"


def test_stat_timeout_fails_when_the_hang_stalls_other_metrics(tmp_path: Path) -> None:
    ctx = context("stat_timeout", tmp_path, activity=None)
    mount = shared.mount_path(ctx, shared.PRIMARY)
    ctx.regional.cpu_python_responder = store_script(
        {"activity": hang_activity(mount, companion=False)}
    )
    result = SPECS["stat_timeout"].execute(ctx)
    assert result["verdict"] == "FAIL", result
    assert any("memory_used_percent" in error for error in result["errors"]), result[
        "errors"
    ]


def test_stat_timeout_is_inconclusive_when_the_pause_does_not_hang(
    tmp_path: Path,
) -> None:
    probe = shared_probe(NODE)
    probe.script["nfs-stat"] = {"statvfs": {"used_percent": 2.0}}
    ctx = context(
        "stat_timeout", tmp_path, activity=None, probes={f"shared@{NODE}": probe}
    )
    mount = shared.mount_path(ctx, shared.PRIMARY)
    ctx.regional.cpu_python_responder = store_script({"activity": hang_activity(mount)})
    with pytest.raises(VariantInconclusive, match="still answered"):
        SPECS["stat_timeout"].execute(ctx)
    assert servers_by_suffix()["a"].paused is False, "resumed on the way out"


def test_the_export_must_be_discovered_before_any_injection(tmp_path: Path) -> None:
    ctx = context("stat_timeout", tmp_path, activity=quiet_activity("/elsewhere"))
    with pytest.raises(VariantInconclusive, match="not discovered"):
        SPECS["stat_timeout"].execute(ctx)
    assert servers_by_suffix()["a"].calls == ["create", "wait_ready"], "never paused"


def test_an_unbounded_export_capacity_is_a_fixture_problem(tmp_path: Path) -> None:
    probe = shared_probe(NODE)
    probe.script["nfs-mount"] = {"statvfs": {"total_bytes": 900 * 1024**3}}
    ctx = context(
        "stat_timeout", tmp_path, activity=None, probes={f"shared@{NODE}": probe}
    )
    with pytest.raises(VariantNotRun, match="not bounded"):
        SPECS["stat_timeout"].execute(ctx)


def test_a_missing_export_image_or_client_tooling_is_not_run(tmp_path: Path) -> None:
    ctx = context("stat_timeout", tmp_path, activity=None, image="")
    with pytest.raises(VariantNotRun, match="export server image"):
        SPECS["stat_timeout"].execute(ctx)
    value = survey()
    value["tools"]["mount.nfs4"] = False
    ctx = context(
        "stat_timeout",
        tmp_path,
        activity=None,
        probes={f"node@{NODE}": FakeProbe({"survey": value})},
    )
    with pytest.raises(VariantNotRun, match="mount.nfs4"):
        SPECS["stat_timeout"].execute(ctx)
    assert FakeExportServer.instances == [], "refused before any export was created"


def test_nfsv4_is_loaded_under_the_token_when_missing(tmp_path: Path) -> None:
    value = survey()
    value["modules"]["nfsv4"] = {"loaded": False, "available": True}
    ctx = context(
        "stat_timeout",
        tmp_path,
        activity=None,
        probes={f"node@{NODE}": FakeProbe({"survey": value})},
    )
    mount = shared.mount_path(ctx, shared.PRIMARY)
    ctx.regional.cpu_python_responder = store_script({"activity": hang_activity(mount)})
    assert SPECS["stat_timeout"].execute(ctx)["verdict"] == "PASS"
    mounted = next(c for c in ctx.probes[f"shared@{NODE}"].calls if c[0] == "nfs-mount")
    assert "--load-module" in mounted, mounted


def test_an_export_landing_on_the_client_node_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def wait_ready(self: FakeExportServer, **_: Any) -> dict[str, Any]:
        self.ready = {"ip": "10.0.9.9", "node": NODE, "uid": "u", "export": "/a"}
        return dict(self.ready)

    monkeypatch.setattr(FakeExportServer, "wait_ready", wait_ready)
    ctx = context("stat_timeout", tmp_path, activity=None)
    with pytest.raises(RegionalFixtureError, match="client node"):
        SPECS["stat_timeout"].execute(ctx)
    assert ctx.cleanup.pending() == ["export-a"], "the delete is already registered"


# --------------------------------------------------------------------------- #
# capacity_full
# --------------------------------------------------------------------------- #
def full_activity(mount: str, *, isolated: bool = False) -> dict[str, Any]:
    chain = diagnostic(mount, shared.USED)
    if isolated:
        chain["workflow"]["official_steps"].append(
            {"operation": "QUARANTINE", "parameters": {}}
        )
    return {
        "incidents": [chain["incident"]],
        "workflows": [chain["workflow"]],
        "batches": [
            batch(
                0,
                sample(shared.UNAVAILABLE, mount, 0.0),
                sample(shared.USED, mount, 2.0),
            ),
            batch(2, sample(shared.USED, mount, 99.2)),
            batch(9, sample(shared.USED, mount, 1.0)),
        ],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
    }


def test_capacity_full_runs_a_diagnosis_without_isolation(tmp_path: Path) -> None:
    ctx = context("capacity_full", tmp_path, activity=None)
    mount = shared.mount_path(ctx, shared.PRIMARY)
    ctx.regional.cpu_python_responder = store_script({"activity": full_activity(mount)})
    result = SPECS["capacity_full"].execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    probe = ctx.probes[f"shared@{NODE}"]
    names = [call[0] for call in probe.calls]
    assert names.index("nfs-fill") < names.index("nfs-unfill"), names
    assert "unfill" not in ctx.cleanup.pending(), ctx.cleanup.pending()


def test_capacity_full_fails_on_a_quarantine(tmp_path: Path) -> None:
    ctx = context("capacity_full", tmp_path, activity=None)
    mount = shared.mount_path(ctx, shared.PRIMARY)
    ctx.regional.cpu_python_responder = store_script(
        {"activity": full_activity(mount, isolated=True)}
    )
    result = SPECS["capacity_full"].execute(ctx)
    assert result["verdict"] == "FAIL", result
    assert any("QUARANTINE" in error for error in result["errors"]), result["errors"]


def test_capacity_full_is_inconclusive_below_98(tmp_path: Path) -> None:
    probe = shared_probe(NODE)
    probe.script["nfs-fill"] = {"statvfs": {"used_percent": 90.0}}
    ctx = context(
        "capacity_full", tmp_path, activity=None, probes={f"shared@{NODE}": probe}
    )
    mount = shared.mount_path(ctx, shared.PRIMARY)
    ctx.regional.cpu_python_responder = store_script({"activity": full_activity(mount)})
    with pytest.raises(VariantInconclusive, match="only reached"):
        SPECS["capacity_full"].execute(ctx)
    assert "unfill" in ctx.cleanup.pending(), "the fill is undone by the stack"


# --------------------------------------------------------------------------- #
# the expected-FAIL gaps
# --------------------------------------------------------------------------- #
def test_mount_disappears_records_the_gap_and_passes_once_reported(
    tmp_path: Path,
) -> None:
    spec = SPECS["mount_disappears"]
    ctx = context("mount_disappears", tmp_path, activity=None)
    mount = shared.mount_path(ctx, shared.PRIMARY)
    ctx.regional.cpu_python_responder = store_script(
        {"activity": quiet_activity(mount)}
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL", result
    assert any("never reported" in error for error in result["errors"]), result[
        "errors"
    ]
    assert spec.expected_outcome == "FAIL" and "rediscovers" in spec.product_gap
    probe = ctx.probes[f"shared@{NODE}"]
    assert any(call[0] == "nfs-umount" for call in probe.calls), probe.calls
    assert servers_by_suffix()["a"].paused is False, "the server was never paused"
    ctx = context("mount_disappears", tmp_path / "fixed", activity=None)
    ctx.regional.cpu_python_responder = store_script({"activity": hang_activity(mount)})
    assert spec.execute(ctx)["verdict"] == "PASS", "a product that reports it passes"


def test_io_only_failure_records_the_gap_and_restores_io(tmp_path: Path) -> None:
    spec = SPECS["io_only_failure"]
    ctx = context("io_only_failure", tmp_path, activity=None)
    mount = shared.mount_path(ctx, shared.PRIMARY)
    ctx.regional.cpu_python_responder = store_script(
        {"activity": quiet_activity(mount)}
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL", result
    assert any("statvfs only" in error for error in result["errors"]), result["errors"]
    assert result["stages"]["io_restored"] == [], result["stages"]
    server = servers_by_suffix()["a"]
    assert server.calls == ["create", "wait_ready", "mode-000", "mode-1777"], (
        server.calls
    )
    assert "export-a-mode" not in ctx.cleanup.pending(), ctx.cleanup.pending()
    assert result["io"]["errno"] == 13, result["io"]


def test_io_only_failure_is_inconclusive_when_io_still_works(tmp_path: Path) -> None:
    probe = shared_probe(NODE)
    probe.script["nfs-io"] = {"io": {"ok": True}}
    ctx = context(
        "io_only_failure", tmp_path, activity=None, probes={f"shared@{NODE}": probe}
    )
    mount = shared.mount_path(ctx, shared.PRIMARY)
    ctx.regional.cpu_python_responder = store_script(
        {"activity": quiet_activity(mount)}
    )
    with pytest.raises(VariantInconclusive, match="I/O-failing"):
        SPECS["io_only_failure"].execute(ctx)
    assert servers_by_suffix()["a"].mode == "1777", "mode restored on the way out"


def test_shared_failure_domain_needs_two_clients_and_records_the_gap(
    tmp_path: Path,
) -> None:
    spec = SPECS["shared_failure_domain"]
    with pytest.raises(VariantNotRun, match="node-b"):
        spec.execute(context("shared_failure_domain", tmp_path, activity=None))
    ctx = context("shared_failure_domain", tmp_path, activity=None, peer=PEER)
    mount = shared.mount_path(ctx, shared.PRIMARY)

    def per_node(arguments: tuple[str, ...]) -> dict[str, Any]:
        node = arguments[1]
        return hang_activity(mount, node=node)

    ctx.regional.cpu_python_responder = store_script({"activity": per_node})
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL", result
    assert (
        result["stages"][f"chain@{NODE}"] == []
        and result["stages"][f"chain@{PEER}"] == []
    )
    assert any(
        "common failure" in error or "two unrelated" in error
        for error in result["errors"]
    ), result["errors"]
    server = servers_by_suffix()["a"]
    assert server.settings.avoid_nodes == (NODE, PEER), server.settings
    assert server.calls == ["create", "wait_ready", "pause", "resume"], server.calls
    for node in (NODE, PEER):
        probe = ctx.probes[f"shared@{node}"]
        assert any(call[0] == "nfs-mount" for call in probe.calls), node
    assert ctx.cleanup.pending() == [
        "export-a",
        f"mount-shared@{NODE}",
        f"mount-shared@{PEER}",
    ], ctx.cleanup.pending()


def test_shared_failure_domain_passes_when_one_incident_names_both_nodes(
    tmp_path: Path,
) -> None:
    ctx = context("shared_failure_domain", tmp_path, activity=None, peer=PEER)
    mount = shared.mount_path(ctx, shared.PRIMARY)

    def per_node(arguments: tuple[str, ...]) -> dict[str, Any]:
        return hang_activity(mount, node=arguments[1], node_ids=[NODE, PEER])

    ctx.regional.cpu_python_responder = store_script({"activity": per_node})
    result = SPECS["shared_failure_domain"].execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


# --------------------------------------------------------------------------- #
# healthy_other_mount and recovery
# --------------------------------------------------------------------------- #
def test_healthy_other_mount_keeps_the_second_export_out_of_the_verdict(
    tmp_path: Path,
) -> None:
    ctx = context("healthy_other_mount", tmp_path, activity=None)
    mount, other = (
        shared.mount_path(ctx, shared.PRIMARY),
        shared.mount_path(ctx, shared.OTHER),
    )
    document = hang_activity(mount, other=other)
    document["batches"][0]["samples"] += [
        sample(shared.UNAVAILABLE, other, 0.0),
        sample(shared.USED, other, 2.0),
    ]
    ctx.regional.cpu_python_responder = store_script({"activity": document})
    result = SPECS["healthy_other_mount"].execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    servers = servers_by_suffix()
    assert servers["a"].calls == ["create", "wait_ready", "pause", "resume"]
    assert servers["b"].calls == ["create", "wait_ready"], (
        "the control export is never paused"
    )
    assert ctx.cleanup.pending() == [
        "export-a",
        "export-b",
        f"mount-shared@{NODE}",
        f"mount-other@{NODE}",
    ], ctx.cleanup.pending()


def test_healthy_other_mount_fails_when_the_healthy_mount_is_graded_or_dropped(
    tmp_path: Path,
) -> None:
    ctx = context("healthy_other_mount", tmp_path, activity=None)
    mount, other = (
        shared.mount_path(ctx, shared.PRIMARY),
        shared.mount_path(ctx, shared.OTHER),
    )
    # The other mount's only readings predate the hang: nothing fresh after it.
    document = hang_activity(mount)
    document["batches"].insert(
        0,
        batch(
            -120,
            sample(shared.UNAVAILABLE, other, 0.0),
            sample(shared.USED, other, 2.0),
        ),
    )
    ctx.regional.cpu_python_responder = store_script({"activity": document})
    result = SPECS["healthy_other_mount"].execute(ctx)
    assert result["verdict"] == "FAIL", result
    assert result["stages"]["other_still_reported"], result["stages"]


def test_recovery_proves_fresh_io_and_samples(tmp_path: Path) -> None:
    ctx = context("recovery", tmp_path, activity=None)
    mount = shared.mount_path(ctx, shared.PRIMARY)
    ctx.regional.cpu_python_responder = store_script({"activity": hang_activity(mount)})
    result = SPECS["recovery"].execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["io"]["ok"] and result["stages"]["fresh_samples"] == [], result
    probe = ctx.probes[f"shared@{NODE}"]
    io_calls = [call for call in probe.calls if call[0] == "nfs-io"]
    assert io_calls and io_calls[0][io_calls[0].index("--mib") + 1] == "4", io_calls


def test_recovery_fails_without_real_io_after_the_resume(tmp_path: Path) -> None:
    probe = shared_probe(NODE)
    probe.script["nfs-io"] = {"io": {"ok": False, "error": "OSError: 5", "errno": 5}}
    ctx = context("recovery", tmp_path, activity=None, probes={f"shared@{NODE}": probe})
    mount = shared.mount_path(ctx, shared.PRIMARY)
    ctx.regional.cpu_python_responder = store_script({"activity": hang_activity(mount)})
    result = SPECS["recovery"].execute(ctx)
    assert result["verdict"] == "FAIL", result
    assert result["stages"]["fresh_io"], result["stages"]


# --------------------------------------------------------------------------- #
# verdict helpers
# --------------------------------------------------------------------------- #
def test_common_domain_errors_accepts_a_marker_naming_both_nodes_and_the_export() -> (
    None
):
    mount = "/mnt/gfrs-live/abc/shared"
    activities = {
        NODE: {"incidents": [], "markers": [{"text": f"{NODE} {PEER} {mount}"}]},
        PEER: {"incidents": []},
    }
    assert shared.common_domain_errors(activities, (NODE, PEER), mount) == []
    activities[NODE]["markers"] = [{"text": f"{NODE} only {mount}"}]
    assert shared.common_domain_errors(activities, (NODE, PEER), mount), (
        "one node is not a domain"
    )


def test_delivery_errors_need_two_batches_with_the_companion_metric() -> None:
    since = T0 + timedelta(minutes=1)
    document = {"batches": [batch(2, sample(shared.COMPANION_METRIC, None, 1.0))]}
    assert shared.delivery_errors(document, since), "one batch is not continuity"
    document["batches"].append(batch(3, sample(shared.UNAVAILABLE, "/m", 1.0)))
    assert shared.delivery_errors(document, since) == []
    before = {"batches": [batch(-5, sample(shared.COMPANION_METRIC, None, 1.0))] * 2}
    assert shared.delivery_errors(before, since), (
        "batches from before the injection do not count"
    )
