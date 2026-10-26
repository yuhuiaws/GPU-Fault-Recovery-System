"""GF-GENERAL-LIVE-005 injector/verdict contract over scripted records.

A fake test export server stands in for ``shared_export_fixture.TestExportServer``
(no cluster), a scripted shared probe answers the node side (no node), and the
product's records are scripted documents. The tests pin: the fixture order on
the LIFO stack, the oracle gates (a pause must really hang statvfs, a fill must
really reach 98 %), the diagnostic-only chain every shared-storage finding must
run, the ``failure_mode``/``source`` labels ``mount_disappears`` and
``io_only_failure`` now demand of the product, and the remaining expected-FAIL
product gap (``shared_failure_domain``), which flips to PASS the moment the
product records what the spec asks for.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import general_live_shared as shared
from scripts.e2e.regional import general_live_shared_recovery as recovery
from scripts.e2e.regional.general_live_fixture import VariantInconclusive, VariantNotRun
from scripts.e2e.regional.host_probe_fixture import HostProbeError
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
        self.name = f"gfrs-export-fake-{settings.suffix}"
        self.log_text = (
            "export_commit_common :CONFIG :INFO :Export 1 created at pseudo (/a)\n"
        )
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

    def logs(self, tail: int = 40) -> str:
        self.calls.append(f"logs-{tail}")
        return self.log_text


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
            "nfs-fill": {
                "fill": {"fill_bytes": 253 << 20, "wanted_bytes": 253 << 20},
                "statvfs": {"used_percent": 99.2, "total_bytes": SIZE},
            },
            "nfs-unfill": {"unfill": {"removed": 1}, "statvfs": {"used_percent": 1.0}},
            "nfs-umount": {
                "host_mount": None,
                "unmounted": f"/mnt/x/{node}",
                "directory_present": True,
            },
            "nfs-remount": {
                "host_mount": {"fstype": "nfs4"},
                "statvfs": {"used_percent": 2.0, "total_bytes": SIZE},
            },
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


def sample(
    name: str, device: str | None, value: float, labels: dict[str, str] | None = None
) -> dict[str, Any]:
    return {
        "name": name,
        "device": device,
        "value": value,
        "labels": dict(labels or {}),
    }


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
    labels: dict[str, str] | None = None,
    source: str | None = None,
    domain: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Baseline, the fault (two batches, ``labels`` on them), the recovery readings.

    ``source`` puts the collector's export label on every shared sample of the
    mount (batch 176); ``domain`` adds a failure-domain record as ``failure_domain``
    builds it, its back-reference on the node's incident and its notification.
    """

    companion_sample = (
        [sample(recovery.COMPANION_METRIC, None, 40.0)] if companion else []
    )
    other_samples = [sample(shared.UNAVAILABLE, other, 0.0)] if other else []
    identity = {"source": source, "fstype": "nfs4"} if source else {}
    fault_labels = {**identity, **(labels or {})}
    batches = [
        batch(
            0,
            sample(shared.UNAVAILABLE, mount, 0.0, identity),
            sample(shared.USED, mount, 2.0, identity),
        ),
        batch(
            2,
            sample(shared.UNAVAILABLE, mount, 1.0, fault_labels),
            *companion_sample,
            *other_samples,
        ),
        batch(
            3,
            sample(shared.UNAVAILABLE, mount, 1.0, fault_labels),
            *companion_sample,
            *other_samples,
        ),
        batch(
            9,
            sample(shared.UNAVAILABLE, mount, 0.0, identity),
            sample(shared.USED, mount, 2.0, identity),
        ),
        batch(10, sample(shared.UNAVAILABLE, mount, 0.0, identity)),
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
        if domain is not None:
            record = domain["record"]
            if domain.get("back_reference", True):
                chain["incident"]["reasons"].append(
                    f"common failure domain {record['incident_id']}: shared filesystem "
                    f"{domain['export']} is unavailable on other nodes as well "
                    "(shared-storage outage, not a node fault)"
                )
            document["incidents"].append(record)
            document["workflows"] += domain.get("workflows", [])
            document["notifications"] = list(domain.get("notifications", []))
    return document


def failure_domain(
    export: str,
    *,
    nodes: tuple[str, ...] = (NODE, PEER),
    state: str = "RECOVERED",
    action: str | None = None,
    workflow_id: str | None = None,
    notifications: int = 1,
    first_reason: str | None = None,
    members: bool = True,
    back_reference: bool = True,
) -> dict[str, Any]:
    """The record batch 177 writes for ``export`` on ``nodes``, and its mail."""

    domain_id = "inc-domain-1"
    reasons = [
        first_reason
        if first_reason is not None
        else (
            f"common failure domain: shared filesystem {export} unavailable on "
            f"{len(nodes)} nodes ({', '.join(nodes)}); a shared-storage outage, not a "
            "per-node hardware fault"
        )
    ]
    if members:
        reasons += [
            f"member {node}: incident inc-{node[-3:]} ({shared.UNAVAILABLE} on /m, "
            "observed 2026-10-25T00:00:00+00:00)"
            for node in nodes
        ]
    record = {
        "incident_id": domain_id,
        "event_id": "shared-storage-failure-domain-x",
        "event_type": shared.DOMAIN_EVENT_TYPE,
        "event_source": shared.UNAVAILABLE,
        "cluster_id": "gpu-eks-cluster",
        "node_ids": list(nodes),
        "policy_source": "SITE_SHARED_STORAGE_FAILURE_DOMAIN",
        "policy_reference": f"shared-filesystem:{export}",
        "effective_action": action,
        "state": state,
        "reasons": reasons,
        "workflow_request_id": workflow_id,
        "created_at": at(2.5),
    }
    workflows = (
        [workflow(workflow_id, domain_id, ["FREEZE_EVIDENCE"])] if workflow_id else []
    )
    mails = [
        {
            "notification_id": f"ntf-domain-{index}",
            "incident_id": domain_id,
            "category": shared.DOMAIN_NOTIFICATION_CATEGORY,
            "deduplication_key": f"shared-storage-failure-domain/{index}",
            "created_at": at(2.5),
            "subject": f"[GPU fault shared storage] {export} unavailable on 2 nodes",
            "status": "SENT",
        }
        for index in range(notifications)
    ]
    return {
        "record": record,
        "export": export,
        "workflows": workflows,
        "notifications": mails,
        "back_reference": back_reference,
    }


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
    assert gaps == {}, "batches 176/177 closed every 005 product gap"
    assert all(spec.product_gap == "" for spec in SPECS.values()), (
        "no 005 variant carries a product gap"
    )
    assert SPECS["shared_failure_domain"].flags == "X", "both lane locks on EKS"
    assert shared.DOMAIN_EVENT_TYPE in SPECS["shared_failure_domain"].expected
    assert all(
        spec.flags == "S"
        for name, spec in SPECS.items()
        if name != "shared_failure_domain"
    ), "single-node variants hold the lane and site locks only"


def test_preflight_needs_an_immutable_export_image_and_node_b_for_the_pair() -> None:
    spec = SPECS["stat_timeout"]
    assert spec.preflight is not None
    regional = FakeRegional()
    assert (
        spec.preflight(make_settings(spec), regional)
        and "export" in spec.preflight(make_settings(spec), regional)[0]
    )
    settings = dataclasses.replace(make_settings(spec), export_image="repo/img:latest")
    assert spec.preflight(settings, regional), "a tag is not an immutable digest"
    settings = dataclasses.replace(make_settings(spec), export_image=IMAGE)
    assert spec.preflight(settings, regional) == []
    pair = SPECS["shared_failure_domain"]
    assert pair.preflight is not None
    assert any("--node-b" in item for item in pair.preflight(settings, regional)), (
        "the pair variant must demand --node-b"
    )
    assert pair.preflight(dataclasses.replace(settings, peer_node=PEER), regional) == []


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


def store_view(
    document: dict[str, Any],
    *,
    delivered_minutes: float | None = 1.0,
    latest: Iterable[tuple[str, str, float, dict[str, str] | None]] = (),
) -> dict[str, Any]:
    """``document`` as the probe prints it when it also lists the host collector's
    status row and the latest-metric records: ``since`` just before the read,
    one ``host-`` HOST_TELEMETRY status delivered at ``delivered_minutes`` (None:
    no delivery), ``latest`` as ``(name, device, value, labels)`` observed after
    ``since``."""

    # Before the fakes' fixed incident ``created_at`` (``minted_in_read``) and
    # before every timestamp ``at`` yields.
    view = {**document, "since": "2026-10-24T00:00:00+00:00"}
    view["collector_statuses"] = (
        []
        if delivered_minutes is None
        else [
            {
                "collector": "HOST_TELEMETRY",
                "batch_id": f"host-{NODE}-{int(delivered_minutes * 60)}",
                "node_id": NODE,
                "last_success_at": at(delivered_minutes),
                "observed_at": at(delivered_minutes),
                "sample_count": 40,
                "errors": [],
            }
        ]
    )
    view["metric_latest"] = [
        {
            "name": name,
            "device": device,
            "value": value,
            "observed_at": at(2),
            "labels": dict(labels or {}),
        }
        for name, device, value, labels in latest
    ]
    return view


def test_a_summary_only_baseline_is_read_from_the_latest_metric_records(
    tmp_path: Path,
) -> None:
    """Live 2026-10-26, both lanes: the mounted export read healthy, but a healthy
    new mount is no edge for the host collector, so its readings travelled with
    the health summary -- never raw evidence, only the latest-metric records
    (``LATEST_METRICS_REQUIRED``). The baseline is read from those too."""

    ctx = context("stat_timeout", tmp_path, activity=None)
    mount = shared.mount_path(ctx, shared.PRIMARY)
    document = hang_activity(mount, source="10.0.46.250:/a")
    first = document["batches"].pop(0)
    assert {item["name"] for item in first["samples"]} == {
        shared.UNAVAILABLE,
        shared.USED,
    }
    identity = {"source": "10.0.46.250:/a", "fstype": "nfs4"}
    ctx.regional.cpu_python_responder = store_script(
        {
            "activity": store_view(
                document,
                latest=[
                    (shared.UNAVAILABLE, mount, 0.0, identity),
                    (shared.USED, mount, 0.0, identity),
                ],
            )
        }
    )
    result = SPECS["stat_timeout"].execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    baseline = json.loads((ctx.case_dir / "baseline-shared.json").read_text())
    readings = shared.baseline_readings(baseline, shared.UNAVAILABLE, mount)
    # The scripted document is static, so the later fault and recovery batches
    # show as evidence too; the healthy reading that made the baseline is the
    # latest record (no evidence batch carries a healthy used % any more).
    assert any(
        item["record"] == "metric_latest" and item["value"] == 0 for item in readings
    ), readings
    assert [
        item["record"]
        for item in shared.baseline_readings(baseline, shared.USED, mount)
    ] == ["evidence", "metric_latest"], "batch 9's recovery reading and the record"
    assert shared.export_sources(baseline, mount) == {"10.0.46.250:/a"}, (
        "the failure-domain identity is read from the latest record's labels too"
    )
    timeline = json.loads((ctx.case_dir / f"baseline-{NODE}-timeline.json").read_text())
    assert timeline["entries"][-1]["accepted"] is True, timeline


def test_the_export_must_be_discovered_before_any_injection(tmp_path: Path) -> None:
    """A delivering collector whose shared records name another mount only: the
    export is not discovered, and the error says which mounts the product does
    list; the last read is saved."""

    ctx = context("stat_timeout", tmp_path, activity=None)
    ctx.regional.cpu_python_responder = store_script(
        {
            "activity": store_view(
                quiet_activity("/elsewhere"),
                latest=[
                    (shared.UNAVAILABLE, "/elsewhere", 0.0, None),
                    (shared.USED, "/elsewhere", 3.0, None),
                ],
            )
        }
    )
    with pytest.raises(VariantInconclusive, match="not discovered") as caught:
        SPECS["stat_timeout"].execute(ctx)
    message = str(caught.value)
    assert "['/elsewhere']" in message and f"delivered until {at(1)}" in message, (
        message
    )
    assert servers_by_suffix()["a"].calls == ["create", "wait_ready"], "never paused"
    last = json.loads((ctx.case_dir / f"baseline-{NODE}-last-read.json").read_text())
    assert [item["device"] for item in last["shared_latest"]] == ["/elsewhere"] * 2
    assert last["host_delivery"]["batch_id"].startswith("host-"), last


def test_a_silent_host_collector_is_named_instead_of_the_mount(tmp_path: Path) -> None:
    ctx = context("stat_timeout", tmp_path, activity=None)
    ctx.regional.cpu_python_responder = store_script(
        {"activity": store_view(quiet_activity("/elsewhere"), delivered_minutes=None)}
    )
    with pytest.raises(VariantInconclusive, match="delivered no batch since the mount"):
        SPECS["stat_timeout"].execute(ctx)


def test_a_mount_read_unhealthy_from_the_start_is_not_a_discovery(
    tmp_path: Path,
) -> None:
    ctx = context("stat_timeout", tmp_path, activity=None)
    mount = shared.mount_path(ctx, shared.PRIMARY)
    ctx.regional.cpu_python_responder = store_script(
        {
            "activity": store_view(
                quiet_activity("/elsewhere"),
                latest=[(shared.UNAVAILABLE, mount, 1.0, {"failure_mode": "IO_ERROR"})],
            )
        }
    )
    with pytest.raises(VariantInconclusive, match="read but never healthy") as caught:
        SPECS["stat_timeout"].execute(ctx)
    assert f"{shared.UNAVAILABLE}=[1.0]" in str(caught.value), str(caught.value)


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
    """A fill the export stopped short (ENOSPC before the target, a quota) is a
    reading the probe hands over, not a failed call: the verdict names the
    reached percentage, the bytes written against the target and the fill's
    first failure by class, errno and errno name."""

    probe = shared_probe(NODE)
    probe.script["nfs-fill"] = {
        "fill": {
            "fill_bytes": 160 << 20,
            "wanted_bytes": 251 << 20,
            "enospc": True,
            "error": "OSError",
            "errno": 28,
        },
        "statvfs": {"used_percent": 61.2},
    }
    ctx = context(
        "capacity_full", tmp_path, activity=None, probes={f"shared@{NODE}": probe}
    )
    mount = shared.mount_path(ctx, shared.PRIMARY)
    ctx.regional.cpu_python_responder = store_script({"activity": full_activity(mount)})
    with pytest.raises(VariantInconclusive) as caught:
        SPECS["capacity_full"].execute(ctx)
    assert str(caught.value) == (
        "the test export only reached 61.2 % used (fill wrote 160 of 251 MiB; "
        "OSError errno 28 ENOSPC)"
    )
    assert "unfill" in ctx.cleanup.pending(), "the fill is undone by the stack"


def test_capacity_full_is_inconclusive_without_a_statvfs_reading(
    tmp_path: Path,
) -> None:
    probe = shared_probe(NODE)
    probe.script["nfs-fill"] = {
        "fill": {
            "fill_bytes": 0,
            "wanted_bytes": 0,
            "enospc": False,
            "error": "NotADirectoryError",
            "errno": 20,
        }
    }
    ctx = context(
        "capacity_full", tmp_path, activity=None, probes={f"shared@{NODE}": probe}
    )
    mount = shared.mount_path(ctx, shared.PRIMARY)
    ctx.regional.cpu_python_responder = store_script({"activity": full_activity(mount)})
    with pytest.raises(VariantInconclusive) as caught:
        SPECS["capacity_full"].execute(ctx)
    assert str(caught.value) == (
        "the fill left no statvfs reading of the test export (fill wrote 0 of 0 MiB; "
        "NotADirectoryError errno 20 ENOTDIR)"
    )
    assert "unfill" in ctx.cleanup.pending(), "the fill is undone by the stack"


# --------------------------------------------------------------------------- #
# mount_disappears and io_only_failure: closed in the product, labelled
# --------------------------------------------------------------------------- #
def export_source() -> str:
    server = servers_by_suffix()["a"]
    return f"{server.ready['ip']}:{server.ready['export']}"


def labelled(mode: str) -> dict[str, str]:
    return {"failure_mode": mode, "source": export_source(), "fstype": "nfs4"}


def test_mount_disappears_passes_on_mount_missing_and_remounts(tmp_path: Path) -> None:
    spec = SPECS["mount_disappears"]
    ctx = context("mount_disappears", tmp_path, activity=None)
    mount = shared.mount_path(ctx, shared.PRIMARY)
    ctx.regional.cpu_python_responder = store_script(
        {
            "activity": lambda _args: hang_activity(
                mount, labels=labelled("MOUNT_MISSING")
            )
        }
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert spec.expected_outcome == "PASS" and spec.product_gap == ""
    probe = ctx.probes[f"shared@{NODE}"]
    names = [call[0] for call in probe.calls]
    umount = next(call for call in probe.calls if call[0] == "nfs-umount")
    assert "--forget" not in umount, (
        "the injection must leave the mount point directory"
    )
    assert names.index("nfs-umount") < names.index("nfs-remount"), names
    assert servers_by_suffix()["a"].paused is False, "the server was never paused"
    # The mount undo (umount + rmdir, which makes the product forget) stays on
    # the stack above the export delete.
    assert ctx.cleanup.pending() == ["export-a", f"mount-shared@{NODE}"], (
        ctx.cleanup.pending()
    )


def test_mount_disappears_fails_when_the_product_stays_quiet(tmp_path: Path) -> None:
    ctx = context("mount_disappears", tmp_path, activity=None)
    mount = shared.mount_path(ctx, shared.PRIMARY)
    ctx.regional.cpu_python_responder = store_script(
        {"activity": quiet_activity(mount)}
    )
    result = SPECS["mount_disappears"].execute(ctx)
    assert result["verdict"] == "FAIL", result
    assert any("never reported" in error for error in result["errors"]), result[
        "errors"
    ]


def test_mount_disappears_fails_on_another_failure_mode_or_a_nameless_export(
    tmp_path: Path,
) -> None:
    ctx = context("mount_disappears", tmp_path, activity=None)
    mount = shared.mount_path(ctx, shared.PRIMARY)
    ctx.regional.cpu_python_responder = store_script(
        {
            "activity": lambda _args: hang_activity(
                mount, labels=labelled("PROBE_TIMEOUT")
            )
        }
    )
    result = SPECS["mount_disappears"].execute(ctx)
    assert result["verdict"] == "FAIL", result
    assert any("not ['MOUNT_MISSING']" in error for error in result["errors"]), result[
        "errors"
    ]
    ctx = context("mount_disappears", tmp_path / "nameless", activity=None)
    mount = shared.mount_path(ctx, shared.PRIMARY)
    ctx.regional.cpu_python_responder = store_script(
        {"activity": hang_activity(mount, labels={"failure_mode": "MOUNT_MISSING"})}
    )
    result = SPECS["mount_disappears"].execute(ctx)
    assert result["verdict"] == "FAIL", result
    assert any("source label" in error for error in result["errors"]), result["errors"]


def test_mount_disappears_is_inconclusive_when_the_directory_went_too(
    tmp_path: Path,
) -> None:
    probe = shared_probe(NODE)
    probe.script["nfs-umount"] = {"host_mount": None, "directory_present": False}
    ctx = context(
        "mount_disappears", tmp_path, activity=None, probes={f"shared@{NODE}": probe}
    )
    mount = shared.mount_path(ctx, shared.PRIMARY)
    ctx.regional.cpu_python_responder = store_script(
        {
            "activity": lambda _args: hang_activity(
                mount, labels=labelled("MOUNT_MISSING")
            )
        }
    )
    with pytest.raises(VariantInconclusive, match="teardown"):
        SPECS["mount_disappears"].execute(ctx)


def test_io_only_failure_passes_on_io_error_and_restores_io(tmp_path: Path) -> None:
    spec = SPECS["io_only_failure"]
    ctx = context("io_only_failure", tmp_path, activity=None)
    mount = shared.mount_path(ctx, shared.PRIMARY)
    ctx.regional.cpu_python_responder = store_script(
        {"activity": lambda _args: hang_activity(mount, labels=labelled("IO_ERROR"))}
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert spec.expected_outcome == "PASS" and spec.product_gap == ""
    assert result["stages"]["io_restored"] == [], result["stages"]
    server = servers_by_suffix()["a"]
    assert server.calls == ["create", "wait_ready", "mode-000", "mode-1777"], (
        server.calls
    )
    assert "export-a-mode" not in ctx.cleanup.pending(), ctx.cleanup.pending()
    assert result["io"]["errno"] == 13, result["io"]


def test_io_only_failure_accepts_a_timed_out_read_but_not_a_quiet_product(
    tmp_path: Path,
) -> None:
    ctx = context("io_only_failure", tmp_path, activity=None)
    mount = shared.mount_path(ctx, shared.PRIMARY)
    ctx.regional.cpu_python_responder = store_script(
        {"activity": lambda _args: hang_activity(mount, labels=labelled("IO_TIMEOUT"))}
    )
    assert SPECS["io_only_failure"].execute(ctx)["verdict"] == "PASS"
    ctx = context("io_only_failure", tmp_path / "quiet", activity=None)
    mount = shared.mount_path(ctx, shared.PRIMARY)
    ctx.regional.cpu_python_responder = store_script(
        {"activity": quiet_activity(mount)}
    )
    result = SPECS["io_only_failure"].execute(ctx)
    assert result["verdict"] == "FAIL", result
    assert any("statvfs answered" in error for error in result["errors"]), result[
        "errors"
    ]
    assert servers_by_suffix()["a"].mode == "1777", "mode restored after the verdict"


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


def _pair_context(tmp_path: Path, build) -> Any:
    """A two-client context whose per-node reads come from ``build(node, mount)``."""

    ctx = context("shared_failure_domain", tmp_path, activity=None, peer=PEER)
    mount = shared.mount_path(ctx, shared.PRIMARY)

    def per_node(arguments: tuple[str, ...]) -> dict[str, Any]:
        return build(arguments[1], mount)

    ctx.regional.cpu_python_responder = store_script({"activity": per_node})
    return ctx


def test_shared_failure_domain_needs_two_clients_with_one_export_source(
    tmp_path: Path,
) -> None:
    spec = SPECS["shared_failure_domain"]
    with pytest.raises(VariantNotRun, match="node-b"):
        spec.execute(context("shared_failure_domain", tmp_path, activity=None))
    # No source label on the peer: nothing the control plane could correlate.
    ctx = _pair_context(
        tmp_path,
        lambda node, mount: hang_activity(
            mount, node=node, source=export_source() if node == NODE else None
        ),
    )
    with pytest.raises(VariantNotRun, match="no source label"):
        spec.execute(ctx)
    # Two different sources: not one export.
    ctx = _pair_context(
        tmp_path,
        lambda node, mount: hang_activity(
            mount, node=node, source=f"10.0.9.{1 if node == NODE else 2}:/a"
        ),
    )
    with pytest.raises(VariantNotRun, match="one and the same export source"):
        spec.execute(ctx)
    assert servers_by_suffix()["a"].calls == ["create", "wait_ready"], (
        "refused before the export was paused"
    )


def test_shared_failure_domain_fails_on_two_unrelated_node_diagnoses(
    tmp_path: Path,
) -> None:
    spec = SPECS["shared_failure_domain"]
    ctx = _pair_context(
        tmp_path,
        lambda node, mount: hang_activity(mount, node=node, source=export_source()),
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL", result
    assert (
        result["stages"][f"chain@{NODE}"] == []
        and result["stages"][f"chain@{PEER}"] == []
    )
    assert any("two unrelated node diagnoses" in error for error in result["errors"]), (
        result["errors"]
    )
    assert result["export"] == export_source() and result["domain_records"] == []
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


def test_shared_failure_domain_passes_on_the_designed_record(tmp_path: Path) -> None:
    spec = SPECS["shared_failure_domain"]
    ctx = _pair_context(
        tmp_path,
        lambda node, mount: hang_activity(
            mount,
            node=node,
            source=export_source(),
            domain=failure_domain(export_source()),
        ),
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["domain_records"] == ["inc-domain-1"]
    assert result["stages"]["domain_recovered"] == [], result["stages"]
    assert spec.expected_outcome == "PASS" and spec.product_gap == ""


@pytest.mark.parametrize(
    ("domain", "needle"),
    [
        (lambda e: failure_domain(e, nodes=(NODE,)), "does not name"),
        (lambda e: failure_domain(e, action="COLLECT_EVIDENCE"), "action/workflow"),
        (
            lambda e: failure_domain(e, workflow_id="wf-domain"),
            "planned on the failure-domain",
        ),
        (lambda e: failure_domain(e, notifications=0), "expected exactly one"),
        (lambda e: failure_domain(e, notifications=2), "expected exactly one"),
        (
            lambda e: failure_domain(e, first_reason="shared storage outage"),
            "first reason",
        ),
        (lambda e: failure_domain(e, members=False), "no member line"),
        (
            lambda e: failure_domain(e, back_reference=False),
            "reason back to the record",
        ),
        (lambda e: failure_domain(e, state="DETECTED"), "did not clear RECOVERED"),
    ],
)
def test_shared_failure_domain_fails_when_the_record_is_not_the_designed_one(
    tmp_path: Path, domain, needle: str
) -> None:
    ctx = _pair_context(
        tmp_path,
        lambda node, mount: hang_activity(
            mount, node=node, source=export_source(), domain=domain(export_source())
        ),
    )
    result = SPECS["shared_failure_domain"].execute(ctx)
    assert result["verdict"] == "FAIL", result
    assert any(needle in error for error in result["errors"]), (
        needle,
        result["errors"],
    )


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
def test_failure_domain_errors_read_one_record_from_both_nodes_once() -> None:
    export = "10.0.9.1:/a"
    mount = "/mnt/gfrs-live/abc/shared"
    activities = {
        node: hang_activity(
            mount, node=node, source=export, domain=failure_domain(export)
        )
        for node in (NODE, PEER)
    }
    ids = {node: [f"inc-{node[-3:]}"] for node in (NODE, PEER)}
    assert shared.failure_domain_errors(activities, (NODE, PEER), export, ids) == []
    assert len(shared.domain_records(activities, export)) == 1, (
        "the record both nodes list is counted once"
    )
    assert shared.domain_records(activities, "10.0.9.2:/a") == [], (
        "another export's record does not count"
    )
    missing = {NODE: [], PEER: ids[PEER]}
    assert any(
        "has no shared_filesystem_unavailable incident" in error
        for error in shared.failure_domain_errors(
            activities, (NODE, PEER), export, missing
        )
    ), "a node without its own incident cannot be a member"
    assert shared.export_sources(activities[NODE], mount) == {export}
    assert shared.require_same_export(activities, mount) == export
    assert shared.domain_state(activities[NODE], "inc-domain-1") == "RECOVERED"


def test_delivery_errors_need_two_batches_with_the_companion_metric() -> None:
    since = T0 + timedelta(minutes=1)
    document = {"batches": [batch(2, sample(recovery.COMPANION_METRIC, None, 1.0))]}
    assert shared.delivery_errors(document, since), "one batch is not continuity"
    document["batches"].append(batch(3, sample(shared.UNAVAILABLE, "/m", 1.0)))
    assert shared.delivery_errors(document, since) == []
    before = {"batches": [batch(-5, sample(recovery.COMPANION_METRIC, None, 1.0))] * 2}
    assert shared.delivery_errors(before, since), (
        "batches from before the injection do not count"
    )


def test_a_failed_client_mount_keeps_the_server_log_and_names_the_cause(
    tmp_path: Path,
) -> None:
    """Live attempt 5 ended every variant on "host probe failed ...; output
    withheld": the export Pod was Ready, the mount got ENOENT from the server,
    and the Pod was deleted with its log. The module now saves the server log
    beside the case before the LIFO cleanup deletes the Pod, and the error
    names the node, the probe's sentence and the evidence file."""

    probe = shared_probe(NODE)
    failure = HostProbeError(
        "host probe failed (exit 1; error MountError 0123456789abcdef at "
        "nfs_mount:560; nfs mount failed: export server answered ENOENT for the "
        "export path); output withheld"
    )

    def refuse(_arguments: tuple[str, ...]) -> dict[str, Any]:
        raise failure

    probe.script["nfs-mount"] = refuse
    ctx = context(
        "stat_timeout", tmp_path, activity=None, probes={f"shared@{NODE}": probe}
    )
    with pytest.raises(RegionalFixtureError) as caught:
        SPECS["stat_timeout"].execute(ctx)
    text = str(caught.value)
    server = servers_by_suffix()["a"]
    assert server.name in text and NODE in text and "answered ENOENT" in text, text
    assert "export-a-log-at-mount-failure" in text, text
    assert any(call.startswith("logs-") for call in server.calls), server.calls
    saved = json.loads(
        (ctx.case_dir / "export-a-log-at-mount-failure.json").read_text()
    )
    assert saved["log"] == server.log_text and saved["node"] == NODE, saved
    assert saved["server"] == server.ready and "answered ENOENT" in saved["error"], (
        saved
    )
    assert ctx.cleanup.pending() == ["export-a", f"mount-shared@{NODE}"], (
        "the unmount and the delete are still on the stack for the runner"
    )


def waiting_full_activity(mount: str) -> dict[str, Any]:
    """The first snapshots after the fill: the incident is minted (the 98 %
    rule has no sustain) but VALIDATE_HOST still waits for the health summary
    after it -- the evidence newer than itself that it is judged on."""

    doc = full_activity(mount)
    doc["workflows"][0]["status"] = "RUNNING"
    doc["workflows"][0]["step_executions"] = [
        {
            "operation": "VALIDATE_HOST",
            "status": "WAITING",
            "details": {"node_pending": {NODE: ["host_diagnostic_probe"]}},
        }
    ]
    return doc


def test_capacity_full_waits_for_validate_host_to_be_judgeable(tmp_path: Path) -> None:
    ctx = context("capacity_full", tmp_path, activity=None)
    mount = shared.mount_path(ctx, shared.PRIMARY)
    probe = ctx.probes[f"shared@{NODE}"]
    after_fill = itertools.count()

    def product(_script: str, *_arguments: str, **_: Any) -> dict[str, Any]:
        filled = any(call[0] == "nfs-fill" for call in probe.calls)
        if filled and next(after_fill) < 3:
            return waiting_full_activity(mount)
        return full_activity(mount)

    ctx.regional.cpu_python_responder = product
    result = SPECS["capacity_full"].execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    detection = json.loads((ctx.case_dir / "full-detection-timeline.json").read_text())
    assert [entry["accepted"] for entry in detection["entries"]] == [
        False,
        False,
        False,
        True,
    ], "the chain is graded on the terminal validation, not the first snapshot"


def test_capacity_full_detection_budget_covers_the_validations_summary() -> None:
    # No sustain on the 98 % rule: the fill's edge mints the finding and only
    # VALIDATE_HOST's summary is still ahead -- one summary, the interval and
    # the margin, read from the node's cadence rather than a fixed 420 s.
    from scripts.e2e.regional.general_live_common import sustained_detection_budget

    site = SimpleNamespace(host_summary=300, host_interval=15)
    assert sustained_detection_budget(site, 0, summaries=1) == 300 + 15 + 180
    assert sustained_detection_budget(site, 0, summaries=1) > shared.DETECTION_BUDGET
