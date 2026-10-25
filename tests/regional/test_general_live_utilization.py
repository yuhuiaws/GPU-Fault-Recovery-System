"""GF-GENERAL-LIVE-003 injector/verdict contract over scripted records."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.host_health import HostMetricSample, NodeHealthPolicy
from gpu_fault.models import WorkflowStatus
from scripts.e2e.regional import general_live_utilization as util
from scripts.e2e.regional.general_live_common import STORE_POLL_SECONDS
from scripts.e2e.regional.general_live_fixture import VariantInconclusive, VariantNotRun
from scripts.e2e.regional.general_live_store import ACTIVITY_PROBE
from scripts.e2e.regional.general_live_verdicts import finding_incidents
from tests._builders import build_store, host_telemetry_batch, workflow_request
from tests.host_health._support import NOW, observe_running_attempt
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    activity_read,
    host_batch,
    incident,
    make_context,
    make_settings,
    save_diagnostic_incident,
    store_script,
    workflow,
)

SPECS = {spec.variant: spec for spec in util.VARIANTS}
NODE = "ip-10-0-43-5.us-west-2.compute.internal"


class FakeJob:
    instances: list["FakeJob"] = []

    def __init__(self, ctx: Any, *, name: str, gpu_count: int = 8) -> None:
        self.name = name
        self.gpu_count = gpu_count
        FakeJob.instances.append(self)

    def start(self) -> dict[str, Any]:
        return {"pods": []}

    def identity(self) -> dict[str, Any]:
        return {
            "job_id": self.name,
            "attempt_id": f"{self.name}-a001",
            "gpu_uuids": ["GPU-1"],
            "pods": [],
        }

    def cleanup(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _patch_job(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeJob.instances = []
    monkeypatch.setattr(util, "ManagedIdleJob", FakeJob)


def survey() -> dict[str, Any]:
    return {
        "cpu_count": 8,
        "tools": {
            name: True
            for name in ("losetup", "mkfs.ext4", "mount", "fallocate", "systemd-run")
        },
        "modules": {},
        "backing": [
            {
                "root": "/var/lib/gpu-fault-acceptance",
                "mount": {"fstype": "ext4"},
                "statvfs": {"available_bytes": 400 * 1024**3, "used_percent": 20.0},
            }
        ],
        "collector_env": {
            "GPU_FAULT_HOST_INTERVAL_SECONDS": "15",
            "GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS": "300",
        },
        "residue": {"units": [], "paused_units": {}, "paused_processes": {}},
    }


def node_probe(oracle: dict[str, Any] | None = None) -> FakeProbe:
    return FakeProbe(
        {
            "survey": survey(),
            "oracle-start": {"unit": "oracle"},
            "oracle-stop": {},
            "oracle-read": oracle or {"records": [], "next_offset": 0, "total": 0},
            "memory-start": {"unit": "mem", "deadman": "d"},
            "memory-stop": {"memory": {}},
            "page-cache-start": {"unit": "pc", "file": "/x", "deadman": "d"},
            "page-cache-stop": {"removed": ["/x"]},
        }
    )


def storage_probe() -> FakeProbe:
    return FakeProbe(
        {
            "fs-create": {"name": "fs", "mountpoint": "/mnt/gfrs-live/x/fs"},
            "fs-fill": {"statvfs": {"used_percent": 92.0}},
            "fs-unfill": {"statvfs": {"used_percent": 20.0}},
            "fs-destroy": {"destroyed": True},
        }
    )


def window_probe() -> FakeProbe:
    return FakeProbe(
        {
            "open-window": {"after": {"ActiveState": "active"}},
            "close-window": {"after": {"ActiveState": "active"}},
        }
    )


def probes(oracle: dict[str, Any] | None = None) -> dict[str, FakeProbe]:
    return {
        f"node@{NODE}": node_probe(oracle),
        f"storage@{NODE}": storage_probe(),
        f"window@{NODE}": window_probe(),
    }


def product_rule(metric: str) -> str:
    """The rule id the product mints ``metric``'s sustained finding under.

    Read from the product's table, never from the runner: a fake built from
    the runner's own constant agrees with the runner even when both are wrong
    (the threshold family ``memory`` kept these tests green while the live
    run never matched the product's LOW_MEMORY_AVAILABLE incident).
    """

    return NodeHealthPolicy.SUSTAINED_METRIC_RULES[metric][0]


def low_doc(
    metric: str,
    validation: str,
    device: str = "node",
    *,
    probe: str | None = None,
    attempt: str | None = None,
) -> dict[str, Any]:
    inc = incident(
        "inc-u",
        metric,
        device=device,
        rule_id=product_rule(metric),
        state="ACTION_PENDING",
        workflow="wf-u",
    )
    if attempt is not None:
        inc["attempt_id"] = attempt
    executions = [
        {
            "operation": validation,
            "status": "FAILED" if probe else "SUCCEEDED",
            "details": {"host_diagnostic_failure": bool(probe)},
        }
    ]
    wf = workflow(
        "wf-u",
        "inc-u",
        ["FREEZE_EVIDENCE", validation],
        status="FAILED" if probe else "SUCCEEDED",
        probe_metric=probe,
        executions=executions,
    )
    return {
        "incidents": [inc],
        "workflows": [wf],
        "batches": [
            host_batch(
                "2026-10-24T00:05:00+00:00",
                metric,
                2.0,
                device=None if device == "node" else device,
            )
        ],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
        "collector_statuses": [],
    }


def empty() -> dict[str, Any]:
    return {
        "incidents": [],
        "workflows": [],
        "batches": [],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
        "collector_statuses": [],
    }


def test_gpu_idle_passes(tmp_path: Path) -> None:
    spec = SPECS["gpu_idle"]
    settings = _settings(spec, tmp_path)
    regional = FakeRegional(
        cpu_python=store_script(
            {
                "activity": low_doc(
                    "host_gpu_utilization_percent",
                    "VALIDATE_GPU",
                    attempt="gl3-idle-1-a001",
                )
            }
        )
    )
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes=probes(), settings=settings
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_gpu_idle_fails_when_dcgm_diagnostic_runs(tmp_path: Path) -> None:
    spec = SPECS["gpu_idle"]
    settings = _settings(spec, tmp_path)
    doc = low_doc(
        "host_gpu_utilization_percent", "VALIDATE_GPU", attempt="gl3-idle-1-a001"
    )
    doc["workflows"][0]["official_steps"].insert(
        1, {"operation": "RUN_DCGM_DIAGNOSTIC", "parameters": {}}
    )
    regional = FakeRegional(cpu_python=store_script({"activity": doc}))
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes=probes(), settings=settings
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL"


def test_rank_cpu_idle_passes(tmp_path: Path) -> None:
    spec = SPECS["rank_cpu_idle"]
    settings = _settings(spec, tmp_path)
    regional = FakeRegional(
        cpu_python=store_script(
            {
                "activity": low_doc(
                    "training_rank_cpu_percent",
                    "VALIDATE_HOST",
                    attempt="gl3-rankcpu-1-a001",
                )
            }
        )
    )
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes=probes(), settings=settings
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_low_available_memory_requires_approval(tmp_path: Path) -> None:
    spec = SPECS["low_available_memory"]
    settings = _settings(spec, tmp_path, approvals=())
    regional = FakeRegional(cpu_python=store_script({"activity": empty()}))
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes=probes(), settings=settings
    )
    with pytest.raises(VariantNotRun):
        spec.execute(ctx)


def test_low_available_memory_passes(tmp_path: Path) -> None:
    spec = SPECS["low_available_memory"]
    settings = _settings(spec, tmp_path, approvals=("HOST_MEMORY_PRESSURE",))
    oracle = {
        "records": [{"memory_available_percent": 3.0} for _ in range(40)],
        "next_offset": 40,
        "total": 40,
    }
    regional = FakeRegional(
        cpu_python=store_script(
            {
                "activity": low_doc(
                    "memory_available_percent",
                    "VALIDATE_HOST",
                    probe="memory_available_percent",
                )
            }
        )
    )
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes=probes(oracle),
        settings=settings,
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_high_page_cache_passes(tmp_path: Path) -> None:
    spec = SPECS["high_page_cache"]
    settings = _settings(spec, tmp_path)
    regional = FakeRegional(
        cpu_python=store_script(
            {
                "activity": low_doc(
                    "page_cache_percent", "VALIDATE_HOST", probe="page_cache_percent"
                )
            }
        )
    )
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes=probes(), settings=settings
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_scratch_full_passes(tmp_path: Path) -> None:
    spec = SPECS["scratch_full"]
    settings = _settings(spec, tmp_path)
    mount = util._fs_mount(
        make_context(
            spec,
            tmp_path=tmp_path,
            regional=FakeRegional(cpu_python=store_script({"activity": empty()})),
            probes=probes(),
            settings=settings,
        )
    )
    regional = FakeRegional(
        cpu_python=store_script(
            {
                "activity": low_doc(
                    "local_filesystem_used_percent",
                    "VALIDATE_HOST",
                    device=mount,
                    probe="local_filesystem_used_percent",
                )
            }
        )
    )
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes=probes(), settings=settings
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_scratch_full_inconclusive_below_90(tmp_path: Path) -> None:
    spec = SPECS["scratch_full"]
    settings = _settings(spec, tmp_path)
    store = storage_probe()
    store.script["fs-fill"] = {"statvfs": {"used_percent": 50.0}}
    regional = FakeRegional(cpu_python=store_script({"activity": empty()}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={
            f"node@{NODE}": node_probe(),
            f"storage@{NODE}": store,
            f"window@{NODE}": window_probe(),
        },
        settings=settings,
    )
    with pytest.raises(VariantInconclusive):
        spec.execute(ctx)


def test_idle_node_and_ambiguous_ownership_pass(tmp_path: Path) -> None:
    for variant in ("idle_node", "ambiguous_ownership"):
        spec = SPECS[variant]
        settings = _settings(spec, tmp_path)
        regional = FakeRegional(cpu_python=store_script({"activity": empty()}))
        ctx = make_context(
            spec,
            tmp_path=tmp_path,
            regional=regional,
            probes=probes(),
            settings=settings,
        )
        result = spec.execute(ctx)
        assert result["verdict"] == "PASS", (variant, result["errors"])


def test_attempt_change_attributes_to_b(tmp_path: Path) -> None:
    spec = SPECS["attempt_change"]
    settings = _settings(spec, tmp_path)
    regional = FakeRegional(
        cpu_python=store_script(
            {
                "activity": low_doc(
                    "host_gpu_utilization_percent",
                    "VALIDATE_GPU",
                    attempt="gl3-attB-1-a001",
                )
            }
        )
    )
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes=probes(), settings=settings
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


# --------------------------------------------------------------------------- #
# the product's sustained rule ids
# --------------------------------------------------------------------------- #
def test_sustained_rule_ids_are_the_products() -> None:
    product = {
        metric: rule[0]
        for metric, rule in NodeHealthPolicy.SUSTAINED_METRIC_RULES.items()
    }
    assert util.SUSTAINED_RULES == product, (
        "the runner's sustained rule ids drifted from host_health.py"
    )


#: The device each sustained sample carries, and the device its variant asks
#: the matcher for (None: any device of the metric).
MINTED_DEVICES = {
    "training_rank_cpu_percent": ("GPU-0", None),
    "host_gpu_utilization_percent": ("GPU-0", None),
    "local_filesystem_used_percent": ("/mnt/gfrs-live/x/fs", "/mnt/gfrs-live/x/fs"),
}


@pytest.mark.parametrize("metric", sorted(NodeHealthPolicy.SUSTAINED_METRIC_RULES))
def test_runner_selects_the_sustained_incident_the_product_mints(
    monkeypatch: pytest.MonkeyPatch, metric: str
) -> None:
    for knob in (
        "LOW_UTILIZATION",
        "MEMORY_PRESSURE",
        "PAGE_CACHE",
        "LOCAL_FILESYSTEM",
    ):
        monkeypatch.setenv(f"GPU_FAULT_{knob}_DURATION_SECONDS", "15")
    comparison = NodeHealthPolicy.SUSTAINED_METRIC_RULES[metric][1]
    sample_device, wanted_device = MINTED_DEVICES.get(metric, (None, None))
    store = build_store()
    # The low-utilisation rules count only on a GPU a running attempt holds.
    observe_running_attempt(store, NOW)
    policy = NodeHealthPolicy(store)

    def batch(batch_id: str, seconds: int) -> Any:
        sample = HostMetricSample(
            name=metric,
            value=1.0 if comparison == "lte" else 99.0,
            unit="percent",
            device=sample_device,
        )
        return host_telemetry_batch(
            batch_id,
            NOW + timedelta(seconds=seconds),
            [sample],
            workload_state="ACTIVE",
            affected_workload_ids=["training/job/job-a"],
        )

    assert not policy.evaluate_metrics(batch("initial", 0)), "one reading is no breach"
    findings = policy.evaluate_metrics(batch("sustained", 15))
    assert len(findings) == 1, findings
    event_id = findings[0].event_id
    document = {
        "incidents": [
            {
                "incident_id": f"inc-{event_id}",
                "event_id": event_id,
                "event_type": "NODE_HEALTH",
                "state": "RECOVERED",
            }
        ]
    }
    selected = finding_incidents(
        document, metric, device=wanted_device, rule_id=util.SUSTAINED_RULES[metric]
    )
    assert [item["incident_id"] for item in selected] == [f"inc-{event_id}"], event_id


# --------------------------------------------------------------------------- #
# low_available_memory over the real activity probe (live 2026-10-25)
# --------------------------------------------------------------------------- #
CLUSTER = "gpu-eks-cluster"
OTHER_NODE = "ip-10-0-77-9.us-west-2.compute.internal"
#: Detection polls the store every 20 s; the product's records first show on
#: this read, the incident already RECOVERED (it was created, validated and
#: closed within one second between two reads).
FIRST_SEEN_READ = 3
#: The ramp to 5 % MemAvailable plus the 60-s sustain, as live.
FIRED_AFTER = timedelta(seconds=1084)


def host_event(metric: str, *, batch: int = 1761363470000000, node: str = NODE) -> str:
    """The product's event id of a node-level host finding (``host_health.py``)."""

    event = f"host-{node}-{batch}-{metric}-node"
    if metric in NodeHealthPolicy.SUSTAINED_METRIC_RULES:
        event += f"-{product_rule(metric).lower()}"
    return event


class ReplayedStore:
    """The node's records read by read, through the real activity probe.

    ``fire(store, since)`` writes the product's records just before read
    ``FIRST_SEEN_READ``; every read runs ``ACTIVITY_PROBE`` over the store.
    """

    def __init__(self, fire: Callable[[Any, datetime], None]) -> None:
        self.store = build_store()
        self.fire = fire
        self.reads = 0

    def __call__(self, script: str, *arguments: str) -> dict[str, Any]:
        assert script is ACTIVITY_PROBE, "the variant reads only node activity"
        self.reads += 1
        if self.reads == FIRST_SEEN_READ:
            self.fire(self.store, datetime.fromisoformat(arguments[2]))
        return activity_read(self.store, arguments)


def low_memory_pair(store: Any, created: datetime, *, node: str = NODE) -> str:
    """LOW_MEMORY_AVAILABLE and the co-firing 002 rule, both RECOVERED."""

    save_diagnostic_incident(
        store,
        host_event("memory_used_percent", node=node),
        "memory_used_percent",
        cluster_id=CLUSTER,
        node=node,
        created=created,
        reason="host memory pressure",
    )
    return save_diagnostic_incident(
        store,
        host_event("memory_available_percent", node=node),
        "memory_available_percent",
        cluster_id=CLUSTER,
        node=node,
        created=created,
        reason="available host memory remained below the warning threshold",
    )


def fired(store: Any, since: datetime) -> None:
    low_memory_pair(store, since + FIRED_AFTER)


def fired_before_the_run(store: Any, since: datetime) -> None:
    """An earlier run's pair (002 co-fires this rule), pulled into the read by
    a validated restore created since: the probe adds any such workflow's
    incident however old it is."""

    incident_id = low_memory_pair(store, since - timedelta(minutes=30))
    store.save_workflow(
        workflow_request(
            "wf-restore",
            incident_id,
            WorkflowStatus.SUCCEEDED,
            created_at=since + FIRED_AFTER,
            updated_at=since + FIRED_AFTER,
        )
    )


def fired_on_another_node(store: Any, since: datetime) -> None:
    low_memory_pair(store, since + FIRED_AFTER, node=OTHER_NODE)


def other_metrics_fired(store: Any, since: datetime) -> None:
    for metric, reason in (
        ("memory_used_percent", "host memory pressure"),
        ("page_cache_percent", "page cache remained near total physical memory"),
    ):
        save_diagnostic_incident(
            store,
            host_event(metric),
            metric,
            cluster_id=CLUSTER,
            node=NODE,
            created=since + FIRED_AFTER,
            reason=reason,
        )


def low_memory_run(tmp_path: Path, replay: ReplayedStore) -> dict[str, Any]:
    spec = SPECS["low_available_memory"]
    settings = _settings(spec, tmp_path, approvals=("HOST_MEMORY_PRESSURE",))
    oracle = {
        "records": [{"memory_available_percent": 3.0} for _ in range(40)],
        "next_offset": 40,
        "total": 40,
    }
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=FakeRegional(cluster_id=CLUSTER, cpu_python=replay),
        probes=probes(oracle),
        settings=settings,
    )
    # One store poll per clock reading, as live: a read every 20 s.
    ticks = itertools.count()
    start = datetime(2026, 10, 25, 3, 19, 46, tzinfo=timezone.utc)
    ctx.clock = lambda: start + timedelta(seconds=STORE_POLL_SECONDS * next(ticks))
    return spec.execute(ctx)


def test_low_available_memory_takes_the_incident_recovered_between_reads(
    tmp_path: Path,
) -> None:
    # Live 2026-10-25 (ip-10-0-43-5, release bf1bb0037847): the product opened
    # LOW_MEMORY_AVAILABLE and memory_used_percent at 03:37:50Z and closed both
    # RECOVERED "diagnostic inconclusive" in the same second; the runner kept
    # polling until 03:43 and failed "no memory_available_percent finding
    # reached an incident" -- it matched on the threshold family "memory".
    replay = ReplayedStore(fired)
    result = low_memory_run(tmp_path, replay)
    assert result["verdict"] == "PASS", result["errors"]
    assert replay.reads == FIRST_SEEN_READ, "detection stops at the first read"


@pytest.mark.parametrize(
    "fire",
    [fired_before_the_run, fired_on_another_node, other_metrics_fired],
    ids=["older", "other-node", "other-metric"],
)
def test_low_available_memory_takes_no_incident_it_did_not_cause(
    tmp_path: Path, fire: Callable[[Any, datetime], None]
) -> None:
    replay = ReplayedStore(fire)
    result = low_memory_run(tmp_path, replay)
    assert result["verdict"] == "FAIL", result
    assert result["stages"]["chain"] == [
        "no memory_available_percent finding reached an incident"
    ], result["stages"]
    assert replay.reads > FIRST_SEEN_READ, "nothing to take: detection runs out"


def _settings(spec: Any, tmp_path: Path, *, approvals: tuple[str, ...] | None = None):
    settings = make_settings(spec)
    updates: dict[str, Any] = {"site_file": tmp_path / "site.yaml"}
    (tmp_path / "site.yaml").write_text("site: test\n")
    if approvals is not None:
        updates["approvals"] = approvals
    return type(settings)(**{**settings.__dict__, **updates})
