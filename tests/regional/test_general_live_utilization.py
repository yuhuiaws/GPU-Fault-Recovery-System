"""GF-GENERAL-LIVE-003 injector/verdict contract over scripted records."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.execution.terminal_state import (
    failure_incident_state,
    inconclusive_reason,
)
from gpu_fault.health_recovery_contract import recovery_contract
from gpu_fault.host_diagnostics import (
    diagnostic_failure_details,
    diagnostic_probe,
    validate_diagnostic_probe,
)
from gpu_fault.host_health import HostMetricSample, NodeHealthPolicy
from gpu_fault.models import (
    IncidentState,
    RecoveryAction,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.telemetry import TelemetryMetricLatest
from scripts.e2e.regional import general_live_utilization as util
from scripts.e2e.regional.general_live_common import STORE_POLL_SECONDS
from scripts.e2e.regional.general_live_fixture import VariantInconclusive, VariantNotRun
from scripts.e2e.regional.general_live_store import ACTIVITY_PROBE
from scripts.e2e.regional.general_live_verdicts import finding_incidents
from tests._builders import (
    build_store,
    fault_incident,
    host_telemetry_batch,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)
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


def idle_doc(metric: str, validation: str, attempt: str) -> dict[str, Any]:
    """An attempt-bound low-utilisation chain as the product leaves it: the
    validation's host diagnostic probe failed on the still-idle metric."""

    return low_doc(metric, validation, probe=metric, attempt=attempt)


def test_gpu_idle_passes(tmp_path: Path) -> None:
    spec = SPECS["gpu_idle"]
    settings = _settings(spec, tmp_path)
    doc = idle_doc("host_gpu_utilization_percent", "VALIDATE_GPU", "gl3-idle-1-a001")
    regional = FakeRegional(cpu_python=store_script({"activity": doc}))
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes=probes(), settings=settings
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_gpu_idle_fails_when_dcgm_diagnostic_runs(tmp_path: Path) -> None:
    spec = SPECS["gpu_idle"]
    settings = _settings(spec, tmp_path)
    doc = idle_doc("host_gpu_utilization_percent", "VALIDATE_GPU", "gl3-idle-1-a001")
    doc["workflows"][0]["official_steps"].insert(
        1, {"operation": "RUN_DCGM_DIAGNOSTIC", "parameters": {}}
    )
    regional = FakeRegional(cpu_python=store_script({"activity": doc}))
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes=probes(), settings=settings
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL"
    assert result["stages"]["chain"] == [
        "host_gpu_utilization_percent workflow wf-u plans ['RUN_DCGM_DIAGNOSTIC']"
    ], result["stages"]


def test_rank_cpu_idle_passes(tmp_path: Path) -> None:
    spec = SPECS["rank_cpu_idle"]
    settings = _settings(spec, tmp_path)
    doc = idle_doc("training_rank_cpu_percent", "VALIDATE_HOST", "gl3-rankcpu-1-a001")
    regional = FakeRegional(cpu_python=store_script({"activity": doc}))
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


def poll_clock(ctx: Any) -> Any:
    """One store poll per clock reading, as live: a read every 20 s.

    The default fake clock moves 1 s per reading, so a wait that runs out its
    budget polls hundreds of times; the timelines it writes grow with it.
    """

    ticks = itertools.count()
    start = datetime(2026, 10, 25, 3, 19, 46, tzinfo=timezone.utc)
    ctx.clock = lambda: start + timedelta(seconds=STORE_POLL_SECONDS * next(ticks))
    return ctx


def cache_oracle(*values: float) -> dict[str, Any]:
    """The node probe's meminfo oracle: one ``page_cache_percent`` per 5 s."""

    records = [{"page_cache_percent": value} for value in values]
    return {"records": records, "next_offset": len(records), "total": len(records)}


#: Five minutes at or above HIGH_PAGE_CACHE's 90 % (60 readings 5 s apart).
CACHE_HELD = cache_oracle(*([40.0] * 20 + [91.5] * 60))


def page_cache_run(
    tmp_path: Path, oracle: dict[str, Any], activity: dict[str, Any]
) -> dict[str, Any]:
    spec = SPECS["high_page_cache"]
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=FakeRegional(cpu_python=store_script({"activity": activity})),
        probes=probes(oracle),
        settings=_settings(spec, tmp_path),
    )
    return spec.execute(poll_clock(ctx))


def page_cache_doc() -> dict[str, Any]:
    return low_doc("page_cache_percent", "VALIDATE_HOST", probe="page_cache_percent")


def test_high_page_cache_passes(tmp_path: Path) -> None:
    result = page_cache_run(tmp_path, CACHE_HELD, page_cache_doc())
    assert result["verdict"] == "PASS", result["errors"]


def test_high_page_cache_fails_only_on_an_established_condition(tmp_path: Path) -> None:
    result = page_cache_run(tmp_path, CACHE_HELD, empty())
    assert result["verdict"] == "FAIL", result
    assert result["stages"]["chain"] == [
        "no page_cache_percent finding reached an incident"
    ], result["stages"]


@pytest.mark.parametrize(
    "oracle",
    [
        # Live 2026-10-25 shape: the cache never got near the threshold.
        cache_oracle(*[20.0 + index * 0.5 for index in range(80)]),
        # Reached it, but not for the 300-s sustain window.
        cache_oracle(*([40.0] * 20 + [91.5] * 59 + [89.0])),
        # No reading at all.
        cache_oracle(),
    ],
    ids=["never-high", "not-held", "no-reading"],
)
@pytest.mark.parametrize("activity", ["empty", "incident"])
def test_high_page_cache_is_inconclusive_without_the_condition(
    tmp_path: Path, oracle: dict[str, Any], activity: str
) -> None:
    # Never FAIL (nor PASS) on a condition the runner did not see held,
    # whatever the product recorded.
    document = page_cache_doc() if activity == "incident" else empty()
    with pytest.raises(VariantInconclusive, match="condition not established"):
        page_cache_run(tmp_path, oracle, document)


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


def mint_sustained_finding(monkeypatch: pytest.MonkeyPatch, metric: str) -> Any:
    """The finding the product's own policy mints for a sustained ``metric``
    breach on NODE, held by the running attempt ``job-a``/``attempt-a``."""

    for knob in (
        "LOW_UTILIZATION",
        "MEMORY_PRESSURE",
        "PAGE_CACHE",
        "LOCAL_FILESYSTEM",
    ):
        monkeypatch.setenv(f"GPU_FAULT_{knob}_DURATION_SECONDS", "15")
    comparison = NodeHealthPolicy.SUSTAINED_METRIC_RULES[metric][1]
    sample_device, _wanted = MINTED_DEVICES.get(metric, (None, None))
    store = build_store()
    # The low-utilisation rules count only on a GPU a running attempt holds.
    observe_running_attempt(store, NOW, node_ids=(NODE,))
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
            node_id=NODE,
            workload_state="ACTIVE",
            affected_workload_ids=["training/job/job-a"],
        )

    assert not policy.evaluate_metrics(batch("initial", 0)), "one reading is no breach"
    findings = policy.evaluate_metrics(batch("sustained", 15))
    assert len(findings) == 1, findings
    return findings[0]


@pytest.mark.parametrize("metric", sorted(NodeHealthPolicy.SUSTAINED_METRIC_RULES))
def test_runner_selects_the_sustained_incident_the_product_mints(
    monkeypatch: pytest.MonkeyPatch, metric: str
) -> None:
    _sample_device, wanted_device = MINTED_DEVICES.get(metric, (None, None))
    event_id = mint_sustained_finding(monkeypatch, metric).event_id
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
    return spec.execute(poll_clock(ctx))


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


# --------------------------------------------------------------------------- #
# gpu_idle / rank_cpu_idle: the attempt-bound diagnostic chain (live 2026-10-25)
# --------------------------------------------------------------------------- #
#: metric -> (validation step, the product's sample device, the fake job's attempt)
IDLE_CHAINS = {
    "gpu_idle": ("host_gpu_utilization_percent", "VALIDATE_GPU", "gl3-idle-1-a001"),
    "rank_cpu_idle": (
        "training_rank_cpu_percent",
        "VALIDATE_HOST",
        "gl3-rankcpu-1-a001",
    ),
}
#: Live 2026-10-25 10:25Z (release 6a95b90fcd82): the incident first showed on
#: the third store read; gpu_idle's VALIDATE_GPU was WAITING on it and
#: terminal two reads later (the settled wait accepted at 10:33:47Z);
#: rank_cpu_idle's VALIDATE_HOST had already FAILED on the first sight.
WAITING_READS = {3, 4}
FAILED_READ = 5


def fresh_sample(finding: Any, value: float, *, seconds: int = 15) -> Any:
    return TelemetryMetricLatest(
        cluster_id=finding.cluster_id,
        node_id=finding.node_id,
        observed_at=finding.observed_at + timedelta(seconds=seconds),
        name=finding.metric_name,
        value=value,
        unit="percent",
        device=finding.device,
        labels={},
    )


def validation_steps(finding: Any, validation: str) -> list[Any]:
    """FREEZE_EVIDENCE then the validation, parameterised as ``health.py``
    does: the metric's host diagnostic probe and the recovery contract."""

    parameters: dict[str, Any] = {"host_diagnostic_probe": diagnostic_probe(finding)}
    contract = recovery_contract(finding)
    if contract is not None:
        parameters["health_recovery_contract"] = contract
    return [
        workflow_step(WorkflowOperation.FREEZE_EVIDENCE, node_ids=[NODE]),
        workflow_step(
            WorkflowOperation[validation], node_ids=[NODE], parameters=parameters
        ),
    ]


@pytest.mark.parametrize("variant", sorted(IDLE_CHAINS))
def test_idle_validation_waits_then_fails_on_the_held_metric(
    monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    """What the product does with an idle attempt, pinned to its own code.

    The validation step carries a host diagnostic probe on the metric
    (``health.py``); the probe is pending -- the step WAITS -- until a sample
    newer than the finding arrives, fails ``host_diagnostic:<metric>`` while
    the attempt stays idle, and passes only once the metric recovered; a
    FAILED validation-only workflow closes its incident RECOVERED "diagnostic
    inconclusive", never ESCALATED (``terminal_state.py``).
    """

    metric, validation, _attempt = IDLE_CHAINS[variant]
    finding = mint_sustained_finding(monkeypatch, metric)
    assert finding.recommended_action is RecoveryAction.RUN_DIAGNOSTICS
    probe = diagnostic_probe(finding)
    assert probe is not None and probe["metric"] == metric, probe
    assert (probe["comparison"], probe["threshold"]) == ("lte", 5.0), probe

    pending, failure = validate_diagnostic_probe(probe, [])
    assert pending == {"missing_diagnostic_metric": [metric], "device": "GPU-0"}
    assert failure is None, "no fresh sample: the step waits, it does not judge"

    pending, failure = validate_diagnostic_probe(probe, [fresh_sample(finding, 0.0)])
    assert (pending, failure) == (None, f"host_diagnostic:{metric}")
    details = diagnostic_failure_details(probe, {}, {NODE: [failure]})
    assert details == {"host_diagnostic_failure": True}

    pending, failure = validate_diagnostic_probe(probe, [fresh_sample(finding, 42.0)])
    assert (pending, failure) == (None, None), "a busy attempt passes the probe"

    workflow = workflow_request(
        "wf-idle",
        f"inc-{finding.event_id}",
        WorkflowStatus.FAILED,
        official_steps=validation_steps(finding, validation),
        step_executions=[
            workflow_step_execution(0, WorkflowOperation.FREEZE_EVIDENCE),
            workflow_step_execution(
                1,
                WorkflowOperation[validation],
                WorkflowStepStatus.FAILED,
                error=f"validation failed on nodes: {NODE} (host_diagnostic:{metric})",
                details={"node_failures": {NODE: [f"host_diagnostic:{metric}"]}},
            ),
        ],
    )
    assert failure_incident_state(workflow) is IncidentState.RECOVERED
    error = workflow.step_executions[1].error
    assert inconclusive_reason(error) == f"diagnostic inconclusive: {error}"


class RecordedStore:
    """The node's records read by read, through the real activity probe.

    ``stages`` maps a store read number to a writer ``(store, since)`` that
    puts the product's records into their state as of that read.
    """

    def __init__(self, stages: dict[int, Callable[[Any, datetime], None]]) -> None:
        self.store = build_store()
        self.stages = stages
        self.reads = 0

    def __call__(self, script: str, *arguments: str) -> dict[str, Any]:
        assert script is ACTIVITY_PROBE, "the variant reads only node activity"
        self.reads += 1
        writer = self.stages.get(self.reads)
        if writer is not None:
            writer(self.store, datetime.fromisoformat(arguments[2]))
        return activity_read(self.store, arguments)


def idle_chain(
    finding: Any, validation: str, attempt: str, stage: str
) -> Callable[[Any, datetime], None]:
    """Writer of the attempt-bound chain in one of its recorded states.

    ``waiting``: FREEZE_EVIDENCE done, the validation WAITING for a fresh
    probe sample, workflow RUNNING, incident ACTION_PENDING. ``failed``: the
    probe failed on the held metric, workflow FAILED, incident RECOVERED
    "diagnostic inconclusive". ``succeeded``: the validation passed while the
    attempt was idle (what a product defect would look like).
    """

    metric = finding.metric_name
    operation = WorkflowOperation[validation]
    error = f"validation failed on nodes: {NODE} (host_diagnostic:{metric})"

    def write(store: Any, since: datetime) -> None:
        created = since + timedelta(seconds=405)
        now = created + timedelta(seconds=30 if stage == "waiting" else 55)
        if stage == "waiting":
            execution = workflow_step_execution(
                1,
                operation,
                WorkflowStepStatus.WAITING,
                details={
                    "pending_nodes": [NODE],
                    "node_pending": {
                        NODE: {"missing_diagnostic_metric": [metric], "device": "GPU-0"}
                    },
                    "failed_nodes_observed": [],
                    "node_failures_observed": {},
                },
                started_at=created,
                updated_at=now,
            )
            status, state, reasons = (
                WorkflowStatus.RUNNING,
                IncidentState.ACTION_PENDING,
                [finding.reason],
            )
        elif stage == "failed":
            execution = workflow_step_execution(
                1,
                operation,
                WorkflowStepStatus.FAILED,
                error=error,
                details={
                    "failed_nodes": [NODE],
                    "node_failures": {NODE: [f"host_diagnostic:{metric}"]},
                    "validation": validation,
                    "host_diagnostic_failure": True,
                },
                started_at=created,
                updated_at=now,
            )
            status, state, reasons = (
                WorkflowStatus.FAILED,
                IncidentState.RECOVERED,
                [finding.reason, inconclusive_reason(error)],
            )
        else:
            execution = workflow_step_execution(
                1, operation, started_at=created, updated_at=now
            )
            status, state, reasons = (
                WorkflowStatus.SUCCEEDED,
                IncidentState.RECOVERED,
                [finding.reason],
            )
        incident_id = f"inc-{finding.event_id}"
        store.save_incident_and_workflow(
            fault_incident(
                incident_id,
                finding.event_id,
                "NODE_HEALTH",
                CLUSTER,
                node_ids=[NODE],
                policy_version="site-host-resource-health",
                policy_source="SITE_HOST_RESOURCE_HEALTH",
                job_id=attempt.rsplit("-", 1)[0],
                attempt_id=attempt,
                effective_action=RecoveryAction.RUN_DIAGNOSTICS,
                state=state,
                workflow_request_id="wf-idle",
                reasons=reasons,
                created_at=created,
                updated_at=now,
            ),
            workflow_request(
                "wf-idle",
                incident_id,
                status,
                official_steps=validation_steps(finding, validation),
                step_executions=[
                    workflow_step_execution(
                        0,
                        WorkflowOperation.FREEZE_EVIDENCE,
                        started_at=created,
                        updated_at=created,
                    ),
                    execution,
                ],
                created_at=created,
                updated_at=now,
            ),
        )

    return write


def idle_run(
    tmp_path: Path, variant: str, recorded: RecordedStore
) -> tuple[dict[str, Any], Any]:
    spec = SPECS[variant]
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=FakeRegional(cluster_id=CLUSTER, cpu_python=recorded),
        probes=probes(),
        settings=_settings(spec, tmp_path),
    )
    return spec.execute(poll_clock(ctx)), ctx


def accepted_flags(ctx: Any, name: str) -> list[bool]:
    import json

    timeline = json.loads((ctx.case_dir / f"{name}-timeline.json").read_text())
    return [bool(entry["accepted"]) for entry in timeline["entries"]]


def test_gpu_idle_judges_the_validation_once_it_finished(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Live 2026-10-25 003.gpu_idle (ip-10-0-43-5): the third read showed the
    # incident with VALIDATE_GPU WAITING for a fresh probe sample; the chain
    # was judged off that snapshot ("VALIDATE_GPU ended WAITING: ") while the
    # step finished FAILED on the held metric within the next two reads.
    metric, validation, attempt = IDLE_CHAINS["gpu_idle"]
    finding = mint_sustained_finding(monkeypatch, metric)
    recorded = RecordedStore(
        {
            **{
                read: idle_chain(finding, validation, attempt, "waiting")
                for read in WAITING_READS
            },
            FAILED_READ: idle_chain(finding, validation, attempt, "failed"),
        }
    )
    result, ctx = idle_run(tmp_path, "gpu_idle", recorded)
    assert result["verdict"] == "PASS", result["errors"]
    assert accepted_flags(ctx, "gpu-idle") == [False] * (FAILED_READ - 1) + [True]
    import json

    chain = json.loads((ctx.case_dir / "gpu-idle-chain.json").read_text())
    assert chain["accepted"] and chain["validation"] == "VALIDATE_GPU"
    assert chain["workflows"][0]["status"] == "FAILED", "the judged records"


def test_rank_cpu_idle_takes_the_failed_validation_as_the_held_fault(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Live 2026-10-25 003.rank_cpu_idle (ip-10-0-32-162): VALIDATE_HOST FAILED
    # "validation failed on nodes: ... (host_diagnostic:training_rank_cpu_
    # percent)" while the ranks slept -- the product judging the held fault,
    # which the runner reported as "VALIDATE_HOST ended FAILED".
    metric, validation, attempt = IDLE_CHAINS["rank_cpu_idle"]
    finding = mint_sustained_finding(monkeypatch, metric)
    recorded = RecordedStore(
        {min(WAITING_READS): idle_chain(finding, validation, attempt, "failed")}
    )
    result, ctx = idle_run(tmp_path, "rank_cpu_idle", recorded)
    assert result["verdict"] == "PASS", result["errors"]
    assert accepted_flags(ctx, "rank-cpu") == [False, False, True]


@pytest.mark.parametrize("variant", sorted(IDLE_CHAINS))
def test_idle_validation_that_passes_while_idle_fails_the_variant(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, variant: str
) -> None:
    metric, validation, attempt = IDLE_CHAINS[variant]
    finding = mint_sustained_finding(monkeypatch, metric)
    recorded = RecordedStore(
        {min(WAITING_READS): idle_chain(finding, validation, attempt, "succeeded")}
    )
    result, _ctx = idle_run(tmp_path, variant, recorded)
    assert result["verdict"] == "FAIL", result
    # Both the step and the close are wrong then: a pass while idle, and a
    # RECOVERED that does not say "diagnostic inconclusive".
    assert result["stages"]["chain"] == [
        f"{validation} succeeded while the fault was still held",
        f"{metric} incident is RECOVERED while the fault was held, without the "
        "'diagnostic inconclusive' reason",
    ], result["stages"]


def test_gpu_idle_validation_that_never_finishes_is_the_products_fail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    metric, validation, attempt = IDLE_CHAINS["gpu_idle"]
    finding = mint_sustained_finding(monkeypatch, metric)
    recorded = RecordedStore(
        {min(WAITING_READS): idle_chain(finding, validation, attempt, "waiting")}
    )
    result, ctx = idle_run(tmp_path, "gpu_idle", recorded)
    assert result["verdict"] == "FAIL", result
    assert result["stages"]["chain"] == [
        "VALIDATE_GPU did not finish: still WAITING "
        f"({NODE}: device, missing_diagnostic_metric)"
    ], result["stages"]
    assert not any(accepted_flags(ctx, "gpu-idle")), "the wait ran out"


def _settings(spec: Any, tmp_path: Path, *, approvals: tuple[str, ...] | None = None):
    settings = make_settings(spec)
    updates: dict[str, Any] = {"site_file": tmp_path / "site.yaml"}
    (tmp_path / "site.yaml").write_text("site: test\n")
    if approvals is not None:
        updates["approvals"] = approvals
    return type(settings)(**{**settings.__dict__, **updates})
