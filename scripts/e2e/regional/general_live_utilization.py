"""GF-GENERAL-LIVE-003 low utilisation and sustained resource anomalies.

The ``SITE_HOST_RESOURCE_HEALTH`` rules are sustained and workload-gated
(``host_health.py``, ``host_low_utilization.py``): LOW_GPU_UTILIZATION and
LOW_CPU_UTILIZATION only count while the control plane resolves one real
managed attempt with a live container on the node, on a GPU that attempt holds;
LOW_MEMORY_AVAILABLE/HIGH_PAGE_CACHE/HIGH_LOCAL_FILESYSTEM_USAGE are host-wide.
The idle-GPU variants run a managed job whose GPU duty is 0 (the acceptance
load's idle heartbeat); the memory, page-cache and scratch variants use the
host and storage probes. Every low-utilisation finding is bound to its attempt
and drives a diagnostic VALIDATE_GPU/VALIDATE_HOST, never a reset.
"""

from __future__ import annotations

from typing import Any

from scripts.e2e.regional.general_live_common import (
    Oracle,
    activity,
    close_window,
    open_window,
    run_survey,
    wait_for_incident,
    wait_settled,
)
from scripts.e2e.regional.general_live_fixture import (
    VariantContext,
    VariantInconclusive,
    VariantNotRun,
    backing_root,
    require_tools,
    utc_now,
)
from scripts.e2e.regional.general_live_registry import NATIVE, VariantSpec
from scripts.e2e.regional.general_live_verdicts import (
    DIAGNOSTIC_ONLY_STATES,
    ChainExpectation,
    chain_errors,
    finding_incidents,
    hardware_errors,
    oracle_values,
    samples,
    sustained,
    verdict,
)

CASE = "GF-GENERAL-LIVE-003"
LOW_DURATION = 300
LOW_BUDGET = LOW_DURATION + 300 + 120
MEMORY_DURATION = 60
PAGE_CACHE_DURATION = 300
LOCAL_FS_DURATION = 60
TEST_FS = "fs"
SINGLE_NODE_MANIFEST = "single-node-warm-spare-pytorchjob.yaml"


# The managed idle job is a thin wrapper over the proven ManagedWorkloadFixture
# and ImagePrewarmFixture; tests monkeypatch ``ManagedIdleJob`` so the verdict
# logic runs over scripted findings without a cluster, and
# test_general_live_managed_job_start.py runs its start with the real fixtures.
class ManagedIdleJob:
    def __init__(self, ctx: VariantContext, *, name: str, gpu_count: int = 8) -> None:
        from pathlib import Path

        from scripts.e2e.regional.acceptance_gpu_load import set_gpu_duty_percent
        from scripts.e2e.regional.managed_workload_fixture import (
            ImagePrewarmFixture,
            ManagedWorkloadFixture,
            ManagedWorkloadSettings,
            render_node_pinned_manifest,
        )

        yaml = __import__("yaml")
        if ctx.settings.site_file is None:
            raise VariantNotRun(
                "FIXTURE_NOT_READY", "a managed idle job needs --site-file"
            )
        source = (
            Path(__file__).with_name("manifests") / "training" / SINGLE_NODE_MANIFEST
        )
        rendered = render_node_pinned_manifest(
            source, ctx.case_dir / f"{name}.source.yaml", node=ctx.settings.node
        )
        document = yaml.safe_load(rendered.read_text(encoding="utf-8"))
        set_gpu_duty_percent(document, 0)
        _request_gpus(document, gpu_count)
        document.setdefault("metadata", {}).setdefault("labels", {})
        document["metadata"]["labels"].update(
            {"gpu-fault.io/job-id": name, "gpu-fault.io/attempt-id": f"{name}-a001"}
        )
        rendered.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        self.name = name
        self.node = ctx.settings.node
        self.prewarm = ImagePrewarmFixture(
            ctx.regional,
            case_id=ctx.spec.key,
            run_id=f"{ctx.run_id}-{name}",
            state_path=ctx.case_dir / f"{name}.prewarm.json",
        )
        self.fixture = ManagedWorkloadFixture(
            ctx.regional,
            ManagedWorkloadSettings(
                manifest=rendered,
                site_file=ctx.settings.site_file,
                job_id=name,
                attempt_id=f"{name}-a001",
                restart_budget=0,
                expected_pods=1,
                expected_gpu_count=gpu_count,
            ),
            state_path=ctx.case_dir / f"{name}.ownership.json",
        )

    def start(self) -> dict[str, Any]:
        # The node the manifest is pinned to. This used to pass
        # ``self.fixture.regional.settings`` -- the RegionalLiveSettings -- and
        # the prewarm failed serializing it (live 2026-10-24, 003.gpu_idle).
        self.prewarm.create([self.node])
        self.fixture.submit()
        return self.fixture.wait_running(timeout_seconds=900)

    def identity(self) -> dict[str, Any]:
        pods = self.fixture.pods()
        return {
            "job_id": self.name,
            "attempt_id": f"{self.name}-a001",
            "gpu_uuids": sorted(
                {uuid for pod in pods for uuid in pod.get("gpu_uuids", [])}
            ),
            "pods": pods,
        }

    def cleanup(self) -> dict[str, Any]:
        from scripts.e2e.regional.managed_workload_fixture import remove_managed_job

        return remove_managed_job(self.fixture, self.prewarm)


def _request_gpus(document: dict[str, Any], gpu_count: int) -> None:
    for replica in (document.get("spec", {}).get("pytorchReplicaSpecs") or {}).values():
        for container in ((replica.get("template") or {}).get("spec") or {}).get(
            "containers"
        ) or []:
            resources = container.setdefault("resources", {})
            for bucket in ("requests", "limits"):
                if "nvidia.com/gpu" in (resources.get(bucket) or {}):
                    resources[bucket]["nvidia.com/gpu"] = str(gpu_count)


def _managed_job(ctx: VariantContext, name: str, gpu_count: int = 8) -> ManagedIdleJob:
    job = ManagedIdleJob(ctx, name=name, gpu_count=gpu_count)
    ctx.cleanup.push(f"job-{name}", job.cleanup)
    job.start()
    return job


def gpu_idle(ctx: VariantContext) -> dict[str, Any]:
    run_survey(ctx)
    job = _managed_job(ctx, f"gl3-idle-{ctx.attempt}")
    identity = job.identity()
    started = utc_now()
    ctx.hold(LOW_DURATION + ctx.host_interval * 3, name="gpu-idle-hold")
    detection = wait_for_incident(
        ctx,
        started,
        "host_gpu_utilization_percent",
        rule_id="LOW_GPU_UTILIZATION",
        timeout_seconds=LOW_BUDGET,
        metrics={"host_gpu_utilization_percent"},
        name="gpu-idle",
    )
    incidents = finding_incidents(
        detection, "host_gpu_utilization_percent", rule_id="LOW_GPU_UTILIZATION"
    )
    stages = {
        "chain": chain_errors(
            detection,
            ChainExpectation(
                metric="host_gpu_utilization_percent",
                rule_id="LOW_GPU_UTILIZATION",
                device=None,
                operations=("FREEZE_EVIDENCE", "VALIDATE_GPU"),
                validation="VALIDATE_GPU",
                forbidden=_hardware() | {"RUN_DCGM_DIAGNOSTIC"},
                single=False,
                incident_states=DIAGNOSTIC_ONLY_STATES,
            ),
        ),
        "attribution": _attribution_errors(
            incidents, detection, identity["attempt_id"]
        ),
        "hardware": hardware_errors(detection),
    }
    wait_settled(
        ctx,
        started,
        [item["incident_id"] for item in incidents],
        timeout_seconds=LOW_BUDGET,
        metrics={"host_gpu_utilization_percent"},
        name="gpu-idle-settled",
    )
    return _result(stages, extra={"attempt": identity})


def rank_cpu_idle(ctx: VariantContext) -> dict[str, Any]:
    run_survey(ctx)
    job = _managed_job(ctx, f"gl3-rankcpu-{ctx.attempt}")
    identity = job.identity()
    started = utc_now()
    ctx.hold(LOW_DURATION + ctx.host_interval * 3, name="rank-cpu-hold")
    detection = wait_for_incident(
        ctx,
        started,
        "training_rank_cpu_percent",
        rule_id="LOW_CPU_UTILIZATION",
        timeout_seconds=LOW_BUDGET,
        metrics={"training_rank_cpu_percent"},
        name="rank-cpu",
    )
    incidents = finding_incidents(
        detection, "training_rank_cpu_percent", rule_id="LOW_CPU_UTILIZATION"
    )
    stages = {
        "chain": chain_errors(
            detection,
            ChainExpectation(
                metric="training_rank_cpu_percent",
                rule_id="LOW_CPU_UTILIZATION",
                device=None,
                operations=("FREEZE_EVIDENCE", "VALIDATE_HOST"),
                validation="VALIDATE_HOST",
                single=False,
                incident_states=DIAGNOSTIC_ONLY_STATES,
            ),
        ),
        "attribution": _attribution_errors(
            incidents, detection, identity["attempt_id"]
        ),
        "hardware": hardware_errors(detection),
    }
    wait_settled(
        ctx,
        started,
        [item["incident_id"] for item in incidents],
        timeout_seconds=LOW_BUDGET,
        metrics={"training_rank_cpu_percent"},
        name="rank-cpu-settled",
    )
    return _result(stages, extra={"attempt": identity})


#: How long the memory allocator may take to bring MemAvailable to the threshold.
MEMORY_RAMP_BUDGET = 1080


def low_available_memory(ctx: VariantContext) -> dict[str, Any]:
    survey = run_survey(ctx)
    require_tools(survey, "systemd-run")
    if "HOST_MEMORY_PRESSURE" not in ctx.settings.approvals:
        raise VariantNotRun(
            "NOT_AUTHORIZED", "memory pressure needs HOST_MEMORY_PRESSURE"
        )
    started = utc_now()
    hold = MEMORY_DURATION + 300 + 120
    # The allocator touches memory at a fixed pace: on a 2 TiB host it needs
    # about 14 minutes to reach the threshold (live 2026-10-24: MemAvailable
    # still 36 % when the detection budget ran out), so the load is held, and
    # detection timed, from the moment the oracle reads the threshold.
    ramp = MEMORY_RAMP_BUDGET
    oracle = Oracle(ctx, "mem")
    oracle.start(ramp + hold + 120, interval=5, meminfo=True)
    ctx.node(
        "memory-start",
        "--target-used-percent",
        "96.5",
        "--seconds",
        str(ramp + hold + 120),
        "--approved",
    )
    ctx.cleanup.push("memory", lambda: ctx.node("memory-stop"))

    def ramped() -> bool | None:
        oracle.read()
        values = oracle_values(
            oracle.records, lambda record: record.get("memory_available_percent")
        )
        return True if values and values[-1] <= 5.0 else None

    if not ctx.wait_until(
        ramped, timeout_seconds=ramp, poll_seconds=15, name="low-mem-ramp"
    ):
        raise VariantInconclusive(
            f"the allocator did not bring MemAvailable to 5 % within {ramp} s"
        )
    detection = wait_for_incident(
        ctx,
        started,
        "memory_available_percent",
        rule_id="memory",
        timeout_seconds=hold,
        metrics={"memory_available_percent", "memory_used_percent"},
        name="low-mem",
    )
    oracle.read()
    available = oracle_values(
        oracle.records, lambda record: record.get("memory_available_percent")
    )
    if not sustained(available, lambda value: value <= 5.0, MEMORY_DURATION // 5):
        raise VariantInconclusive(f"MemAvailable never stayed <= 5 %: {available[-6:]}")
    stages = {
        "chain": chain_errors(
            detection,
            ChainExpectation(
                metric="memory_available_percent",
                rule_id="memory",
                device=None,
                operations=("FREEZE_EVIDENCE", "VALIDATE_HOST"),
                validation="VALIDATE_HOST",
                validation_must_fail=True,
                probe_metric="memory_available_percent",
                incident_states=DIAGNOSTIC_ONLY_STATES,
            ),
        ),
        "hardware": hardware_errors(detection),
    }
    ctx.node("memory-stop")
    ctx.cleanup.discard("memory")
    return _result(
        stages,
        notes=["LOW_MEMORY_AVAILABLE co-fires with memory_used_percent (002)"],
        extra={"attempt": None},
    )


def high_page_cache(ctx: VariantContext) -> dict[str, Any]:
    survey = run_survey(ctx)
    require_tools(survey, "systemd-run")
    root = backing_root(survey)
    started = utc_now()
    hold = PAGE_CACHE_DURATION + 300 + 120
    opened = ctx.node(
        "page-cache-start",
        "--fraction",
        "0.92",
        "--seconds",
        str(hold + 120),
        "--backing",
        root,
    )
    ctx.cleanup.push("page-cache", lambda: ctx.node("page-cache-stop"))
    ctx.evidence.write("page-cache-start", opened)
    detection = wait_for_incident(
        ctx,
        started,
        "page_cache_percent",
        rule_id="page_cache",
        timeout_seconds=hold,
        metrics={"page_cache_percent", "memory_used_percent"},
        name="page-cache",
    )
    incidents = finding_incidents(detection, "page_cache_percent", rule_id="page_cache")
    stages = {
        "chain": chain_errors(
            detection,
            ChainExpectation(
                metric="page_cache_percent",
                rule_id="page_cache",
                device=None,
                operations=("FREEZE_EVIDENCE", "VALIDATE_HOST"),
                validation="VALIDATE_HOST",
                validation_must_fail=True,
                probe_metric="page_cache_percent",
                incident_states=DIAGNOSTIC_ONLY_STATES,
            ),
        ),
        "hardware": hardware_errors(detection),
        "memory_low": _memory_used_low(detection),
    }
    ctx.node("page-cache-stop")
    ctx.cleanup.discard("page-cache")
    wait_settled(
        ctx,
        started,
        [item["incident_id"] for item in incidents],
        timeout_seconds=hold,
        metrics={"page_cache_percent"},
        name="page-cache-settled",
    )
    return _result(stages)


def scratch_full(ctx: VariantContext) -> dict[str, Any]:
    survey = run_survey(ctx)
    require_tools(survey, "losetup", "mkfs.ext4", "mount", "fallocate")
    root = backing_root(survey)
    created = ctx.storage(
        "fs-create",
        "--name",
        TEST_FS,
        "--size-mib",
        "2048",
        "--backing",
        root,
        "--deadman-seconds",
        str(ctx.spec.max_minutes * 60 + 600),
    )
    ctx.cleanup.push("fs", lambda: ctx.storage("fs-destroy", "--name", TEST_FS))
    mount = created["mountpoint"]
    open_window(
        ctx, f"GPU_FAULT_FILESYSTEMS={_fs_mount(ctx)}", restore_seconds=LOW_BUDGET + 600
    )
    started = utc_now()
    filled = ctx.storage("fs-fill", "--name", TEST_FS, "--percent", "92")
    used = (filled.get("statvfs") or {}).get("used_percent")
    if not isinstance(used, (int, float)) or not 90 <= used < 98:
        raise VariantInconclusive(f"the scratch fs reached {used}% (wanted 90..98)")
    detection = wait_for_incident(
        ctx,
        started,
        "local_filesystem_used_percent",
        device=_fs_mount(ctx),
        rule_id="local_filesystem",
        timeout_seconds=LOW_BUDGET,
        metrics={"local_filesystem_used_percent", "filesystem_used_percent"},
        name="scratch",
    )
    stages = {
        "chain": chain_errors(
            detection,
            ChainExpectation(
                metric="local_filesystem_used_percent",
                device=_fs_mount(ctx),
                rule_id="local_filesystem",
                operations=("FREEZE_EVIDENCE", "VALIDATE_HOST"),
                validation="VALIDATE_HOST",
                validation_must_fail=True,
                probe_metric="local_filesystem_used_percent",
                forbidden=_hardware() | {"QUARANTINE"},
                incident_states=DIAGNOSTIC_ONLY_STATES,
            ),
        ),
        "no_quarantine": []
        if not finding_incidents(
            detection, "filesystem_used_percent", device=_fs_mount(ctx)
        )
        else ["the 98 % QUARANTINE rule fired at 92 % used"],
    }
    ctx.storage("fs-unfill", "--name", TEST_FS)
    close_window(ctx)
    ctx.cleanup.discard("collector-window")
    return _result(stages, extra={"mountpoint": mount, "used_percent": used})


def idle_node(ctx: VariantContext) -> dict[str, Any]:
    run_survey(ctx)
    # No managed attempt on the node; GPUs idle. The workload gate must suppress
    # every LOW_* finding.
    started = utc_now()
    ctx.hold(LOW_DURATION + ctx.host_interval * 3, name="idle-node-hold")
    observed = activity(
        ctx,
        started,
        metrics={"host_gpu_utilization_percent", "training_rank_cpu_percent"},
    )
    stages = {
        "no_low_gpu": chain_errors(
            observed,
            ChainExpectation(
                metric="host_gpu_utilization_percent",
                rule_id="LOW_GPU_UTILIZATION",
                present=False,
            ),
        ),
        "no_low_cpu": chain_errors(
            observed,
            ChainExpectation(
                metric="training_rank_cpu_percent",
                rule_id="LOW_CPU_UTILIZATION",
                present=False,
            ),
        ),
    }
    return _result(
        stages,
        notes=["workload gating: no managed attempt means no low-utilisation finding"],
    )


def attempt_change(ctx: VariantContext) -> dict[str, Any]:
    run_survey(ctx)
    job_a = _managed_job(ctx, f"gl3-attA-{ctx.attempt}")
    identity_a = job_a.identity()
    ctx.hold(200, name="attempt-a")
    # Replace attempt A with a fresh attempt B on the same node.
    job_a.cleanup()
    ctx.cleanup.discard(f"job-gl3-attA-{ctx.attempt}")
    job_b = _managed_job(ctx, f"gl3-attB-{ctx.attempt}")
    identity_b = job_b.identity()
    started_b = utc_now()
    ctx.hold(LOW_DURATION + ctx.host_interval * 3, name="attempt-b")
    detection = wait_for_incident(
        ctx,
        started_b,
        "host_gpu_utilization_percent",
        rule_id="LOW_GPU_UTILIZATION",
        timeout_seconds=LOW_BUDGET,
        metrics={"host_gpu_utilization_percent"},
        name="attempt-change",
    )
    incidents = finding_incidents(
        detection, "host_gpu_utilization_percent", rule_id="LOW_GPU_UTILIZATION"
    )
    stages = {
        "attributed_to_b": _attribution_errors(
            incidents, detection, identity_b["attempt_id"]
        ),
        "not_attributed_to_a": [
            f"a finding is attributed to the retired attempt {identity_a['attempt_id']}"
        ]
        if any(
            item.get("attempt_id") == identity_a["attempt_id"]
            for item in detection.get("incidents") or []
        )
        else [],
        "hardware": hardware_errors(detection),
    }
    wait_settled(
        ctx,
        started_b,
        [item["incident_id"] for item in incidents],
        timeout_seconds=LOW_BUDGET,
        metrics={"host_gpu_utilization_percent"},
        name="attempt-change-settled",
    )
    return _result(stages, extra={"attempt_a": identity_a, "attempt_b": identity_b})


def ambiguous_ownership(ctx: VariantContext) -> dict[str, Any]:
    run_survey(ctx)
    job_a = _managed_job(ctx, f"gl3-ambA-{ctx.attempt}", gpu_count=4)
    job_b = _managed_job(ctx, f"gl3-ambB-{ctx.attempt}", gpu_count=4)
    started = utc_now()
    ctx.hold(LOW_DURATION + ctx.host_interval * 3, name="ambiguous-hold")
    observed = activity(ctx, started, metrics={"host_gpu_utilization_percent"})
    # Two attempts on one node: resolve_managed_attempt resolves none, so no
    # finding (the product refuses to guess ownership).
    stages = {
        "no_finding": chain_errors(
            observed,
            ChainExpectation(
                metric="host_gpu_utilization_percent",
                rule_id="LOW_GPU_UTILIZATION",
                present=False,
            ),
        ),
    }
    return _result(
        stages,
        notes=[
            "two managed attempts on one node resolve to none (MIG is a separate capability gap)"
        ],
        extra={"jobs": [job_a.identity(), job_b.identity()]},
    )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _hardware() -> frozenset[str]:
    from scripts.e2e.regional.general_live_registry import HARDWARE_OPERATIONS

    return HARDWARE_OPERATIONS


def _fs_mount(ctx: VariantContext) -> str:
    import hashlib

    digest = hashlib.sha256(ctx.run_id.encode()).hexdigest()[:12]
    return f"/mnt/gfrs-live/{digest}/{TEST_FS}"


def _attribution_errors(
    incidents: list[dict[str, Any]], detection: dict[str, Any], attempt_id: str
) -> list[str]:
    if not incidents:
        return ["no low-utilisation finding reached an incident"]
    errors = []
    for incident in incidents:
        if incident.get("attempt_id") not in (None, attempt_id):
            errors.append(
                f"finding attempt {incident.get('attempt_id')} != {attempt_id}"
            )
    return errors


def _memory_used_low(detection: dict[str, Any]) -> list[str]:
    used = [
        sample["value"]
        for sample in samples(detection, "memory_used_percent")
        if isinstance(sample.get("value"), (int, float))
    ]
    return (
        [f"memory_used_percent rose to {max(used)} while page cache was high"]
        if used and max(used) >= 95
        else []
    )


def _result(
    stages: dict[str, list[str]],
    *,
    notes: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "verdict": verdict(stages),
        "stages": stages,
        "errors": [
            f"{stage}: {error}" for stage, items in stages.items() for error in items
        ],
        "notes": notes or [],
        **(extra or {}),
    }


def _spec(
    variant: str,
    injection: str,
    expected: str,
    execute,
    *,
    risk="L",
    max_minutes=20,
    approvals=(),
    lanes=("eks", "hp"),
) -> VariantSpec:
    return VariantSpec(
        case_id=CASE,
        variant=variant,
        proof_mode=NATIVE,
        lanes=lanes,
        risk=risk,
        flags="S",
        injection=injection,
        expected=expected,
        cleanup="stop the load/job; check the window reset and the original rank's progress",
        max_minutes=max_minutes,
        execute=execute,
        approvals=tuple(approvals),
    )


VARIANTS = [
    _spec(
        "gpu_idle",
        "a 1-node 8-GPU managed job at GPU duty 0 for >= 12 min",
        "LOW_GPU_UTILIZATION on the job's UUIDs -> VALIDATE_GPU (no DCGM diag, no reset); the attempt id matches",
        gpu_idle,
    ),
    _spec(
        "rank_cpu_idle",
        "the same managed job; ranks sleep holding GPU contexts",
        "LOW_CPU_UTILIZATION bound to the rank PIDs -> VALIDATE_HOST",
        rank_cpu_idle,
    ),
    _spec(
        "low_available_memory",
        "share the 002 memory cgroup; MemAvailable <= 5 % for >= 7 min",
        "LOW_MEMORY_AVAILABLE -> VALIDATE_HOST probe on memory_available_percent",
        low_available_memory,
        risk="H",
        approvals=("HOST_MEMORY_PRESSURE",),
        max_minutes=45,
    ),
    _spec(
        "high_page_cache",
        "read a sparse file so page cache stays >= 90 % for 300 s",
        "HIGH_PAGE_CACHE -> VALIDATE_HOST; memory_used stays low (cache is reclaimable)",
        high_page_cache,
        risk="M",
    ),
    _spec(
        "scratch_full",
        "a loop fs added to GPU_FAULT_FILESYSTEMS filled to 92 %",
        "HIGH_LOCAL_FILESYSTEM_USAGE (>= 90 % for 60 s) -> VALIDATE_HOST; no 98 % QUARANTINE",
        scratch_full,
        risk="M",
    ),
    _spec(
        "idle_node",
        "no managed attempt on the node; GPUs idle for >= 15 min",
        "no LOW_* finding (workload gating)",
        idle_node,
    ),
    _spec(
        "attempt_change",
        "attempt A idle 200 s, then a fresh attempt B idle 400 s",
        "no finding for A; B's finding is attributed only to B and its window starts at B",
        attempt_change,
    ),
    _spec(
        "ambiguous_ownership",
        "two 4-GPU managed jobs on one node, both idle",
        "resolve_managed_attempt resolves none, so no finding (MIG recorded as a capability gap)",
        ambiguous_ownership,
    ),
]
