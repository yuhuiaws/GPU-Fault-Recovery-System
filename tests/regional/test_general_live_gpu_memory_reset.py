"""GF-GENERAL-LIVE-019 ``pending_supported`` / ``post_reset``: the flag for one
scrape, the chain judged on the records, and a runner that never has to reach
the node while the product acts on it (the HyperPod quiesce stops kubelet and
the probe Pod with it)."""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_gpu_memory as mem
from scripts.e2e.regional.host_probe_fixture import HostProbeMissingResponseError
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    make_context,
    make_settings,
    store_poll_clock,
)
from tests.regional._general_live_gpu_fakes import (
    audit_records,
    dcgm_probe,
    gpu_cpu_python,
    gpu_incident,
    gpu_workflow,
    survey,
)

SPECS = {spec.variant: spec for spec in mem.VARIANTS}
GPU = "GPU-11111111-1111-4111-8111-111111111111"


# --------------------------------------------------------------------------- #
# pending_supported / post_reset: the flag for one scrape, the chain's records,
# and a runner that never has to reach the node while the product acts on it
# --------------------------------------------------------------------------- #
#: The live eksb 2026-10-26 07:47:57-07:50:03Z record of a pending incident's
#: service-window reset (024 wrong_scope_source's repair GPU): every step
#: SUCCEEDED and the incident RECOVERED by itself.
RESET_CHAIN = (
    "FREEZE_EVIDENCE",
    "MARK_UNSCHEDULABLE",
    "QUARANTINE",
    "QUIESCE_GPU_SERVICES",
    "VERIFY_NO_GPU_CLIENTS",
    "RESET_GPU",
    "RESTORE_GPU_SERVICES",
    "VALIDATE_GPU",
    "RESTORE_SCHEDULING",
)
#: The ladder's next rung after a validation that fails on a reset node
#: (live 2026-10-26 14:54-15:06Z, HyperPod, a held plan): the successor record.
ESCALATION_CHAIN = (
    "FREEZE_EVIDENCE",
    "MARK_UNSCHEDULABLE",
    "QUARANTINE",
    "RESTART_NODE",
    "VALIDATE_GPU",
    "VALIDATE_HOST",
    "VALIDATE_FABRIC",
    "RESTORE_SCHEDULING",
)
PENDING_RECORDS = audit_records({f"{GPU}/row_remap_pending": 1.0})
MISSING = "host probe response is missing (exit 1)"


def _pending_incident(state: str = "ACTION_PENDING") -> dict[str, Any]:
    return gpu_incident(
        "inc-pend",
        f"b-{GPU}-row_remap_pending",
        state=state,
        action="RESET_GPU",
        workflow="wf-r",
    )


def _reset_record(
    executed: int, *, status: str = "RUNNING", failed: str | None = None
) -> dict[str, Any]:
    """``wf-r`` with the first ``executed`` steps of the chain SUCCEEDED, and
    ``failed`` FAILED after them."""

    executions = _succeeded(*RESET_CHAIN[:executed])
    if failed:
        executions.append(
            {
                "operation": failed,
                "status": "FAILED",
                "updated_at": f"2026-10-24T00:03:{executed:02d}+00:00",
                "error": (
                    "validation failed on nodes: ip-10-0-43-5: "
                    "active_gpu_health_findings (row_remap_pending)"
                ),
            }
        )
    return gpu_workflow(
        "wf-r", "inc-pend", list(RESET_CHAIN), status=status, executions=executions
    )


def _reset_stages() -> list[dict[str, Any]]:
    """The product's records as the runner reads them: minted and quiescing
    (twice: the gate's records read, then the detection), the reset done with
    the restore ahead, then the record ended RECOVERED."""

    minted = _activity([_pending_incident()], [_reset_record(4)])
    reset_done = _activity([_pending_incident()], [_reset_record(6)])
    recovered = _activity(
        [_pending_incident("RECOVERED")],
        [_reset_record(len(RESET_CHAIN), status="SUCCEEDED")],
    )
    return [minted, minted, reset_done, recovered]


def _pending_ctx(
    tmp_path: Path,
    variant: str,
    *,
    activity: Any,
    probe: FakeProbe | None = None,
    gpu_state: dict[str, Any] | None = None,
    node: dict[str, Any] | None = None,
) -> Any:
    spec = SPECS[variant]
    regional = FakeRegional(
        cpu_python=gpu_cpu_python(
            activity=activity, gpu_state=gpu_state or {"active_findings": []}
        ),
        node_snapshot=node,
    )
    settings = make_settings(spec, node=regional.node_snapshot("")["name"])
    object.__setattr__(settings, "approvals", ("GPU_RESET",))
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes={}, settings=settings
    )
    ctx.probes[f"dcgm@{ctx.settings.node}"] = probe or dcgm_probe(audit=PENDING_RECORDS)
    return ctx


def _kubelet_gap(
    stages: list[dict[str, Any]], *, back_at: int
) -> tuple[Callable[..., dict[str, Any]], dict[str, bool]]:
    """An activity probe that walks ``stages`` one read at a time while the
    node is away (kubelet stopped by the quiesce) until the read at index
    ``back_at`` -- RESTORE_GPU_SERVICES started kubelet again."""

    node = {"down": True}
    reads = {"n": 0}

    def activity(arguments: tuple[str, ...]) -> dict[str, Any]:
        index = min(reads["n"], len(stages) - 1)
        reads["n"] += 1
        node["down"] = index < back_at
        return dict(stages[index])

    return activity, node


def _gap_probe(node: dict[str, bool]) -> FakeProbe:
    """A probe the node refuses while ``node["down"]``; a plan advance is a
    harness error (the bounded plan has nothing to advance)."""

    def audit_read(arguments: tuple[str, ...]) -> dict[str, Any]:
        if node["down"]:
            raise HostProbeMissingResponseError(MISSING)
        offset = int(arguments[arguments.index("--offset") + 1])
        chunk = PENDING_RECORDS[offset:]
        return {
            "records": list(chunk),
            "next_offset": offset + len(chunk),
            "total": len(PENDING_RECORDS),
        }

    def close_run(arguments: tuple[str, ...]) -> dict[str, Any]:
        if node["down"]:
            raise HostProbeMissingResponseError(MISSING)
        return dcgm_probe().script["close-run"]

    def advance(arguments: tuple[str, ...]) -> dict[str, Any]:
        raise AssertionError("the bounded plan is never advanced")

    return dcgm_probe(
        extra={"audit-read": audit_read, "close-run": close_run, "advance": advance}
    )


def _installed_plan(probe: FakeProbe) -> dict[str, Any]:
    """The field plan the controller installed on the node (``install-source``)."""

    for call in probe.calls:
        if (
            call[0] == "install-source"
            and call[call.index("--name") + 1] == "plan.json"
        ):
            return json.loads(base64.b64decode(call[call.index("--b64") + 1]))
    raise AssertionError("no plan.json was installed")


def _exposition(pending: float = 0) -> bytes:
    """One GPU's scrape of the replayed field, in the exporter's format."""

    labels = (
        f'gpu="1",UUID="{GPU}",pci_bus_id="00000000:02:00.0",modelName="NVIDIA H200"'
    )
    return f"DCGM_FI_DEV_ROW_REMAP_PENDING{{{labels}}} {pending}\n".encode()


def test_pending_supported_proves_the_executed_chain_on_the_records(
    tmp_path: Path,
) -> None:
    spec = SPECS["pending_supported"]
    assert spec.approvals == ("GPU_RESET",) and spec.flags == "X" and spec.risk == "H"
    ctx = _pending_ctx(tmp_path, "pending_supported", activity=_reset_stages())
    out = mem.pending_supported(ctx)
    assert out["verdict"] == "PASS", out["errors"]
    assert set(out["stages"]) == {"chain", "hardware", "recovery"}, out["stages"]
    assert out["fault_reads"] == 1 and out["oracle_gate"] == "journal", out
    assert out["incident_ids"] == ["inc-pend"]
    calls = [call[0] for call in ctx.probes[f"dcgm@{ctx.settings.node}"].calls]
    assert "advance" not in calls, "nothing is held, nothing is advanced"
    assert calls.count("close-run") == 1
    settled = json.loads((ctx.case_dir / "settled-timeline.json").read_text())
    assert [entry["accepted"] for entry in settled["entries"]] == [
        False,
        False,
        True,
    ], "the chain is judged once the record has ended"
    assert ctx.cleanup.pending() == []
    assert "RESTART_NODE escalation" in spec.expected, spec.expected
    assert "until_advance" not in json.dumps(
        _installed_plan(ctx.probes[f"dcgm@{ctx.settings.node}"])
    )


def test_pending_plan_serves_the_flag_once_then_the_real_reading_for_good(
    tmp_path: Path,
) -> None:
    """The installed plan, run through the field tool over a real-shaped
    scrape: three pass-through reads, ONE read of the flag, then the device's
    own reading byte for byte -- what the first scrape after
    RESTORE_GPU_SERVICES serves to the post-reset VALIDATE_GPU, however many
    scrapes the quiesce window swallowed in between."""

    from gpu_fault.collectors.gpu.dcgm import parse_dcgm_samples
    from tools import gpu_dcgm_field_replay as tool

    ctx = _pending_ctx(tmp_path, "post_reset", activity=_reset_stages())
    out = mem.post_reset(ctx)
    assert out["verdict"] == "PASS", out["errors"]
    document = _installed_plan(ctx.probes[f"dcgm@{ctx.settings.node}"])
    assert document["steps"] == [
        {"reads": 3, "ops": []},
        {
            "reads": 1,
            "ops": [
                {"op": "set", "field": "row_remap_pending", "gpus": [GPU], "value": 1}
            ],
        },
    ], document
    replay = tool.Replay(tool.validate_plan(document), 600)
    replay.arm()
    body = _exposition()
    served = []
    verbatim = []
    for _ in range(8):
        output, receipt = replay.render(body)
        replay.reads += 1
        (sample,) = parse_dcgm_samples(output.decode())
        served.append((sample.value, receipt.get("step")))
        verbatim.append(output == body)
    assert served == [
        (0.0, 0),
        (0.0, 0),
        (0.0, 0),
        (1.0, 1),
        (0.0, None),
        (0.0, None),
        (0.0, None),
        (0.0, None),
    ], served
    assert verbatim[4:] == [True] * 4, (
        "after the plan the scrape is the exporter's own, byte for byte"
    )
    assert replay.active() is None and not replay.disarmed
    with pytest.raises(ValueError, match="not on a step"):
        replay.advance()


def test_post_reset_grades_the_records_through_the_kubelet_gap(tmp_path: Path) -> None:
    """HyperPod: the quiesce stops kubelet ~20 s after the mint and the probe
    Pod with it until RESTORE_GPU_SERVICES. The oracle gate falls back to the
    product's records, the detection and the settle wait read the store only,
    the plan is never advanced, and the close comes after the chain -- when
    the node answers again -- with the journal read first."""

    activity, node = _kubelet_gap(_reset_stages(), back_at=3)
    probe = _gap_probe(node)
    ctx = _pending_ctx(
        tmp_path,
        "post_reset",
        activity=activity,
        probe=probe,
        node={**FakeRegional().node_snapshot(""), "boot_id": survey()["boot_id"]},
    )
    out = mem.post_reset(ctx)
    assert out["verdict"] == "PASS", out["errors"]
    assert set(out["stages"]) == {"chain", "hardware", "recovery", "post_reset"}
    assert out["oracle_gate"] == "records"
    deferred = json.loads((ctx.case_dir / "replay-gate-deferred.json").read_text())
    assert deferred["keys"] == [f"{GPU}/row_remap_pending"]
    assert deferred["probe_error"].startswith(
        "HostProbeMissingResponseError: dcgm audit-read: host probe response is missing"
    ), deferred
    calls = [call[0] for call in probe.calls]
    assert "advance" not in calls
    assert calls.count("close-run") == 1
    assert calls.index("close-run") > calls.index("arm")
    close = json.loads((ctx.case_dir / "dcgm-replay-close.json").read_text())
    assert "close_error" not in close, "the node was back before the close"
    assert "audit_error" not in close, "the journal was read at the close"
    assert close["close"]["clean"] is True
    oracle = [
        json.loads(line)
        for line in (ctx.case_dir / "oracle.jsonl").read_text().splitlines()
    ]
    assert [item["served"] for item in oracle] == [PENDING_RECORDS[0]["served"]]
    assert node["down"] is False
    assert ctx.cleanup.pending() == []


def test_post_reset_fails_on_the_reset_ladders_restart_node_rung(
    tmp_path: Path,
) -> None:
    """The record a still-served flag leaves behind (live 2026-10-26 14:54Z):
    VALIDATE_GPU FAILED on the reset GPU, a successor workflow QUARANTINE ->
    RESTART_NODE, and the incident RECOVERED at the end of the reboot. Every
    stage but the recovery fails it; the incident's final state alone would
    not."""

    failed = _reset_record(7, status="FAILED", failed="VALIDATE_GPU")
    successor = {
        **gpu_workflow(
            "wf-esc",
            "inc-pend",
            list(ESCALATION_CHAIN),
            status="SUCCEEDED",
            executions=_succeeded(*ESCALATION_CHAIN),
        ),
        "created_at": "2026-10-24T00:05:00+00:00",
    }
    activity = _activity([_pending_incident("RECOVERED")], [failed, successor])
    ctx = _pending_ctx(tmp_path, "post_reset", activity=activity)
    out = mem.post_reset(ctx)
    assert out["verdict"] == "FAIL", out
    chain = out["stages"]["chain"]
    assert any("workflow wf-esc plans ['RESTART_NODE']" in e for e in chain), chain
    assert any("VALIDATE_GPU ended FAILED" in e for e in chain), chain
    assert any("plans ['RESTART_NODE']" in e for e in out["stages"]["hardware"]), out
    post = out["stages"]["post_reset"]
    assert any(
        "went on to workflow wf-esc planning" in e and "RESTART_NODE" in e for e in post
    ), post
    assert any("post-reset VALIDATE_GPU ended FAILED" in e for e in post), post
    assert out["stages"]["recovery"] == [], "RECOVERED at the end is not the proof"


def test_post_reset_fails_when_the_pending_finding_stays_active(tmp_path: Path) -> None:
    ctx = _pending_ctx(
        tmp_path,
        "post_reset",
        activity=_reset_stages(),
        gpu_state={
            "active_findings": [
                {"canonical_name": "row_remap_pending", "gpu_uuid": GPU, "value": 1.0}
            ]
        },
    )
    out = mem.post_reset(ctx)
    assert out["verdict"] == "FAIL", out
    (error,) = out["stages"]["recovery"]
    assert "1 row_remap_pending finding still active after the chain ended" in error
    assert out["stages"]["post_reset"] == [], out["stages"]


def test_post_reset_fails_a_second_reset_or_a_validation_before_the_reset() -> None:
    label = "row_remap_pending on gpu"
    incidents = [_pending_incident("RECOVERED")]
    twice = _reset_record(len(RESET_CHAIN), status="SUCCEEDED")
    twice["step_executions"].append(
        {
            "operation": "RESET_GPU",
            "status": "SUCCEEDED",
            "updated_at": "2026-10-24T00:04:00+00:00",
        }
    )
    errors = mem.post_reset_errors(
        _activity(incidents, [twice]), incidents, label=label
    )
    assert errors == [f"{label} RESET_GPU succeeded 2 times (expected exactly one)"]
    early = _reset_record(len(RESET_CHAIN), status="SUCCEEDED")
    for execution in early["step_executions"]:
        if execution["operation"] == "VALIDATE_GPU":
            execution["updated_at"] = "2026-10-24T00:02:00+00:00"
    errors = mem.post_reset_errors(
        _activity(incidents, [early]), incidents, label=label
    )
    assert len(errors) == 1 and "before RESET_GPU completed" in errors[0], errors
    assert mem.post_reset_errors(_activity(incidents, []), [], label=label) == []


def test_reset_chain_still_running_at_the_settle_bound_fails_the_chain(
    tmp_path: Path,
) -> None:
    ctx = _pending_ctx(tmp_path, "pending_supported", activity=_reset_stages()[2])
    store_poll_clock(ctx)
    out = mem.pending_supported(ctx)
    assert out["verdict"] == "FAIL", out
    chain = out["stages"]["chain"]
    assert any("never executed VALIDATE_GPU" in e for e in chain), chain
    assert any(
        "is ACTION_PENDING, expected one of ['RECOVERED']" in e for e in chain
    ), chain
    recovery = out["stages"]["recovery"]
    assert any("had not ended when the settle wait ran out" in e for e in recovery), (
        recovery
    )
    assert any("is ACTION_PENDING, not RECOVERED" in e for e in recovery), recovery
    settled = json.loads((ctx.case_dir / "settled-timeline.json").read_text())
    polls = len(settled["entries"])
    budget = (300 + mem.RESET_SETTLE_SECONDS) // 20
    assert budget // 2 <= polls <= budget + 2, f"polled to the settle bound: {polls}"
    calls = [call[0] for call in ctx.probes[f"dcgm@{ctx.settings.node}"].calls]
    assert calls.count("close-run") == 1, "the replay is still closed"


def test_pending_fault_reads_rest_on_the_products_own_cadence() -> None:
    """One scrape at the 15-s collector, and why: the pending flag mints on
    one reading and clears on the next real one; the quiesce stops the
    collector; a scrape the collector could not make is not a read the plan
    consumes; RESET_GPU is the validation's readiness barrier; and a validation
    that fails after a reset escalates to RESTART_NODE."""

    import io
    import threading
    import urllib.error
    import urllib.request
    from datetime import datetime, timedelta, timezone

    from gpu_fault.adapters.gpu_validation import TELEMETRY_READINESS_OPERATIONS
    from gpu_fault.gpu_metrics import (
        GpuMetricBatch,
        GpuMetricSample,
        GpuMetricSource,
        GpuMetricsService,
    )
    from gpu_fault.models import WorkflowOperation
    from gpu_fault.node_agent.common import (
        DEFAULT_QUIESCE_CONTAINERS,
        DEFAULT_QUIESCE_SERVICES,
    )
    from gpu_fault.orchestration.escalation import next_rung
    from tools import gpu_dcgm_field_replay as tool

    assert mem.QUIESCE_LEAD_SECONDS == 10
    assert mem.pending_fault_reads(15) == 1, "the live collectors"
    assert mem.pending_fault_reads(10) == 1
    assert mem.pending_fault_reads(30) == 1
    assert mem.pending_fault_reads(5) == 2, "a second read only where it lands early"
    assert mem.pending_fault_reads(1) == 10

    # The product mints the WARNING RESET_GPU finding on one reading > 0 and
    # closes it on the next reading of 0: no delta, confirmation or sustain.
    service = GpuMetricsService()
    start = datetime(2026, 10, 26, 7, 47, tzinfo=timezone.utc)

    def batch(tick: int, value: float) -> GpuMetricBatch:
        return GpuMetricBatch(
            batch_id=f"b{tick}",
            cluster_id="gpu-eks-cluster",
            node_id="ip-10-0-43-5.us-west-2.compute.internal",
            observed_at=start + timedelta(seconds=15 * tick),
            source=GpuMetricSource.DCGM_EXPORTER,
            samples=[
                GpuMetricSample(
                    metric_name="DCGM_FI_DEV_ROW_REMAP_PENDING",
                    canonical_name="row_remap_pending",
                    value=value,
                    gpu_index="1",
                    gpu_uuid=GPU,
                    pci_bdf="00000000:02:00.0",
                    labels={"modelName": "NVIDIA H200"},
                )
            ],
        )

    (finding,) = service.ingest(batch(0, 1.0)).new_findings
    assert finding.automatic_action == "RESET_GPU", finding
    assert finding.official_action == "RESET_GPU_IN_SERVICE_WINDOW", finding
    assert finding.confirmation_count == 0
    assert [
        f.canonical_name for f in service.findings("gpu-eks-cluster", finding.node_id)
    ] == ["row_remap_pending"]
    assert service.ingest(batch(1, 0.0)).new_findings == []
    assert service.findings("gpu-eks-cluster", finding.node_id) == [], (
        "the first real reading clears the finding"
    )

    # The quiesce stops the collector and the exporter on both providers.
    assert {"gpu-fault-metrics-collector", "kubelet"} <= set(DEFAULT_QUIESCE_SERVICES)
    assert "gpu-fault-system/exporter" in DEFAULT_QUIESCE_CONTAINERS
    # The post-reset VALIDATE_GPU grades readings newer than the reset, and a
    # validation that fails after a reset is the reboot rung.
    assert WorkflowOperation.RESET_GPU in TELEMETRY_READINESS_OPERATIONS
    assert (
        next_rung(WorkflowOperation.VALIDATE_GPU, {WorkflowOperation.RESET_GPU})
        is WorkflowOperation.RESTART_NODE
    )

    # A scrape the tool could not serve (the exporter gone) is journaled as an
    # error and consumes no read: the fault read waits for the next scrape.
    replay = tool.Replay(
        tool.validate_plan(
            {
                "profile": "memory",
                "steps": [
                    {
                        "reads": 1,
                        "ops": [
                            {
                                "op": "set",
                                "field": "row_remap_pending",
                                "gpus": [GPU],
                                "value": 1,
                            }
                        ],
                    }
                ],
            }
        ),
        600,
    )
    replay.arm()
    audit = io.StringIO()
    server = tool.make_server("http://127.0.0.1:1/metrics", 0, replay, audit)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        with pytest.raises(urllib.error.HTTPError) as refused:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10)
        assert refused.value.code == 503
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
    (record,) = [json.loads(line) for line in audit.getvalue().splitlines()]
    assert record["read"] == 1 and "error" in record, record
    assert replay.reads == 0 and replay.active() is not None, (
        "the fault read is still ahead, for the first scrape the collector makes"
    )


def test_fault_reads_cover_the_wait_and_refuse_the_tools_step_cap() -> None:
    from scripts.e2e.regional.general_live_fixture import VariantInconclusive

    assert mem.FAULT_SLACK_READS == 2 and mem.DETECTION_MARGIN_SECONDS == 240
    assert mem.fault_reads(15, 540) == 38, "the diagnostic variants' detection wait"
    assert mem.fault_reads(15, 720) == 50, "detection plus CONTAINMENT_SECONDS"
    assert mem.fault_reads(60, 540) == 11
    with pytest.raises(VariantInconclusive, match="at most 200 per step"):
        mem.fault_reads(1, 400)


def _succeeded(*operations: str) -> list[dict[str, Any]]:
    return [
        {
            "operation": operation,
            "status": "SUCCEEDED",
            "updated_at": f"2026-10-24T00:03:{index:02d}+00:00",
        }
        for index, operation in enumerate(operations)
    ]


def _activity(incidents: list, workflows: list) -> dict[str, Any]:
    return {
        "incidents": incidents,
        "workflows": workflows,
        "batches": [],
        "markers": [],
        "commands": [],
        "notifications": [],
        "metric_latest": [],
        "collector_statuses": [],
        "agent": None,
    }
