"""GF-GENERAL-LIVE-024 ``xid_twin_*`` judged on the product's own records.

The variant's code runs end to end (``general_live_xid_twins.run_twin``): the
product-generation gate, the GPU survey, the field replay armed before the
repoint, the gauge baseline, the kmsg window, the placement of both reports on
the node's clock, the negative control, the settle and the oracles. The node is
``_xid_twin_node.TwinNode`` -- the metrics collector scraping the field tool on
a virtual clock, the kernel collector posting each armed line -- and the
control plane is a real ``ApplicationContext`` over an in-memory store with the
production dispatcher and executor over a recording adapter, every store probe
of the variant run in-process over it. On this tree (batch 237) every variant
PASSes; with the twin rule taken out (the product before it) and the gate
bypassed, the same variant FAILs on the stage that names the double report.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_xid_twin_delivery as delivery_module
from scripts.e2e.regional import general_live_xid_twin_records as records_module
from scripts.e2e.regional import general_live_xid_twins as twins
from scripts.e2e.regional.general_live_fixture import (
    EKS_NODEGROUP_LABEL,
    HYPERPOD_LABEL,
)
from scripts.e2e.regional.general_live_gpu_store import GPU_STATE_PROBE
from scripts.e2e.regional.workload_restart_custody import (
    UNBOUND_IDENTITY_SOURCE,
    WorkloadOwnershipError,
)
from tests.regional._general_live_fakes import FakeRegional, make_context, make_settings
from tests.regional._xid_twin_node import (
    EIGHT_GPUS,
    NODE,
    START,
    TwinNode,
    TwinProduct,
    VirtualClock,
    without_twin_rule,
)

#: The real Agent prerequisite (``Run`` replaces it: the in-memory store has
#: no Agent record).
REQUIRE_AGENT = twins.require_agent


class Run:
    """One variant run over a faked node and the in-memory product."""

    def __init__(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        variant: str,
        *,
        interval: int = 15,
        phase: float = 4.0,
        gpus: tuple[str, ...] | None = None,
        provider: str = "eks",
        exec_away: tuple[float, float] | None = None,
    ) -> None:
        self.spec = next(item for item in twins.VARIANTS if item.variant == variant)
        self.case = twins.CASES[variant]
        self.clock = VirtualClock()
        self.product = TwinProduct(
            self.clock, remediate=self.case.expectation.remediates
        )
        self.node = TwinNode(
            self.product,
            self.clock,
            interval=interval,
            phase=phase,
            exec_away=exec_away,
            **({"gpus": gpus} if gpus else {}),
        )
        labels = (
            {HYPERPOD_LABEL: "hyperpod"}
            if provider == "hp"
            else {EKS_NODEGROUP_LABEL: "gpu-p5en-efa-spot-node-group"}
        )
        regional = FakeRegional(
            cpu_python=self.product.probe,
            node_metadata={"name": NODE, "labels": labels},
        )
        settings = make_settings(self.spec, node=NODE)
        self.ctx = make_context(
            self.spec,
            tmp_path=tmp_path,
            regional=regional,
            probes={},
            settings=settings,
        )
        self.ctx.clock = self.clock.clock
        self.ctx.sleep = self.clock.sleep
        self.ctx.deadline = START + timedelta(hours=2)
        self.ctx.probes[f"dcgm@{NODE}"] = self.node.dcgm()  # type: ignore[assignment]
        self.ctx.probes[f"kernel-replay-kmsg@{NODE}"] = self.node.kernel()  # type: ignore[assignment]
        monkeypatch.setattr(twins, "utc_now", self.clock.clock)
        monkeypatch.setattr(twins, "require_agent", lambda *_args, **_kw: {})
        self.monkeypatch = monkeypatch

    def bypass_gate(self) -> None:
        """Hand the twin to a product without the rule (the must-fail shape)."""

        generation = records_module.TwinGeneration(
            self.case.xid, {"present": True, "note": "gate bypassed by the test"}
        )
        self.monkeypatch.setattr(
            twins, "require_twin_generation", lambda *_args, **_kw: generation
        )

    def run(self, *, before_rule: bool = False) -> dict[str, Any]:
        guard = without_twin_rule() if before_rule else contextlib.nullcontext()
        with guard:
            return twins.run_twin(self.ctx, self.case)


def _events(product: TwinProduct) -> list[Any]:
    return sorted(
        product.store.list_xid_events("gpu-eks-cluster", NODE),
        key=lambda item: item.observed_at,
    )


# --------------------------------------------------------------------------- #
# The safe variants (XID 92, catalog IGNORE)
# --------------------------------------------------------------------------- #
def test_kernel_first_is_one_occurrence_and_the_repeat_a_new_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = Run(tmp_path, monkeypatch, "xid_twin_kernel_first")
    out = run.run()

    assert out["verdict"] == "PASS", out["errors"]
    delivery = out["delivery"]
    lag = delivery["gauge_minus_kernel_seconds"]
    assert 15 <= lag <= 30, f"the change trails the line by {lag} s"
    assert delivery["repeat_after_gauge_seconds"] >= 150, delivery
    kernel, gauge, repeat = _events(run.product)
    assert (kernel.event_source, gauge.event_source, repeat.event_source) == (
        "KERNEL_LOG",
        "DCGM_EXPORTER",
        "KERNEL_LOG",
    )
    store = run.product.store
    first = store.get_xid_policy_decision(kernel.event_id)
    joined = store.get_xid_policy_decision(gauge.event_id)
    again = store.get_xid_policy_decision(repeat.event_id)
    assert joined.duplicate and joined.correlated_event_id == kernel.event_id, (
        "the change is booked against its kernel line"
    )
    assert joined.incident_id == first.incident_id == f"inc-{kernel.event_id}"
    assert store.get_incident_by_event(gauge.event_id).incident_id == first.incident_id
    assert again.correlated_event_id is None and not again.duplicate
    assert again.incident_id == f"inc-{repeat.event_id}", "a new occurrence"
    assert out["pair"]["joiner"] == gauge.event_id, out["pair"]
    assert out["repeat_event_id"] == repeat.event_id
    plan = run.node.plan or {}
    assert plan["profile"] == "xid_twin", plan
    assert [item["ops"][0]["value"] for item in plan["steps"]] == [0, 92], plan
    assert out["gauge_baseline"]["value"] == 0, out["gauge_baseline"]


def test_gauge_first_is_one_occurrence_on_the_changes_incident(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = Run(tmp_path, monkeypatch, "xid_twin_gauge_first", phase=11.0)
    out = run.run()

    assert out["verdict"] == "PASS", out["errors"]
    gauge, kernel, repeat = _events(run.product)
    assert (gauge.event_source, kernel.event_source) == ("DCGM_EXPORTER", "KERNEL_LOG")
    gap = (kernel.observed_at - gauge.observed_at).total_seconds()
    assert 30 <= gap <= 45, f"the line follows the change by {gap} s"
    store = run.product.store
    first = store.get_xid_policy_decision(gauge.event_id)
    joined = store.get_xid_policy_decision(kernel.event_id)
    assert joined.duplicate and joined.correlated_event_id == gauge.event_id, (
        "the line joins the change decided first"
    )
    assert joined.incident_id == first.incident_id == f"inc-{gauge.event_id}"
    assert store.get_xid_policy_decision(repeat.event_id).incident_id == (
        f"inc-{repeat.event_id}"
    )


@pytest.mark.parametrize("variant", ["xid_twin_kernel_first", "xid_twin_gauge_first"])
def test_the_safe_twin_fails_on_a_product_without_the_twin_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    """Must-fail: before batch 237 the second report opened its own incident.
    (Live, the gate ends such a product INCONCLUSIVE before anything is
    deployed; here it is bypassed to prove the oracle sees the shape.)"""

    run = Run(tmp_path, monkeypatch, variant)
    run.bypass_gate()
    out = run.run(before_rule=True)

    assert out["verdict"] == "FAIL", out
    assert any("two incidents" in item for item in out["stages"]["one_incident"]), out[
        "stages"
    ]
    assert out["stages"]["twin"], "the second report names no twin"
    first, second = _events(run.product)[:2]
    store = run.product.store
    assert (
        store.get_incident_by_event(first.event_id).incident_id
        != store.get_incident_by_event(second.event_id).incident_id
    ), "the product before batch 237 opened one incident per report"


def test_the_gate_refuses_a_product_without_the_rule_before_touching_the_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = Run(tmp_path, monkeypatch, "xid_twin_kernel_first")
    with without_twin_rule(), pytest.raises(twins.VariantInconclusive) as refused:
        twins.run_twin(run.ctx, run.case)

    text = str(refused.value)
    assert text.startswith(records_module.PRE_TWIN_GENERATION), text
    assert "Nothing was deployed and the node was not touched" in text, text
    assert run.node.plan is None and not run.node.kernel_posts, "nothing deployed"
    for key in (f"dcgm@{NODE}", f"kernel-replay-kmsg@{NODE}"):
        assert run.ctx.probes[key].calls == [], f"{key} was called"
    generation = (run.ctx.case_dir / "product-generation.json").read_text()
    assert '"twin_rule": false' in generation, generation


# --------------------------------------------------------------------------- #
# The X variants: the product's own chain, run by the production dispatcher
# --------------------------------------------------------------------------- #
def _executed(product: TwinProduct) -> dict[str, int]:
    return {
        operation.value: count
        for operation, count in product.adapter.executions().items()
        if count
    }


def test_reset_gpu_twin_runs_one_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = Run(tmp_path, monkeypatch, "xid_twin_reset_gpu")
    out = run.run()

    assert out["verdict"] == "PASS", out["errors"]
    assert _executed(run.product) == {"QUIESCE_GPU_SERVICES": 1, "RESET_GPU": 1}, (
        "the recording adapter ran one reset chain"
    )
    gauge, kernel = _events(run.product)
    assert (gauge.event_source, kernel.event_source) == ("DCGM_EXPORTER", "KERNEL_LOG")
    joined = run.product.store.get_xid_policy_decision(kernel.event_id)
    assert joined.duplicate and joined.correlated_event_id == gauge.event_id
    assert joined.workflow_request_id == out["workflow_request_id"]
    assert out["workflow_status"] == "SUCCEEDED"
    counts = out["executions"]["RESET_GPU"]["steps"]
    assert len(counts) == 1, counts
    closes = [
        call
        for call in run.ctx.probes[f"kernel-replay-kmsg@{NODE}"].calls
        if call[0] == "close-window"
    ]
    assert len(closes) == 1, "the kmsg window closed early, once"


def test_reset_gpu_twin_fails_on_a_product_without_the_twin_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In flight on an idle node the product before batch 237 already kept the
    reset single (the store's node-group merge into the open workflow); what it
    lacked is the twin link the ``twin`` stage reads."""

    run = Run(tmp_path, monkeypatch, "xid_twin_reset_gpu")
    run.bypass_gate()
    out = run.run(before_rule=True)

    assert out["verdict"] == "FAIL", out
    assert any("correlated_event_id" in item for item in out["stages"]["twin"]), out[
        "stages"
    ]
    assert out["stages"]["executions"] == [], "the merge kept one reset in flight"


# --------------------------------------------------------------------------- #
# HyperPod: the quiesce stops kubelet, the probes' exec channel
# --------------------------------------------------------------------------- #
#: Live 2026-11-02 (R16, HyperPod): QUIESCE_GPU_SERVICES claimed 6 s after the
#: scrape that served the change, kubelet back about 74 s after it.
HP_KUBELET_AWAY = (6.0, 68.0)


def _refused_before_the_line(run: Run) -> list[str]:
    line = run.node.kernel_posts[0]["observed_at"]
    return [
        name
        for name, at in run.node.refused
        if at < datetime.fromisoformat(line).timestamp()
    ]


def test_reset_gpu_twin_on_hyperpod_places_the_line_before_the_quiesce(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On HyperPod the XID 48 line follows the product's record of the change at
    once -- no journal read first -- and lands before kubelet stops."""

    run = Run(
        tmp_path,
        monkeypatch,
        "xid_twin_reset_gpu",
        provider="hp",
        exec_away=HP_KUBELET_AWAY,
    )
    out = run.run()

    assert out["verdict"] == "PASS", out["errors"]
    delivery = out["delivery"]
    assert (delivery["served_from"], delivery["written_from"]) == (
        "product record",
        "probe",
    ), delivery
    assert 0 <= -delivery["gauge_minus_kernel_seconds"] < HP_KUBELET_AWAY[0], delivery
    assert _refused_before_the_line(run) == [], run.node.refused
    assert _executed(run.product) == {"QUIESCE_GPU_SERVICES": 1, "RESET_GPU": 1}


def test_reset_gpu_twin_on_hyperpod_writes_the_line_once_kubelet_is_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live 2026-11-02 the journal read and the line's exec came back empty
    while kubelet was stopped, and the run ended INCONCLUSIVE "harness fixture
    failed". Kubelet stopped as the change was recorded: the line is retried
    and written once the node answers, inside the twin window, and the product
    still runs one reset."""

    run = Run(
        tmp_path,
        monkeypatch,
        "xid_twin_reset_gpu",
        provider="hp",
        exec_away=(0.0, 68.0),
    )
    out = run.run()

    assert out["verdict"] == "PASS", out["errors"]
    delivery = out["delivery"]
    assert delivery["written_from"] == "probe-retry", delivery
    lag = -delivery["gauge_minus_kernel_seconds"]
    assert (
        68 <= lag <= twins.TWIN_WINDOW_SECONDS - delivery_module.REPEAT_MARGIN_SECONDS
    )
    assert run.node.refused[0][0] == "arm", run.node.refused
    assert _executed(run.product) == {"QUIESCE_GPU_SERVICES": 1, "RESET_GPU": 1}
    retry = json.loads((run.ctx.case_dir / "twin-arm-retry.json").read_text())
    assert retry["outcome"] == "probe-retry", retry
    assert retry["attempts"][-1]["arm"] == "ok", retry


def test_reset_gpu_twin_is_inconclusive_when_kubelet_outlasts_the_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A quiesce longer than the twin window with margin: no line is written
    late, nothing is graded, the reason is named."""

    run = Run(
        tmp_path,
        monkeypatch,
        "xid_twin_reset_gpu",
        provider="hp",
        exec_away=(0.0, 150.0),
    )
    with pytest.raises(twins.VariantInconclusive) as raised:
        run.run()

    assert "could not be written inside the twin window" in str(raised.value)
    assert "QUIESCE_GPU_SERVICES stops it" in str(raised.value)
    assert run.node.kernel_posts == [], "no line outside the window"


def test_the_eks_twin_still_reads_the_journal_before_the_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plain EKS keeps kubelet through the quiesce: the served moment is the
    journal's read, as before."""

    run = Run(tmp_path, monkeypatch, "xid_twin_reset_gpu")
    out = run.run()

    assert out["verdict"] == "PASS", out["errors"]
    assert (out["delivery"]["served_from"], out["delivery"]["written_from"]) == (
        "journal",
        "probe",
    ), out["delivery"]


@pytest.mark.parametrize("before_rule", [False, True])
def test_a_line_after_the_finished_reset_is_still_one_remediation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, before_rule: bool
) -> None:
    """The recording adapter finishes the chain at once, so a line 30 s after
    the change meets a finished remediation inside the twin window: batch 237
    joins it, the product before it ran the reset again -- the execution count
    the oracle reads."""

    run = Run(tmp_path, monkeypatch, "xid_twin_reset_gpu")
    run.case = dataclasses.replace(run.case, kernel_delay_seconds=30)
    if before_rule:
        run.bypass_gate()
    out = run.run(before_rule=before_rule)

    resets = _executed(run.product).get("RESET_GPU")
    if not before_rule:
        assert out["verdict"] == "PASS", out["errors"]
        assert resets == 1, _executed(run.product)
        return
    assert resets == 2, "the product before batch 237 reset the GPU twice"
    assert out["verdict"] == "FAIL", out
    assert any(
        "RESET_GPU executed 2 time(s)" in item for item in out["stages"]["executions"]
    ), out["stages"]["executions"]


def test_drain_p2p_change_waits_for_its_line_and_drains_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = Run(tmp_path, monkeypatch, "xid_twin_drain_p2p")
    out = run.run()

    assert out["verdict"] == "PASS", out["errors"]
    assert _executed(run.product) == {
        "QUIESCE_GPU_SERVICES": 1,
        "RECOVER_DRAIN_P2P": 1,
    }, "one Drain P2P chain, no quarantine"
    gauge, kernel = _events(run.product)
    store = run.product.store
    joined = store.get_xid_policy_decision(gauge.event_id)
    assert joined.official_action == "DRAIN_P2P" and joined.duplicate
    assert joined.correlated_event_id == kernel.event_id, (
        "a kernel-decoded XID's change always names its line"
    )
    assert out["pair"]["joiner_source"] == "DCGM_EXPORTER", out["pair"]
    for workflow in store.list_workflows(limit=50):
        steps = [step.operation.value for step in workflow.official_steps]
        assert "QUARANTINE" not in steps, steps


def test_drain_p2p_twin_fails_where_the_change_quarantines_the_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Before batch 237 the label-less change was decided alone,
    BLOCKED_MISSING_EVIDENCE, and its safety plan quarantined the node."""

    run = Run(tmp_path, monkeypatch, "xid_twin_drain_p2p")
    run.bypass_gate()
    out = run.run(before_rule=True)

    assert out["verdict"] == "FAIL", out
    assert any("QUARANTINE" in item for item in out["stages"]["workflows"]), out[
        "stages"
    ]


# --------------------------------------------------------------------------- #
# xid_twin_unbound_job: a job on four GPUs, the XID on a fifth
# --------------------------------------------------------------------------- #
JOB_GPUS = list(EIGHT_GPUS[:4])


class FakeFixture:
    """The job's workload fixture: like ``ManagedWorkloadFixture`` it refuses
    the restarted Pods until the case bound the restart (``authorize_restart``;
    the real custody over live-shaped records is
    ``test_general_live_xid_twin_job_restart``)."""

    def __init__(self) -> None:
        self.authorized: list[tuple[dict[str, Any], bool]] = []

    def pods(self) -> list[dict[str, Any]]:
        return [{"name": "gl24tw-pod", "uid": "uid-original"}]

    def authorize_restart(self, state: dict[str, Any], *, unbound: bool) -> None:
        self.authorized.append((state, unbound))

    def wait_restarted(
        self, old_uids: set[str], *, timeout_seconds: int
    ) -> dict[str, Any]:
        assert old_uids == {"uid-original"}, old_uids
        if not self.authorized:
            raise WorkloadOwnershipError("managed workload ownership or UID changed")
        return {"pods": [{"name": "gl24tw-pod-r", "uid": "uid-restarted"}]}


class FakeJob:
    """The managed job as the variant sees it; the product sees its attempt."""

    name = "gl24tw-deadbeef-a1"
    attempt_id = "train-a001"

    def __init__(self, product: TwinProduct) -> None:
        product.job(self.name, self.attempt_id, JOB_GPUS)
        self.fixture = FakeFixture()
        self.removed = False

    def identity(self) -> dict[str, Any]:
        return {
            "job_id": self.name,
            "attempt_id": self.attempt_id,
            "pods": self.fixture.pods(),
            "pod_uids": ["uid-original"],
        }

    def cleanup(self) -> dict[str, Any]:
        self.removed = True
        return {"removed": True}


def _job_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, ledger: int = 1
) -> tuple[Run, FakeJob]:
    run = Run(tmp_path, monkeypatch, "xid_twin_unbound_job", gpus=EIGHT_GPUS)
    job = FakeJob(run.product)

    def start(ctx: Any) -> FakeJob:
        ctx.cleanup.push(f"job-{job.name}", job.cleanup)
        return job

    monkeypatch.setattr(twins, "start_job", start)
    monkeypatch.setattr(twins, "pod_gpus", lambda _ctx, _job: set(JOB_GPUS))
    monkeypatch.setattr(
        twins,
        "attempt_records",
        lambda _regional, _cluster, _job_id, attempts: {
            "attempts": {
                attempts[0]: {
                    "event": {"terminal_status": "STOPPED"},
                    "decision": {"status": "NO_ACTION"},
                    "containment_incident": None,
                }
            },
            "restart_budget": {"restart_count": ledger, "budget": 2},
        },
    )
    return run, job


def test_unbound_job_twin_stops_resets_and_restarts_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, job = _job_run(tmp_path, monkeypatch)
    out = run.run()

    assert out["verdict"] == "PASS", out["errors"]
    assert out["gpu_uuid"] == EIGHT_GPUS[-1] and out["gpu_uuid"] not in JOB_GPUS
    assert _executed(run.product) == {
        "STOP_WORKLOADS": 1,
        "QUIESCE_GPU_SERVICES": 1,
        "RESET_GPU": 1,
        "RESTART_WORKLOAD": 1,
    }, _executed(run.product)
    assert out["restart_attempt_id"] == "train-a001-r-1", out
    assert out["job"]["watcher_gpus"] == sorted(JOB_GPUS), out["job"]
    # The restarted Pods were adopted only under the occurrence's restart: its
    # workflow and incident -- which, the XID's GPU holding no managed attempt,
    # names no job (the product's own ingest here) -- bound as unbound.
    [(state, unbound)] = job.fixture.authorized
    assert unbound is True
    assert state["workflow"]["request_id"] == out["workflow_request_id"], state
    assert state["incident"]["incident_id"] == out["incident_id"], state
    assert state["incident"]["job_id"] is None, state["incident"]
    assert state["incident"]["workload_identity_source"] == UNBOUND_IDENTITY_SOURCE, (
        state["incident"]
    )
    assert all(
        item.get("workflow_request_id") == out["workflow_request_id"]
        for item in state["commands"]
    ), state["commands"]


def test_unbound_job_twin_fails_when_the_restart_is_not_bound_to_the_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The custody refuses (here: records that do not bind the restart): the
    job stage fails by name and the restarted Pods are never adopted."""

    run, job = _job_run(tmp_path, monkeypatch)
    waited: list[set[str]] = []

    def refuse(state: dict[str, Any], *, unbound: bool) -> None:
        raise WorkloadOwnershipError(
            "managed workload restart custody: restart execution is missing or "
            "ambiguous"
        )

    monkeypatch.setattr(job.fixture, "authorize_restart", refuse)
    monkeypatch.setattr(
        job.fixture,
        "wait_restarted",
        lambda old, **_kw: waited.append(old) or {"pods": []},
    )
    out = run.run()

    assert out["verdict"] == "FAIL", out
    assert out["stages"]["job"] == [
        "the product's restart is not bound to the job: WorkloadOwnershipError: "
        "managed workload restart custody: restart execution is missing or "
        "ambiguous"
    ], out["stages"]
    assert waited == [], "nothing is adopted before the restart is bound"


def test_unbound_job_twin_fails_on_a_second_restart_in_the_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, _job = _job_run(tmp_path, monkeypatch, ledger=2)
    out = run.run()

    assert out["verdict"] == "FAIL", out
    assert any("counts 2 restarts" in item for item in out["stages"]["job"]), out[
        "stages"
    ]


def test_unbound_job_twin_refuses_a_gpu_the_job_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every GPU in the job: no GPU outside it to name, before any replay."""

    run, _job = _job_run(tmp_path, monkeypatch)
    monkeypatch.setattr(twins, "JOB_GPU_COUNT", 8)
    monkeypatch.setattr(twins, "pod_gpus", lambda _ctx, _job: set(EIGHT_GPUS))
    run.product.job(_job.name, _job.attempt_id, list(EIGHT_GPUS))
    with pytest.raises(twins.VariantNotRun) as refused:
        twins.run_twin(run.ctx, run.case)
    assert refused.value.reason == "CAPABILITY_ABSENT", refused.value
    assert run.node.plan is None, "nothing was deployed"


def test_the_job_variant_reads_only_code_and_the_agent_before_its_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The managed job is the variant's first cluster mutation; before it the
    variant asked the control plane only the code-only generation probe and
    read the Agent's allowed operations (the GPU state probe), and touched no
    node probe."""

    run, job = _job_run(tmp_path, monkeypatch)
    monkeypatch.setattr(twins, "require_agent", REQUIRE_AGENT)
    order: list[str] = []
    probe = run.product.probe
    names = {
        records_module.XID_TWIN_GENERATION_PROBE: "generation",
        records_module.XID_TWIN_RECORDS_PROBE: "records",
        records_module.JOB_GPUS_PROBE: "job-gpus",
        GPU_STATE_PROBE: "gpu-state",
    }

    def spy(script: str, *arguments: str, **options: Any) -> dict[str, Any]:
        order.append(names.get(script, "store"))
        answer = probe(script, *arguments, **options)
        if script == GPU_STATE_PROBE:
            # The node's Agent as the live store holds it.
            answer["agent"] = {
                "allowed_operations": ["QUIESCE_GPU_SERVICES", "RESET_GPU"]
            }
        return answer

    run.ctx.regional.cpu_python_responder = spy
    started = twins.start_job

    def start(ctx: Any) -> Any:
        order.append("start-job")
        assert all(not item.calls for item in ctx.probes.values()), (
            "no node probe before the job"
        )
        return started(ctx)

    monkeypatch.setattr(twins, "start_job", start)
    out = run.run()

    assert out["verdict"] == "PASS", out["errors"]
    assert order[:3] == ["generation", "gpu-state", "start-job"], order
    assert job.removed is False, "the job is the cleanup stack's, run by the runner"
