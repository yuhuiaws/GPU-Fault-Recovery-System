"""GF-GENERAL-LIVE-024 ``xid_twin_*``: the gates, job readers and stages at their edges.

``test_general_live_xid_twins`` and ``test_general_live_xid_twins_product``
run the variants and their premises on the main paths. The cases here pin the
rest to the record that produces it: a catalog without the XID or without the
site's family; a node with no GPU the twin can name; the job's GPUs as its
container and the product's watcher see them, and the three ways they fail to
agree; the grading of a twin that never reached the product, of a pair one of
whose reports is still undecided, of a repeat that never arrived or moved the
gauge again; the job stage on the job's own GPU and on a restart that never
ran; the hand-back of a node the chain left isolated; each registered
variant's ``execute`` running its own case; and a kernel line the product
never recorded, which ends the run INCONCLUSIVE with the probe's oracle.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_xid_twins as twins
from scripts.e2e.regional.general_live_fixture import (
    VariantContext,
    VariantInconclusive,
    VariantNotRun,
)
from scripts.e2e.regional.general_live_gpu_replay import GpuSurvey
from scripts.e2e.regional.general_live_kernel_xid import CatalogEntry
from scripts.e2e.regional.general_live_xid_twin_delivery import Delivery
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    make_context,
    make_settings,
)
from tests.regional._general_live_gpu_fakes import survey
from tests.regional.test_general_live_xid_twins import G0, G1, NODE, _event
from tests.regional.test_general_live_xid_twins_product import Run

START = datetime(2026, 11, 1, 12, tzinfo=timezone.utc)
SAFE = twins.CASES["xid_twin_kernel_first"]


class Cluster(FakeRegional):
    """The read surface of the job readers: each Pod's ``nvidia-smi`` and the
    product's watcher observations."""

    def __init__(
        self,
        *,
        pod_output: dict[str, str] | None = None,
        watcher: dict[str, Any] | None = None,
        node: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            cpu_python=lambda _script, *_arguments: dict(watcher or {}),
            node_snapshot=node,
        )
        self.pod_output = pod_output or {}
        self.execs: list[tuple[str, ...]] = []

    def kubectl(self, plane: str, *arguments: str, **options: Any) -> str:
        self.execs.append((plane, *arguments))
        return self.pod_output[arguments[1]]


def context(tmp_path: Path, regional: FakeRegional, variant: str) -> VariantContext:
    spec = next(item for item in twins.VARIANTS if item.variant == variant)
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={},
        settings=make_settings(spec, node=NODE),
    )
    now = {"value": START}
    ctx.clock = lambda: now["value"]

    def sleep(seconds: float) -> None:
        now["value"] += timedelta(seconds=seconds)

    ctx.sleep = sleep
    return ctx


class Fixture:
    def __init__(self, pods: list[str], *, restarted: Exception | None = None) -> None:
        self.names = pods
        self.restarted = restarted
        self.authorized: list[tuple[dict[str, Any], bool]] = []

    def pods(self) -> list[dict[str, Any]]:
        return [{"name": name, "uid": f"uid-{name}"} for name in self.names]

    def authorize_restart(self, state: dict[str, Any], *, unbound: bool) -> None:
        self.authorized.append((state, unbound))

    def wait_restarted(self, old: set[str], *, timeout_seconds: int) -> dict:
        if self.restarted is not None:
            raise self.restarted
        return {"pods": [{"name": "restarted", "uid": "uid-new"}]}


class Job:
    name = "gl24tw-deadbeef-a1"
    attempt_id = "train-a001"

    def __init__(self, fixture: Fixture) -> None:
        self.fixture = fixture


def watcher(*gpus: str, attempt: str = "train-a001") -> dict[str, Any]:
    return {
        "job_id": Job.name,
        "attempts": [
            {
                "attempt_id": attempt,
                "containers": [
                    {"node_id": NODE, "gpu_uuids": list(gpus), "terminated": False},
                    {"node_id": NODE, "gpu_uuids": ["GPU-dead"], "terminated": True},
                ],
            }
        ],
    }


# --------------------------------------------------------------------------- #
# the pinned catalog
# --------------------------------------------------------------------------- #
def test_a_catalog_without_the_xid_leaves_the_twin_not_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(twins, "catalog_entry", lambda _xid: None)

    with pytest.raises(VariantNotRun) as refused:
        twins.require_catalog(SAFE)

    assert refused.value.reason == "FIXTURE_NOT_READY"
    assert refused.value.detail == "XID 92 is absent from the pinned catalog"


def test_a_catalog_row_without_the_sites_family_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = CatalogEntry(
        xid=48,
        mnemonic="ROBUST_CHANNEL_GPU_ECC_DBE",
        immediate_action="WORKFLOW_XID_48",
        investigatory_action=None,
        products=("A100",),
        catalog_version="615",
    )
    monkeypatch.setattr(twins, "catalog_entry", lambda _xid: row)

    with pytest.raises(VariantNotRun) as refused:
        twins.require_catalog(twins.CASES["xid_twin_reset_gpu"])

    assert refused.value.detail == "XID 48 does not apply to the H100 family (['A100'])"


# --------------------------------------------------------------------------- #
# choosing the GPU
# --------------------------------------------------------------------------- #
def test_no_gpu_without_a_bdf_or_of_another_family_can_carry_the_twin() -> None:
    value = survey(gpus=(G0, G1), fields=("xid_last_error",))
    value["exporter"]["gpus"][G0]["pci_bdf"] = ""
    value["exporter"]["gpus"][G1]["model"] = "Tesla V100-SXM2-16GB"

    with pytest.raises(VariantNotRun) as refused:
        twins.choose_gpu(GpuSurvey(value), SAFE)

    assert refused.value.reason == "CAPABILITY_ABSENT"
    assert refused.value.detail == (
        f"no GPU the twin can name: {G1}: 'Tesla V100-SXM2-16GB' is family None; "
        f"{G0}: no PCI BDF"
    )


# --------------------------------------------------------------------------- #
# the job's GPUs
# --------------------------------------------------------------------------- #
def test_the_containers_gpus_are_the_uuids_its_nvidia_smi_lists(tmp_path: Path) -> None:
    regional = Cluster(
        pod_output={
            "trainer-0": "GPU-a\nGPU-b\n",
            "trainer-1": "  GPU-c  \nNo devices were found\n\n",
        }
    )
    ctx = context(tmp_path, regional, "xid_twin_unbound_job")

    found = twins.pod_gpus(ctx, Job(Fixture(["trainer-0", "trainer-1"])))

    assert found == {"GPU-a", "GPU-b", "GPU-c"}
    assert regional.execs[0] == (
        "gpu",
        "exec",
        "trainer-0",
        "-c",
        "pytorch",
        "--",
        "nvidia-smi",
        "--query-gpu=uuid",
        "--format=csv,noheader",
    )
    assert len(regional.execs) == 2, "one read per Pod"


@pytest.mark.parametrize(
    "pods, mapped, message",
    [
        (
            "",
            (),
            "neither the job's container nor the product's watcher names the GPUs it "
            "holds; a GPU the job does not use cannot be chosen",
        ),
        (
            "GPU-a\nGPU-b\nGPU-c\nGPU-d\n",
            ("GPU-a", "GPU-b", "GPU-c", "GPU-e"),
            "the job's container sees ['GPU-a', 'GPU-b', 'GPU-c', 'GPU-d'], the "
            "watcher maps ['GPU-a', 'GPU-b', 'GPU-c', 'GPU-e']; the job's GPUs are "
            "not established",
        ),
        ("GPU-a\nGPU-b\nGPU-c\n", (), "the job holds 3 GPUs, not the 4 it asked for"),
    ],
    ids=["nothing-seen", "disagree", "short"],
)
def test_the_jobs_gpus_must_be_established_before_one_is_chosen(
    tmp_path: Path, pods: str, mapped: tuple[str, ...], message: str
) -> None:
    regional = Cluster(pod_output={"trainer-0": pods}, watcher=watcher(*mapped))
    ctx = context(tmp_path, regional, "xid_twin_unbound_job")

    with pytest.raises(VariantInconclusive) as raised:
        twins.job_gpus(ctx, Job(Fixture(["trainer-0"])))

    assert str(raised.value) == message
    document = json.loads((ctx.case_dir / "job-gpus.json").read_text())
    assert document["watcher_gpus"] == sorted(mapped), document


# --------------------------------------------------------------------------- #
# grading
# --------------------------------------------------------------------------- #
def test_a_twin_that_never_reached_the_product_is_graded_inconclusive() -> None:
    stages, inconclusive, facts = twins.grade(
        {"events": [], "incidents": []},
        {"workflows": [], "commands": []},
        SAFE,
        gpu=G1,
        delivery=Delivery(order="kernel-first"),
    )

    assert inconclusive == [
        f"no field-230 change of XID 92 on {G1} reached the product",
        f"no kernel line of XID 92 on {G1} reached the product",
        "the repeated kernel line was not written",
    ]
    assert stages == {
        "workflows": [],
        "executions": [],
        "hardware": [],
        "negative_control": [],
    }
    assert (facts["pair"], facts["repeat_event_id"]) == (None, None)


def test_a_pair_with_an_undecided_report_is_a_premise_not_a_verdict() -> None:
    kernel = _event(
        "k", "KERNEL_LOG", 0, incident_id="inc-k", disposition="PENDING_CORRELATION"
    )
    gauge = _event(
        "g",
        "DCGM_EXPORTER",
        20,
        incident_id="inc-k",
        duplicate=True,
        correlated_event_id="k",
    )
    single = dataclasses.replace(SAFE, repeat=False)

    _stages, inconclusive, facts = twins.grade(
        {
            "events": [kernel, gauge],
            "incidents": [{"incident_id": "inc-k", "state": "RECOVERED"}],
        },
        {"workflows": [], "commands": []},
        single,
        gpu=G1,
        delivery=Delivery(order="kernel-first"),
    )

    assert inconclusive == [
        "a report of the pair has no final decision within the budget"
    ]
    assert facts["pair"]["joiner"] == "g", facts["pair"]


def test_a_repeat_that_never_reached_the_product_is_a_premise() -> None:
    kernel = _event("k", "KERNEL_LOG", 0, incident_id="inc-k")
    gauge = _event("g", "DCGM_EXPORTER", 20, incident_id="inc-k")

    errors, premise, repeat = twins.grade_repeat(
        {"events": [kernel, gauge]},
        SAFE,
        gpu=G1,
        delivery=Delivery(order="kernel-first", repeat_written_at=170.0),
        incident_id="inc-k",
    )

    assert (errors, premise, repeat) == (
        [],
        ["the repeated kernel line never reached the product"],
        None,
    )


def test_a_repeat_that_moved_the_gauge_again_is_a_product_error() -> None:
    events = [
        _event("k", "KERNEL_LOG", 0, incident_id="inc-k"),
        _event("g", "DCGM_EXPORTER", 20, incident_id="inc-k", duplicate=True),
        _event("g2", "DCGM_EXPORTER", 150, incident_id="inc-g2"),
        _event("r", "KERNEL_LOG", 200, incident_id="inc-r"),
    ]
    records = {
        "events": events,
        "incidents": [
            {"incident_id": "inc-k", "state": "RECOVERED"},
            {"incident_id": "inc-r", "state": "RECOVERED"},
        ],
    }

    errors, premise, repeat = twins.grade_repeat(
        records,
        SAFE,
        gpu=G1,
        delivery=Delivery(order="kernel-first", repeat_written_at=200.0),
        incident_id="inc-k",
    )

    assert (premise, repeat) == ([], "r")
    assert errors == ["2 field-230 changes: the gauge moved again for the repeat"]


# --------------------------------------------------------------------------- #
# the job stage
# --------------------------------------------------------------------------- #
def job_document() -> tuple[dict[str, Any], dict[str, Any]]:
    workflow = {
        "request_id": "wf-twin",
        "incident_id": "inc-twin",
        "official_steps": [{"operation": "RESTART_WORKLOAD"}],
        "step_executions": [
            {
                "operation": "RESTART_WORKLOAD",
                "status": "SUCCEEDED",
                "updated_at": "2026-11-01T12:10:00+00:00",
                "details": {"restart_attempt_id": "train-a001-r-1"},
            }
        ],
    }
    document = {
        "workflows": [workflow],
        "incidents": [{"incident_id": "inc-twin"}],
        "commands": [{"workflow_request_id": "wf-twin", "command_id": "c1"}],
    }
    return document, {"workflow_request_id": "wf-twin"}


def stopped_once(*_arguments: Any) -> dict[str, Any]:
    return {
        "attempts": {
            "train-a001": {
                "event": {"terminal_status": "STOPPED"},
                "decision": {"status": "NO_ACTION"},
                "containment_incident": None,
            }
        },
        "restart_budget": {"restart_count": 1, "budget": 2},
    }


def test_the_job_stage_names_the_jobs_own_gpu_and_a_restart_that_never_ran(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(twins, "attempt_records", stopped_once)
    ctx = context(tmp_path, Cluster(), "xid_twin_unbound_job")
    fixture = Fixture(["trainer-0"], restarted=TimeoutError("no Pod Running in 900 s"))
    document, facts = job_document()

    errors = twins.judge_job(
        ctx,
        Job(fixture),
        document,
        facts,
        {"container_gpus": [G1], "watcher_gpus": [], "pod_uids": ["uid-trainer-0"]},
        gpu=G1,
    )

    assert errors == [
        f"the XID GPU {G1} is one of the job's",
        "the restarted job did not run: TimeoutError: no Pod Running in 900 s",
    ]
    assert facts["restart_attempt_id"] == "train-a001-r-1"
    ((state, unbound),) = fixture.authorized
    assert unbound is True and state["workflow"]["request_id"] == "wf-twin", state
    assert state["commands"] == [{"workflow_request_id": "wf-twin", "command_id": "c1"}]
    assert not (ctx.case_dir / "job-restarted.json").exists(), (
        "a restart that never ran is not recorded as restarted"
    )


# --------------------------------------------------------------------------- #
# the hand-back of an isolated node
# --------------------------------------------------------------------------- #
ISOLATED = {
    "name": NODE,
    "uid": "node-uid",
    "boot_id": "boot-1",
    "ready": "True",
    "unschedulable": True,
    "taints": [],
    "ownership_annotations": {},
    "spare_pool_annotations": {},
    "gpu_allocatable": "8",
}


def test_an_isolated_node_no_twin_incident_owns_is_named(tmp_path: Path) -> None:
    ctx = context(tmp_path, Cluster(node=ISOLATED), "xid_twin_reset_gpu")
    records = {"incidents": [{"incident_id": "inc-k", "state": "RECOVERED"}, {}]}

    assert twins.restore_if_left_isolated(ctx, records, {}) == [
        "the node is isolated and no incident of the twin owns it"
    ]


def test_a_refused_validated_restore_is_the_isolation_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(ctx: Any, incidents: Any, *, reason: str) -> list[dict[str, Any]]:
        raise RuntimeError("validated restore of inc-k ended FAILED")

    monkeypatch.setattr(twins, "restore_incidents", refuse)
    ctx = context(tmp_path, Cluster(node=ISOLATED), "xid_twin_reset_gpu")
    facts: dict[str, Any] = {}

    errors = twins.restore_if_left_isolated(
        ctx, {"incidents": [{"incident_id": "inc-k", "state": "ACTION_PENDING"}]}, facts
    )

    assert errors == [
        "validated restore failed: RuntimeError: validated restore of inc-k ended FAILED"
    ]
    assert "restored_incidents" not in facts


def test_the_twins_open_incidents_are_handed_back_through_the_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    restored: list[tuple[list[dict[str, Any]], str]] = []

    def restore(ctx: Any, incidents: Any, *, reason: str) -> list[dict[str, Any]]:
        restored.append((list(incidents), reason))
        return []

    monkeypatch.setattr(twins, "restore_incidents", restore)
    ctx = context(tmp_path, Cluster(node=ISOLATED), "xid_twin_reset_gpu")
    facts: dict[str, Any] = {}
    records = {
        "incidents": [
            {"incident_id": "inc-k", "state": "ESCALATED"},
            {"incident_id": "inc-done", "state": "RECOVERED"},
        ]
    }

    assert twins.restore_if_left_isolated(ctx, records, facts) == []
    assert restored == [
        (
            [{"incident_id": "inc-k"}],
            "GF-GENERAL-LIVE-024.xid_twin_reset_gpu acceptance restore",
        )
    ]
    assert facts["restored_incidents"] == ["inc-k"]


# --------------------------------------------------------------------------- #
# the registered variants
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "variant",
    [
        "xid_twin_gauge_first",
        "xid_twin_reset_gpu",
        "xid_twin_unbound_job",
        "xid_twin_drain_p2p",
    ],
)
def test_each_registered_variant_runs_its_own_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    ran: list[twins.TwinCase] = []

    def run_twin(ctx: Any, case: twins.TwinCase) -> dict[str, Any]:
        ran.append(case)
        return {"verdict": "PASS", "variant": case.variant}

    monkeypatch.setattr(twins, "run_twin", run_twin)
    spec = next(item for item in twins.VARIANTS if item.variant == variant)
    ctx = context(tmp_path, Cluster(), variant)

    assert spec.execute(ctx) == {"verdict": "PASS", "variant": variant}
    assert ran == [twins.CASES[variant]]


def test_a_kernel_line_the_product_never_recorded_ends_the_run_inconclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = Run(tmp_path, monkeypatch, "xid_twin_kernel_first")
    clock = run.clock

    def arm(arguments: tuple[str, ...]) -> dict[str, Any]:
        # The probe wrote the line; the kernel collector never delivered it.
        return {"armed": True, "written_at": clock.epoch(), "lines_written": 2}

    silent = FakeProbe(
        {
            "install-records": {"records": 2},
            "open-window": {"after": {"ActiveState": "active"}},
            "arm": arm,
            "status": {"injected": {"mode": "kmsg", "fixture_lines": 2}},
            "close-window": {"clean": True},
        }
    )
    run.ctx.probes[f"kernel-replay-kmsg@{NODE}"] = silent  # type: ignore[assignment]

    with pytest.raises(VariantInconclusive) as raised:
        run.spec.execute(run.ctx)

    assert str(raised.value) == (
        "the kernel line of XID 92 never reached the product (the fixture holds 2 "
        "lines)"
    )
    assert run.node.kernel_posts == [], "the product never saw the line"
    timeline = json.loads(
        (run.ctx.case_dir / "twin-kernel-recorded-timeline.json").read_text()
    )
    assert not any(entry["accepted"] for entry in timeline["entries"]), timeline
