"""Unit coverage of the GF-GENERAL-LIVE registry, verdict helpers and store reads."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.models import WorkflowStatus
from scripts.e2e.regional import general_live_registry as reg
from scripts.e2e.regional import general_live_store as store
from scripts.e2e.regional import general_live_verdicts as v
from tests._builders import build_store, workflow_request
from tests.regional._general_live_fakes import activity_read, save_diagnostic_incident


def test_every_registered_variant_runs_on_both_clusters() -> None:
    """The site has two EFA GPU nodes per cluster: no variant is single-lane now.

    The two-node 011 job, the exclusive-node 021 resets, the 014 freezes and
    the 030 spare declaration run on both; plan-only deferred variants record
    their NOT_RUN reason on both boards (``deferred_variant`` defaults to both).
    """

    registry = reg.load_registry()
    single = sorted(key for key, item in registry.items() if item.lanes != reg.LANES)
    assert single == [], single
    plan_only = reg.deferred_variant(
        "GF-GENERAL-LIVE-031",
        "inactive_cluster",
        proof_mode=reg.CAPABILITY,
        reason="CAPABILITY_ABSENT: x",
    )
    assert plan_only.lanes == reg.LANES and plan_only.flags == "P"


def spec(**overrides):
    base = dict(
        case_id="GF-GENERAL-LIVE-001",
        variant="cpu_busy",
        proof_mode=reg.NATIVE,
        lanes=("eks",),
        risk="L",
        injection="i",
        expected="e",
        cleanup="c",
        max_minutes=10,
        execute=lambda ctx: {"verdict": "PASS"},
    )
    base.update(overrides)
    return reg.VariantSpec(**base)


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def test_variant_spec_properties() -> None:
    s = spec()
    assert s.key == "GF-GENERAL-LIVE-001.cpu_busy"
    assert s.number == 1
    assert s.confirmation == "GENERALLIVE001_EXECUTE"
    assert s.implemented is True
    described = s.describe()
    assert described["key"] == s.key and "execute" not in described


def test_deferred_variant_is_not_implemented() -> None:
    s = reg.deferred_variant(
        "GF-GENERAL-LIVE-004",
        "probe_error",
        proof_mode="live-replay",
        reason="FIXTURE_NOT_READY: x",
    )
    assert s.implemented is False and s.execute is None
    reg.validate_spec(s)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"case_id": "GF-X-1"}, "bad case id"),
        ({"variant": "Bad Name"}, "bad variant"),
        ({"proof_mode": "nope"}, "unknown proof mode"),
        ({"lanes": ("mars",)}, "unknown lane"),
        ({"risk": "Z"}, "unknown risk"),
        ({"flags": "ZZ"}, "bad list flags"),
        ({"approvals": ("NOPE",)}, "unknown approvals"),
        ({"expected_outcome": "MAYBE"}, "expected outcome"),
        ({"expected_outcome": "FAIL"}, "must cite its product gap"),
        ({"max_minutes": 999}, "max_minutes is outside"),
        ({"proof_mode": "vendor-lab"}, "only native, observation and live-replay"),
        ({"execute": None}, "needs execute"),
    ],
)
def test_validate_spec_rejects(overrides, match) -> None:
    with pytest.raises(reg.RegistryError, match=match):
        reg.validate_spec(spec(**overrides))


def test_validate_spec_deferred_reason_and_executor() -> None:
    with pytest.raises(reg.RegistryError, match="deferred reason"):
        reg.validate_spec(spec(deferred="NOPE: x", execute=None))
    with pytest.raises(reg.RegistryError, match="has no executor"):
        reg.validate_spec(spec(deferred="FIXTURE_NOT_READY: x"))


def test_build_registry_detects_missing_and_extra() -> None:
    rows = {"GF-GENERAL-LIVE-001": ["cpu_busy", "recovery"]}
    with pytest.raises(reg.RegistryError, match="missing="):
        reg.build_registry([spec()], rows=rows, categories=[1])
    with pytest.raises(reg.RegistryError, match="extra="):
        reg.build_registry(
            [spec(), spec(variant="recovery"), spec(variant="extra")],
            rows={"GF-GENERAL-LIVE-001": ["cpu_busy", "recovery"]},
            categories=[1],
        )
    with pytest.raises(reg.RegistryError, match="duplicate"):
        reg.build_registry(
            [spec(), spec()], rows={"GF-GENERAL-LIVE-001": ["cpu_busy"]}, categories=[1]
        )


def test_select_and_coverage_rows() -> None:
    registry = {spec().key: spec()}
    assert reg.select(registry, "GF-GENERAL-LIVE-001", "cpu_busy").variant == "cpu_busy"
    with pytest.raises(reg.RegistryError, match="no variant"):
        reg.select(registry, "GF-GENERAL-LIVE-001", "ghost")
    with pytest.raises(reg.RegistryError, match="not implemented"):
        reg.select(registry, "GF-GENERAL-LIVE-099", "x")
    rows = reg.coverage_rows()
    assert rows["GF-GENERAL-LIVE-001"][0] == "cpu_busy"


def test_full_registry_matches_coverage() -> None:
    registry = reg.load_registry()
    rows = reg.coverage_rows()
    for number in reg.PHASE_ONE_CATEGORIES + reg.PHASE_TWO_CATEGORIES:
        case = f"GF-GENERAL-LIVE-{number:03d}"
        for variant in rows[case]:
            assert f"{case}.{variant}" in registry


# --------------------------------------------------------------------------- #
# verdicts
# --------------------------------------------------------------------------- #
def _incident(metric, device="node", rule=None, state="ACTION_PENDING"):
    event = f"b-{metric}-{v.safe_device(device)}" + (f"-{rule.lower()}" if rule else "")
    return {
        "incident_id": f"inc-{metric}",
        "event_id": event,
        "event_type": "NODE_HEALTH",
        "state": state,
        "created_at": "2026-10-24T00:03:00+00:00",
    }


def _wf(incident_id, ops, status="RUNNING", execs=(), probe=None):
    steps = [
        {
            "operation": op,
            "parameters": (
                {"host_diagnostic_probe": {"metric": probe}}
                if probe and op.startswith("VALIDATE")
                else {}
            ),
        }
        for op in ops
    ]
    return {
        "request_id": "wf",
        "incident_id": incident_id,
        "status": status,
        "official_steps": steps,
        "step_executions": list(execs),
    }


def test_finding_incidents_device_and_rule() -> None:
    doc = {
        "incidents": [
            _incident("cpu_usage_percent"),
            _incident("load1_per_cpu", rule="LOW"),
        ]
    }
    assert v.finding_incidents(doc, "cpu_usage_percent"), (
        "v.finding_incidents(doc, 'cpu_usage_percent')"
    )
    assert v.finding_incidents(doc, "load1_per_cpu", rule_id="LOW"), (
        "v.finding_incidents(doc, 'load1_per_cpu', rule_id='LOW')"
    )
    assert not v.finding_incidents(doc, "load1_per_cpu", rule_id="OTHER"), (
        "not v.finding_incidents(doc, 'load1_per_cpu', rule_id='OTHER')"
    )
    assert not v.finding_incidents(doc, "cpu_usage_percent", device="dm-7"), (
        "not v.finding_incidents(doc, 'cpu_usage_percent', device='dm-7')"
    )


#: What ``ACTIVITY_PROBE`` names about its own read.
READ = {"since": "2026-10-25T03:19:46+00:00", "cluster_id": "c", "node_id": "node-a"}


def _minted(**changes) -> dict:
    """A LOW_MEMORY_AVAILABLE incident as the probe prints it (UTC ``Z``)."""

    incident = {
        **_incident("memory_available_percent", rule="LOW_MEMORY_AVAILABLE"),
        "cluster_id": "c",
        "node_ids": ["node-a"],
        "created_at": "2026-10-25T03:37:50.000000Z",
    }
    return {**incident, **changes}


def _low_memory(doc: dict) -> list:
    return v.finding_incidents(
        doc, "memory_available_percent", rule_id="LOW_MEMORY_AVAILABLE"
    )


@pytest.mark.parametrize(
    "state", ["DETECTED", "ACTION_PENDING", "RECOVERED", "ESCALATED", "QUARANTINED"]
)
def test_finding_incidents_takes_the_reads_incident_in_any_state(state: str) -> None:
    assert _low_memory({**READ, "incidents": [_minted(state=state)]}), state


def test_finding_incidents_reads_a_zoneless_creation_time_as_utc() -> None:
    doc = {**READ, "incidents": [_minted(created_at="2026-10-25T03:19:46")]}
    assert _low_memory(doc), "created at the read's since, written without a zone"


@pytest.mark.parametrize(
    ("changes", "why"),
    [
        ({"created_at": "2026-10-25T03:19:45+00:00"}, "created before the read"),
        ({"created_at": "not-a-time"}, "no readable creation time"),
        ({"created_at": None}, "no creation time"),
        ({"node_ids": ["node-b"]}, "another node's incident"),
        ({"cluster_id": "other"}, "another cluster's incident"),
        ({"event_id": "b-memory_used_percent-node"}, "another metric's finding"),
        ({"event_id": "b-memory_available_percent-node-memory"}, "threshold family"),
    ],
)
def test_finding_incidents_refuses_what_the_read_did_not_mint(
    changes: dict, why: str
) -> None:
    assert not _low_memory({**READ, "incidents": [_minted(**changes)]}), why


def test_finding_incidents_takes_a_scripted_document_as_it_is() -> None:
    # No since/cluster/node: the document is all the matcher has to go on.
    doc = {"incidents": [_minted(created_at="2026-01-01T00:00:00+00:00")]}
    assert _low_memory(doc), "a scripted document names no read"


@pytest.mark.parametrize(
    "label", ["NODE_HEALTH_GROUP", "NODE_RESOURCE_GROUP", "GPU_FAULT_GROUP"]
)
def test_finding_incidents_takes_the_minted_record_under_a_group_label(
    label: str,
) -> None:
    # Live 2026-10-25 008 required_link_down: the veth peer's finding joined
    # the incident, which the product re-labelled NODE_HEALTH_GROUP; its id and
    # event id still name the finding that minted it.
    assert _low_memory({**READ, "incidents": [_minted(event_type=label)]}), label


@pytest.mark.parametrize("label", ["XID", "SXID", "XID_BATCH", "NODE", None])
def test_finding_incidents_refuses_an_incident_no_node_health_finding_minted(
    label: str | None,
) -> None:
    assert not _low_memory({**READ, "incidents": [_minted(event_type=label)]}), label


def test_chain_errors_present_and_absent() -> None:
    doc = {
        "incidents": [_incident("cpu_usage_percent")],
        "workflows": [
            _wf(
                "inc-cpu_usage_percent",
                ["FREEZE_EVIDENCE", "VALIDATE_HOST"],
                status="FAILED",
                execs=[
                    {
                        "operation": "VALIDATE_HOST",
                        "status": "FAILED",
                        "details": {"host_diagnostic_failure": True},
                    }
                ],
                probe="cpu_usage_percent",
            )
        ],
    }
    errors = v.chain_errors(
        doc,
        v.ChainExpectation(
            metric="cpu_usage_percent",
            operations=("FREEZE_EVIDENCE", "VALIDATE_HOST"),
            validation="VALIDATE_HOST",
            validation_must_fail=True,
            probe_metric="cpu_usage_percent",
            incident_states=frozenset({"ACTION_PENDING"}),
        ),
    )
    assert errors == []
    assert v.chain_errors(
        {"incidents": []}, v.ChainExpectation(metric="cpu_usage_percent")
    ), "v.chain_errors( {'incidents': []}, v.ChainExpectation(metric='cpu_usag"
    assert v.chain_errors(
        doc, v.ChainExpectation(metric="cpu_usage_percent", present=False)
    ), "v.chain_errors( doc, v.ChainExpectation(metric='cpu_usage_percent', pr"


def _held_cpu_fault(state: str, reasons: list[str]) -> dict:
    incident = {**_incident("cpu_usage_percent", state=state), "reasons": reasons}
    return {
        "incidents": [incident],
        "workflows": [
            _wf(
                "inc-cpu_usage_percent",
                ["FREEZE_EVIDENCE", "VALIDATE_HOST"],
                status="FAILED",
                execs=[
                    {
                        "operation": "VALIDATE_HOST",
                        "status": "FAILED",
                        "details": {"host_diagnostic_failure": True},
                    }
                ],
                probe="cpu_usage_percent",
            )
        ],
    }


HELD_CPU = v.ChainExpectation(
    metric="cpu_usage_percent",
    operations=("FREEZE_EVIDENCE", "VALIDATE_HOST"),
    validation="VALIDATE_HOST",
    validation_must_fail=True,
    probe_metric="cpu_usage_percent",
    incident_states=v.DIAGNOSTIC_ONLY_STATES,
)


def test_a_failed_diagnostic_only_chain_closes_recovered_inconclusive() -> None:
    # Live 2026-10-24 GF-GENERAL-LIVE-001.cpu_busy: the product closes a failed
    # evidence/validation-only workflow RECOVERED with "diagnostic
    # inconclusive" (detailed design 6.3); the runner expected ESCALATED.
    doc = _held_cpu_fault(
        "RECOVERED",
        [
            "CPU saturation threshold exceeded",
            "diagnostic inconclusive: validation failed",
        ],
    )
    assert v.chain_errors(doc, HELD_CPU) == []


@pytest.mark.parametrize(
    ("state", "reasons", "fragment"),
    [
        ("ESCALATED", ["diagnostic inconclusive"], "incident is ESCALATED"),
        ("RECOVERED", ["CPU saturation threshold exceeded"], "without the"),
    ],
)
def test_a_held_fault_is_never_escalated_or_silently_recovered(
    state: str, reasons: list[str], fragment: str
) -> None:
    errors = v.chain_errors(_held_cpu_fault(state, reasons), HELD_CPU)
    assert any(fragment in item for item in errors), errors


def test_chain_errors_detects_every_shape() -> None:
    doc = {
        "incidents": [_incident("m", state="RECOVERED"), _incident("m")],
        "workflows": [
            _wf(
                "inc-m",
                ["FREEZE_EVIDENCE"],
                status="SUCCEEDED",
                execs=[{"operation": "VALIDATE_HOST", "status": "SUCCEEDED"}],
            )
        ],
    }
    errors = v.chain_errors(
        doc,
        v.ChainExpectation(
            metric="m",
            operations=("VALIDATE_HOST",),
            validation="VALIDATE_HOST",
            validation_must_fail=True,
            probe_metric="m",
            incident_states=frozenset({"ACTION_PENDING"}),
            single=True,
        ),
    )
    joined = " ".join(errors)
    assert "incidents (expected one)" in joined
    assert "lack" in joined
    assert "succeeded while the fault" in joined
    assert "no validation step carries" in joined
    doc2 = {"incidents": [_incident("m")], "workflows": []}
    assert "has no workflow" in " ".join(
        v.chain_errors(doc2, v.ChainExpectation(metric="m"))
    )


def test_validation_missing_and_failed_without_marker() -> None:
    doc = {
        "incidents": [_incident("m")],
        "workflows": [
            _wf(
                "inc-m",
                ["VALIDATE_HOST"],
                execs=[
                    {"operation": "VALIDATE_HOST", "status": "FAILED", "details": {}}
                ],
            )
        ],
    }
    assert "without naming the held fault" in " ".join(
        v.chain_errors(
            doc,
            v.ChainExpectation(
                metric="m", validation="VALIDATE_HOST", validation_must_fail=True
            ),
        )
    )
    # Live 2026-10-24 GF-GENERAL-LIVE-001.load_average: no marker on that
    # contract, but node_failures names the held metric's probe -- accepted.
    named = {
        "incidents": [_incident("load1_per_cpu")],
        "workflows": [
            _wf(
                "inc-load1_per_cpu",
                ["VALIDATE_HOST"],
                execs=[
                    {
                        "operation": "VALIDATE_HOST",
                        "status": "FAILED",
                        "details": {
                            "node_failures": {
                                "node-a": [
                                    "cpu_usage_percent",
                                    "host_diagnostic:load1_per_cpu",
                                    "load1_per_cpu",
                                ]
                            }
                        },
                    }
                ],
            )
        ],
    }
    assert not any(
        "held fault" in item
        for item in v.chain_errors(
            named,
            v.ChainExpectation(
                metric="load1_per_cpu",
                validation="VALIDATE_HOST",
                validation_must_fail=True,
            ),
        )
    ), "a failure naming host_diagnostic:load1_per_cpu judged the held fault"
    doc2 = {
        "incidents": [_incident("m")],
        "workflows": [_wf("inc-m", ["FREEZE_EVIDENCE"])],
    }
    assert "never executed VALIDATE_HOST" in " ".join(
        v.chain_errors(doc2, v.ChainExpectation(metric="m", validation="VALIDATE_HOST"))
    )
    doc3 = {
        "incidents": [_incident("m")],
        "workflows": [
            _wf(
                "inc-m",
                ["VALIDATE_HOST"],
                execs=[
                    {"operation": "VALIDATE_HOST", "status": "FAILED", "error": "boom"}
                ],
            )
        ],
    }
    assert "ended FAILED" in " ".join(
        v.chain_errors(doc3, v.ChainExpectation(metric="m", validation="VALIDATE_HOST"))
    )


def test_hardware_isolation_settled_detection_notifications() -> None:
    doc = {
        "workflows": [
            _wf(
                "i",
                ["RESET_GPU", "MARK_UNSCHEDULABLE", "STOP_WORKLOADS"],
                status="SUCCEEDED",
            )
        ]
    }
    assert v.hardware_errors(doc), "v.hardware_errors(doc)"
    assert v.hardware_errors(doc, allowed={"RESET_GPU"}) == []
    assert set(x.split(":")[1] for x in v.isolation_operations(doc)) == {
        "MARK_UNSCHEDULABLE",
        "STOP_WORKLOADS",
    }
    assert v.settled({"workflows": [_wf("i", ["x"], status="SUCCEEDED")]}, ["i"]), (
        "v.settled({'workflows': [_wf('i', ['x'], status='SUCCEEDED')]}, ['i'])"
    )
    assert not v.settled({"workflows": [_wf("i", ["x"], status="RUNNING")]}, ["i"]), (
        "not v.settled({'workflows': [_wf('i', ['x'], status='RUNNING')]}, ['i'"
    )
    assert (
        v.detection_seconds(
            {}, _incident("m"), datetime(2026, 10, 24, tzinfo=timezone.utc)
        )
        == 180.0
    )
    assert v.notification_errors(
        {"notifications": [{"incident_id": "i"}, {"incident_id": "i"}]},
        incident_ids=["i"],
    ), "v.notification_errors( {'notifications': [{'incident_id': 'i'}, {'inci"
    assert (
        v.notification_errors(
            {"notifications": [{"incident_id": "i"}]}, incident_ids=["i"]
        )
        == []
    )


def test_oracle_and_sustained_and_node_isolated() -> None:
    assert v.oracle_values(
        [{"a": 1.0}, {"a": "x"}, {"a": float("nan")}], lambda r: r.get("a")
    ) == [1.0]
    assert v.sustained([1, 1, 1], lambda x: x >= 1, 3), (
        "v.sustained([1, 1, 1], lambda x: x >= 1, 3)"
    )
    assert not v.sustained([1, 0, 1], lambda x: x >= 1, 2), (
        "not v.sustained([1, 0, 1], lambda x: x >= 1, 2)"
    )
    assert v.node_isolated({"unschedulable": True}), (
        "v.node_isolated({'unschedulable': True})"
    )
    assert v.node_isolated({"taints": [{"key": "gpu-fault.io/quarantined"}]}), (
        "v.node_isolated({'taints': [{'key': 'gpu-fault.io/quarantined'}]})"
    )
    assert not v.node_isolated(
        {"unschedulable": False, "taints": [], "ownership_annotations": {}}
    ), "not v.node_isolated( {'unschedulable': False, 'taints': [], 'ownership"
    assert v.verdict({"a": []}) == "PASS"
    assert v.verdict({"a": ["x"]}) == "FAIL"
    assert v.verdict({"a": []}, inconclusive=["y"]) == "INCONCLUSIVE"


def test_samples_breaching_steps_and_subsequence() -> None:
    doc = {
        "batches": [
            {
                "observed_at": "t",
                "batch_id": "b",
                "samples": [{"name": "m", "device": "d", "value": 5.0}],
            }
        ]
    }
    assert v.samples(doc, "m", device="d")[0]["value"] == 5.0
    assert v.breaching({"value": 99.0}, 98.0), "v.breaching({'value': 99.0}, 98.0)"
    assert v.breaching({"value": 1.0}, 2.0, "lte"), (
        "v.breaching({'value': 1.0}, 2.0, 'lte')"
    )
    assert not v.breaching({"value": float("inf")}, 1.0), (
        "not v.breaching({'value': float('inf')}, 1.0)"
    )
    wf = _wf("i", ["VALIDATE_HOST"], probe="m")
    assert (
        v.step_parameters(wf, "VALIDATE_HOST")[0]["host_diagnostic_probe"]["metric"]
        == "m"
    )
    assert v.subsequence(["a", "c"], ["a", "b", "c"]), (
        "v.subsequence(['a', 'c'], ['a', 'b', 'c'])"
    )
    assert not v.subsequence(["c", "a"], ["a", "b", "c"]), (
        "not v.subsequence(['c', 'a'], ['a', 'b', 'c'])"
    )
    assert (
        v.final_execution(
            _wf("i", ["X"], execs=[{"operation": "X", "status": "WAITING"}]), "X"
        )["status"]
        == "WAITING"
    )


# --------------------------------------------------------------------------- #
# store reads
# --------------------------------------------------------------------------- #
class Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def cpu_python(self, script, *arguments, **_):
        self.calls.append((script, arguments))
        return {"ok": True}


def test_activity_probe_lists_every_state_and_workflow_pulled_incidents() -> None:
    """The real probe over a store: it hides no state, and names its read.

    Live 2026-10-25 the variant missed a RECOVERED LOW_MEMORY_AVAILABLE
    incident; the probe did list it -- the runner's rule id was wrong. The
    probe also adds an older incident that a later workflow names, which only
    the matcher's ``minted_in_read`` keeps out.
    """

    since = datetime(2026, 10, 25, 3, 19, 46, tzinfo=timezone.utc)
    created = since + timedelta(seconds=1084)
    low = "memory_available_percent-node-low_memory_available"
    records = build_store()

    def save(batch: str, node: str, at: datetime) -> str:
        return save_diagnostic_incident(
            records,
            f"{batch}-{low}",
            "memory_available_percent",
            cluster_id="c",
            node=node,
            created=at,
            reason="available host memory remained below the warning threshold",
        )

    fresh = save("b2", "node-a", created)
    older = save("b1", "node-a", since - timedelta(hours=1))
    other_node = save("b3", "node-b", created)
    records.save_workflow(
        workflow_request(
            "wf-restore",
            older,
            WorkflowStatus.SUCCEEDED,
            created_at=created,
            updated_at=created,
        )
    )
    doc = activity_read(
        records, ["c", "node-a", since.isoformat(), "", "", "HOST_TELEMETRY"]
    )
    listed = {item["incident_id"]: item["state"] for item in doc["incidents"]}
    assert listed == {fresh: "RECOVERED", older: "RECOVERED"}, listed
    assert other_node not in listed, "the probe reads one node"
    assert {key: doc[key] for key in READ} == {
        "since": since.isoformat(),
        "cluster_id": "c",
        "node_id": "node-a",
    }
    assert [item["incident_id"] for item in _low_memory(doc)] == [fresh]


def test_store_read_helpers_pass_through_and_validate() -> None:
    rec = Recorder()
    store.preflight_state(rec, "c", "n")
    assert rec.calls[-1][0] is store.PREFLIGHT_PROBE
    store.node_activity(
        rec,
        "c",
        "n",
        datetime(2026, 10, 24, tzinfo=timezone.utc),
        metrics=["m"],
        signal_keys=["s"],
        kinds=["HOST_TELEMETRY"],
    )
    assert rec.calls[-1][0] is store.ACTIVITY_PROBE
    store.attempt_records(rec, "c", "job", ["a1"])
    assert rec.calls[-1][0] is store.ATTEMPT_PROBE
    assert store.signal_key("c", "n", "m", "dev", "RULE") == "c/n/m/dev/RULE"
    assert store.signal_key("c", "n", "m") == "c/n/m/node"
    with pytest.raises(ValueError):
        store.node_activity(rec, "c", "n", datetime.now(timezone.utc), metrics=["a,b"])
    with pytest.raises(ValueError):
        store.attempt_records(rec, "c", "job", [])


# --------------------------------------------------------------------------- #
# a validation still running is not a verdict (live 2026-10-25, 003.gpu_idle)
# --------------------------------------------------------------------------- #
WAITING_VALIDATION = {
    "operation": "VALIDATE_GPU",
    "status": "WAITING",
    "details": {
        "pending_nodes": ["node-a"],
        "node_pending": {
            "node-a": {
                "missing_diagnostic_metric": ["host_gpu_utilization_percent"],
                "device": "GPU-0",
            }
        },
    },
}


def _idle_doc(execs: list[dict], status: str = "RUNNING") -> dict:
    return {
        "incidents": [
            _incident("host_gpu_utilization_percent", rule="LOW_GPU_UTILIZATION")
        ],
        "workflows": [
            _wf(
                "inc-host_gpu_utilization_percent",
                ["FREEZE_EVIDENCE", "VALIDATE_GPU"],
                status=status,
                execs=execs,
                probe="host_gpu_utilization_percent",
            )
        ],
    }


@pytest.mark.parametrize("must_fail", [False, True])
def test_chain_errors_names_a_validation_still_waiting(must_fail: bool) -> None:
    # A WAITING step used to read "ended WAITING: " and, with
    # validation_must_fail, as no error at all.
    errors = v.chain_errors(
        _idle_doc([WAITING_VALIDATION]),
        v.ChainExpectation(
            metric="host_gpu_utilization_percent",
            rule_id="LOW_GPU_UTILIZATION",
            validation="VALIDATE_GPU",
            validation_must_fail=must_fail,
            probe_metric="host_gpu_utilization_percent",
            incident_states=v.DIAGNOSTIC_ONLY_STATES,
        ),
    )
    assert errors == [
        "VALIDATE_GPU did not finish: still WAITING "
        "(node-a: device, missing_diagnostic_metric)"
    ], errors


def test_chain_judgeable_waits_for_a_terminal_validation() -> None:
    judge = lambda doc: v.chain_judgeable(  # noqa: E731
        doc,
        "host_gpu_utilization_percent",
        rule_id="LOW_GPU_UTILIZATION",
        validation="VALIDATE_GPU",
    )
    assert not judge({"incidents": []}), "no incident yet"
    assert not judge({**_idle_doc([]), "workflows": []}), "no workflow yet"
    assert not judge(_idle_doc([WAITING_VALIDATION])), "the step is still running"
    for status in ("FAILED", "SUCCEEDED"):
        terminal = _idle_doc([{"operation": "VALIDATE_GPU", "status": status}])
        assert judge(terminal), f"a {status} validation is a verdict"
    assert judge(_idle_doc([], status="BLOCKED")), "a terminal workflow is a verdict"
    assert v.pending_summary({"status": "WAITING", "error": "lease lost"}) == (
        "lease lost"
    )
