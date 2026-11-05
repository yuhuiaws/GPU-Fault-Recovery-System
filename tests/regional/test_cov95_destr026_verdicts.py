"""GF-REGIONAL-DESTR-026 verdicts: every refusal the pass-shape tests leave unnamed.

The neighbouring ``test_destr026_same_node_two_gpu_reset`` proves the pass shape
and the negatives the brief names. Here each remaining refusal of
``destr026_verdicts`` is driven from that pass shape with exactly one defect,
and the verdict must name that defect and nothing else: the preflight's node,
agent, profile, queue and workload gates, an injected fault the store recorded
wrongly, a terminal record that is not one clean window, a reset receipt or
ledger that miscounts, Fabric Manager restarted by the restore, the detached
sampler's evidence, and the job's second node or another GPU node changing.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from scripts.e2e.regional import destr026_verdicts as verdicts
from tests.orchestration.test_same_node_job_reset_merge import (
    GPU_A,
    GPU_B,
    NODE,
    PARTNER,
)
from tests.regional.test_destr026_same_node_two_gpu_reset import (
    INCIDENT,
    KEPT,
    OTHERS,
    REQUEST,
    STOPPED,
    TARGETS,
    at,
    event_state,
    gpu_nodes,
    hosts,
    inventory,
    partner_snapshot,
    pass_incident,
    pass_workflow,
    preflight,
    reset_receipt,
    row_of,
    workflow_errors,
)

LABELS = {"a": GPU_A, "b": GPU_B}
QUIESCE_INDEX = 3
RESTORE_INDEX = 6
RESTART_INDEX = 9


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #
def snapshot(**changes: Any) -> dict[str, Any]:
    """A Ready, schedulable, untainted eight-GPU node of the current boot."""

    return {
        "ready": "True",
        "unschedulable": False,
        "taints": [],
        "ownership_annotations": {},
        "gpu_allocatable": "8",
        "boot_id": "boot-a",
        **changes,
    }


def profile(gpu_reset: dict[str, Any] | None, **extra: Any) -> dict[str, Any]:
    """The runtime profile the preflight admits, with ``gpuReset`` replaced."""

    owner = {
        "mode": "OWN",
        "owner": "gpu-fault-kubernetes-adapter",
        "adapter": "regional-cluster-executor",
    }
    capabilities = [
        {"capability": "workloadStop", **owner},
        {"capability": "workloadRestart", **owner},
    ]
    if gpu_reset is not None:
        capabilities.insert(0, {"capability": "gpuReset", **gpu_reset})
    return {"capabilities": capabilities, **extra}


AGENT_OWNED_RESET = {"mode": "OWN", "owner": "gpu-fault-node-agent"}


@pytest.mark.parametrize(
    "defect",
    [
        {"ready": "False"},
        {"unschedulable": True},
        {"taints": [{"key": "node.kubernetes.io/unreachable"}]},
        {"ownership_annotations": {"gpu-fault.io/workflow": "older"}},
    ],
    ids=["not-ready", "cordoned", "tainted", "owned"],
)
def test_the_preflight_refuses_a_job_node_that_is_not_a_clean_schedulable_node(
    defect: dict[str, Any],
) -> None:
    errors = preflight(node_snapshots={NODE: snapshot(), PARTNER: snapshot(**defect)})

    assert errors == [f"{PARTNER} is not Ready, schedulable and untainted"], (
        "the job's second node is gated like the fault node, with one refusal",
        errors,
    )


@pytest.mark.parametrize("allocatable", ["4", None])
def test_the_preflight_refuses_a_node_that_does_not_allocate_eight_gpus(
    allocatable: str | None,
) -> None:
    errors = preflight(
        node_snapshots={
            NODE: snapshot(),
            PARTNER: snapshot(gpu_allocatable=allocatable),
        }
    )

    assert errors == [f"{PARTNER} does not allocate 8 GPUs"], (
        "a node short of its eight GPUs cannot host the 16-GPU job",
        errors,
    )


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"tests": {"passed": False}}, ["focused regression tests failed"]),
        (
            {"business_workloads": {PARTNER: ["kube-system/coredns-1"]}},
            [
                f"{PARTNER} carries a critical system workload: ['kube-system/coredns-1']"
            ],
        ),
        (
            {
                "agent": {
                    "lifecycle_state": "DRAINING",
                    "allowed_operations": list(verdicts.AGENT_OPERATIONS),
                }
            },
            [f"{NODE} Node Agent is not ACTIVE"],
        ),
        ({"profile": profile(None)}, ["runtime profile has no gpuReset capability"]),
        (
            {"profile": profile({"mode": "OBSERVE", "owner": "gpu-fault-node-agent"})},
            ["gpuReset is not OWN by the Node Agent"],
        ),
        (
            {"profile": profile({"mode": "OWN", "owner": "nvidia-gpu-operator"})},
            ["gpuReset is not OWN by the Node Agent"],
        ),
        (
            {"profile": profile(AGENT_OWNED_RESET, warnings=["capability drift"])},
            ["runtime profile has warnings"],
        ),
        ({"queue": {"depth": 3}}, ["processor queue is not empty"]),
        (
            {"queue": {"fault_backlog_depth": 2, "depth": 0}},
            ["processor queue is not empty"],
        ),
        ({"queue": {"fault_backlog_depth": 0, "depth": 40}}, []),
        (
            {"gpu_workloads": [{"name": "train-x", "node": "node-c"}]},
            [
                "GPU cluster already has a GPU workload: "
                "[{'name': 'train-x', 'node': 'node-c'}]"
            ],
        ),
    ],
    ids=[
        "tests-failed",
        "system-workload",
        "agent-not-active",
        "no-gpu-reset",
        "reset-observed",
        "reset-foreign-owner",
        "profile-warnings",
        "queue-depth",
        "fault-backlog",
        "telemetry-only-backlog",
        "gpu-workload",
    ],
)
def test_the_preflight_names_each_remaining_gate_alone(
    overrides: dict[str, Any], expected: list[str]
) -> None:
    errors = preflight(**overrides)

    assert errors == expected, (
        "one defect yields exactly its own refusal; routine telemetry depth "
        "never blocks when the fault tier is reported",
        overrides,
        errors,
    )


def test_the_preflight_profile_builder_is_the_admitted_profile() -> None:
    assert preflight(profile=profile(AGENT_OWNED_RESET)) == [], (
        "the local profile builder must equal the admitted profile, or the "
        "single-defect checks above would measure two defects"
    )


# --------------------------------------------------------------------------- #
# Injection
# --------------------------------------------------------------------------- #
def injection_errors(second: dict[str, Any]) -> list[str]:
    return verdicts.injection_errors(
        {"a": event_state("a", GPU_A), "b": second}, node=NODE, targets=LABELS
    )


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("xid", 79, "GPU b event is not XID 46: 79"),
        (
            "evidence_ref",
            f"file:///var/log/kern.log#{NODE}",
            "GPU b XID evidence is not backed by kmsg://",
        ),
        ("evidence_ref", None, "GPU b XID evidence is not backed by kmsg://"),
        ("node_id", PARTNER, f"GPU b event names node {PARTNER}"),
    ],
    ids=["wrong-xid", "not-kmsg", "no-evidence", "wrong-node"],
)
def test_an_injected_event_recorded_wrongly_is_named(
    field: str, value: Any, expected: str
) -> None:
    second = event_state("b", GPU_B)
    second["event"][field] = value

    assert injection_errors(second) == [expected], (
        "the event of the second fault must be an XID 46 kmsg event of the node",
        field,
    )


def test_an_event_without_a_workflow_is_named_and_splits_the_workflows() -> None:
    second = event_state("b", GPU_B)
    second["workflow"] = {}
    second["incident"] = {"incident_id": INCIDENT}

    errors = injection_errors(second)

    assert len(errors) == 2 and errors[0] == "GPU b event has no workflow", errors
    assert errors[1].startswith(
        "the two faults landed in 2 workflows: ['None', 'workflow-destr026']; "
        "decisions: "
    ), errors


def test_the_incident_workflow_link_alone_carries_the_fault() -> None:
    second = event_state("b", GPU_B)
    second["workflow"] = {}

    assert injection_errors(second) == [], (
        "an incident's workflow_request_id is enough when the workflow read is empty"
    )


def test_two_faults_that_share_one_event_are_not_two_faults() -> None:
    second = event_state("b", GPU_B)
    second["event"]["event_id"] = "kernel-xid-46-a"

    assert injection_errors(second) == [
        "the two faults did not produce two distinct events"
    ], "two writes must yield two distinct events"


def test_a_fault_the_store_never_recorded_fails_every_account_of_it() -> None:
    errors = verdicts.injection_errors(
        {"a": event_state("a", GPU_A)}, node=NODE, targets=LABELS
    )

    expected_prefixes = [
        "GPU b event is not XID 46: None",
        "GPU b XID evidence is not backed by kmsg://",
        "GPU b event names node None",
        f"GPU b event resolved to None, not {GPU_B}",
        "GPU b policy did not resolve to RESET_GPU: ",
        "GPU b event has no workflow",
        "the two faults opened 2 incidents: ['None', 'inc-destr026']",
        "the two faults landed in 2 workflows: ['None', 'workflow-destr026']",
    ]
    assert len(errors) == len(expected_prefixes), errors
    for error, prefix in zip(errors, expected_prefixes, strict=True):
        assert error.startswith(prefix), (prefix, error)


# --------------------------------------------------------------------------- #
# Control plane: status, executions, restart
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("status", "FAILED", "workflow status is not SUCCEEDED: FAILED"),
        (
            "superseded_step_indexes",
            [5],
            "a single-window reset has superseded steps: [5]",
        ),
        (
            "branch_escalation_counts",
            {"reset-branch": 1},
            "the reset escalated or exhausted a branch: {'reset-branch': 1} []",
        ),
        (
            "exhausted_branch_ids",
            ["reset-branch"],
            "the reset escalated or exhausted a branch: {} ['reset-branch']",
        ),
        (
            "workload_withdrawn_at",
            at(900),
            "the workflow was withdrawn; the job was stopped externally",
        ),
        (
            "terminal_failure_reason",
            "agent lost",
            "workflow carries a failure reason: agent lost",
        ),
    ],
    ids=[
        "not-succeeded",
        "superseded",
        "escalated",
        "exhausted",
        "withdrawn",
        "failure-reason",
    ],
)
def test_a_terminal_record_that_is_not_one_clean_window_is_named(
    field: str, value: Any, expected: str
) -> None:
    workflow = pass_workflow()
    workflow[field] = value

    assert workflow_errors(workflow) == [expected], (
        "one defect of the terminal record yields exactly its refusal",
        field,
    )


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("state", "OPEN", "incident state is not RECOVERED: OPEN"),
        (
            "gpu_uuids",
            [GPU_A],
            "incident does not name exactly the two faulted GPUs: ['GPU-a0']",
        ),
        (
            "node_ids",
            [NODE, PARTNER],
            "incident does not name the fault node alone: ['node-a', 'node-b']",
        ),
    ],
    ids=["not-recovered", "one-gpu", "two-nodes"],
)
def test_an_incident_that_is_not_the_recovered_two_gpu_fault_is_named(
    field: str, value: Any, expected: str
) -> None:
    incident = {**pass_incident(), field: value}

    assert workflow_errors(pass_workflow(), incident=incident) == [expected], (
        "the incident must be RECOVERED and name both GPUs of the fault node only",
        field,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("started_at", "not-a-time"),
        ("started_at", "2026-11-05T00:00:00"),
        ("started_at", None),
        ("updated_at", at(60 * QUIESCE_INDEX - 10)),
    ],
    ids=["unparseable", "naive", "absent", "ends-before-start"],
)
def test_an_execution_with_unusable_timestamps_is_named(field: str, value: Any) -> None:
    workflow = pass_workflow()
    workflow["step_executions"][QUIESCE_INDEX][field] = value

    assert workflow_errors(workflow) == [
        "QUIESCE_GPU_SERVICES execution has incomplete or unordered timestamps"
    ], ("only an aware, ordered start/end pair proves the execution", field, value)


def test_the_restart_must_carry_the_job_gpu_count_from_either_source() -> None:
    assert workflow_errors(pass_workflow(), expected_gpu_count=8) == [
        "RESTART_WORKLOAD source_gpu_count is not 8: 16",
        "RESTART_WORKLOAD target_gpu_count is not 8: 16",
    ], "a restart of another job size is not this job's restart"
    workflow = pass_workflow()
    workflow["step_executions"][RESTART_INDEX]["details"] = {
        "notification_context": {"source_gpu_count": 16, "target_gpu_count": 8}
    }
    assert workflow_errors(workflow) == [
        "RESTART_WORKLOAD target_gpu_count is not 16: 8"
    ], "the notification context answers when the details carry no count"


# --------------------------------------------------------------------------- #
# Node Agent receipt and ledger
# --------------------------------------------------------------------------- #
def test_a_receipt_that_omits_a_gpu_from_its_result_is_named() -> None:
    receipt = reset_receipt()
    receipt["reset_gpu_uuids"] = [GPU_A]

    assert verdicts.reset_receipt_errors(receipt, targets=TARGETS, source="cp") == [
        "cp RESET_GPU did not target exactly the two GPUs: ['GPU-a0']",
        "cp did not reset the GPUs one after another in command order: "
        "['GPU-a0', 'GPU-a1']",
    ], "the result list must name both GPUs, in the order the calls reset them"


def test_a_receipt_ignores_calls_that_are_neither_resets_nor_releases() -> None:
    receipt = reset_receipt()
    receipt["physical_calls"][1:1] = [
        {"command": "nvidia-smi -q -d PERSISTENCE", "outcome": "ok"},
        None,
        {"command": "modprobe nvidia_drm", "outcome": "loaded"},
    ]

    assert verdicts.reset_receipt_errors(receipt, targets=TARGETS, source="cp") == [], (
        "queries, reloads and empty slots are neither resets nor display releases"
    )


def test_a_reset_outcome_other_than_reset_or_busy_is_named() -> None:
    receipt = reset_receipt()
    receipt["physical_calls"][0]["outcome"] = "timeout"

    assert verdicts.reset_receipt_errors(receipt, targets=TARGETS, source="ledger") == [
        "ledger carries reset outcomes other than reset/busy: ['timeout']"
    ], "only the documented busy retry may precede a GPU's reset"


def call_ledger(after: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    before, _ = hosts()
    return verdicts.ledger_errors(
        before,
        after,
        targets=TARGETS,
        incident_id=INCIDENT,
        workflow_request_id=REQUEST,
    )


def test_a_ledger_without_a_successful_reset_row_stops_before_grading_it() -> None:
    _, after = hosts()
    after["ledger"] = [
        row
        for row in after["ledger"]
        if not (
            row["operation"] == "RESET_GPU" and row["workflow_request_id"] == REQUEST
        )
    ]

    errors, rows = call_ledger(after)

    assert errors == [
        "Node Agent ledger does not add exactly one successful RESET_GPU: []"
    ], "the older workflow's reset row is baseline, not this window's"
    assert set(rows) == {"QUIESCE_GPU_SERVICES", "RESTORE_GPU_SERVICES"}, (
        "the rows that are present are still returned for the caller",
        rows,
    )


def test_a_failed_reset_row_is_not_the_window_reset() -> None:
    _, after = hosts()
    row_of(after, "RESET_GPU")["state"] = "FAILED"

    errors, rows = call_ledger(after)

    assert errors == [
        "Node Agent ledger does not add exactly one successful RESET_GPU: ['FAILED']"
    ], errors
    assert "RESET_GPU" not in rows, "a failed row is not graded as the reset"


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("attempt", 2, "RESET_GPU ledger attempt is not one: 2"),
        (
            "incident_id",
            "inc-other",
            "RESET_GPU ledger incident_id does not match this workflow",
        ),
    ],
    ids=["retried", "other-incident"],
)
def test_the_ledger_reset_row_must_be_the_first_attempt_of_this_incident(
    field: str, value: Any, expected: str
) -> None:
    _, after = hosts()
    row_of(after, "RESET_GPU")[field] = value

    errors, rows = call_ledger(after)

    assert errors == [expected], (field, errors)
    assert rows["RESET_GPU"][field] == value, "the graded row is returned as read"


# --------------------------------------------------------------------------- #
# Fabric Manager
# --------------------------------------------------------------------------- #
def fabric_errors(
    workflow: dict[str, Any], ledger_rows: dict[str, Any] | None = None
) -> list[str]:
    before, after = hosts()
    if ledger_rows is None:
        _, ledger_rows = call_ledger(after)
    return verdicts.fabric_manager_errors(
        before, after, workflow=workflow, ledger_rows=ledger_rows, node=NODE
    )


def test_a_restore_that_starts_fabric_manager_again_is_named() -> None:
    workflow = pass_workflow()
    details = workflow["step_executions"][RESTORE_INDEX]["details"]
    details["restored_services"][NODE] = [*STOPPED, "nvidia-fabricmanager"]

    assert fabric_errors(workflow) == [
        "RESTORE_GPU_SERVICES started Fabric Manager again"
    ], "a kept unit is never restored"


@pytest.mark.parametrize(
    "ledger_rows",
    [
        {
            "QUIESCE_GPU_SERVICES": {
                "evidence": {
                    "active_services": STOPPED,
                    "kept_services": ["nvidia-persistenced"],
                }
            }
        },
        {},
    ],
    ids=["kept-other", "no-row"],
)
def test_the_ledger_quiesce_row_must_name_fabric_manager_kept(
    ledger_rows: dict[str, Any],
) -> None:
    assert fabric_errors(pass_workflow(), ledger_rows) == [
        "the Node Agent ledger's quiesce row does not name Fabric Manager kept"
    ], "the node's own record must say Fabric Manager was kept"


def test_kept_services_entries_without_a_service_name_are_skipped() -> None:
    workflow = pass_workflow()
    details = workflow["step_executions"][QUIESCE_INDEX]["details"]
    details["kept_services"][NODE] = [{"reason": "unnamed"}, 7, None, *KEPT]

    assert fabric_errors(workflow) == [], (
        "an unnamed entry neither hides nor fakes the kept Fabric Manager"
    )
    details["kept_services"][NODE] = [{"reason": "unnamed"}, 7, None]
    assert fabric_errors(workflow) == [
        "QUIESCE_GPU_SERVICES did not keep Fabric Manager running"
    ], "unnamed entries alone do not keep Fabric Manager"


# --------------------------------------------------------------------------- #
# Detached sampler
# --------------------------------------------------------------------------- #
def sampler() -> dict[str, Any]:
    _, after = hosts()
    return copy.deepcopy(after["sampler"])


@pytest.mark.parametrize(
    "baseline",
    [
        inventory()[:-1],
        [
            {**row, "uuid": "GPU-a9"} if row["uuid"] == GPU_B else row
            for row in inventory()
        ],
    ],
    ids=["seven-gpus", "target-absent"],
)
def test_a_baseline_inventory_without_both_targets_stops_the_sampler_verdict(
    baseline: list[dict[str, str]],
) -> None:
    assert verdicts.sampler_errors(baseline, sampler(), targets=TARGETS) == [
        "the baseline GPU inventory does not contain both target GPUs"
    ], "the sampler is judged against an eight-GPU baseline holding both targets"


@pytest.mark.parametrize(
    "reading", [{"sample_count": 1}, None], ids=["one-sample", "no-sampler"]
)
def test_a_sampler_with_fewer_than_two_samples_proves_nothing(
    reading: dict[str, Any] | None,
) -> None:
    assert verdicts.sampler_errors(inventory(), reading, targets=TARGETS) == [
        "detached host GPU sampler has insufficient samples"
    ], "one sample cannot show a GPU leave and return"


def test_a_non_target_gpu_leaving_is_named() -> None:
    reading = sampler()
    every = set(reading["last"]["gpu_uuids"])
    reading["observed_gpu_uuid_sets"].append(sorted(every - {OTHERS[0]}))

    assert verdicts.sampler_errors(inventory(), reading, targets=TARGETS) == [
        "the sampler saw a GPU other than the two targets leave"
    ], "only the two faulted GPUs may drop off the bus"


@pytest.mark.parametrize(
    "last",
    [{"gpu_count": 7}, {"gpu_uuids": sorted(set(TARGETS) | set(OTHERS[1:]))}],
    ids=["count", "set"],
)
def test_a_final_sample_that_is_not_the_baseline_is_named(last: dict[str, Any]) -> None:
    reading = sampler()
    reading["last"] = {**reading["last"], **last}

    assert verdicts.sampler_errors(inventory(), reading, targets=TARGETS) == [
        "the sampler's final GPU set differs from the baseline"
    ], "the node must end with every GPU back"


# --------------------------------------------------------------------------- #
# The job's second node and the other GPU nodes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("change", ["readdressed", "lost"])
def test_a_partner_whose_gpu_inventory_changed_is_named(change: str) -> None:
    after = partner_snapshot()
    if change == "readdressed":
        after["gpu_inventory"][3]["pci_bdf"] = "0000:99:00"
    else:
        after["gpu_inventory"].pop()

    assert verdicts.partner_host_errors(partner_snapshot(), after, node=PARTNER) == [
        f"{PARTNER} GPU inventory changed"
    ], "node B's GPUs are never reset, so its inventory never changes"


@pytest.mark.parametrize("change", ["failed", "absent"])
def test_a_partner_service_that_did_not_stay_active_is_named(change: str) -> None:
    after = partner_snapshot()
    if change == "failed":
        after["services"]["kubelet.service"]["ActiveState"] = "failed"
    else:
        del after["services"]["kubelet.service"]

    assert verdicts.partner_host_errors(partner_snapshot(), after, node=PARTNER) == [
        f"{PARTNER} service did not stay active: kubelet.service"
    ], "every unit active on node B before stays active"


def test_a_gpu_node_that_disappeared_is_named() -> None:
    after = [item for item in gpu_nodes() if item["name"] != "node-c"]

    assert verdicts.other_node_errors(gpu_nodes(), after, fault_node=NODE) == [
        "node-c disappeared from the GPU node list"
    ], "a GPU node that left the list cannot be shown untouched"
