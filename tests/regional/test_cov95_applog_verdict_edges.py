"""GF-GENERAL-LIVE-017 CRI attribution readers: the edges each stage reader names.

``general_live_applog_verdicts`` turns the scripted ``APPLOG_PROBE`` documents
into stage errors. ``test_general_live_applogs`` drives the variants end to end;
these cases feed the pure readers one crafted record at a time so that every
reason a reader can give is pinned to the input that produces it: a record from
another source, Pod, container or clock, an incident or workflow that is absent,
a bundle step whose attribution parameters are wrong, a stitched record read
from the live file instead of the rotated one.
"""

from __future__ import annotations

from typing import Any

import pytest

from scripts.e2e.regional import general_live_applog_verdicts as verdicts

NODE = "ip-10-0-43-5.us-west-2.compute.internal"
PEER = "ip-10-0-77-9.us-west-2.compute.internal"
CLUSTER = "gpu-eks-cluster"
POD_UID = "0f3b7c2e-4d5a-4b6c-8d9e-0a1b2c3d4e5f"
PID = 4242
NEW_PID = 5151
STAMP = "2026-10-24T00:00:30.000000000Z"
PATH = f"/var/log/pods/gpu-fault-system_pod-0_{POD_UID}/pytorch/gfrs-live.log"
ROTATED = PATH + ".1"
MESSAGE = "torch.OutOfMemoryError: CUDA out of memory."


def entry(**changes: Any) -> dict[str, Any]:
    fields = {
        "path": PATH,
        "pod_uid": POD_UID,
        "container_name": "pytorch",
        "timestamp_source": "cri",
        "stream": "stdout",
    }
    fields.update(changes.pop("fields", {}))
    value: dict[str, Any] = {
        "entry_id": "e" * 64,
        "source": "training-log",
        "observed_at": "2026-10-24T00:00:30+00:00",
        "message": MESSAGE,
        "fields": fields,
    }
    value.update(changes)
    return value


def parameters(**changes: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "attribution": "OBSERVED_PROCESS",
        "process_state": "LIVE",
        "log_source": "training-log",
        "capture_process_state": True,
        "target_pids_by_node": {NODE: [PID]},
    }
    value.update(changes)
    return value


def workflow(*steps: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "request_id": "wf-log",
        "official_steps": [
            {"operation": operation, "parameters": params}
            for operation, params in steps
        ],
        "step_executions": [],
    }


def bundle_workflow(params: dict[str, Any]) -> dict[str, Any]:
    return workflow(("FREEZE_EVIDENCE", {}), ("COLLECT_DIAGNOSTIC_BUNDLE", params))


INCIDENT = {
    "incident_id": "inc-1",
    "job_id": "JOB",
    "attempt_id": "JOB-a001",
    "effective_action": "RUN_DIAGNOSTICS",
}


# --------------------------------------------------------------------------- #
# record readers
# --------------------------------------------------------------------------- #
def test_incident_of_skips_incidents_minted_for_other_entries() -> None:
    delivered = entry()
    wanted = verdicts.log_event_id(CLUSTER, NODE, delivered["entry_id"])
    other = verdicts.log_event_id(CLUSTER, NODE, "f" * 64)
    records = {
        "cluster_id": CLUSTER,
        "node_id": NODE,
        "incidents": [
            {"incident_id": "inc-other", "event_id": other},
            {"incident_id": "inc-mine", "event_id": wanted},
        ],
    }

    found = verdicts.incident_of(records, delivered)

    assert found == {"incident_id": "inc-mine", "event_id": wanted}
    assert (
        verdicts.incident_of(
            {**records, "incidents": records["incidents"][:1]}, delivered
        )
        is None
    ), "an incident of another entry is never this entry's"


def test_attempt_of_returns_none_when_the_watcher_recorded_another_attempt() -> None:
    records = {"attempts": [{"attempt_id": "JOB-a000"}, {"attempt_id": "JOB-a002"}]}

    assert verdicts.attempt_of(records, "JOB-a002") == {"attempt_id": "JOB-a002"}
    assert verdicts.attempt_of(records, "JOB-a001") is None, (
        "an attempt the watcher never recorded has no observation"
    )
    assert verdicts.attempt_of({}, "JOB-a001") is None, "no attempts recorded at all"


def test_workflow_readers_have_nothing_to_read_without_a_workflow() -> None:
    assert verdicts.diagnostic_parameters(None) == []
    assert verdicts.diagnostic_parameters({}) == []
    assert verdicts.capture_steps(None) == []
    assert verdicts.capture_steps({}) == []
    sampled = workflow(
        ("FREEZE_EVIDENCE", {}),
        ("COLLECT_DIAGNOSTIC_BUNDLE", {"target_pids_by_node": {NODE: [PID]}}),
        ("COLLECT_HUNG_TRIAGE", {"capture_process_state": True}),
    )
    assert verdicts.capture_steps(sampled) == [
        "COLLECT_DIAGNOSTIC_BUNDLE",
        "COLLECT_HUNG_TRIAGE",
    ]
    assert verdicts.diagnostic_parameters(sampled) == [
        {"target_pids_by_node": {NODE: [PID]}},
        {"capture_process_state": True},
    ]


# --------------------------------------------------------------------------- #
# entry_identity_errors
# --------------------------------------------------------------------------- #
def test_entry_identity_accepts_the_genuine_pod_line_at_its_cri_stamp() -> None:
    assert (
        verdicts.entry_identity_errors(
            entry(), pod_uid=POD_UID, container="pytorch", stamp=STAMP, message=MESSAGE
        )
        == []
    )


@pytest.mark.parametrize(
    "changes, expected",
    [
        ({"source": "kernel"}, ["entry source 'kernel' is not training-log"]),
        (
            {"fields": {"container_name": "sidecar"}},
            ["entry container 'sidecar' != pytorch"],
        ),
        (
            {"fields": {"timestamp_source": "collector"}},
            ["timestamp source 'collector' is not cri"],
        ),
        (
            {"observed_at": None},
            ["the entry or the fixture stamp has no readable time"],
        ),
        ({"message": "another line"}, [f"entry message 'another line' != {MESSAGE!r}"]),
        (
            {"observed_at": "2026-10-24T00:00:33+00:00"},
            [f"event time 2026-10-24T00:00:33+00:00 is not the CRI stamp {STAMP}"],
        ),
    ],
    ids=["source", "container", "timestamp-source", "no-time", "message", "late"],
)
def test_entry_identity_names_each_mismatch(
    changes: dict[str, Any], expected: list[str]
) -> None:
    errors = verdicts.entry_identity_errors(
        entry(**changes),
        pod_uid=POD_UID,
        container="pytorch",
        stamp=STAMP,
        message=MESSAGE,
    )

    assert errors == expected


def test_entry_identity_without_a_fixture_stamp_cannot_time_the_line() -> None:
    errors = verdicts.entry_identity_errors(
        entry(), pod_uid=POD_UID, container="pytorch", stamp=None
    )

    assert errors == ["the entry or the fixture stamp has no readable time"]


# --------------------------------------------------------------------------- #
# attribution_errors
# --------------------------------------------------------------------------- #
def attribution(
    incident: dict[str, Any] | None, flow: dict[str, Any] | None
) -> list[str]:
    return verdicts.attribution_errors(
        incident,
        flow,
        job_id="JOB",
        attempt_id="JOB-a001",
        process_state="LIVE",
        node=NODE,
        pid=PID,
    )


def test_attribution_without_an_incident_is_one_error() -> None:
    assert attribution(None, bundle_workflow(parameters())) == [
        "the delivered line produced no incident"
    ]


def test_attribution_accepts_the_live_process_bundle() -> None:
    assert attribution(INCIDENT, bundle_workflow(parameters())) == []


def test_attribution_names_a_bundle_read_from_another_log_without_capture() -> None:
    flow = bundle_workflow(
        parameters(log_source="journal", capture_process_state=False)
    )

    assert attribution(INCIDENT, flow) == [
        "log_source 'journal' != training-log",
        "capture_process_state is not set for a live process",
    ]


def test_attribution_requires_the_capture_flag_to_be_exactly_true() -> None:
    flow = bundle_workflow(parameters(capture_process_state="yes"))

    assert attribution(INCIDENT, flow) == [
        "capture_process_state is not set for a live process"
    ]


# --------------------------------------------------------------------------- #
# diagnostic_chain_errors
# --------------------------------------------------------------------------- #
def test_diagnostic_chain_without_an_incident_is_one_error() -> None:
    assert verdicts.diagnostic_chain_errors(
        None, bundle_workflow(parameters()), expect_bundle=True, node=NODE
    ) == ["no incident"]


def test_diagnostic_chain_names_the_action_and_the_missing_workflow() -> None:
    incident = {**INCIDENT, "effective_action": "NO_ACTION"}

    errors = verdicts.diagnostic_chain_errors(
        incident, None, expect_bundle=True, node=NODE
    )

    assert errors == [
        "effective action 'NO_ACTION' != RUN_DIAGNOSTICS",
        "the incident has no workflow",
    ]


def test_diagnostic_chain_without_a_workflow_keeps_the_action_verdict_alone() -> None:
    errors = verdicts.diagnostic_chain_errors(INCIDENT, None, expect_bundle=False)

    assert errors == ["the incident has no workflow"]


# --------------------------------------------------------------------------- #
# new_attempt_errors
# --------------------------------------------------------------------------- #
def new_attempt(
    incident: dict[str, Any] | None,
    flow: dict[str, Any] | None,
    *,
    new_pid: int | None = NEW_PID,
) -> list[str]:
    return verdicts.new_attempt_errors(
        incident, flow, new_attempt_id="JOB-a002", new_pid=new_pid, node=NODE
    )


def test_new_attempt_without_an_incident_is_one_error() -> None:
    assert new_attempt(None, None) == ["the delivered line produced no incident"]


def test_new_attempt_accepts_an_old_pod_line_graded_exited() -> None:
    flow = bundle_workflow(
        parameters(process_state="EXITED", target_pids_by_node={NODE: [PID]})
    )

    assert new_attempt(INCIDENT, flow) == []


def test_new_attempt_without_a_new_pid_still_judges_the_process_state() -> None:
    flow = bundle_workflow(parameters(target_pids_by_node={NODE: [NEW_PID]}))

    assert new_attempt(INCIDENT, flow, new_pid=None) == [
        "an old Pod's line was graded as a LIVE process"
    ]


def test_new_attempt_names_the_new_attempt_and_its_targeted_pid() -> None:
    incident = {**INCIDENT, "attempt_id": "JOB-a002"}
    flow = bundle_workflow(
        parameters(process_state="EXITED", target_pids_by_node={NODE: [NEW_PID]})
    )

    assert new_attempt(incident, flow) == [
        "the old Pod's line was attributed to the new attempt JOB-a002",
        f"the new attempt's PID {NEW_PID} was targeted for sampling",
    ]


# --------------------------------------------------------------------------- #
# stitch_errors
# --------------------------------------------------------------------------- #
def stitch(entries: list[dict[str, Any]], head_stamp: str | None = STAMP) -> list[str]:
    return verdicts.stitch_errors(
        entries, message=MESSAGE, rotated_path=ROTATED, head_stamp=head_stamp
    )


def test_stitch_without_any_delivered_record_is_one_error() -> None:
    assert stitch([]) == ["no record was delivered from the fixture file"]


def test_stitch_accepts_one_whole_record_on_the_rotated_file() -> None:
    assert stitch([entry(fields={"path": ROTATED})]) == []


def test_stitch_names_a_record_read_from_the_live_file() -> None:
    assert stitch([entry()]) == [
        f"the stitched record was read from {PATH}, not the rotated file {ROTATED}"
    ]


def test_stitch_without_a_head_stamp_cannot_time_the_record() -> None:
    assert stitch([entry(fields={"path": ROTATED})], head_stamp=None) == [
        "the stitched record or the P stamp has no readable time"
    ]


# --------------------------------------------------------------------------- #
# attributable_errors
# --------------------------------------------------------------------------- #
def test_attributable_names_a_container_the_watcher_recorded_on_another_node() -> None:
    records = {
        "attempts": [
            {
                "attempt_id": "JOB-a001",
                "containers": [
                    {
                        "pod_uid": POD_UID,
                        "container_name": "pytorch",
                        "node_id": PEER,
                        "terminated": False,
                    }
                ],
            }
        ]
    }

    errors = verdicts.attributable_errors(
        records,
        entry(),
        attempt_id="JOB-a001",
        pod_uid=POD_UID,
        container="pytorch",
        node=NODE,
    )

    assert errors == [f"the attempt's container is recorded on {PEER}, not {NODE}"]


def test_attributable_accepts_a_container_recorded_without_a_node() -> None:
    records = {
        "attempts": [
            {
                "attempt_id": "JOB-a001",
                "containers": [{"pod_uid": POD_UID, "container_name": "pytorch"}],
            }
        ]
    }

    errors = verdicts.attributable_errors(
        records,
        entry(),
        attempt_id="JOB-a001",
        pod_uid=POD_UID,
        container="pytorch",
        node=NODE,
    )

    assert errors == []
