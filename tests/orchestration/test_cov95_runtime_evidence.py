from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

import pytest

from tests._builders import container_observation
from tests.orchestration._cov95_runtime_builder import xid
from tests.orchestration._cov95_runtime_evidence import EvidenceHarness


@pytest.mark.parametrize(
    ("field", "value", "matches"),
    [
        ("pod_uid", "pod-a", True),
        ("pod_uid", "other-pod", False),
        ("container_id", "docker://container-a", True),
        ("container_id", "container-b", False),
        ("host_pid", 100, True),
        ("host_pid", 101, False),
        ("cgroup_path", "/kubepods/pod-a/child/", True),
        ("cgroup_path", "/kubepods", True),
        ("cgroup_path", "/kubepods/pod-ab", False),
    ],
)
def test_explicit_process_identity_narrows_observation_candidates(
    field, value, matches
) -> None:
    h = EvidenceHarness()
    observation = h.observation()
    event = xid(observed_at=h.now, gpu_uuid=None, **{field: value})
    result = h.service.attempt_observation(event)
    assert result == (observation if matches else None), (
        "process identity must match a live container without prefix collisions",
        field,
        value,
        result,
    )
    assert h.service.ambiguous_attempt_ownership_total() == 0, (
        "a refused identity is not an ambiguous ownership event"
    )


def test_ambiguity_precheck_does_not_double_count_the_family_resolution() -> None:
    h = EvidenceHarness()
    h.observation()
    h.observation(job="other-job", attempt="other-attempt")
    event = xid(observed_at=h.now)
    assert h.service.attempt_observation(event, record_ambiguity=False) is None, (
        "two active jobs cannot be resolved by choosing the newest"
    )
    assert h.service.ambiguous_attempt_ownership_total() == 0, "precheck must not count"
    assert h.service.attempt_observation(event) is None, (
        "family resolution must remain ambiguous"
    )
    assert h.service.ambiguous_attempt_ownership_total() == 1, (
        "count the resolving call once"
    )


@pytest.mark.parametrize("has_generation_start", [False, True])
def test_delayed_explicit_generation_requires_the_attempt_to_have_started_before_the_fault(
    has_generation_start: bool,
) -> None:
    h = EvidenceHarness()
    observation = h.observation(
        observed_at=h.now,
        started_at=h.now - timedelta(minutes=2) if has_generation_start else None,
    )
    event = xid(
        observed_at=h.now - timedelta(seconds=60),
        ingested_at=h.now,
        job_id="job-a",
        attempt_id="attempt-a",
    )
    result = h.service.attempt_observation(event)
    assert result == (observation if has_generation_start else None), (
        "late ingestion cannot infer that an unknown generation owned an earlier fault",
        result,
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"job_id": "other-job"},
        {"attempt_id": "other-attempt"},
        {"affected_workload_ids": ["training/job/other-job"]},
    ],
)
def test_active_recovery_fallback_never_overrides_explicit_workload_identity(
    updates,
) -> None:
    h = EvidenceHarness()
    h.observation()
    workflow = h.active_recovery()
    event = xid(observed_at=h.now, **updates)
    result = h.service.attempt_observation(event)
    assert result is None, (
        "the active recovery on this node does not own a conflicting explicit workload identity",
        updates,
        result,
    )
    assert h.store.get_workflow(workflow.request_id) == workflow, (
        "evidence resolution must not rewrite the incumbent workflow"
    )


def test_active_recovery_can_recover_unbound_identity_after_its_workload_stopped() -> (
    None
):
    h = EvidenceHarness()
    observation = h.observation(
        containers=[
            container_observation(
                "pod-a",
                "trainer",
                0,
                "node-a",
                gpu_uuids=["GPU-a"],
                terminated=True,
                exit_code=0,
            )
        ],
        workload_phase="SUCCEEDED",
    )
    h.active_recovery()
    result = h.service.attempt_observation(xid(observed_at=h.now))
    assert result == observation, (
        "the scoped active recovery remains evidence when its own stopped containers are gone"
    )


def test_ownership_metrics_count_unique_attempts_and_keep_stale_and_terminated_rows_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = EvidenceHarness()
    first = h.observation()
    second = h.observation(job="job-b", attempt="attempt-b")
    stale = h.observation(
        job="job-stale",
        attempt="attempt-stale",
        observed_at=h.now - timedelta(seconds=121),
        containers=[container_observation("pod-b", "trainer", 0, "node-b")],
    )
    complete = h.observation(
        job="completed", attempt="completed", workload_phase="SUCCEEDED"
    )
    terminated = h.observation(
        job="terminated",
        attempt="terminated",
        containers=[
            container_observation(
                "pod-c", "trainer", 0, "node-b", terminated=True, exit_code=0
            )
        ],
    )
    mixed = first.model_copy(
        update={
            "containers": [
                *first.containers,
                container_observation("no-node", "trainer", 1, None),
                container_observation(
                    "stopped", "trainer", 2, "node-b", terminated=True, exit_code=0
                ),
            ]
        }
    )
    states = [
        SimpleNamespace(observation=value)
        for value in (mixed, first, second, stale, complete, terminated)
    ]

    def no_scan(*args, **kwargs):
        raise AssertionError(
            "preloaded metric inputs must not trigger another Store scan"
        )

    monkeypatch.setattr(h.store, "list_agents", no_scan)
    monkeypatch.setattr(h.store, "list_attempt_observation_states", no_scan)
    snapshot = h.service.ownership_metric_snapshot(
        now=h.now,
        agents=[
            SimpleNamespace(cluster_id="cluster-a", node_id="node-idle"),
            SimpleNamespace(cluster_id="", node_id="incomplete"),
        ],
        observation_states=states,
    )
    assert snapshot == {
        "current": {
            ("cluster-a", "node-a"): 2,
            ("cluster-a", "node-b"): 0,
            ("cluster-a", "node-idle"): 0,
        },
        "stale": {
            ("cluster-a", "node-a"): 0,
            ("cluster-a", "node-b"): 1,
            ("cluster-a", "node-idle"): 0,
        },
    }, snapshot


def test_quiesce_cgroup_scope_keeps_only_live_nonempty_paths_without_mutating_parameters() -> (
    None
):
    h = EvidenceHarness()
    observation = h.observation(
        containers=[
            container_observation(
                "pod-a", "trainer", 0, "node-a", cgroup_path="/workload/a/"
            ),
            container_observation(
                "pod-b", "trainer", 1, "node-a", cgroup_path="/workload/a"
            ),
            container_observation("root", "trainer", 2, "node-a", cgroup_path="/"),
            container_observation(
                "stopped",
                "trainer",
                3,
                "node-b",
                cgroup_path="/workload/b",
                terminated=True,
                exit_code=0,
            ),
        ]
    )
    parameters = {"timeout_seconds": 30}
    result = h.service.quiesce_parameters(parameters, observation)
    assert result == {
        "timeout_seconds": 30,
        "workload_cgroup_paths_by_node": {"node-a": ["/workload/a"]},
    }, result
    assert parameters == {"timeout_seconds": 30}, (
        "scope enrichment must not mutate shared parameters"
    )
