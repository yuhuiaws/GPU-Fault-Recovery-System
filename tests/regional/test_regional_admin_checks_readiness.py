"""The verify judges collector readiness with a bounded post-roll grace.

Twice a release whose deploy had completed cleanly was rolled back by its own
verify: the ``control_api`` check read collector readiness from one snapshot
taken seconds after the ``api-ha`` roll, when a collector that posts every
~300 s and lost one post to the roll showed an age above its 420 s threshold.
A rerun minutes later passed with nothing changed, and the failed record did
not say which node or collector had been stale. These cases pin the judgement
that replaced it: wait out one missed posting cycle, re-asking only the cluster
that is not ready; never outlive the release's remaining budget; fail closed
after that; and keep the judged snapshot in the record either way.
"""

from __future__ import annotations

import io
import json
import runpy
import urllib.request
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin import deadlines
from gpu_fault.collector_kinds import collector_silent_thresholds
from gpu_fault_release import regional_admin_checks as CHECKS
from gpu_fault_release import regional_admin_checks_readiness as READINESS
from gpu_fault_release import regional_release_probes as PROBES
from gpu_fault_release.regional_release_config import CheckFailure, ReleaseError

THRESHOLDS = {"GPU_INVENTORY": 180.0, "GPU_METRICS": 420.0, "HOST_TELEMETRY": 420.0}
STALE_LINE = "gpu-a/node-a/HOST_TELEMETRY: age 512s exceeds threshold 420s"


@pytest.fixture(autouse=True)
def _no_inherited_release_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (deadlines.DEADLINE_ENV, deadlines.HARD_DEADLINE_ENV):
        monkeypatch.delenv(name, raising=False)


class Clock:
    """A clock that only moves when the judgement sleeps."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class Fetcher:
    """The re-poll, answering from a script; the last answer repeats."""

    def __init__(self, answers: list[Any]) -> None:
        self.answers = list(answers)
        self.asked: list[list[str]] = []

    def __call__(self, release: Any, cluster_ids: list[str]) -> dict[str, Any]:
        self.asked.append(list(cluster_ids))
        if not self.answers:
            pytest.fail("the re-poll ran for a cluster that was already ready")
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, BaseException):
            raise answer
        return dict(answer)


def snapshot(
    *,
    ready: bool,
    age: float | None,
    cluster_id: str = "gpu-a",
    unit_state: str = "active",
) -> dict[str, Any]:
    return {
        "cluster_id": cluster_id,
        "ready": ready,
        "nodes": [
            {
                "node_id": "node-a",
                "ready": ready,
                "collectors": {
                    "HOST_TELEMETRY": {
                        "unit": "gpu-fault-host-collector",
                        "unit_state": unit_state,
                        "unit_enabled": "enabled",
                        "last_success_at": None if age is None else "fixture",
                        "age_seconds": age,
                        "pending_first_report": False,
                        "ready": ready,
                    }
                },
            }
        ],
        "retired_nodes": [],
    }


def stale(cluster_id: str = "gpu-a") -> dict[str, Any]:
    return snapshot(ready=False, age=512.0, cluster_id=cluster_id)


def fresh(cluster_id: str = "gpu-a") -> dict[str, Any]:
    return snapshot(ready=True, age=41.0, cluster_id=cluster_id)


def answer(*snapshots: dict[str, Any]) -> dict[str, Any]:
    return {
        "clusters": {item["cluster_id"]: item for item in snapshots},
        "collector_thresholds": THRESHOLDS,
    }


def report(*snapshots: dict[str, Any]) -> dict[str, Any]:
    return {
        "clusters": {
            item["cluster_id"]: {"collector_readiness": item} for item in snapshots
        },
        "collector_thresholds": THRESHOLDS,
    }


def judge(
    document: dict[str, Any], *, grace: float, fetch: Fetcher, clock: Clock
) -> dict[str, Any]:
    return READINESS.judge_collector_readiness(
        object(),
        document,
        grace_seconds=grace,
        fetch=fetch,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )


def host_collector(evidence: dict[str, Any], cluster_id: str = "gpu-a") -> dict:
    nodes = evidence["clusters"][cluster_id]["nodes"]
    return dict(nodes[0]["collectors"]["HOST_TELEMETRY"])


def test_a_collector_that_posts_late_passes_within_the_grace() -> None:
    clock = Clock()
    fetch = Fetcher([answer(stale()), answer(fresh())])

    evidence = judge(report(stale()), grace=600.0, fetch=fetch, clock=clock)

    assert evidence["ready"] is True
    assert (evidence["polls"], evidence["waited_seconds"]) == (2, 20.0)
    assert clock.sleeps == [10.0, 10.0]
    assert fetch.asked == [["gpu-a"], ["gpu-a"]]
    assert evidence["stale"] == []
    collector = host_collector(evidence)
    assert (collector["ready"], collector["age_seconds"]) == (True, 41.0)
    assert collector["threshold_seconds"] == 420.0


def test_a_collector_still_silent_after_the_grace_fails_naming_it() -> None:
    clock = Clock()
    fetch = Fetcher([answer(stale())])

    with pytest.raises(CheckFailure) as failure:
        judge(report(stale()), grace=600.0, fetch=fetch, clock=clock)

    message = str(failure.value)
    assert "gpu-a collector readiness failed after 600s of post-roll grace" in message
    assert "(60 polls)" in message
    assert STALE_LINE in message
    evidence = CheckFailure.details_of(failure.value)
    assert evidence["ready"] is False
    assert (evidence["polls"], evidence["waited_seconds"]) == (60, 600.0)
    assert (evidence["grace_seconds"], evidence["requested_grace_seconds"]) == (
        600.0,
        600.0,
    )
    assert evidence["stale"] == [STALE_LINE]
    collector = host_collector(evidence)
    assert (collector["ready"], collector["age_seconds"]) == (False, 512.0)
    assert collector["threshold_seconds"] == 420.0
    assert len(fetch.asked) == 60


def test_a_ready_first_snapshot_is_judged_without_a_poll() -> None:
    clock = Clock()
    fetch = Fetcher([])

    evidence = judge(report(fresh()), grace=600.0, fetch=fetch, clock=clock)

    assert (evidence["ready"], evidence["polls"]) == (True, 0)
    assert (evidence["waited_seconds"], clock.sleeps, fetch.asked) == (0.0, [], [])
    assert host_collector(evidence)["threshold_seconds"] == 420.0


def test_the_remaining_release_budget_caps_the_grace() -> None:
    clock = Clock()
    fetch = Fetcher([answer(stale())])

    with deadlines.deadline_scope("release/verify", 30):
        with pytest.raises(CheckFailure) as failure:
            judge(report(stale()), grace=600.0, fetch=fetch, clock=clock)

    evidence = CheckFailure.details_of(failure.value)
    assert evidence["requested_grace_seconds"] == 600.0
    assert 0.0 < evidence["grace_seconds"] <= 30.0
    assert evidence["polls"] == 3
    assert evidence["waited_seconds"] <= 30.0
    assert "failed after 30s of post-roll grace (3 polls)" in str(failure.value)


def test_an_exhausted_release_budget_leaves_no_grace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def expired(seconds: float) -> float:
        raise deadlines.DeploymentDeadlineExceeded("deployment deadline exceeded")

    monkeypatch.setattr(READINESS, "remaining_timeout", expired)
    clock = Clock()
    fetch = Fetcher([])

    with pytest.raises(CheckFailure) as failure:
        judge(report(stale()), grace=600.0, fetch=fetch, clock=clock)

    evidence = CheckFailure.details_of(failure.value)
    assert (evidence["grace_seconds"], evidence["polls"]) == (0.0, 0)
    assert (clock.sleeps, fetch.asked) == ([], [])


def test_a_status_read_judges_the_snapshot_in_hand() -> None:
    assert READINESS.grace_seconds(True) == READINESS.POST_ROLL_GRACE_SECONDS
    assert READINESS.grace_seconds(False) == 0.0
    clock = Clock()
    fetch = Fetcher([])

    with pytest.raises(CheckFailure) as failure:
        judge(report(stale()), grace=0.0, fetch=fetch, clock=clock)

    assert str(failure.value) == f"gpu-a collector readiness failed: {STALE_LINE}"
    evidence = CheckFailure.details_of(failure.value)
    assert (evidence["polls"], evidence["waited_seconds"]) == (0, 0.0)
    assert (clock.sleeps, fetch.asked) == ([], [])


def test_only_the_cluster_that_is_not_ready_is_asked_again() -> None:
    clock = Clock()
    fetch = Fetcher([answer(fresh("gpu-b"))])

    evidence = judge(
        report(fresh("gpu-a"), stale("gpu-b")), grace=600.0, fetch=fetch, clock=clock
    )

    assert fetch.asked == [["gpu-b"]]
    assert evidence["ready"] is True
    assert sorted(evidence["clusters"]) == ["gpu-a", "gpu-b"]
    assert host_collector(evidence, "gpu-b")["age_seconds"] == 41.0


def test_a_failed_re_poll_is_asked_again_but_an_expired_release_is_not() -> None:
    clock = Clock()
    fetch = Fetcher(
        [ReleaseError("the collector readiness report failed"), answer(fresh())]
    )

    evidence = judge(report(stale()), grace=600.0, fetch=fetch, clock=clock)

    assert (evidence["ready"], evidence["polls"]) == (True, 2)
    assert evidence["probe_errors"] == ["the collector readiness report failed"]

    clock = Clock()
    fetch = Fetcher(
        [deadlines.DeploymentDeadlineExceeded("deployment deadline exceeded: command")]
    )
    with pytest.raises(CheckFailure) as failure:
        judge(report(stale()), grace=600.0, fetch=fetch, clock=clock)

    evidence = CheckFailure.details_of(failure.value)
    assert (evidence["polls"], len(fetch.asked)) == (1, 1)
    assert evidence["probe_errors"] == ["deployment deadline exceeded: command"]


@pytest.mark.parametrize(
    ("document", "reason"),
    [
        (
            report(snapshot(ready=False, age=12.0, unit_state="failed")),
            "unit gpu-fault-host-collector is failed",
        ),
        (report(snapshot(ready=False, age=None)), "never reported (threshold 420s)"),
        (
            {"clusters": {"gpu-a": {"collector_readiness": stale()}}},
            "age 512s, threshold unknown",
        ),
        (
            {"clusters": {"gpu-a": {"collector_readiness": None}}},
            "no collector readiness answer",
        ),
        (
            {
                "clusters": {
                    "gpu-a": {"collector_readiness": {"ready": False, "nodes": []}}
                }
            },
            "no ACTIVE Agent has reported",
        ),
    ],
)
def test_the_stale_line_says_why_a_collector_holds_its_cluster_back(
    document: dict[str, Any], reason: str
) -> None:
    with pytest.raises(CheckFailure) as failure:
        judge(document, grace=0.0, fetch=Fetcher([]), clock=Clock())

    assert reason in str(failure.value)
    assert reason in CheckFailure.details_of(failure.value)["stale"][0]


def test_a_plain_release_error_carries_no_details() -> None:
    assert CheckFailure.details_of(ReleaseError("refused")) is None
    assert CheckFailure.details_of(CheckFailure("refused", {"seen": 1})) == {"seen": 1}


# --- through the health report -------------------------------------------------


def control_api_release() -> SimpleNamespace:
    return SimpleNamespace(
        node_wheel_sha="a" * 64,
        executor_wheel_sha="d" * 64,
        config=SimpleNamespace(
            site_name="test-site",
            agent_config_digest="b" * 64,
            component_digests={"node_runtime": "e" * 64, "executor": "f" * 64},
            runtime_profile_version="hyperpod-v1",
            health=SimpleNamespace(remote_command_max_unclaimed_seconds=300),
            clusters=[SimpleNamespace(cluster_id="gpu-a")],
        ),
        _load_state=lambda: {"previous": {"executor_internal_error_total": 0}},
    )


def control_api_report(first: dict[str, Any]) -> dict[str, Any]:
    return {
        "healthz": {"status": "ok"},
        "version": {
            "deployment_mode": "regional",
            "required_agent_artifact_sha256": "a" * 64,
            "required_agent_config_digest": "b" * 64,
            "required_agent_compatibility_digest": "e" * 64,
            "required_runtime_profile_version": "hyperpod-v1",
            "required_node_action_key_version": 2,
            "required_regional_executor_artifact_sha256": "d" * 64,
            "required_regional_executor_compatibility_digest": "f" * 64,
        },
        "registry": [{"cluster_id": "gpu-a"}],
        "clusters": {
            "gpu-a": {
                "expected_node_ids": ["node-a"],
                "agents": [
                    {
                        "node_id": "node-a",
                        "lifecycle_state": "ACTIVE",
                        "runtime_profile_version": "hyperpod-v1",
                    }
                ],
                "fleet_readiness": {"ready": True},
                "collector_readiness": first,
            }
        },
        "collector_thresholds": THRESHOLDS,
        "remote_commands": {"oldest_unclaimed_age_seconds": 0},
    }


OTHER_CHECKS = (
    "_check_contexts",
    "check_cpu_secrets",
    "_check_cpu_workloads",
    "check_email_notifications",
    "check_aurora_refresh",
    "check_adot_self_metrics",
    "_verify_profile",
    "run_read_only_verifiers",
    "_check_runtime_component_identity",
    "_check_gpu_cluster",
    "_check_nlb_runtime",
    "_check_aurora",
    "_check_monitoring",
)


def control_api_finding(
    monkeypatch: pytest.MonkeyPatch, *, mode: str, answers: list[Any]
) -> tuple[dict[str, Any], list[float], Clock]:
    """The ``control_api`` record of a health report, with the wait made fake."""

    for name in OTHER_CHECKS:
        monkeypatch.setattr(
            CHECKS, name, lambda *_args, name=name, **_kwargs: CHECKS.CheckValue(name)
        )
    monkeypatch.setattr(
        CHECKS, "_control_api_report", lambda _release: control_api_report(stale())
    )
    clock = Clock()
    fetch = Fetcher(answers)
    graces: list[float] = []
    original = READINESS.judge_collector_readiness

    def judged(release: Any, document: dict[str, Any], **kwargs: Any) -> dict:
        graces.append(kwargs["grace_seconds"])
        return original(
            release,
            document,
            grace_seconds=kwargs["grace_seconds"],
            fetch=fetch,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )

    monkeypatch.setattr(READINESS, "judge_collector_readiness", judged)
    if mode == "quick":
        health = CHECKS.build_quick_health_report(control_api_release())
    else:
        health = CHECKS.build_health_report(control_api_release(), mode=mode)
    finding = next(item for item in health["checks"] if item["name"] == "control_api")
    return finding, graces, clock


def test_the_verify_record_carries_the_readiness_it_waited_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finding, graces, clock = control_api_finding(
        monkeypatch, mode="verify", answers=[answer(stale()), answer(fresh())]
    )

    assert graces == [600.0]
    assert finding["status"] == "PASS"
    assert clock.sleeps == [10.0, 10.0]
    judged = finding["details"]["collector_readiness"]
    assert (judged["polls"], judged["waited_seconds"]) == (2, 20.0)
    assert host_collector(judged)["threshold_seconds"] == 420.0
    # The first snapshot stays in the report, beside the one that was judged.
    first = finding["details"]["clusters"]["gpu-a"]["collector_readiness"]
    assert (first["ready"], judged["clusters"]["gpu-a"]["ready"]) == (False, True)


def test_a_failed_verify_record_names_the_stale_collector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finding, graces, clock = control_api_finding(
        monkeypatch, mode="verify", answers=[answer(stale())]
    )

    assert graces == [600.0]
    assert finding["status"] == "FAIL"
    assert "gpu-a collector readiness failed after 600s" in finding["summary"]
    assert STALE_LINE in finding["summary"]
    assert finding["details"]["stale"] == [STALE_LINE]
    assert (finding["details"]["polls"], clock.now) == (60, 600.0)
    assert host_collector(finding["details"])["age_seconds"] == 512.0


@pytest.mark.parametrize("mode", ["status", "quick"])
def test_a_status_report_does_not_wait_for_collectors(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    finding, graces, clock = control_api_finding(monkeypatch, mode=mode, answers=[])

    assert graces == [0.0]
    assert finding["status"] == "FAIL"
    assert finding["summary"] == f"gpu-a collector readiness failed: {STALE_LINE}"
    assert (finding["details"]["polls"], clock.sleeps) == (0, [])


# --- the probes -----------------------------------------------------------------


class Response(io.BytesIO):
    def __enter__(self) -> Response:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def expected_thresholds() -> dict[str, float]:
    return {
        kind.value: seconds for kind, seconds in collector_silent_thresholds().items()
    }


def test_the_re_poll_asks_the_ingress_pod_only_about_the_named_clusters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def execute(release: Any, **kwargs: Any) -> str:
        calls.append(kwargs)
        return json.dumps(answer(fresh()))

    monkeypatch.setattr(READINESS, "exec_cpu_ingress", execute)

    document = READINESS.fetch_collector_readiness(object(), ["gpu-a"])

    assert document["clusters"]["gpu-a"]["ready"] is True
    (call,) = calls
    arguments = list(call["arguments"])
    assert arguments[:2] == ["env", 'ADMIN_CLUSTER_IDS_JSON=["gpu-a"]']
    assert arguments[2:4] == [CHECKS.CONTROL_PLANE_PYTHON, "-c"]
    assert arguments[4] == PROBES.probe_source("collector_readiness_inspect")
    assert (call["sensitive"], call["interactive"]) == (True, False)

    monkeypatch.setattr(READINESS, "exec_cpu_ingress", lambda release, **kwargs: "[]")
    with pytest.raises(ReleaseError, match="non-object evidence"):
        READINESS.fetch_collector_readiness(object(), ["gpu-a"])


def test_the_readiness_probe_answers_per_cluster_with_the_thresholds_it_applies(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    requested: list[tuple[str, str | None]] = []

    def urlopen(request: urllib.request.Request, timeout: float) -> Response:
        requested.append(
            (request.full_url, request.get_header("X-gpu-fault-execution-token"))
        )
        return Response(json.dumps({"ready": True, "nodes": []}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setenv("GPU_FAULT_EXECUTION_TOKEN", "fixture-token")
    monkeypatch.setenv("ADMIN_CLUSTER_IDS_JSON", json.dumps(["gpu-a", "gpu b"]))

    runpy.run_path(
        str(PROBES.PROBE_DIR / "collector_readiness_inspect.py"), run_name="__main__"
    )

    document = json.loads(capsys.readouterr().out)
    assert sorted(document["clusters"]) == ["gpu b", "gpu-a"]
    assert document["clusters"]["gpu b"] == {"ready": True, "nodes": []}
    assert document["collector_thresholds"] == expected_thresholds()
    assert [url.rsplit("/", 1)[1] for url, _token in requested] == ["gpu-a", "gpu%20b"]
    assert {token for _url, token in requested} == {"fixture-token"}


def test_the_full_control_api_probe_reports_the_thresholds_too(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from gpu_fault.app import ApplicationContext

    store = SimpleNamespace(
        remote_command_stats=lambda: {"oldest_unclaimed_age_seconds": 0}
    )
    monkeypatch.setattr(
        ApplicationContext,
        "from_environment",
        staticmethod(lambda: SimpleNamespace(store=store)),
    )

    def urlopen(request: urllib.request.Request, timeout: float) -> Response:
        path = request.full_url.removeprefix("http://127.0.0.1:8080")
        body: Any = {
            "/healthz": {"status": "ok"},
            "/v1/version": {"deployment_mode": "regional"},
            "/v1/regional/clusters": [],
        }.get(path)
        if body is None:
            body = [] if path.startswith("/v1/fleet/agents") else {"ready": True}
        return Response(json.dumps(body).encode())

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setenv("GPU_FAULT_EXECUTION_TOKEN", "fixture-token")
    monkeypatch.setenv("ADMIN_CLUSTER_IDS_JSON", '["gpu-a"]')
    monkeypatch.setenv("ADMIN_EXPECTED_NODES_JSON", '{"gpu-a": ["node-a"]}')

    runpy.run_path(
        str(PROBES.PROBE_DIR / "control_api_inspect.py"), run_name="__main__"
    )

    document = json.loads(capsys.readouterr().out)
    assert document["collector_thresholds"] == expected_thresholds()
    assert document["clusters"]["gpu-a"]["collector_readiness"] == {"ready": True}
    assert document["remote_commands"] == {"oldest_unclaimed_age_seconds": 0}
