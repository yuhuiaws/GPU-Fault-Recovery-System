"""The final rotate-token publish against a control plane it has just rolled.

Twice live the publish failed within seconds of ``CONTROL_PLANE_ROLLED`` and
succeeded unchanged minutes later. These cases pin the three defences:
selecting the exec target from the current ReplicaSet only, waiting for the
registry runtime in every current Pod, and retrying only the transient classes.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from gpu_fault.admin.command_log import FAILURE_EXIT_CODE_ATTRIBUTE
from gpu_fault_release import regional_release_control_plane_ready as READY
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_runtime_identity import (
    CPU_INGRESS_POD_ATTRIBUTE,
)

NAMESPACE = "gpu-fault-system"
API_HA = READY.inventory.CPU_INGRESS_DEPLOYMENT
REVISION = READY.CURRENT_REVISION_ANNOTATION
HASH = READY.POD_TEMPLATE_HASH_LABEL


def _deployment(
    name: str, *, revision: str = "7", replicas: int = 2, healthz: bool = True
) -> dict:
    container: dict = {"name": name}
    if healthz:
        container["readinessProbe"] = {"httpGet": {"path": "/healthz", "port": 8080}}
    return {
        "metadata": {"name": name, "annotations": {REVISION: revision}},
        "spec": {
            "replicas": replicas,
            "selector": {"matchLabels": {"app": name}},
            "template": {"spec": {"containers": [container]}},
        },
    }


def _replicaset(name: str, revision: str, template_hash: str) -> dict:
    return {
        "metadata": {
            "name": f"{name}-{template_hash}",
            "annotations": {REVISION: revision},
            "labels": {"app": name, HASH: template_hash},
        }
    }


def _pod(
    app: str,
    name: str,
    *,
    template_hash: str = "new",
    ready: bool = True,
    terminating: bool = False,
) -> dict:
    metadata: dict = {"name": name, "labels": {"app": app, HASH: template_hash}}
    if terminating:
        metadata["deletionTimestamp"] = "2026-01-01T00:00:00Z"
    return {
        "metadata": metadata,
        "status": {
            "phase": "Running",
            "conditions": [{"type": "Ready", "status": "True" if ready else "False"}],
        },
    }


def _health(*, ready: bool = True, generation: int = 42, http_status: int = 200):
    return {
        "http_status": http_status,
        "status": "ok" if ready else "unhealthy",
        "service_role": "ingress",
        "regional_registry": {
            "member_id": "m",
            "service_role": "ingress",
            "ready": ready,
            "generation": generation,
            "content_sha256": "f" * 64,
            "target_generation": generation,
            "secret_drift": True,
            "error": None,
        },
    }


def _selector(args: list[str]) -> dict[str, str]:
    value = args[args.index("-l") + 1]
    return dict(item.split("=", 1) for item in value.split(","))


class ControlPlane:
    """A kubectl that answers for one CPU control plane mid-roll."""

    def __init__(self) -> None:
        self.deployments: dict[str, dict] = {}
        self.replicasets: list[dict] = []
        self.pods: list[dict] = []
        self.health: dict[str, object] = {}
        self.commands: list[list[str]] = []
        self.health_calls: list[tuple[str, int]] = []
        self.on_list: list = []

    def with_rolled(self, name: str, **options) -> ControlPlane:
        self.deployments[name] = _deployment(name, **options)
        self.replicasets.append(_replicaset(name, "6", "old"))
        self.replicasets.append(_replicaset(name, "7", "new"))
        return self

    def get_json(self, args: list[str]) -> dict:
        self.commands.append(list(args))
        kind = args[args.index("get") + 1]
        if kind == "deployment":
            return self.deployments[args[args.index("get") + 2]]
        selector = _selector(args)
        source = self.replicasets if kind == "replicaset" else self.pods
        if kind == "pods":
            for hook in self.on_list:
                hook()
        return {
            "items": [
                item
                for item in source
                if all(
                    item["metadata"]["labels"].get(key) == value
                    for key, value in selector.items()
                )
            ]
        }

    def run(self, args: list[str], **options) -> str:
        self.commands.append(list(args))
        pod = args[args.index("exec") + 2]
        port = json.loads(options["input_text"])["port"]
        self.health_calls.append((pod, port))
        answer = self.health[pod]
        if isinstance(answer, Exception):
            raise answer
        return json.dumps(answer)

    def probe_output(self, args: list[str], *, timeout_seconds=None):
        name = args[args.index("pod") + 1]
        if not any(item["metadata"]["name"] == name for item in self.pods):
            return (0, "", "")
        return (
            0,
            json.dumps(
                {
                    "kind": "Pod",
                    "metadata": {"name": name, "namespace": NAMESPACE, "uid": "u"},
                }
            ),
            "",
        )


def _release(cluster: ControlPlane, *, dry_run: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        runner=SimpleNamespace(
            dry_run=dry_run, run=cluster.run, probe_output=cluster.probe_output
        ),
        config=SimpleNamespace(namespace=NAMESPACE),
        _cpu=lambda *arguments: ["kubectl", "--kubeconfig", "cpu", *arguments],
        _get_json=cluster.get_json,
    )


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _mid_roll() -> ControlPlane:
    cluster = ControlPlane().with_rolled(API_HA)
    cluster.pods.extend(
        [
            _pod(API_HA, "api-old-1", template_hash="old", terminating=True),
            _pod(API_HA, "api-old-2", template_hash="old"),
            _pod(API_HA, "api-new-1"),
            _pod(API_HA, "api-new-2", ready=False),
            _pod(API_HA, "api-new-3", terminating=True),
        ]
    )
    return cluster


# --------------------------------------------------------------------------
# Pod selection


def test_current_replicaset_lists_only_the_current_hash_and_marks_each_pod() -> None:
    cluster = _mid_roll()

    current = READY.current_replicaset(_release(cluster), API_HA)

    assert [pod.name for pod in current.pods] == ["api-new-1", "api-new-2", "api-new-3"]
    assert current.revision == "7" and current.pod_template_hash == "new"
    assert [pod.name for pod in current.serving_pods] == ["api-new-1"]
    assert not current.complete, "a not-Ready and a Terminating Pod are pending"
    listing = next(call for call in cluster.commands if "pods" in call)
    assert _selector(listing) == {"app": API_HA, HASH: "new"}, (
        "Pods are listed by the current ReplicaSet's hash, never by app alone"
    )
    assert current.healthz_port == 8080
    assert current.evidence()["pending_pods"] == ["api-new-2", "api-new-3"]


def test_select_current_ingress_pod_skips_old_terminating_and_not_ready_pods() -> None:
    cluster = _mid_roll()
    release = _release(cluster)
    setattr(release, CPU_INGRESS_POD_ATTRIBUTE, "api-old-2")

    assert READY.select_current_ingress_pod(release) == "api-new-1"
    assert getattr(release, CPU_INGRESS_POD_ATTRIBUTE) == "api-new-1", (
        "the publish's memoised exec target is replaced, not left pre-roll"
    )


def test_select_current_ingress_pod_fails_closed_without_a_ready_current_pod() -> None:
    cluster = ControlPlane().with_rolled(API_HA)
    cluster.pods.extend(
        [
            _pod(API_HA, "api-old-1", template_hash="old"),
            _pod(API_HA, "api-new-1", ready=False),
            _pod(API_HA, "api-new-2", terminating=True),
        ]
    )

    with pytest.raises(ReleaseError, match="no Ready CPU ingress Pod in the current"):
        READY.select_current_ingress_pod(_release(cluster))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda c: c.replicasets.append(_replicaset(API_HA, "7", "dup")), "has 2 Rep"),
        (lambda c: c.replicasets.clear(), "has 0 ReplicaSets"),
        (
            lambda c: c.deployments[API_HA]["metadata"].pop("annotations"),
            "no current revision",
        ),
        (
            lambda c: c.replicasets[1]["metadata"]["labels"].pop(HASH),
            f"no {HASH} label",
        ),
    ],
)
def test_current_replicaset_fails_closed_on_an_unreasoned_deployment(
    mutate, message
) -> None:
    cluster = _mid_roll()
    mutate(cluster)

    with pytest.raises(ReleaseError, match=message):
        READY.current_replicaset(_release(cluster), API_HA)


# --------------------------------------------------------------------------
# Readiness wait


def test_wait_times_out_fail_closed_while_a_runtime_never_reports_ready() -> None:
    cluster = ControlPlane().with_rolled(API_HA)
    cluster.pods.extend([_pod(API_HA, "api-new-1"), _pod(API_HA, "api-new-2")])
    cluster.health["api-new-1"] = _health(ready=True, generation=41)
    cluster.health["api-new-2"] = _health(ready=False, generation=40, http_status=503)
    clock = Clock()

    with pytest.raises(ReleaseError) as info:
        READY.wait_control_plane_ready(
            _release(cluster),
            (API_HA,),
            timeout_seconds=10,
            poll_seconds=4,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )

    message = str(info.value)
    assert message.startswith("control plane did not become ready within 10 seconds"), (
        message
    )
    assert f"{API_HA}/api-new-2: registry runtime not ready (generation=40" in message
    assert clock.sleeps == [4, 4, 2], "polls are bounded by the deadline"
    assert all(port == 8080 for _pod_name, port in cluster.health_calls), (
        "the health port comes from the Deployment's own readiness probe"
    )


def test_wait_passes_once_every_current_pod_is_ready_on_one_generation() -> None:
    cluster = ControlPlane().with_rolled(API_HA)
    cluster.with_rolled("gpu-fault-control-worker", replicas=1)
    cluster.pods.extend(
        [
            _pod(API_HA, "api-old-1", template_hash="old", terminating=True),
            _pod(API_HA, "api-new-1"),
            _pod(API_HA, "api-new-2", ready=False),
            _pod("gpu-fault-control-worker", "worker-new-1"),
        ]
    )
    cluster.health["api-new-1"] = _health(generation=42)
    cluster.health["api-new-2"] = _health(generation=42)
    cluster.health["worker-new-1"] = _health(generation=42)

    listings: list[int] = []

    def second_poll_ready() -> None:
        # The hook runs before each Pod listing is answered: the first poll
        # sees api-new-2 not Ready, the second sees it Ready.
        listings.append(len(listings))
        if len(listings) == 2:
            cluster.pods[2]["status"]["conditions"][0]["status"] = "True"

    cluster.on_list.append(second_poll_ready)
    clock = Clock()

    evidence = READY.wait_control_plane_ready(
        _release(cluster),
        (API_HA, "gpu-fault-control-worker"),
        timeout_seconds=60,
        poll_seconds=2,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert clock.sleeps == [2], "one poll was pending on a not-Ready Pod"
    assert evidence["registry_generation"] == 42
    assert evidence["deployments"][API_HA]["ready_pods"] == ["api-new-1", "api-new-2"]
    assert evidence["deployments"][API_HA]["pending_pods"] == []
    assert evidence["deployments"]["gpu-fault-control-worker"]["replicas"] == 1
    assert "api-old-1" not in json.dumps(evidence), "old-hash Pods are not judged"
    assert sorted(set(cluster.health_calls)) == [
        ("api-new-1", 8080),
        ("api-new-2", 8080),
        ("worker-new-1", 8080),
    ], "every current Pod of a health-probed Deployment is asked"


def test_wait_treats_divergent_generations_as_not_ready() -> None:
    cluster = ControlPlane().with_rolled(API_HA)
    cluster.pods.extend([_pod(API_HA, "api-new-1"), _pod(API_HA, "api-new-2")])
    cluster.health["api-new-1"] = _health(generation=41)
    cluster.health["api-new-2"] = _health(generation=42)
    clock = Clock()

    with pytest.raises(ReleaseError, match="registry generations differ"):
        READY.wait_control_plane_ready(
            _release(cluster),
            (API_HA,),
            timeout_seconds=1,
            poll_seconds=1,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )


def test_wait_counts_a_failed_health_exec_as_pending_not_as_a_verdict() -> None:
    cluster = ControlPlane().with_rolled(API_HA, replicas=1)
    cluster.pods.append(_pod(API_HA, "api-new-1"))
    cluster.health["api-new-1"] = ReleaseError("command failed (137): kubectl")
    clock = Clock()

    def recover() -> None:
        # Once the first health exec has failed, the Pod answers normally.
        if len(cluster.health_calls) >= 1:
            cluster.health["api-new-1"] = _health(generation=5)

    cluster.on_list.append(recover)

    evidence = READY.wait_control_plane_ready(
        _release(cluster),
        (API_HA,),
        timeout_seconds=30,
        poll_seconds=3,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert clock.sleeps == [3]
    assert evidence["registry_generation"] == 5


def test_wait_judges_a_deployment_without_a_healthz_probe_on_ready_alone() -> None:
    cluster = ControlPlane().with_rolled(
        "gpu-fault-telemetry-spool-worker", healthz=False
    )
    cluster.pods.extend(
        [
            _pod("gpu-fault-telemetry-spool-worker", "spool-new-1"),
            _pod("gpu-fault-telemetry-spool-worker", "spool-new-2"),
        ]
    )

    evidence = READY.wait_control_plane_ready(
        _release(cluster),
        ("gpu-fault-telemetry-spool-worker",),
        timeout_seconds=5,
        sleep=lambda _seconds: None,
    )

    assert cluster.health_calls == [], "no HTTP readiness probe, no in-Pod exec"
    assert evidence["registry_generation"] is None


def test_wait_in_dry_run_reads_nothing() -> None:
    cluster = ControlPlane()

    evidence = READY.wait_control_plane_ready(_release(cluster, dry_run=True))

    assert evidence["dry_run"] is True
    assert cluster.commands == []


# --------------------------------------------------------------------------
# Bounded retry


def _ready_cluster() -> ControlPlane:
    cluster = ControlPlane().with_rolled(API_HA)
    cluster.pods.extend([_pod(API_HA, "api-new-1"), _pod(API_HA, "api-new-2")])
    cluster.health["api-new-1"] = _health()
    cluster.health["api-new-2"] = _health()
    return cluster


def _failure(message: str, exit_code: int | None = None) -> ReleaseError:
    error = ReleaseError(message)
    if exit_code is not None:
        setattr(error, FAILURE_EXIT_CODE_ATTRIBUTE, exit_code)
    return error


class Publisher:
    def __init__(self, *outcomes) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self) -> dict:
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _publish(cluster: ControlPlane, publisher: Publisher, **options) -> dict:
    recorded: list[list[dict]] = []
    clock = Clock()
    outcome = READY.publish_after_control_plane_roll(
        _release(cluster),
        publish=publisher,
        record=lambda history: recorded.append([dict(item) for item in history]),
        deployments=(API_HA,),
        sleep=clock.sleep,
        **options,
    )
    outcome["recorded"] = recorded
    outcome["sleeps"] = clock.sleeps
    return outcome


def test_an_exec_killed_in_flight_is_retried_and_every_attempt_is_recorded() -> None:
    cluster = _ready_cluster()
    publisher = Publisher(
        _failure("command failed (137): kubectl", 137), {"generation": 43}
    )

    outcome = _publish(cluster, publisher)

    assert outcome["result"] == {"generation": 43}
    assert publisher.calls == 2
    attempts = outcome["publish_attempts"]
    assert [item["outcome"] for item in attempts] == ["failed", "published"]
    assert attempts[0]["failure_class"] == READY.FAILURE_EXEC_KILLED
    assert attempts[0]["pod"] == "api-new-1" and attempts[1]["pod"] == "api-new-1"
    assert outcome["sleeps"] == [READY.PUBLISH_RETRY_DELAY_SECONDS]
    assert [len(history) for history in outcome["recorded"]] == [1, 2], (
        "the journal is told after every attempt, failed or not"
    )
    assert outcome["control_plane_ready"]["registry_generation"] == 42


def test_an_exec_target_that_vanished_is_retried_against_a_fresh_pod() -> None:
    cluster = _ready_cluster()
    calls: list[str] = []

    def publish() -> dict:
        calls.append("publish")
        if len(calls) == 1:
            # The rollout reaps the exec target under the publish: kubectl
            # exits 1 and, the output being sensitive, says nothing usable.
            cluster.pods[:] = [
                pod for pod in cluster.pods if pod["metadata"]["name"] != "api-new-1"
            ]
            raise _failure("command failed (1): kubectl", 1)
        return {"generation": 44}

    outcome = _publish(cluster, publish)

    attempts = outcome["publish_attempts"]
    assert attempts[0]["failure_class"] == READY.FAILURE_EXEC_TARGET_GONE
    assert attempts[0]["pod"] == "api-new-1"
    assert attempts[1]["pod"] == "api-new-2", (
        "the next attempt re-selects from the current ReplicaSet"
    )
    assert attempts[1]["outcome"] == "published"
    assert calls == ["publish", "publish"]


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            "command failed (1): kubectl: NotFound: <sensitive output redacted>",
            READY.FAILURE_EXEC_TARGET_GONE,
        ),
        (
            "command failed (1): kubectl: Unable to connect: <sensitive output redacted>",
            READY.FAILURE_CONTROL_PLANE_UNAVAILABLE,
        ),
        (
            "command failed (1): kubectl: ServiceUnavailable: <sensitive output redacted>",
            READY.FAILURE_CONTROL_PLANE_UNAVAILABLE,
        ),
        ("regional registry generation conflict", READY.FAILURE_REFUSED),
        ("command failed (1): kubectl: Forbidden: <redacted>", READY.FAILURE_REFUSED),
        ("regional registry generation 9 did not converge", READY.FAILURE_REFUSED),
        ("regional registry client returned a non-object", READY.FAILURE_REFUSED),
    ],
)
def test_failure_classes_come_from_exit_codes_and_surviving_error_codes(
    message: str, expected: str
) -> None:
    cluster = _ready_cluster()

    assert (
        READY.classify_publish_failure(
            _release(cluster), _failure(message), pod="api-new-1"
        )
        == expected
    )


def test_a_refusal_is_raised_at_once_without_a_second_publish() -> None:
    cluster = _ready_cluster()
    publisher = Publisher(_failure("regional registry generation conflict"))
    recorded: list[list[dict]] = []
    clock = Clock()

    with pytest.raises(ReleaseError, match="generation conflict") as info:
        READY.publish_after_control_plane_roll(
            _release(cluster),
            publish=publisher,
            record=lambda history: recorded.append(list(history)),
            deployments=(API_HA,),
            sleep=clock.sleep,
        )

    assert publisher.calls == 1, "a drift refusal is never repeated"
    assert clock.sleeps == []
    assert recorded[-1][0]["failure_class"] == READY.FAILURE_REFUSED
    assert info.value.__notes__ == [
        "registry publish failed on attempt 1/3 (refused); exec target api-new-1"
    ]


def test_the_retry_budget_is_bounded() -> None:
    cluster = _ready_cluster()
    publisher = Publisher(
        _failure("command failed (143): kubectl", 143),
        _failure("command failed (143): kubectl", 143),
        _failure("command failed (143): kubectl", 143),
        {"generation": 1},
    )
    clock = Clock()

    with pytest.raises(ReleaseError, match=r"command failed \(143\)") as info:
        READY.publish_after_control_plane_roll(
            _release(cluster),
            publish=publisher,
            deployments=(API_HA,),
            sleep=clock.sleep,
        )

    assert publisher.calls == READY.PUBLISH_RETRY_ATTEMPTS == 3
    assert clock.sleeps == [READY.PUBLISH_RETRY_DELAY_SECONDS] * 2
    assert info.value.__notes__[-1].startswith(
        "registry publish failed on attempt 3/3 (exec-killed)"
    ), info.value.__notes__


def test_the_publish_does_not_run_while_the_control_plane_is_not_ready() -> None:
    cluster = ControlPlane().with_rolled(API_HA)
    cluster.pods.extend(
        [_pod(API_HA, "api-new-1"), _pod(API_HA, "api-new-2", ready=False)]
    )
    cluster.health["api-new-1"] = _health()
    publisher = Publisher({"generation": 1})
    clock = Clock()

    with pytest.raises(ReleaseError, match="did not become ready within 6 seconds"):
        READY.publish_after_control_plane_roll(
            _release(cluster),
            publish=publisher,
            deployments=(API_HA,),
            timeout_seconds=6,
            sleep=clock.sleep,
        )

    assert publisher.calls == 0, "fail closed: no publish before readiness"


def test_product_bounds_are_the_documented_numbers() -> None:
    assert READY.CONTROL_PLANE_READY_TIMEOUT_SECONDS == 120
    assert READY.CONTROL_PLANE_READY_POLL_SECONDS == 2
    assert READY.PUBLISH_RETRY_ATTEMPTS == 3
    assert READY.PUBLISH_RETRY_DELAY_SECONDS == 5
    assert READY.TRANSIENT_FAILURE_CLASSES == {
        "exec-killed",
        "exec-target-gone",
        "control-plane-unavailable",
    }
