"""A control-plane Deployment that cannot create its Pods is named, with why.

The world modelled is the live one of 2026-11-01, sanitised: the CPU nodes were
re-imaged together, Kueue's Pod webhook ``mpod.kb.io`` (HyperPod task
governance, ``failurePolicy: Fail``) restarted with them, and while it was down
every Pod create in the product namespace failed. The control-worker
Deployment carried ``ReplicaFailure=True FailedCreate`` and stayed at 4/6 for
minutes after Kueue answered again -- the ReplicaSet controller's backoff --
until a metadata-only annotation of its ReplicaSet made it retry.

``status`` already reported each Deployment's readiness (``cpu_workloads``,
and each GPU cluster's check); the readiness now keeps the condition, the
check names every short Deployment with it, and the header prints one line per
control-plane Deployment with the reason, the redacted and capped message and
the operator hint. Readiness alone still decides the verdict.
"""

from __future__ import annotations

from typing import Any

import pytest

from gpu_fault_release import regional_admin_checks as checks
from gpu_fault_release import regional_release_replica_failure as failure
from gpu_fault_release.regional_admin_commands import (
    replica_failure_lines,
    status_header_lines,
)
from gpu_fault_release.regional_release_config import CheckFailure
from tests.regional._cov95_release_checks import (
    CheckedRelease,
    finding,
    install_api_transport,
)

NAMESPACE = "gpu-fault-system"
WORKER = "gpu-fault-control-worker"
WORKER_RS = "gpu-fault-control-worker-b85bcdcbf"
KUEUE_MESSAGE = (
    'Internal error occurred: failed calling webhook "mpod.kb.io": failed to call '
    'webhook: Post "https://kueue-webhook-service.kueue-system.svc:443/'
    'mutate--v1-pod?timeout=10s": no endpoints available for service '
    '"kueue-webhook-service"'
)
RETRY = (
    f"kubectl -n {NAMESPACE} annotate replicaset {WORKER_RS} "
    'gpu-fault.io/replica-retry-requested-at="$(date -u +%Y%m%dT%H%M%SZ)" '
    "--overwrite"
)


def _failing(
    document: dict[str, Any],
    *,
    ready: int,
    message: str = KUEUE_MESSAGE,
    progressing: str | None = f'ReplicaSet "{WORKER_RS}" has successfully progressed.',
) -> dict[str, Any]:
    """``document`` with ``ready`` Ready replicas and the live condition set."""

    document["status"]["readyReplicas"] = ready
    conditions = [
        {
            "type": "Available",
            "status": "False",
            "reason": "MinimumReplicasUnavailable",
        },
        {
            "type": "ReplicaFailure",
            "status": "True",
            "reason": "FailedCreate",
            "message": message,
            "lastTransitionTime": "2026-11-01T08:00:31Z",
        },
    ]
    if progressing is not None:
        conditions.insert(
            0,
            {
                "type": "Progressing",
                "status": "True",
                "reason": "NewReplicaSetAvailable",
                "message": progressing,
            },
        )
    document["status"]["conditions"] = conditions
    document["spec"]["selector"] = {
        "matchLabels": {"app": document["metadata"]["name"]}
    }
    return document


def _worker(ready: int = 4, **options: Any) -> dict[str, Any]:
    document = {
        "metadata": {"name": WORKER, "namespace": NAMESPACE},
        "spec": {"replicas": 6},
        "status": {"readyReplicas": 6},
    }
    return _failing(document, ready=ready, **options)


@pytest.fixture
def release(monkeypatch: pytest.MonkeyPatch) -> CheckedRelease:
    release = CheckedRelease()
    install_api_transport(monkeypatch, release)
    monkeypatch.setattr(checks.shutil, "which", lambda name: f"/example/bin/{name}")
    return release


# --- the record ------------------------------------------------------------------------


def test_the_live_condition_becomes_a_record_with_the_retry_command() -> None:
    record = failure.deployment_readiness(_worker(), namespace=NAMESPACE, name=WORKER)

    assert record == {
        "replicas": 6,
        "ready": 4,
        "replica_failure": {
            "reason": "FailedCreate",
            "message": KUEUE_MESSAGE,
            "since": "2026-11-01T08:00:31Z",
            "webhook": "mpod.kb.io",
            "retry": RETRY,
            "evidence": failure.KUEUE_POD_WEBHOOK_EVIDENCE["url"],
        },
    }, "the ReplicaSet the Progressing condition names is the one annotated"


def test_a_healthy_or_cleared_condition_records_no_failure() -> None:
    healthy = _worker(ready=6)
    healthy["status"]["conditions"][-1]["status"] = "False"

    assert failure.deployment_readiness(healthy, namespace=NAMESPACE, name=WORKER) == {
        "replicas": 6,
        "ready": 6,
    }
    assert failure.replica_failure({"status": {"conditions": "bad"}}) is None
    assert failure.replica_failure({}) is None


def test_the_message_is_redacted_capped_and_on_one_line() -> None:
    secret = "Bearer abcdefghijklmnopqrstuvwxyz0123456789"
    message = (
        f"admission denied:\n  token={'x' * 40} header Authorization: {secret} "
        + "y" * 600
    )

    record = failure.replica_failure(_worker(message=message), namespace=NAMESPACE)

    assert record is not None
    text = record["message"]
    assert "x" * 40 not in text and "abcdefghijklmnop" not in text, text
    assert "\n" not in text and len(text) == failure.MESSAGE_LIMIT
    assert text.endswith("..."), "a capped message says it was cut"
    assert record["webhook"] is None and "evidence" not in record, (
        "no webhook named: no Kueue pointer"
    )


def test_another_webhook_is_named_without_the_kueue_pointer() -> None:
    message = (
        'Internal error occurred: failed calling webhook "mpod.kb.io": failed to '
        'call webhook: Post "https://amazon-cloudwatch-observability-webhook-'
        'service.amazon-cloudwatch.svc:443/mutate-v1-pod?timeout=10s": context '
        "deadline exceeded"
    )

    record = failure.replica_failure(_worker(message=message), namespace=NAMESPACE)

    assert record is not None and record["webhook"] == "mpod.kb.io"
    assert "evidence" not in record, "the CloudWatch agent's webhook shares the name"


@pytest.mark.parametrize(
    "progressing,selector,target",
    [
        (None, {"app": WORKER}, f"-l app={WORKER}"),
        ("Deployment has minimum availability.", {"app": WORKER}, f"-l app={WORKER}"),
        (None, {"app": WORKER, "tier": "control"}, f"-l app={WORKER},tier=control"),
        (None, None, f"<ReplicaSet of {WORKER}>"),
        (f'Created new replica set "{WORKER_RS}"', None, WORKER_RS),
    ],
    ids=["no-progressing", "no-name", "two-labels", "no-selector", "created"],
)
def test_the_retry_targets_the_named_replica_set_or_the_selector(
    progressing: str | None, selector: dict[str, str] | None, target: str
) -> None:
    document = _worker(progressing=progressing)
    if selector is None:
        document["spec"].pop("selector")
    else:
        document["spec"]["selector"] = {"matchLabels": selector}

    record = failure.replica_failure(document, namespace=NAMESPACE, name=WORKER)

    assert record is not None
    assert record["retry"].startswith(
        f"kubectl -n {NAMESPACE} annotate replicaset {target} "
        f"{failure.RETRY_ANNOTATION}="
    ), record["retry"]


def test_require_ready_names_every_short_deployment_with_its_condition() -> None:
    records = {
        "gpu-fault-api-ha": {"replicas": 3, "ready": 2},
        WORKER: failure.deployment_readiness(
            _worker(), namespace=NAMESPACE, name=WORKER
        ),
        "gpu-fault-adot": {"replicas": 1, "ready": 1},
    }

    with pytest.raises(CheckFailure) as raised:
        failure.require_ready(records)

    assert str(raised.value) == (
        "gpu-fault-api-ha has 2/3 Ready; "
        f"{WORKER} has 4/6 Ready (ReplicaFailure FailedCreate: {KUEUE_MESSAGE})"
    )
    assert raised.value.details == records, "the readiness is the failure's evidence"
    failure.require_ready({"gpu-fault-adot": {"replicas": 1, "ready": 1}})


# --- the checks -----------------------------------------------------------------------


def test_cpu_workloads_fails_naming_both_short_deployments_and_the_webhook(
    release: CheckedRelease,
) -> None:
    worker = release.documents[("cpu", "deployment", WORKER)]
    _failing(worker, ready=1)
    ingress = release.documents[("cpu", "deployment", "gpu-fault-api-ha")]
    ingress["status"]["readyReplicas"] = 1

    result = finding(checks.build_quick_health_report(release), "cpu_workloads")

    assert result["status"] == "FAIL"
    assert result["summary"].startswith("gpu-fault-api-ha has 1/"), result["summary"]
    assert (
        f"; {WORKER} has 1/3 Ready (ReplicaFailure FailedCreate: "
        in (result["summary"])
    ), result["summary"]
    assert result["details"][WORKER]["replica_failure"]["retry"] == RETRY
    assert "replica_failure" not in result["details"]["gpu-fault-api-ha"]


def test_cpu_workloads_passes_at_full_readiness_and_still_records_the_condition(
    release: CheckedRelease,
) -> None:
    worker = release.documents[("cpu", "deployment", WORKER)]
    _failing(worker, ready=3)  # a surge Pod that cannot be created

    result = finding(checks.build_quick_health_report(release), "cpu_workloads")

    assert result["status"] == "PASS", "readiness alone decides the verdict"
    assert result["details"][WORKER]["replica_failure"]["reason"] == "FailedCreate"


def test_a_gpu_cluster_check_names_a_data_plane_deployment_the_same_way(
    release: CheckedRelease,
) -> None:
    name = next(
        key[2]
        for key in release.documents
        if key[:2] == ("gpu-a", "deployment") and key[2]
    )
    _failing(release.documents[("gpu-a", "deployment", name)], ready=0)

    result = finding(
        checks.build_health_report(release, mode="status"), "gpu_cluster:gpu-a"
    )

    assert result["status"] == "FAIL"
    assert result["summary"].startswith(f"{name} has 0/1 Ready (ReplicaFailure "), (
        result["summary"]
    )
    assert result["details"]["deployments"][name]["replica_failure"]["webhook"] == (
        "mpod.kb.io"
    )


# --- the status header ------------------------------------------------------------------


def _status(check: dict[str, Any]) -> dict[str, Any]:
    return {
        "healthy": check["status"] == "PASS",
        "health_scope": "quick",
        "live_release": {"release_id": "r-1", "phase": "complete"},
        "next_deploy": {"kind": "NOOP"},
        "health": {
            "summary": {"PASS": 1, "WARN": 0, "FAIL": 0, "SKIP": 0},
            "checks": [check],
        },
    }


def test_the_header_names_the_deployment_the_reason_and_the_hint(
    release: CheckedRelease,
) -> None:
    _failing(release.documents[("cpu", "deployment", WORKER)], ready=1)
    check = finding(checks.build_quick_health_report(release), "cpu_workloads")

    lines = status_header_lines(_status(check), json_destination="stdout")

    failing = lines.index("failing checks: cpu_workloads")
    assert lines[failing + 1] == (
        f"replica failure: {WORKER} (1/3 Ready) FailedCreate: {KUEUE_MESSAGE}; the "
        "ReplicaSet controller retries a failed Pod create with exponential backoff "
        "(up to ~16 min); once the cause is gone, any metadata update of the "
        f"ReplicaSet makes it retry at once: {RETRY}; mpod.kb.io is Kueue's Pod "
        "webhook (HyperPod task governance), whose namespace selector covers this "
        "namespace: administrator-operations §5.0.5"
    )
    assert lines[-1] == "full JSON report: stdout"


def test_the_header_line_also_follows_a_passing_check(release: CheckedRelease) -> None:
    _failing(release.documents[("cpu", "deployment", WORKER)], ready=3)
    check = finding(checks.build_quick_health_report(release), "cpu_workloads")

    (line,) = replica_failure_lines(_status(check)["health"])

    assert line.startswith(f"replica failure: {WORKER} (3/3 Ready) FailedCreate: "), (
        line
    )


@pytest.mark.parametrize(
    "check",
    [
        {"name": "cpu_workloads", "status": "PASS", "details": {"x": {"ready": 1}}},
        {"name": "cpu_workloads", "status": "FAIL", "details": None},
        {"name": "cpu_workloads", "status": "FAIL", "details": ["bad"]},
        {"name": "control_api", "status": "PASS", "details": {}},
    ],
    ids=["no-condition", "no-details", "malformed", "other-check"],
)
def test_no_condition_no_line(check: dict[str, Any]) -> None:
    assert replica_failure_lines({"checks": [check]}) == []
    assert replica_failure_lines(None) == []


# --- the evidence -------------------------------------------------------------------------


def test_the_backoff_hint_cites_the_controller_source() -> None:
    assert failure.REPLICASET_BACKOFF_EVIDENCE == {
        "vendor": "Kubernetes",
        "version": "v1.33.13",
        "sources": (
            "https://github.com/kubernetes/kubernetes/blob/v1.33.13/"
            "pkg/controller/replicaset/replica_set.go",
            "https://github.com/kubernetes/kubernetes/blob/v1.33.13/"
            "staging/src/k8s.io/client-go/util/workqueue/default_rate_limiters.go",
        ),
        "quotes": (
            "NewTypedItemExponentialFailureRateLimiter[T](5*time.Millisecond, "
            "1000*time.Second)",
            "it is safer to sync any time this function is triggered",
        ),
        "backoff_ceiling_seconds": 1000,
    }
    assert "up to ~16 min" in failure.RETRY_HINT, "1000 s is 16 min 40 s"


def test_the_kueue_pointer_cites_the_version_pinned_documentation() -> None:
    assert failure.KUEUE_POD_WEBHOOK_EVIDENCE == {
        "vendor": "Kueue",
        "version": "v0.18.0",
        "url": (
            "https://github.com/kubernetes-sigs/kueue/blob/v0.18.0/"
            "site/content/en/docs/tasks/run/plain_pods.md"
        ),
        "page": "https://kueue.sigs.k8s.io/docs/tasks/run/plain_pods/#before-you-begin",
        "section": "Run Plain Pods > Before you begin, item 3",
        "quotes": (
            "Kueue will run webhooks for all created pods if the pod integration "
            "is enabled. The webhook namespaceSelector could be used to filter the "
            "pods to reconcile.",
            "When you install Kueue via Helm, the webhook namespace selector will "
            "match the `managedJobsNamespaceSelector` in the `values.yaml`.",
        ),
    }
