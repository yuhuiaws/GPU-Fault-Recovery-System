"""A Deployment that cannot create its Pods: the ``ReplicaFailure`` condition.

When a ReplicaSet's Pod create is refused -- an admission webhook that does
not answer, a quota, a policy -- the ReplicaSet controller records
``ReplicaFailure=True`` (reason ``FailedCreate``, the API error as the message)
and the Deployment controller copies the condition onto the Deployment. The
readiness ``status`` already reports per Deployment (``cpu_workloads``, and each
GPU cluster's check) now keeps it: ``deployment_readiness`` adds the reason, the
redacted and capped message and the command that retries at once, a check below
readiness names every short Deployment with its condition, and the status
header prints one line per control-plane Deployment that has one
(``replica_failure_line``). Recorded, never a verdict of its own: readiness
decides the check as before.

Live 2026-11-01: the CPU control-plane nodes were re-imaged together by their
instance group's ``ScheduledUpdateConfig`` (``automatic_node_updates``); Kueue's
Pod webhook ``mpod.kb.io`` (HyperPod task governance, ``failurePolicy: Fail``)
restarted with them, every Pod create in the product namespace failed
``failed calling webhook "mpod.kb.io"`` while it was down, and the
control-worker Deployment stayed at 4/6 for minutes after Kueue answered again,
until a metadata-only annotation of its ReplicaSet made the controller retry.

Why the retry waits, and why an annotation ends the wait -- the controller's
source at the live control plane's version (Kubernetes v1.33.13):

* ``pkg/controller/replicaset/replica_set.go`` ``NewBaseController`` builds the
  work queue with ``workqueue.DefaultTypedControllerRateLimiter[string]()``;
  ``processNextWorkItem`` re-queues a failed sync with
  ``rsc.queue.AddRateLimited(key)``; ``updateRS`` enqueues the ReplicaSet on
  every update event: "it is safer to sync any time this function is
  triggered".
* ``staging/src/k8s.io/client-go/util/workqueue/default_rate_limiters.go``
  ``DefaultTypedControllerRateLimiter``:
  ``NewTypedItemExponentialFailureRateLimiter[T](5*time.Millisecond,
  1000*time.Second)`` -- a per-item backoff from 5 ms doubling to at most
  1000 s (about 16 min).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from gpu_fault.admin.diagnostics import redact_text
from gpu_fault_release.regional_release_config import CheckFailure

__all__ = [
    "KUEUE_POD_WEBHOOK_EVIDENCE",
    "REPLICASET_BACKOFF_EVIDENCE",
    "RETRY_ANNOTATION",
    "deployment_readiness",
    "replica_failure",
    "replica_failure_line",
    "require_ready",
]

REPLICA_FAILURE = "ReplicaFailure"
#: The longest condition message a record or a header line carries.
MESSAGE_LIMIT = 300
#: The annotation whose update makes the ReplicaSet controller sync at once.
RETRY_ANNOTATION = "gpu-fault.io/replica-retry-requested-at"
SECTION = "administrator-operations §5.0.5"
#: The operator hint every ``ReplicaFailure`` line carries.
RETRY_HINT = (
    "the ReplicaSet controller retries a failed Pod create with exponential "
    "backoff (up to ~16 min); once the cause is gone, any metadata update of "
    "the ReplicaSet makes it retry at once"
)
#: The evidence behind ``RETRY_HINT``.
REPLICASET_BACKOFF_EVIDENCE: dict[str, Any] = {
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
#: Why Kueue's Pod webhook gates Pod creation in the product namespace and
#: which setting takes a namespace out of it: Kueue v0.18.0, the version the
#: HyperPod task governance add-on v1.5.0-eksbuild.1 installs and the live CPU
#: cluster runs (its mpod.kb.io: failurePolicy Fail, namespaceSelector
#: kubernetes.io/metadata.name NotIn [kube-system, kueue-system]). The rendered
#: page carries the same text; the source at the v0.18.0 tag pins the version.
KUEUE_POD_WEBHOOK_EVIDENCE: dict[str, Any] = {
    "vendor": "Kueue",
    "version": "v0.18.0",
    "url": (
        "https://github.com/kubernetes-sigs/kueue/blob/v0.18.0/"
        "site/content/en/docs/tasks/run/plain_pods.md"
    ),
    "page": "https://kueue.sigs.k8s.io/docs/tasks/run/plain_pods/#before-you-begin",
    "section": "Run Plain Pods > Before you begin, item 3",
    "quotes": (
        "Kueue will run webhooks for all created pods if the pod integration is "
        "enabled. The webhook namespaceSelector could be used to filter the pods "
        "to reconcile.",
        "When you install Kueue via Helm, the webhook namespace selector will "
        "match the `managedJobsNamespaceSelector` in the `values.yaml`.",
    ),
}
_KUEUE = re.compile(r"(?i)kueue")
_WEBHOOK = re.compile(r'failed calling webhook "([^"]+)"')
_REPLICA_SET = re.compile(r'(?i)replica ?set "([^"]+)"')


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _message(value: object) -> str:
    """The condition message, redacted, on one line and at most ``MESSAGE_LIMIT``."""

    text = re.sub(r"\s+", " ", redact_text(str(value or ""))).strip()
    if len(text) <= MESSAGE_LIMIT:
        return text
    return text[: MESSAGE_LIMIT - 3] + "..."


def _conditions(deployment: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    conditions = _mapping(deployment.get("status")).get("conditions")
    return [
        item
        for item in (conditions if isinstance(conditions, list) else ())
        if isinstance(item, Mapping)
    ]


def _retry_command(deployment: Mapping[str, Any], namespace: str, name: str) -> str:
    """The metadata-only update that makes the controller sync the ReplicaSet.

    The ReplicaSet is the one the ``Progressing`` condition names (the
    Deployment's new ReplicaSet, the one that creates Pods); without it, every
    ReplicaSet of the Deployment's selector -- an annotation on a scaled-down
    one is a sync that changes nothing.
    """

    progressing = next(
        (item for item in _conditions(deployment) if item.get("type") == "Progressing"),
        {},
    )
    found = _REPLICA_SET.search(str(progressing.get("message") or ""))
    selector = _mapping(_mapping(deployment.get("spec")).get("selector"))
    labels = _mapping(selector.get("matchLabels"))
    if found:
        target = found.group(1)
    elif labels:
        target = "-l " + ",".join(f"{key}={labels[key]}" for key in sorted(labels))
    else:
        target = f"<ReplicaSet of {name}>"
    return (
        f"kubectl -n {namespace} annotate replicaset {target} "
        f'{RETRY_ANNOTATION}="$(date -u +%Y%m%dT%H%M%SZ)" --overwrite'
    )


def replica_failure(
    deployment: Mapping[str, Any], *, namespace: str = "", name: str = ""
) -> dict[str, Any] | None:
    """The Deployment's ``ReplicaFailure=True`` condition as a record, else ``None``.

    ``reason``, the redacted and capped ``message``, ``since`` (the condition's
    ``lastTransitionTime``), ``webhook`` (the admission webhook the message says
    failed, when it names one), ``retry`` (``_retry_command``) and, when the
    webhook is Kueue's, ``evidence``: where Kueue documents the namespace
    selector of its Pod webhook (``KUEUE_POD_WEBHOOK_EVIDENCE``).
    """

    condition = next(
        (
            item
            for item in _conditions(deployment)
            if item.get("type") == REPLICA_FAILURE and str(item.get("status")) == "True"
        ),
        None,
    )
    if condition is None:
        return None
    message = _message(condition.get("message"))
    webhook = _WEBHOOK.search(message)
    metadata = _mapping(deployment.get("metadata"))
    record: dict[str, Any] = {
        "reason": str(condition.get("reason") or "unknown"),
        "message": message,
        "since": condition.get("lastTransitionTime"),
        "webhook": webhook.group(1) if webhook else None,
        "retry": _retry_command(
            deployment,
            namespace or str(metadata.get("namespace") or ""),
            name or str(metadata.get("name") or ""),
        ),
    }
    if webhook and _KUEUE.search(message):
        record["evidence"] = KUEUE_POD_WEBHOOK_EVIDENCE["url"]
    return record


def deployment_readiness(
    deployment: Mapping[str, Any], *, namespace: str = "", name: str = ""
) -> dict[str, Any]:
    """``replicas`` and ``ready`` of one Deployment, and ``replica_failure``
    when its ``ReplicaFailure`` condition is True. Never raises: the caller
    decides with ``require_ready`` once every Deployment is read."""

    record: dict[str, Any] = {
        "replicas": int(_mapping(deployment.get("spec")).get("replicas", 0) or 0),
        "ready": int(_mapping(deployment.get("status")).get("readyReplicas", 0) or 0),
    }
    failure = replica_failure(deployment, namespace=namespace, name=name)
    if failure is not None:
        record["replica_failure"] = failure
    return record


def _summary(name: str, record: Mapping[str, Any]) -> str:
    failure = _mapping(record.get("replica_failure"))
    text = f"{name} has {record.get('ready')}/{record.get('replicas')} Ready"
    if failure:
        text += f" (ReplicaFailure {failure.get('reason')}: {failure.get('message')})"
    return text


def require_ready(
    records: Mapping[str, Mapping[str, Any]], *, details: Any = None
) -> None:
    """Raise one ``CheckFailure`` naming every Deployment whose ``ready`` is not
    its ``replicas``, each with its ``ReplicaFailure``; the failure keeps
    ``details`` (default: the records) as the check's evidence."""

    short = [
        _summary(name, record)
        for name, record in records.items()
        if record.get("ready") != record.get("replicas")
    ]
    if short:
        raise CheckFailure(
            "; ".join(short), dict(records) if details is None else details
        )


def replica_failure_line(name: str, record: Mapping[str, Any]) -> str:
    """The status header line of one Deployment's ``replica_failure`` record:
    reason, message, ``RETRY_HINT`` with the command, and -- for Kueue's
    webhook -- where to read how its namespace selector is set."""

    failure = _mapping(record.get("replica_failure"))
    line = (
        f"replica failure: {name} ({record.get('ready')}/{record.get('replicas')} "
        f"Ready) {failure.get('reason')}: {failure.get('message')}; {RETRY_HINT}: "
        f"{failure.get('retry')}"
    )
    if failure.get("evidence"):
        line += (
            f"; {failure.get('webhook')} is Kueue's Pod webhook (HyperPod task "
            f"governance), whose namespace selector covers this namespace: {SECTION}"
        )
    return line
