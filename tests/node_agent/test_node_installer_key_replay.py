"""A failed installer Job is replaced at once when its node's key arrives.

A Job created before a replacement node had its Node Action key cannot mount
it: the Pod sits in ``FailedMount`` until ``activeDeadlineSeconds`` and the Job
fails. The reconciler then waits out a doubling backoff (up to an hour) before
the next attempt, although that attempt would now mount the key -- live, the
replacement stayed without an Agent long after its key existed. The executor's
node-key sync marks such a node ``installer-state=Retrying`` with
``installer-reason=NodeActionKeyAdded`` right after it added the key; the
reconciler replaces the failed Job in the same pass. What must hold:

* the replacement happens in one pass, without counting an attempt;
* a Job that failed on its own -- the node parked ``Failed``, or the
  reconciler's own ``Retrying`` without that reason -- keeps the backoff;
* no install slot, or the old Job still going away, defers to the next pass
  without losing the request.
"""

from __future__ import annotations

from datetime import timedelta

from gpu_fault.node_installer_reconciler import (
    INSTALLER_ARTIFACT_ANNOTATION,
    INSTALLER_ATTEMPTS_ANNOTATION,
    INSTALLER_BUNDLE_ANNOTATION,
    INSTALLER_DIGEST_ANNOTATION,
    INSTALLER_NODE_UID_ANNOTATION,
    INSTALLER_REASON_ANNOTATION,
    INSTALLER_STATE_ANNOTATION,
    INSTALLER_TEMPLATE_ANNOTATION,
    INSTALLER_VERSION_ANNOTATION,
    NODE_KEY_ADDED_REASON,
)
from tests.node_agent.test_node_installer_reconciler import (
    ARTIFACT,
    BUNDLE,
    NOW,
    TEMPLATE,
    ApiError,
    BatchApi,
    CoreApi,
    job,
    node,
    reconciler,
)


def _installer_annotations(state: str, **extra: str) -> dict[str, str]:
    return {
        INSTALLER_VERSION_ANNOTATION: "0.10.0",
        INSTALLER_DIGEST_ANNOTATION: "config-sha",
        INSTALLER_ARTIFACT_ANNOTATION: ARTIFACT,
        INSTALLER_BUNDLE_ANNOTATION: BUNDLE,
        INSTALLER_TEMPLATE_ANNOTATION: TEMPLATE,
        INSTALLER_NODE_UID_ANNOTATION: "node-uid",
        INSTALLER_STATE_ANNOTATION: state,
        INSTALLER_ATTEMPTS_ANNOTATION: "2",
        **extra,
    }


def _failed_recently():
    # Failed one minute ago: with two attempts recorded the backoff is 1200 s.
    return job(
        condition="Failed", age_seconds=900, failed_at=NOW - timedelta(minutes=1)
    )


def _key_added() -> dict[str, str]:
    return _installer_annotations(
        "Retrying", **{INSTALLER_REASON_ANNOTATION: NODE_KEY_ADDED_REASON}
    )


def _counter(instance, name: str) -> int:
    """One counter of the reconciler's Prometheus family, read off its render."""

    prefix = "gpu_fault_node_installer_" + name + " "
    (line,) = [
        item
        for item in instance.metrics.render().splitlines()
        if item.startswith(prefix)
    ]
    return int(line.split(" ", 1)[1])


def _node_patches(core: CoreApi) -> list[dict[str, str | None]]:
    return [body["metadata"]["annotations"] for _name, body in core.patches]


def test_a_failed_job_whose_node_got_its_key_is_replaced_in_one_pass() -> None:
    core = CoreApi([node(annotations=_key_added())])
    batch = BatchApi(_failed_recently())
    instance = reconciler(core, batch)

    result = instance.reconcile_once()

    assert result["created"] == 1, result
    assert len(batch.deleted) == 1, "the failed Job is removed first"
    (namespace, body) = batch.created[0]
    assert namespace == "gpu-fault-system"
    assert body["spec"]["template"]["spec"]["nodeName"] == "hyperpod-i-123"
    assert batch.deleted[0][0] == body["metadata"]["name"], (
        "the replacement reuses the identity-derived Job name"
    )
    (patch,) = _node_patches(core)
    assert patch[INSTALLER_STATE_ANNOTATION] == "Installing", patch
    assert patch[INSTALLER_ATTEMPTS_ANNOTATION] == "2", (
        "an attempt that could not mount its key is not held against the node"
    )
    assert patch[INSTALLER_REASON_ANNOTATION] is None, "the request is consumed"
    assert _counter(instance, "retries_scheduled_total") == 1
    assert _counter(instance, "jobs_created_total") == 1
    assert _counter(instance, "jobs_failed_total") == 0, (
        "the failure was counted when the node was first parked"
    )


def test_a_job_that_failed_on_its_own_keeps_the_backoff() -> None:
    core = CoreApi([node(annotations=_installer_annotations("Failed"))])
    batch = BatchApi(_failed_recently())

    result = reconciler(core, batch).reconcile_once()

    assert result["failed"] == 1, result
    assert batch.deleted == [] and batch.created == [], "the backoff still holds"
    assert core.patches == [], "a node parked in Failed is not rewritten"


def test_the_reconcilers_own_retrying_keeps_the_backoff() -> None:
    """``Retrying`` without the key reason is not a request to skip the wait."""

    core = CoreApi([node(annotations=_installer_annotations("Retrying"))])
    batch = BatchApi(_failed_recently())

    result = reconciler(core, batch).reconcile_once()

    assert result["failed"] == 1, result
    assert batch.created == [] and batch.deleted == []


def test_without_an_install_slot_the_replacement_waits_for_the_next_one() -> None:
    target = node(annotations=_key_added())
    other = node(name="hyperpod-i-456", uid="other-uid")
    core = CoreApi([target, other])
    running = job()
    batch = BatchApi(
        jobs={"hyperpod-i-123": _failed_recently(), "hyperpod-i-456": running}
    )

    result = reconciler(core, batch, max_unavailable=1).reconcile_once()

    assert result["deferred"] == 1, result
    assert len(batch.deleted) == 1 and batch.created == [], (
        "the failed Job is gone, the slot belongs to the running install"
    )
    assert not any(
        patch.get(INSTALLER_STATE_ANNOTATION) == "Installing"
        for patch in _node_patches(core)
    ), "the node keeps its request until a slot is free"

    del batch.jobs["hyperpod-i-123"]
    batch.jobs["hyperpod-i-456"] = job(condition="Complete")
    again = reconciler(core, batch, max_unavailable=1).reconcile_once()

    assert again["created"] == 1, again


def test_a_replacement_racing_the_old_jobs_removal_is_created_next_pass() -> None:
    class SlowDelete(BatchApi):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.conflicts = 1

        def create_namespaced_job(self, namespace, body, _request_timeout=None):
            if self.conflicts:
                self.conflicts -= 1
                raise ApiError(409)
            super().create_namespaced_job(namespace, body, _request_timeout)

    core = CoreApi([node(annotations=_key_added())])
    batch = SlowDelete(_failed_recently())

    first = reconciler(core, batch).reconcile_once()

    assert first["running"] == 1, first
    assert batch.created == [] and core.patches == [], (
        "a 409 right after the delete is the old Job, not a created one"
    )

    batch.existing = None
    second = reconciler(core, batch).reconcile_once()

    assert second["created"] == 1, second
