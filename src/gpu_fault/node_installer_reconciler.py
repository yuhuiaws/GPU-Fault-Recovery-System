from __future__ import annotations

import base64
import copy
import hashlib
import hmac
import logging
import os
import re
import socket
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault.dcgm_exporter_cadence import DCGM_EXPORTER_COLLECT_INTERVAL_MS
from gpu_fault.gpu_instance_inventory import gpu_instance_inventory
from gpu_fault.logging_setup import configure_logging

LOGGER = logging.getLogger(__name__)

INSTALLER_VERSION_ANNOTATION = "gpu-fault.io/installer-version"
INSTALLER_DIGEST_ANNOTATION = "gpu-fault.io/installer-config-digest"
INSTALLER_ARTIFACT_ANNOTATION = "gpu-fault.io/installer-artifact-sha256"
INSTALLER_BUNDLE_ANNOTATION = "gpu-fault.io/installer-bundle-sha256"
INSTALLER_TEMPLATE_ANNOTATION = "gpu-fault.io/installer-template-sha256"
INSTALLER_NODE_UID_ANNOTATION = "gpu-fault.io/installer-node-uid"
INSTALLER_STATE_ANNOTATION = "gpu-fault.io/installer-state"
# The boot the installation was recorded on. HyperPod UpdateClusterSoftware
# re-images a node in place: same Node object, same UID, every annotation kept,
# and an empty root where the Agent used to be. The UID check cannot see that;
# a boot-id change followed by an Agent that does not answer can.
INSTALLER_BOOT_ID_ANNOTATION = "gpu-fault.io/installer-boot-id"
# How many installer Jobs this node has already burned, and why the last one
# did not take. Both are read back off the Node on the next pass: the backoff
# has to survive a reconciler restart, and a restart is a routine event here
# (rollout, eviction, OOM), so it cannot live in process memory.
INSTALLER_ATTEMPTS_ANNOTATION = "gpu-fault.io/installer-attempts"
INSTALLER_REASON_ANNOTATION = "gpu-fault.io/installer-reason"
# The earliest time a node classified Unsupported may be given another Job.
# The classification deletes the Job to release the install slot at once, so
# without this the very next pass would recreate it and the slot would churn
# every 5 s instead of being held for 840 s.
INSTALLER_RETRY_AFTER_ANNOTATION = "gpu-fault.io/installer-retry-after"
#: Every node annotation this reconciler writes, in one place, because two
#: other things have to remove them all: cluster removal
#: (``gpu_fault.admin.cluster_removal``) and the clean-redeploy inventory
#: (``scripts/generate-cleanup-inventory.py``). Both used to list five of the
#: eleven by hand, so ``installer-attempts`` and ``installer-retry-after``
#: survived a removal and a re-added cluster inherited up to an hour of
#: backoff on its first failed install.
INSTALLER_NODE_ANNOTATIONS: tuple[str, ...] = (
    INSTALLER_VERSION_ANNOTATION,
    INSTALLER_DIGEST_ANNOTATION,
    INSTALLER_ARTIFACT_ANNOTATION,
    INSTALLER_BUNDLE_ANNOTATION,
    INSTALLER_TEMPLATE_ANNOTATION,
    INSTALLER_NODE_UID_ANNOTATION,
    INSTALLER_STATE_ANNOTATION,
    INSTALLER_BOOT_ID_ANNOTATION,
    INSTALLER_ATTEMPTS_ANNOTATION,
    INSTALLER_REASON_ANNOTATION,
    INSTALLER_RETRY_AFTER_ANNOTATION,
)
DEFAULT_AGENT_PORT = 9099
DEFAULT_REBOOT_GRACE_SECONDS = 600
INSTALLER_JOB_LABEL = "gpu-fault.io/node-installer"
# (connect, read) seconds for every apiserver call. Without it a half-open
# connection to the apiserver blocks the reconcile loop forever while the Pod
# stays Running and Ready.
REQUEST_TIMEOUT = (5, 60)
# Touched at the end of every reconcile pass; the Deployment's livenessProbe
# reads its age. A constant path, not an env var: the probe is a shell test in
# the manifest and the two must agree without a second knob to keep in sync.
RECONCILER_HEARTBEAT_PATH = "/tmp/reconciler-heartbeat"  # noqa: S108
# The maximum retry delay for a node that keeps failing to install.
MAX_RETRY_SECONDS = 3600
# Container waiting reasons that mean the installer never ran a single line, so
# there is nothing on the node to roll back and nothing a fast retry can fix.
# The first one is what a node with no key in ``gpu-fault-node-action-keys``
# produces: the reconciler has no ``secrets`` RBAC and cannot check the key
# itself, so it reads the consequence off the Job's Pod instead.
POD_NEVER_STARTED_REASONS = frozenset(
    {
        "CreateContainerConfigError",
        "CreateContainerError",
        "InvalidImageName",
        "ImagePullBackOff",
        "ErrImagePull",
        "RunContainerError",
    }
)
# How long a Pod may sit in one of those reasons before the reconciler calls it.
# Image pulls and container creation are not instant, and a verdict that fires
# on a transient reason would delete a Job that was about to start.
POD_NEVER_STARTED_GRACE_SECONDS = 120


def _value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _metadata(item: Any) -> Any:
    return _value(item, "metadata", {})


def _conditions(item: Any) -> list[Any]:
    return list(_value(_value(item, "status", {}), "conditions", []) or [])


def _is_ready(node: Any) -> bool:
    return any(
        _value(condition, "type") == "Ready"
        and str(_value(condition, "status")).lower() == "true"
        for condition in _conditions(node)
    )


def _boot_id(node: Any) -> str | None:
    info = _value(_value(node, "status"), "node_info")
    if info is None:
        info = _value(_value(node, "status"), "nodeInfo")
    value = _value(info, "boot_id") if info is not None else None
    if value is None and info is not None:
        value = _value(info, "bootID")
    return str(value) if value else None


def _as_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


def _transition_time(item: Any, condition_type: str) -> datetime | None:
    for condition in _conditions(item):
        if _value(condition, "type") != condition_type:
            continue
        value = _value(condition, "last_transition_time")
        if value is None:
            value = _value(condition, "lastTransitionTime")
        return _as_datetime(value)
    return None


def _ready_since(node: Any) -> datetime | None:
    return _transition_time(node, "Ready")


def _failure_time(job: Any) -> datetime | None:
    """When the Job actually failed, not when it was created.

    Retry backoff measured from the creation timestamp is no backoff at all for
    the failure that matters most: a Pod that never started burns the whole
    ``activeDeadlineSeconds`` (840 s) before the Job goes Failed, so every
    delay shorter than that has already elapsed by the time we look.
    """

    failed_at = _transition_time(job, "Failed")
    if failed_at is not None:
        return failed_at
    status = _value(job, "status", {})
    return _as_datetime(_value(status, "completion_time")) or _as_datetime(
        _value(status, "completionTime")
    )


def _attempt_count(annotations: dict[str, str]) -> int:
    try:
        return max(int(annotations.get(INSTALLER_ATTEMPTS_ANNOTATION, "0")), 0)
    except (TypeError, ValueError):
        return 0


def _annotations_match(current: dict[str, str], desired: dict[str, str | None]) -> bool:
    return all(
        (key not in current) if value is None else current.get(key) == value
        for key, value in desired.items()
    )


def _touch_heartbeat() -> None:
    """Prove the reconcile loop turned. Read by the Deployment's livenessProbe."""

    try:
        path = Path(RECONCILER_HEARTBEAT_PATH)
        path.touch(exist_ok=True)
        os.utime(path, None)
    except OSError:
        # Losing the heartbeat means liveness will restart this Pod, which is
        # the right answer for a host that cannot write /tmp. It must not take
        # the reconcile result down with it.
        LOGGER.warning(
            "could not write reconciler heartbeat %s",
            RECONCILER_HEARTBEAT_PATH,
            exc_info=True,
        )


class _InstallBudget:
    """The ``max_unavailable`` slots left in this pass.

    Consumed at the moment a Job is created rather than derived from the
    outcome the node loop returns, so a node whose annotation patch fails
    *after* its Job exists still spends the slot it took.
    """

    def __init__(self, used: int, limit: int) -> None:
        self.used = used
        self.limit = limit

    @property
    def allows_create(self) -> bool:
        return self.used < self.limit

    def consume(self) -> None:
        self.used += 1


def agent_answers(address: str, port: int, *, timeout_seconds: float = 2.0) -> bool:
    """Whether something listens on the Agent port.

    A TCP connect is enough: the Agent serves HTTPS with client certificates,
    so a plain HTTP health call would be refused even by a healthy Agent, and
    the question here is only whether the installed software is running at all.
    """

    try:
        with socket.create_connection((address, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False


def _job_condition(job: Any, condition_type: str) -> bool:
    return any(
        _value(condition, "type") == condition_type
        and str(_value(condition, "status")).lower() == "true"
        for condition in _conditions(job)
    )


def _api_status(error: Exception) -> int | None:
    return getattr(error, "status", None)


def _job_name(node_name: str, node_uid: str, digest: str) -> str:
    suffix = hashlib.sha256(f"{node_uid}:{digest}".encode()).hexdigest()[:10]
    safe_name = re.sub(r"[^a-z0-9-]+", "-", node_name.lower()).strip("-")
    prefix = f"gpu-fault-install-{safe_name}"[: 63 - len(suffix) - 1].rstrip("-")
    return f"{prefix}-{suffix}"


def _internal_ip(node: Any) -> str:
    addresses = _value(_value(node, "status", {}), "addresses", []) or []
    for address in addresses:
        if _value(address, "type") == "InternalIP":
            return str(_value(address, "address"))
    raise ValueError("node has no InternalIP")


class NodeInstallerReconciler:
    def __init__(
        self,
        core_api: Any,
        batch_api: Any,
        *,
        namespace: str,
        cluster_name: str,
        version: str,
        config_digest: str,
        artifact_sha256: str,
        bundle_sha256: str | None = None,
        template_sha256: str,
        job_template: dict[str, Any],
        dcgm_metrics_url_template: str,
        node_action_keys_secret: str = ("gpu-fault-node-action-keys"),
        retry_seconds: int = 300,
        max_unavailable: int = 1,
        job_active_deadline_seconds: int = 840,
        allowed_node_names: frozenset[str] | None = None,
        wave_config_map: str | None = None,
        now: Callable[[], datetime] | None = None,
        agent_port: int = DEFAULT_AGENT_PORT,
        reboot_grace_seconds: int = DEFAULT_REBOOT_GRACE_SECONDS,
        agent_alive: Callable[[str, int], bool] = agent_answers,
    ) -> None:
        self.core = core_api
        self.batch = batch_api
        self.namespace = namespace
        self.cluster_name = cluster_name
        self.version = version
        self.config_digest = config_digest
        if not re.fullmatch(r"[0-9a-f]{64}", artifact_sha256):
            raise ValueError(
                "installer artifact SHA-256 must be 64 lowercase hex characters"
            )
        self.artifact_sha256 = artifact_sha256
        self.bundle_sha256 = bundle_sha256 or artifact_sha256
        # The release's template-inputs digest, stamped on nodes and Jobs so a
        # template change re-installs. It is an identity pin, not an integrity
        # check of the text this process loaded -- that is
        # ``load_job_template`` -- so it has no default: a made-up value here
        # would let every node "match" a template nobody pinned.
        self.template_sha256 = template_sha256
        for name, digest in (
            ("bundle", self.bundle_sha256),
            ("template", self.template_sha256),
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(
                    f"installer {name} SHA-256 must be 64 lowercase hex characters"
                )
        self.job_template = job_template
        self.dcgm_metrics_url_template = dcgm_metrics_url_template
        self.node_action_keys_secret = node_action_keys_secret
        self.retry_seconds = retry_seconds
        if max_unavailable < 1:
            raise ValueError("installer max_unavailable must be positive")
        if job_active_deadline_seconds < 60:
            raise ValueError("installer active deadline must be at least 60 seconds")
        self.max_unavailable = max_unavailable
        self.job_active_deadline_seconds = job_active_deadline_seconds
        self.allowed_node_names = allowed_node_names
        self.wave_config_map = wave_config_map
        self.now = now or (lambda: datetime.now(UTC))
        self.agent_port = agent_port
        self.reboot_grace_seconds = reboot_grace_seconds
        self.agent_alive = agent_alive

    @property
    def node_selector(self) -> str:
        return f"sagemaker.amazonaws.com/cluster-name={self.cluster_name}"

    def reconcile_once(self) -> dict[str, int]:
        try:
            return self._reconcile_pass()
        finally:
            # In ``finally``, and not only on the happy path: liveness asks
            # whether the loop is turning, not whether reconciliation
            # succeeded. A pass that fails closed (an invalid wave ConfigMap)
            # is a live process an operator can fix; a wedged apiserver call is
            # not, and only the second one must trip the probe.
            _touch_heartbeat()

    def _reconcile_pass(self) -> dict[str, int]:
        result = {
            "current": 0,
            "created": 0,
            "running": 0,
            "succeeded": 0,
            "failed": 0,
            "deferred": 0,
            "not_ready": 0,
            "unsupported": 0,
            "rebooted": 0,
            "recovering": 0,
            "error": 0,
        }
        allowed_node_names, max_unavailable = self._wave_settings()
        response = self.core.list_node(
            label_selector=self.node_selector,
            _request_timeout=REQUEST_TIMEOUT,
        )
        nodes = sorted(
            (
                item
                for item in (_value(response, "items", []) or [])
                if allowed_node_names is None
                or str(_value(_metadata(item), "name")) in allowed_node_names
            ),
            key=lambda item: str(_value(_metadata(item), "name")),
        )
        # Counted from the apiserver before the loop, never from the outcomes
        # the loop produces: an in-loop counter lets a node that sorts *before*
        # an already-running install see zero in flight and start a second one,
        # so ``max_unavailable: 1`` took two nodes down at once.
        budget = _InstallBudget(self._active_installer_jobs(), max_unavailable)
        for node in nodes:
            node_name = str(_value(_metadata(node), "name"))
            try:
                outcome = self._reconcile_node(node, budget=budget)
            except Exception:
                # Per node, because the pass is a fleet-wide loop over a sorted
                # list: one node's 404 (deleted between LIST and patch), 5xx or
                # missing InternalIP used to skip every node after it in sort
                # order, every 5 s, for as long as the condition lasted.
                LOGGER.exception(
                    "node %s could not be reconciled; continuing with the rest",
                    node_name,
                )
                result["error"] += 1
                continue
            result[outcome] += 1
        return result

    def _active_installer_jobs(self) -> int:
        response = self.batch.list_namespaced_job(
            self.namespace,
            label_selector=f"{INSTALLER_JOB_LABEL}=true",
            _request_timeout=REQUEST_TIMEOUT,
        )
        active = 0
        for item in _value(response, "items", []) or []:
            if _job_condition(item, "Complete") or _job_condition(item, "Failed"):
                continue
            active += 1
        return active

    def _wave_settings(self) -> tuple[frozenset[str] | None, int]:
        if self.wave_config_map is None:
            return self.allowed_node_names, self.max_unavailable
        value = self.core.read_namespaced_config_map(
            self.wave_config_map,
            self.namespace,
            _request_timeout=REQUEST_TIMEOUT,
        )
        data = dict(_value(value, "data", {}) or {})
        raw_nodes = str(data.get("allowed-nodes") or "").strip()
        if raw_nodes == "*":
            allowed_nodes = None
        else:
            allowed_nodes = frozenset(
                item.strip() for item in raw_nodes.split(",") if item.strip()
            )
        try:
            max_unavailable = int(data.get("max-unavailable") or "")
        except (TypeError, ValueError) as exc:
            raise RuntimeError("installer wave max-unavailable is invalid") from exc
        if max_unavailable < 1:
            raise RuntimeError("installer wave max-unavailable must be positive")
        return allowed_nodes, max_unavailable

    def _reconcile_node(self, node: Any, *, budget: _InstallBudget) -> str:
        metadata = _metadata(node)
        node_name = str(_value(metadata, "name"))
        node_uid = str(_value(metadata, "uid"))
        annotations = dict(_value(metadata, "annotations", {}) or {})
        if not _is_ready(node):
            LOGGER.info("node %s is not Ready; installation deferred", node_name)
            return "not_ready"
        identity_matches = (
            annotations.get(INSTALLER_VERSION_ANNOTATION) == self.version
            and annotations.get(INSTALLER_DIGEST_ANNOTATION) == self.config_digest
            and annotations.get(INSTALLER_ARTIFACT_ANNOTATION) == self.artifact_sha256
            and annotations.get(INSTALLER_BUNDLE_ANNOTATION) == self.bundle_sha256
            and annotations.get(INSTALLER_TEMPLATE_ANNOTATION) == self.template_sha256
            and annotations.get(INSTALLER_NODE_UID_ANNOTATION) == node_uid
        )
        installer_state = annotations.get(INSTALLER_STATE_ANNOTATION)
        boot_id = _boot_id(node)
        if identity_matches and installer_state == "Succeeded":
            recorded_boot = annotations.get(INSTALLER_BOOT_ID_ANNOTATION)
            if boot_id is None or recorded_boot == boot_id:
                return "current"
            if recorded_boot is None:
                # Installed before boot ids were recorded: adopt this boot.
                self._mark_node(
                    node_name, node_uid, "Succeeded", boot_id, annotations=annotations
                )
                return "current"
            # The node booted since the installation was recorded. A plain
            # reboot keeps the software and the Agent comes back on its own; a
            # re-image (HyperPod UpdateClusterSoftware) keeps the Node object
            # and its annotations but not /opt/gpu-fault. Only the Agent can
            # tell the two apart.
            if self.agent_alive(_internal_ip(node), self.agent_port):
                self._mark_node(
                    node_name, node_uid, "Succeeded", boot_id, annotations=annotations
                )
                return "rebooted"
            ready_since = _ready_since(node)
            if (
                ready_since is not None
                and (self.now() - ready_since).total_seconds()
                < self.reboot_grace_seconds
            ):
                return "recovering"
            LOGGER.warning(
                "node %s rebooted (boot %s -> %s) and its Agent does not answer on "
                "port %s; reinstalling",
                node_name,
                recorded_boot,
                boot_id,
                self.agent_port,
            )
            self._mark_node(
                node_name, node_uid, "Retrying", boot_id, annotations=annotations
            )
            installer_state = "Retrying"

        name = _job_name(
            node_name,
            node_uid,
            (
                f"{self.config_digest}:{self.artifact_sha256}:"
                f"{self.bundle_sha256}:{self.template_sha256}"
            ),
        )
        try:
            job = self.batch.read_namespaced_job(
                name, self.namespace, _request_timeout=REQUEST_TIMEOUT
            )
        except Exception as error:
            if _api_status(error) != 404:
                raise
            if identity_matches:
                # A node classified Unsupported holds no Job any more, so this
                # annotation is the only thing keeping the create path from
                # handing it a fresh one on the very next pass. Ignored once the
                # identity has moved on: a new release may be the fix.
                retry_after = _as_datetime(
                    annotations.get(INSTALLER_RETRY_AFTER_ANNOTATION)
                )
                if retry_after is not None and self.now() < retry_after:
                    return "unsupported"
            if not budget.allows_create:
                return "deferred"
            try:
                body = self._build_job(node, name)
            except ValueError as build_error:
                LOGGER.error("node %s cannot be installed: %s", node_name, build_error)
                return "unsupported"
            try:
                self.batch.create_namespaced_job(
                    self.namespace, body, _request_timeout=REQUEST_TIMEOUT
                )
            except Exception as create_error:
                if _api_status(create_error) != 409:
                    raise
            # Before the annotation patch: the slot is spent the moment the Job
            # exists, whether or not this node's patch lands.
            budget.consume()
            self._mark_node(
                node_name, node_uid, "Installing", boot_id, annotations=annotations
            )
            LOGGER.info("created installer Job %s for node %s", name, node_name)
            return "created"

        if _job_condition(job, "Complete"):
            if not identity_matches or installer_state == "Retrying":
                self._delete_job(name)
                self._mark_node(
                    node_name, node_uid, "Retrying", boot_id, annotations=annotations
                )
                LOGGER.warning(
                    "deleted stale completed installer Job %s for replay",
                    name,
                )
                return "running"
            self._mark_node(
                node_name, node_uid, "Succeeded", boot_id, annotations=annotations
            )
            return "succeeded"
        if _job_condition(job, "Failed"):
            attempts = _attempt_count(annotations)
            if self._retry_due(job, attempts):
                self._delete_job(name)
                self._mark_node(
                    node_name,
                    node_uid,
                    "Retrying",
                    boot_id,
                    annotations=annotations,
                    attempts=attempts + 1,
                )
                LOGGER.warning(
                    "deleted failed installer Job %s for retry (attempt %s)",
                    name,
                    attempts + 1,
                )
                return "failed"
            # No Pod lookup here: the Job controller deletes the active Pod in
            # the same action that records DeadlineExceeded, so a Failed Job has
            # nothing left to read -- and the number of simultaneously Failed
            # nodes is not bounded by max_unavailable.
            self._mark_node(
                node_name, node_uid, "Failed", boot_id, annotations=annotations
            )
            return "failed"

        never_started = self._never_started_reason(job, name)
        if never_started is not None:
            # Nothing ran on the node, and nothing will: backoffLimit is 0 and
            # restartPolicy is Never, so this Pod sits here until
            # activeDeadlineSeconds kills it -- 840 s of the fleet's install
            # budget spent on a node that cannot install (most often a node with
            # no key in the node-action Secret, which this identity cannot
            # read). Release the slot now, report why, and back off.
            attempts = _attempt_count(annotations)
            delay = self._retry_delay_seconds(attempts)
            self._delete_job(name)
            LOGGER.error(
                "node %s installer Pod never started (%s); next attempt in %ss",
                node_name,
                never_started,
                delay,
            )
            self._mark_node(
                node_name,
                node_uid,
                "Unsupported",
                boot_id,
                annotations=annotations,
                attempts=attempts + 1,
                reason=never_started,
                retry_after=self.now() + timedelta(seconds=delay),
            )
            return "unsupported"
        return "running"

    def _delete_job(self, name: str) -> None:
        self.batch.delete_namespaced_job(
            name,
            self.namespace,
            propagation_policy="Background",
            _request_timeout=REQUEST_TIMEOUT,
        )

    def _retry_delay_seconds(self, attempts: int) -> int:
        """``retry_seconds`` doubled per recorded attempt, capped at one hour."""

        exponent: int = min(max(attempts, 0), 8)
        delay: int = self.retry_seconds * int(2**exponent)
        return min(delay, MAX_RETRY_SECONDS)

    def _retry_due(self, job: Any, attempts: int = 0) -> bool:
        reference = _failure_time(job)
        if reference is None:
            created = _value(_metadata(job), "creation_timestamp")
            reference = _as_datetime(created)
        if reference is None:
            # No Failed condition, no completion time, no creation timestamp:
            # nothing says the delay has elapsed, so treat it as just failed.
            # Returning True here retried such a Job on every 5 s pass.
            return False
        elapsed = (self.now() - reference).total_seconds()
        return elapsed >= self._retry_delay_seconds(attempts)

    def _never_started_reason(self, job: Any, job_name: str) -> str | None:
        """The waiting reason of an *active* Job whose Pod never ran, if any.

        Only an active Job can answer this. The installer Job is
        ``backoffLimit: 0`` + ``restartPolicy: Never``, so a Pod stuck in
        CreateContainerConfigError never fails the Job; it is killed by
        activeDeadlineSeconds, and the Job controller deletes the Pod in the
        same action. Asking after the Job is Failed always reads back an empty
        list.

        The grace period is checked before the LIST, so a healthy slow start
        (image pull, container creation) costs no extra apiserver call at all.

        Diagnostics: a reconciler without ``pods`` read access simply learns
        nothing here and keeps the plain backoff, so this can never make a
        failure look like a success.
        """

        status = _value(job, "status", {})
        started = _as_datetime(_value(status, "start_time"))
        if started is None:
            started = _as_datetime(_value(status, "startTime"))
        if started is None:
            started = _as_datetime(_value(_metadata(job), "creation_timestamp"))
        if started is None:
            return None
        if (self.now() - started).total_seconds() < POD_NEVER_STARTED_GRACE_SECONDS:
            return None
        try:
            response = self.core.list_namespaced_pod(
                self.namespace,
                label_selector=f"job-name={job_name}",
                _request_timeout=REQUEST_TIMEOUT,
            )
        except Exception:
            LOGGER.warning(
                "could not read the Pods of installer Job %s", job_name, exc_info=True
            )
            return None
        for pod in _value(response, "items", []) or []:
            status = _value(pod, "status", {})
            for field in (
                "init_container_statuses",
                "initContainerStatuses",
                "container_statuses",
                "containerStatuses",
            ):
                for container in _value(status, field, []) or []:
                    waiting = _value(_value(container, "state", {}), "waiting")
                    if waiting is None:
                        continue
                    reason = str(_value(waiting, "reason") or "")
                    if reason not in POD_NEVER_STARTED_REASONS:
                        continue
                    message = str(_value(waiting, "message") or "").strip()
                    return f"{reason}: {message}"[:512] if message else reason
        return None

    def _mark_node(
        self,
        node_name: str,
        node_uid: str,
        state: str,
        boot_id: str | None = None,
        *,
        annotations: dict[str, str],
        attempts: int | None = None,
        reason: str | None = None,
        retry_after: datetime | None = None,
    ) -> None:
        """Patch the Node's installer annotations unless they already say this.

        ``annotations`` is this pass's view of the Node and is updated in place
        on success, so a node parked in ``Failed`` waiting out its backoff is
        not rewritten with identical content every 5 s.
        """

        if state == "Succeeded":
            target_attempts = 0
        elif attempts is None:
            target_attempts = _attempt_count(annotations)
        else:
            target_attempts = max(attempts, 0)
        desired: dict[str, str | None] = {
            INSTALLER_VERSION_ANNOTATION: self.version,
            INSTALLER_DIGEST_ANNOTATION: self.config_digest,
            INSTALLER_ARTIFACT_ANNOTATION: self.artifact_sha256,
            INSTALLER_BUNDLE_ANNOTATION: self.bundle_sha256,
            INSTALLER_TEMPLATE_ANNOTATION: self.template_sha256,
            INSTALLER_NODE_UID_ANNOTATION: node_uid,
            INSTALLER_STATE_ANNOTATION: state,
        }
        if boot_id is not None:
            desired[INSTALLER_BOOT_ID_ANNOTATION] = boot_id
        if target_attempts > 0:
            desired[INSTALLER_ATTEMPTS_ANNOTATION] = str(target_attempts)
        elif INSTALLER_ATTEMPTS_ANNOTATION in annotations:
            desired[INSTALLER_ATTEMPTS_ANNOTATION] = None
        if reason:
            desired[INSTALLER_REASON_ANNOTATION] = reason
        elif INSTALLER_REASON_ANNOTATION in annotations:
            desired[INSTALLER_REASON_ANNOTATION] = None
        if retry_after is not None:
            desired[INSTALLER_RETRY_AFTER_ANNOTATION] = (
                retry_after.astimezone(UTC).isoformat().replace("+00:00", "Z")
            )
        elif INSTALLER_RETRY_AFTER_ANNOTATION in annotations:
            desired[INSTALLER_RETRY_AFTER_ANNOTATION] = None
        if _annotations_match(annotations, desired):
            return
        self.core.patch_node(
            node_name,
            {"metadata": {"annotations": desired}},
            _request_timeout=REQUEST_TIMEOUT,
        )
        for key, value in desired.items():
            if value is None:
                annotations.pop(key, None)
            else:
                annotations[key] = value

    def _build_job(self, node: Any, job_name: str) -> dict[str, Any]:
        body = copy.deepcopy(self.job_template)
        metadata = _metadata(node)
        node_name = str(_value(metadata, "name"))
        node_uid = str(_value(metadata, "uid"))
        labels = dict(_value(metadata, "labels", {}) or {})
        instance_type = labels.get("node.kubernetes.io/instance-type", "")
        expected_gpus, expected_efa = gpu_instance_inventory(instance_type)
        node_ip = _internal_ip(node)
        metrics_url = self.dcgm_metrics_url_template.replace(
            "{node_name}", node_name
        ).replace("{node_ip}", node_ip)

        body.setdefault("metadata", {})["name"] = job_name
        body["metadata"]["namespace"] = self.namespace
        body["metadata"]["annotations"] = {
            INSTALLER_DIGEST_ANNOTATION: self.config_digest,
            INSTALLER_ARTIFACT_ANNOTATION: self.artifact_sha256,
            INSTALLER_BUNDLE_ANNOTATION: self.bundle_sha256,
            INSTALLER_TEMPLATE_ANNOTATION: self.template_sha256,
        }
        body["metadata"]["labels"] = {
            INSTALLER_JOB_LABEL: "true",
            "gpu-fault.io/node-uid": node_uid,
            "gpu-fault.io/config-digest": hashlib.sha256(
                self.config_digest.encode()
            ).hexdigest()[:16],
        }
        body["spec"]["activeDeadlineSeconds"] = self.job_active_deadline_seconds
        pod_template = body["spec"]["template"]
        pod_template.setdefault("metadata", {})["labels"] = dict(
            body["metadata"]["labels"]
        )
        pod_spec = pod_template["spec"]
        pod_spec["nodeName"] = node_name
        node_secret = next(
            (
                item
                for item in pod_spec.get("volumes", [])
                if item.get("name") == "node-secret"
            ),
            None,
        )
        if node_secret is None:
            raise ValueError("installer template has no node-secret volume")
        node_secret["secret"] = {
            "secretName": self.node_action_keys_secret,
            "items": [
                {
                    "key": node_name,
                    "path": "node-action-secret",
                }
            ],
        }
        container = next(
            item for item in pod_spec["containers"] if item["name"] == "installer"
        )
        updates = {
            "TARGET_NODE_NAME": node_name,
            "TARGET_NODE_IP": node_ip,
            "TARGET_NODE_UID": node_uid,
            "NODE_INSTANCE_TYPE": instance_type,
            "EXPECTED_GPU_COUNT": str(expected_gpus),
            "EXPECTED_EFA_DEVICE_COUNT": str(expected_efa),
            "DCGM_METRICS_URL_B64": base64.b64encode(metrics_url.encode()).decode(),
            # The exporter on this node is our DaemonSet, whose period the
            # installer cannot see from the host: without it `--dcgm-exporter
            # existing` leaves the collector's exporter-interval seconds unset
            # and its duty-cycle check never runs in production.
            "DCGM_EXPORTER_INTERVAL_MS": str(DCGM_EXPORTER_COLLECT_INTERVAL_MS),
        }
        unfilled = dict(updates)
        for env in container.get("env", []):
            name = env.get("name")
            if name in unfilled:
                env.clear()
                env.update(name=name, value=unfilled.pop(name))
        if unfilled:
            # The template is pinned by digest, so an entry it lacks is a deploy
            # defect, not a per-node condition: a Job created anyway would run
            # the installer without that value and install the wrong thing
            # silently (the exporter period, say, leaves the collector's
            # duty-cycle check off). Not a ValueError -- the caller reads that
            # as "this node is unsupported" and moves on to the next one.
            raise RuntimeError(
                "installer job template has no env entry for "
                + ", ".join(sorted(unfilled))
                + "; the pinned template predates the reconciler filling it"
            )
        return body


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


TEMPLATE_CONTENT_SHA256_ENV = "GPU_FAULT_INSTALLER_TEMPLATE_CONTENT_SHA256"


def load_job_template(
    template_data: bytes | str,
    *,
    expected_sha256: str | None,
    origin: str,
) -> dict[str, Any]:
    """Parse the installer Job template only if it is the one the deploy pinned.

    The template arrives from a ConfigMap in the reconciler's namespace, and
    whatever it says becomes a privileged hostPath Pod on every GPU node with
    that node's HMAC key mounted. Anyone who can update the ConfigMap would
    therefore own the fleet -- unless the deploy also pins the exact bytes it
    put there. ``expected_sha256`` is that pin
    (:data:`TEMPLATE_CONTENT_SHA256_ENV`, the SHA-256 of the ``job.yaml`` text
    the deploy script placed in the ConfigMap). No pin, or a pin that does not
    match, is fatal at startup: the reconciler must not fall back to trusting
    the cluster's copy.
    """

    raw = template_data if isinstance(template_data, bytes) else template_data.encode()
    if not raw.strip():
        raise RuntimeError(f"{origin} is empty")
    expected = (expected_sha256 or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise RuntimeError(
            f"{TEMPLATE_CONTENT_SHA256_ENV} is required and must be the SHA-256 "
            f"of the installer template text ({origin})"
        )
    actual = hashlib.sha256(raw).hexdigest()
    if not hmac.compare_digest(actual, expected):
        raise RuntimeError(
            f"installer template {origin} does not match "
            f"{TEMPLATE_CONTENT_SHA256_ENV}: expected {expected[:12]}..., "
            f"loaded {actual[:12]}...; refusing to start"
        )
    template = yaml.safe_load(raw)
    if not isinstance(template, dict):
        raise RuntimeError(f"installer template {origin} is not a Job document")
    return template


def main() -> None:
    from kubernetes import client, config

    configure_logging()
    namespace = os.environ.get("GPU_FAULT_NAMESPACE", "gpu-fault-system")
    template_name = os.environ.get(
        "GPU_FAULT_INSTALLER_TEMPLATE_CONFIG_MAP",
        "gpu-fault-node-installer-template",
    )
    poll_seconds = int(os.environ.get("GPU_FAULT_RECONCILE_SECONDS", "5"))
    retry_seconds = int(os.environ.get("GPU_FAULT_INSTALL_RETRY_SECONDS", "300"))
    max_unavailable = int(os.environ.get("GPU_FAULT_INSTALLER_MAX_UNAVAILABLE", "1"))
    active_deadline_seconds = int(
        os.environ.get("GPU_FAULT_INSTALLER_ACTIVE_DEADLINE_SECONDS", "840")
    )
    allowed_nodes_value = os.environ.get(
        "GPU_FAULT_INSTALLER_ALLOWED_NODES",
        "*",
    ).strip()
    allowed_nodes = (
        None
        if allowed_nodes_value == "*"
        else frozenset(
            item.strip() for item in allowed_nodes_value.split(",") if item.strip()
        )
    )
    config.load_incluster_config()
    core = client.CoreV1Api()
    batch = client.BatchV1Api()
    template_path = os.environ.get("GPU_FAULT_INSTALLER_TEMPLATE_PATH", "").strip()
    template_data: bytes | str
    if template_path:
        # Bytes, not text: the pin is over the exact file the deploy wrote, and
        # read_text() would fold line endings before hashing.
        template_data = Path(template_path).read_bytes()
        origin = template_path
    else:
        config_map = core.read_namespaced_config_map(
            template_name, namespace, _request_timeout=REQUEST_TIMEOUT
        )
        template_data = (config_map.data or {}).get("job.yaml", "")
        origin = f"{template_name}/job.yaml"
    job_template = load_job_template(
        template_data,
        expected_sha256=os.environ.get(TEMPLATE_CONTENT_SHA256_ENV),
        origin=origin,
    )
    reconciler = NodeInstallerReconciler(
        core,
        batch,
        namespace=namespace,
        cluster_name=_required_env("GPU_FAULT_HYPERPOD_CLUSTER"),
        version=_required_env("GPU_FAULT_INSTALLER_VERSION"),
        config_digest=_required_env("GPU_FAULT_INSTALLER_CONFIG_DIGEST"),
        artifact_sha256=_required_env("GPU_FAULT_INSTALLER_ARTIFACT_SHA256"),
        bundle_sha256=os.getenv("GPU_FAULT_INSTALLER_BUNDLE_SHA256") or None,
        template_sha256=_required_env("GPU_FAULT_INSTALLER_TEMPLATE_SHA256"),
        job_template=job_template,
        dcgm_metrics_url_template=os.environ.get(
            "GPU_FAULT_DCGM_METRICS_URL",
            "http://127.0.0.1:9400/metrics",
        ),
        node_action_keys_secret=os.environ.get(
            "GPU_FAULT_NODE_ACTION_KEYS_SECRET",
            "gpu-fault-node-action-keys",
        ),
        retry_seconds=retry_seconds,
        max_unavailable=max_unavailable,
        job_active_deadline_seconds=active_deadline_seconds,
        allowed_node_names=allowed_nodes,
        wave_config_map=(os.environ.get("GPU_FAULT_INSTALLER_WAVE_CONFIG_MAP") or None),
        agent_port=int(os.environ.get("GPU_FAULT_NODE_AGENT_PORT", "9099")),
        reboot_grace_seconds=int(
            os.environ.get("GPU_FAULT_INSTALLER_REBOOT_GRACE_SECONDS", "600")
        ),
    )
    while True:
        try:
            LOGGER.info("node installer reconcile: %s", reconciler.reconcile_once())
        except Exception:
            LOGGER.exception("node installer reconcile failed")
        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
