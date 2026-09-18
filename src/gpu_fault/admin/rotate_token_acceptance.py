"""Prove a quiet token window against stable current-log prefixes."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Callable

from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.diagnostics import diagnostic_text
from gpu_fault.admin.execution import (
    OVERALL_DEPLOY_SECONDS,
    deadline_scope,
    remaining_timeout,
    run_command,
)
from gpu_fault.admin.site import RenderedSite
from gpu_fault_release.regional_deployment_inventory import CPU_INGRESS_DEPLOYMENT

CPU_INGRESS_LABEL = f"app={CPU_INGRESS_DEPLOYMENT}"
RETIRING_TOKEN_LOG_FRAGMENT = "authenticated with the retiring token"
API_CONTAINER = "api"
LOG_PREFIX_BYTES = 4096
RunCommand = Callable[..., subprocess.CompletedProcess[str]]
ReadRetiringLogs = Callable[..., list[str]]
_TIMESTAMP = re.compile(
    r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?(Z|[+-]\d{2}:\d{2})"
)
_POD_NAME = re.compile(r"[a-z0-9](?:[-a-z0-9.]{0,251}[a-z0-9])?")
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class LogSource:
    pod_uid: str
    pod_name: str
    container_id: str
    started_at: str
    restart_count: int
    replicaset_uid: str


@dataclass(frozen=True)
class IngressSourceProof:
    deployment_uid: str
    deployment_generation: int
    desired_replicas: int
    sources: dict[str, LogSource]


@dataclass(frozen=True)
class PrefixReceipt:
    length_bytes: int
    sha256: str
    first_timestamp_ns: int


def _timestamp_ns(value: str) -> int:
    match = _TIMESTAMP.fullmatch(value)
    if match is None:
        raise BootstrapError("control-plane logs contain an invalid timestamp")
    try:
        seconds = datetime.fromisoformat(match[1] + match[3]).astimezone(UTC)
    except ValueError:
        raise BootstrapError(
            "control-plane logs contain an invalid timestamp"
        ) from None
    elapsed = seconds - _EPOCH
    return (elapsed.days * 86400 + elapsed.seconds) * 1_000_000_000 + int(
        (match[2] or "").ljust(9, "0")
    )


def _cpu_kubectl(site: RenderedSite) -> list[str]:
    return [
        "kubectl",
        "--kubeconfig",
        str(site.release_config["cpu_kubeconfig"]),
        "-n",
        str(site.release_config["namespace"]),
    ]


def _source_resource(
    kubectl: list[str], arguments: list[str], *, run: RunCommand
) -> dict[str, Any]:
    completed = run(
        [*kubectl, "get", *arguments, "-o", "json"],
        timeout_seconds=120,
    )
    if completed.returncode:
        raise BootstrapError("cannot verify control-plane ingress log sources")
    try:
        document = json.loads(completed.stdout)
        if not isinstance(document, dict):
            raise ValueError("invalid source document")
    except ValueError:
        raise BootstrapError(
            "cannot verify control-plane ingress log sources"
        ) from None
    return document


def _controller_owner(metadata: dict[str, Any], kind: str) -> tuple[str, str]:
    owners = [
        owner
        for owner in metadata["ownerReferences"]
        if owner.get("controller") is True
    ]
    if len(owners) != 1:
        raise ValueError("source has no unique controller")
    owner = owners[0]
    uid, name = owner["uid"], owner["name"]
    if (
        owner.get("apiVersion") != "apps/v1"
        or owner.get("kind") != kind
        or not isinstance(uid, str)
        or not uid
        or not isinstance(name, str)
        or _POD_NAME.fullmatch(name) is None
    ):
        raise ValueError("source controller identity is incomplete")
    return uid, name


def _ingress_deployment(
    kubectl: list[str],
    *,
    namespace: str,
    cutoff_ns: int,
    run: RunCommand,
) -> tuple[str, int, int]:
    deployment = _source_resource(
        kubectl, ["deployment", CPU_INGRESS_DEPLOYMENT], run=run
    )
    metadata, spec, status = (
        deployment["metadata"],
        deployment["spec"],
        deployment["status"],
    )
    uid = metadata["uid"]
    generation, desired = metadata["generation"], spec["replicas"]
    if (
        deployment.get("apiVersion") != "apps/v1"
        or deployment.get("kind") != "Deployment"
        or metadata.get("name") != CPU_INGRESS_DEPLOYMENT
        or metadata.get("namespace") != namespace
        or not isinstance(uid, str)
        or not uid
        or metadata.get("deletionTimestamp")
        or _timestamp_ns(metadata["creationTimestamp"]) > cutoff_ns
        or type(generation) is not int
        or generation < 1
        or type(desired) is not int
        or desired < 1
        or spec.get("paused", False) is not False
        or spec["selector"].get("matchLabels") != {"app": CPU_INGRESS_DEPLOYMENT}
        or spec["selector"].get("matchExpressions", []) != []
        or type(status.get("observedGeneration")) is not int
        or status["observedGeneration"] != generation
        or any(
            type(status.get(field)) is not int or status[field] != desired
            for field in (
                "replicas",
                "updatedReplicas",
                "readyReplicas",
                "availableReplicas",
            )
        )
        or any(
            type(status.get(field, 0)) is not int or status.get(field, 0) != 0
            for field in ("unavailableReplicas", "terminatingReplicas")
        )
    ):
        raise ValueError("managed ingress Deployment replicas are not healthy")
    return uid, generation, desired


def _ingress_log_sources(
    site: RenderedSite,
    *,
    quiet_start: datetime,
    run: RunCommand,
) -> IngressSourceProof:
    kubectl = _cpu_kubectl(site)
    namespace = str(site.release_config["namespace"])
    try:
        cutoff_ns = _timestamp_ns(quiet_start.isoformat())
        identity = _ingress_deployment(
            kubectl, namespace=namespace, cutoff_ns=cutoff_ns, run=run
        )
        deployment_uid, generation, desired = identity
        replicasets = _source_resource(
            kubectl, ["replicasets", "-l", CPU_INGRESS_LABEL], run=run
        )
        if not isinstance(replicasets["items"], list) or (
            replicasets.get("metadata") or {}
        ).get("continue"):
            raise ValueError("incomplete ReplicaSet list")
        owners: dict[str, str] = {}
        for replicaset in replicasets["items"]:
            metadata = replicaset["metadata"]
            uid, name = metadata["uid"], metadata["name"]
            if (
                replicaset.get("apiVersion") != "apps/v1"
                or replicaset.get("kind") != "ReplicaSet"
                or metadata.get("namespace") != namespace
                or not isinstance(uid, str)
                or not uid
                or uid in owners
                or not isinstance(name, str)
                or _POD_NAME.fullmatch(name) is None
                or name in owners.values()
                or metadata.get("deletionTimestamp")
                or _controller_owner(metadata, "Deployment")
                != (deployment_uid, CPU_INGRESS_DEPLOYMENT)
            ):
                raise ValueError("ReplicaSet is not owned by the ingress Deployment")
            owners[uid] = name
        document = _source_resource(kubectl, ["pods", "-l", CPU_INGRESS_LABEL], run=run)
        items = document["items"]
        if (
            not isinstance(items, list)
            or len(items) != desired
            or (document.get("metadata") or {}).get("continue")
        ):
            raise ValueError("ingress pods do not cover all desired replicas")
        sources: dict[str, LogSource] = {}
        names: set[str] = set()
        for pod in items:
            metadata, status = pod["metadata"], pod["status"]
            uid, name = metadata["uid"], metadata["name"]
            owner_uid, owner_name = _controller_owner(metadata, "ReplicaSet")
            if (
                pod.get("apiVersion") != "v1"
                or pod.get("kind") != "Pod"
                or metadata.get("namespace") != namespace
                or (metadata.get("labels") or {}).get("app") != CPU_INGRESS_DEPLOYMENT
                or owners.get(owner_uid) != owner_name
                or not isinstance(uid, str)
                or not uid
                or uid in sources
                or not isinstance(name, str)
                or _POD_NAME.fullmatch(name) is None
                or name in names
                or metadata.get("deletionTimestamp")
                or status.get("phase") != "Running"
                or not any(
                    condition.get("type") == "Ready"
                    and condition.get("status") == "True"
                    for condition in status.get("conditions") or []
                )
            ):
                raise ValueError("ingress pod is not stably Ready")
            containers = [
                item
                for item in status["containerStatuses"]
                if item["name"] == API_CONTAINER
            ]
            declared = [
                item
                for item in pod["spec"]["containers"]
                if item["name"] == API_CONTAINER
            ]
            if len(containers) != 1 or len(declared) != 1:
                raise ValueError("container log identity is missing")
            container = containers[0]
            started = str(container["state"]["running"]["startedAt"])
            container_id = container["containerID"]
            if (
                container.get("ready") is not True
                or type(container.get("restartCount")) is not int
                or container["restartCount"] < 0
                or not isinstance(container_id, str)
                or not container_id
                or _timestamp_ns(started) > cutoff_ns
            ):
                raise ValueError("container logs do not cover the quiet window")
            names.add(name)
            sources[uid] = LogSource(
                uid, name, container_id, started, container["restartCount"], owner_uid
            )
        if identity != _ingress_deployment(
            kubectl, namespace=namespace, cutoff_ns=cutoff_ns, run=run
        ):
            raise BootstrapError(
                "control-plane ingress log sources changed during the quiet probe"
            )
    except (AttributeError, KeyError, TypeError, ValueError):
        raise BootstrapError(
            "control-plane ingress logs lack a complete healthy replica set "
            "with a stable Ready source for the full quiet window"
        ) from None
    return IngressSourceProof(deployment_uid, generation, desired, sources)


def _read_logs(
    kubectl: list[str],
    source: LogSource,
    *,
    options: list[str],
    run: RunCommand,
) -> bytes:
    completed = run(
        [
            *kubectl,
            "logs",
            source.pod_name,
            f"--container={API_CONTAINER}",
            "--prefix",
            "--timestamps",
            "--tail=-1",
            "--max-log-requests=20",
            *options,
        ],
        timeout_seconds=180,
    )
    if completed.returncode:
        raise BootstrapError(
            "cannot read control-plane ingress logs: "
            + (
                diagnostic_text(completed.stderr, sensitive=True)
                or "kubectl logs failed"
            )
        )
    prefix = f"[pod/{source.pod_name}/{API_CONTAINER}] "
    chunks = completed.stdout.split("\n")
    payload = []
    for index, chunk in enumerate(chunks):
        if index == len(chunks) - 1 and not chunk:
            continue
        if not chunk.startswith(prefix):
            raise BootstrapError(
                "control-plane log response has an unexpected source prefix"
            )
        payload.append(chunk[len(prefix) :] + ("\n" if index < len(chunks) - 1 else ""))
    return "".join(payload).encode("utf-8")


def _log_entries(payload: bytes) -> list[tuple[int, str]]:
    if payload and not payload.endswith(b"\n"):
        raise BootstrapError("control-plane log response is incomplete")
    try:
        lines = payload.decode("utf-8").split("\n")[:-1]
    except UnicodeDecodeError:
        raise BootstrapError("control-plane logs are not valid UTF-8") from None
    entries = []
    for line in lines:
        timestamp, separator, _message = line.partition(" ")
        if not separator:
            raise BootstrapError("control-plane log record has no timestamp")
        entries.append((_timestamp_ns(timestamp), line))
    return entries


def _prefix_receipt(
    kubectl: list[str],
    source: LogSource,
    *,
    quiet_start_ns: int,
    run: RunCommand,
    expected: PrefixReceipt | None = None,
) -> PrefixReceipt:
    # CRI ReadLogs applies limitBytes after tail selection. Explicit -1 starts
    # at offset zero; kubectl adds source prefixes outside that byte cap.
    limit = expected.length_bytes if expected is not None else LOG_PREFIX_BYTES
    payload = _read_logs(kubectl, source, options=[f"--limit-bytes={limit}"], run=run)
    if not payload or len(payload) > limit:
        raise BootstrapError(
            "control-plane log prefix is missing or exceeds its byte bound"
        )
    if len(payload) < limit and not payload.endswith(b"\n"):
        raise BootstrapError(
            "control-plane log prefix is incomplete below its byte bound"
        )
    complete = payload[: payload.rfind(b"\n") + 1]
    entries = _log_entries(complete)
    if not entries:
        raise BootstrapError(
            "control-plane log prefix has no complete timestamped record"
        )
    if entries[0][0] > quiet_start_ns:
        raise BootstrapError(
            "control-plane current log prefix starts after the quiet window"
        )
    receipt = PrefixReceipt(
        len(complete), hashlib.sha256(complete).hexdigest(), entries[0][0]
    )
    if expected is not None and (
        len(payload) != limit or complete != payload or receipt != expected
    ):
        raise BootstrapError("control-plane log prefix changed during the quiet probe")
    return receipt


def collect_retiring_token_authentications(
    site: RenderedSite,
    cluster_id: str,
    *,
    since_seconds: int,
    run: RunCommand,
    now: Callable[[], datetime],
) -> list[str]:
    if since_seconds < 1:
        raise BootstrapError("rotate-token quiet period must be positive")
    stamp = now()
    if stamp.utcoffset() is None:
        raise BootstrapError(
            "rotate-token quiet window requires a timezone-aware clock"
        )
    quiet_start = stamp.astimezone(UTC) - timedelta(seconds=since_seconds)
    quiet_start_ns = _timestamp_ns(quiet_start.isoformat())
    kubectl = _cpu_kubectl(site)
    proof = _ingress_log_sources(site, quiet_start=quiet_start, run=run)
    sources = proof.sources
    receipts = {
        uid: _prefix_receipt(kubectl, source, quiet_start_ns=quiet_start_ns, run=run)
        for uid, source in sorted(sources.items())
    }
    # PodLogOptions may serialize SinceTime at whole-second precision. Request
    # the earlier whole second and enforce the exact boundary on auth records.
    since_time = quiet_start.isoformat(timespec="seconds").replace("+00:00", "Z")
    matches: list[str] = []
    needle = f"regional cluster {cluster_id} {RETIRING_TOKEN_LOG_FRAGMENT}"
    for source in sources.values():
        payload = _read_logs(
            kubectl, source, options=[f"--since-time={since_time}"], run=run
        )
        read_finished_ns = _timestamp_ns(now().isoformat())
        for timestamp_ns, line in _log_entries(payload):
            if timestamp_ns > read_finished_ns:
                raise BootstrapError("control-plane logs contain a future timestamp")
            if timestamp_ns >= quiet_start_ns and needle in line:
                matches.append(f"[pod/{source.pod_name}/{API_CONTAINER}] {line}")
    for uid, source in sources.items():
        _prefix_receipt(
            kubectl,
            source,
            quiet_start_ns=quiet_start_ns,
            run=run,
            expected=receipts[uid],
        )
    if proof != _ingress_log_sources(site, quiet_start=quiet_start, run=run):
        raise BootstrapError(
            "control-plane ingress log sources changed during the quiet probe"
        )
    return matches


def wait_for_quiet_token_authentications(
    site: RenderedSite,
    cluster_id: str,
    *,
    quiet_seconds: int,
    timeout_seconds: int,
    read_logs: ReadRetiringLogs,
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
    run: RunCommand | None = None,
    now: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    if not 0 < quiet_seconds < timeout_seconds:
        raise BootstrapError(
            "rotate-token quiet period must be shorter than the timeout"
        )
    started = monotonic()
    deadline = started + timeout_seconds
    lines: list[str] = []
    with deadline_scope(
        "rotate-token acceptance", min(float(timeout_seconds), OVERALL_DEPLOY_SECONDS)
    ):
        run = run or run_command
        quiet_start = now() if now is not None else datetime.now(UTC)
        if quiet_start.utcoffset() is None:
            raise BootstrapError(
                "rotate-token quiet window requires a timezone-aware clock"
            )
        # Pin before sleeping: a scale-down during the quiet window must not
        # make a missing replica's retained authentications disappear.
        proof = _ingress_log_sources(site, quiet_start=quiet_start, run=run)
        sleep(remaining_timeout(float(quiet_seconds)))
        while monotonic() < deadline:
            remaining = deadline - monotonic()
            with deadline_scope(
                "rotate-token retiring-token probe",
                min(remaining, OVERALL_DEPLOY_SECONDS),
            ):
                lines = read_logs(site, cluster_id, since_seconds=quiet_seconds)
                if proof != _ingress_log_sources(
                    site, quiet_start=quiet_start, run=run
                ):
                    raise BootstrapError(
                        "control-plane ingress log sources changed during acceptance"
                    )
                remaining_timeout(1)
            if monotonic() >= deadline:
                break
            if not lines:
                return {
                    "quiet_seconds": quiet_seconds,
                    "waited_seconds": round(monotonic() - started, 1),
                }
            sleep(
                remaining_timeout(
                    min(30.0, float(quiet_seconds), deadline - monotonic())
                )
            )
    raise BootstrapError(
        f"rotate-token acceptance timed out after {timeout_seconds}s; "
        + (
            f"{len(lines)} retiring-token authentications for {cluster_id}; "
            "an Agent or Executor still holds the old token"
            if lines
            else "a full quiet period could not be verified before the deadline"
        )
        + " -- rerun to keep waiting"
    )
