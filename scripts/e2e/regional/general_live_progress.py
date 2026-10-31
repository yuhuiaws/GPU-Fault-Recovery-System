"""GF-GENERAL-LIVE-027 training progress, phases and NaN recovery.

The product's training-progress reporter is optional and OFF by default and
must stay off for normal workloads: the periodic ``TrainingHealthService``
monitor (``GPU_FAULT_ENABLE_TRAINING_HEALTH_MONITOR``, default false) is global
and grades every RUNNING attempt in the store, so this runner never enables it.
The disposable test job runs the reporter itself -- a ``gpu-fault-collector
training-progress`` sidecar in the training Pod -- pointed at a progress file
the test application writes atomically, which is the "enabled reporter" source
path the spec requires (better than POSTing heartbeats). Scan-time signals
(stalled step, reporter silence, straggler, phase budget) are driven by the
execution-token ``POST /v1/training-health/<cluster>/scan`` on this cluster
only, after the runner's preflight has confirmed the test attempt is the only
RUNNING managed attempt (otherwise a cluster-wide scan would flag a normal job
as reporter-silent).

The verdict reads only the product's own records -- the training-progress
state, the ``SITE_TRAINING_HEALTH`` incident, its workflow step executions, the
completion decision and the notification -- through ``cpu_python`` in an API
Pod. ``training_nonfinite-loss`` stops only the attributed attempt
(STOP_WORKLOADS) and escalates; it never cordons the node, resets GPUs or
auto-restarts, and the checkpoint reference is evidence only. A stall is
cleared only by the original rank's newer finite progress, never by healthy
GPU/fabric telemetry alone.

A variant that asserts an incident state judges a settled record: the
incident's workflow terminal and the incident out of its in-flight state
(``_settle``). The first store read that shows an incident shows its chain
mid-flight -- live 2026-10-26 (plain EKS, release 37db33f254a1) ``nan`` read
ACTION_PENDING with COLLECT_DIAGNOSTIC_BUNDLE WAITING 4 s after the heartbeat
and failed "expected ESCALATED"; the product completed ESCALATE_SUPPORT and
moved the incident to ESCALATED 22 s later. The scan-driven variants assert no
state: their VALIDATE_FABRIC legitimately stays in flight until newer progress
arrives, which ``resumed_progress`` follows to RECOVERED.
"""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import PurePosixPath
import re
from typing import Any

from scripts.e2e.regional.general_live_common import (
    CHAIN_MARGIN_SECONDS,
    STORE_POLL_SECONDS,
    run_survey,
)
from scripts.e2e.regional.general_live_fixture import (
    VariantContext,
    VariantInconclusive,
    VariantNotRun,
    run_identity,
    utc_now,
)
from scripts.e2e.regional.general_live_progress_probes import (
    HEARTBEAT_POST,
    SCAN_POST,
    TRAINING_PROBE,
)
from scripts.e2e.regional.general_live_progress_verdicts import (
    RESUMED_STEP,
    STALLED_STEP,
    foreign_refusal,
    foreign_state_errors,
    live_kept_errors,
    rank_states,
    validated_by_progress,
)
from scripts.e2e.regional.general_live_registry import (
    HARDWARE_OPERATIONS,
    NATIVE,
    VariantSpec,
)
from scripts.e2e.regional.general_live_verdicts import (
    TERMINAL_WORKFLOWS,
    operations,
    verdict,
)
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from scripts.e2e.regional.regional_live_fixture import component_python

CASE = "GF-GENERAL-LIVE-027"
SINGLE_NODE_MANIFEST = "single-node-warm-spare-pytorchjob.yaml"
#: Two ranks are two Pods on two nodes. The product binds one rank per Pod --
#: the watcher's rank is the Pod's replica index plus the role's
#: ``gpu-fault.io/rank-offset`` that ``gpu-training-submit`` stamps
#: (``completion_pod_parsing.py``, ``training_submit_cli.py``: Master 0, Worker
#: 1) -- its ingest refuses a heartbeat whose rank has no container in the
#: attempt's allocation (``training_health.py`` ``ingest`` -> ``_allocation``,
#: ``accepted=False``), and its scan compares only the allocated ranks' fresh
#: heartbeats (``scan`` walks the observation's containers; a straggler group
#: needs two). Live 2026-10-26 straggler_group ran both reporters in one
#: single-node Pod: rank 1's heartbeat never reached the store, the group had
#: one rank, and rank 0's fixed step was graded a stall instead.
TWO_NODE_MANIFEST = "two-node-replace-pytorchjob.yaml"
#: The shared progress volume the trainer writes and the reporters read.
PROGRESS_DIR = "/var/run/gpu-fault"
#: The ``pytorchReplicaSpecs`` role of each rank of the two-node job, and the
#: nodes they are pinned to: rank 0 on the lane node, rank 1 on ``--node-b``.
RANK_ROLES = ("Master", "Worker")
HEARTBEAT_TIMEOUT = 120
STARTUP_GRACE = 300
#: straggler_group: rank 1 starts STRAGGLER_LAG steps behind rank 0 and both
#: advance STRAGGLER_ADVANCE steps per poll. The product grades a rank against
#: its group's median step (``_group_stragglers``; two ranks: half the gap, 30)
#: with ``GPU_FAULT_TRAINING_MAX_STEP_LAG`` (20), and a rank whose step moves
#: every 20-s poll never crosses the 120-s stall timeout -- a lagging rank is
#: a straggler, a frozen one a stall (live 2026-10-26: rank 0 written once,
#: ``training_hang`` 145 s later). The finding is bound to the lagging rank
#: and its incident names that rank's node alone, so the records read names
#: every node of the job (``_job_nodes``): live 2026-10-26 17:43Z (HyperPod)
#: the two-node job had both ranks fresh in the store (240/180, one group) and
#: the lane-node-only read never listed rank 1's straggler on ``--node-b``.
STRAGGLER_LAG = 60
STRAGGLER_ADVANCE = 10
#: The product's step cadence the settle budget is derived from
#: (``execution/config.py``, ``execution/step_bounds.py``): the dispatcher
#: re-dispatches a WAITING step one poll interval apart
#: (``GPU_FAULT_WORKFLOW_POLL_INTERVAL_SECONDS``, 5 s) and fails a step that
#: stays non-terminal past the per-step cap
#: (``GPU_FAULT_WORKFLOW_STEP_TIMEOUT_SECONDS``, 600 s), so a record whose
#: remote step never answers still settles -- FAILED -- within the cap.
WORKFLOW_POLL_INTERVAL = 5
WORKFLOW_STEP_CAP = 600
#: Incident states of a record whose workflow is still executing
#: (``orchestration/coordinator.py`` ``_incident_state_for_workflow``:
#: PENDING/RUNNING -> ACTION_PENDING, SAFETY_PENDING -> SAFETY_PENDING). The
#: terminal write of the last step moves the incident with the workflow
#: (``execution/terminal_state.py`` ``terminal_incident_state``: SUCCEEDED with
#: ESCALATE_SUPPORT completed -> ESCALATED, otherwise RECOVERED/QUARANTINED;
#: FAILED/BLOCKED -> the failure state).
IN_FLIGHT_STATES = frozenset({"DETECTED", "ACTION_PENDING", "SAFETY_PENDING"})
#: The reporter sidecar runs the product's own ``gpu-fault-collector`` from the
#: GPU-side runtime image -- the image the release stamps on the executor, the
#: completion watcher and the in-cluster collector, whose console scripts sit
#: next to ``component_python("gpu")``. The training image ships no product:
#: live 2026-10-24 the sidecar named the Node Agent's host install path
#: (``/opt/gpu-fault/current``) inside it, never started, and every 027 Pod
#: stayed Running/NotReady through the 900-s start wait.
RUNTIME_DEPLOYMENT = "gpu-fault-cluster-executor"
RUNTIME_CONTAINER = "executor"
REPORTER_BIN = str(
    PurePosixPath(component_python("gpu")).with_name("gpu-fault-collector")
)
DIGEST_PINNED = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}")
#: The control-plane connection, read as the product's in-cluster collector
#: reads it (``deploy/dataplane/kubernetes-node-resource-collector.yaml``):
#: ``gpu-fault-collector`` refuses to start without the URL and cluster id, and
#: ``POST /v1/training-progress`` takes the cluster token.
CONNECTION_SECRET = "gpu-fault-regional-connection"
CONNECTION_ENV = (
    ("GPU_FAULT_CONTROL_PLANE_URL", "control-plane-url"),
    ("GPU_FAULT_CLUSTER_ID", "cluster-id"),
    ("GPU_FAULT_CONTROL_PLANE_TOKEN", "cluster-token"),
)
CA_DIR = "/etc/gpu-fault/tls"
#: Writable state: the collector's durable outbox.
STATE_DIR = "/var/lib/gpu-fault"
#: Where the reporter's launcher records the reporter's PID: the reporter
#: container's own ``/tmp`` emptyDir.
REPORTER_PID_FILE = "/tmp/gfrs-reporter.pid"
#: The reporter container's PID 1 is this launcher shell; the reporter is its
#: child. pid_namespaces(7): the init of a PID namespace -- a container's
#: PID 1 -- receives only the signals it has handlers for; SIGSTOP and SIGKILL,
#: which cannot be handled, are dropped when sent from inside the namespace,
#: and ``kill`` still returns 0. ``kubectl exec`` runs inside it. Live
#: 2026-10-26 (plain EKS, release 37db33f254a1) the reporter was PID 1 and
#: ``reporter_silent``'s ``kill -STOP 1`` stopped nothing: the reporter kept
#: posting ``step=10`` every 15 s for thirteen minutes and the product graded
#: the stall it was shown (``training_hang``, "training step stopped
#: advancing"). The trap forwards a Pod deletion's SIGTERM so the container
#: still ends at once (as PID 1, a shell without a handler would ignore it
#: until the grace period); the reporter's own exit status is the container's.
REPORTER_LAUNCHER = (
    "trap 'kill -TERM \"$reporter\" 2>/dev/null' TERM INT\n"
    '"$@" & reporter=$!\n'
    f"printf '%s\\n' \"$reporter\" > {REPORTER_PID_FILE}\n"
    'wait "$reporter"\n'
)
#: ``kubectl exec`` script that stops the reporter and prints its PID and its
#: ``/proc/<pid>/status`` ``State`` line -- the injection's proof, judged by
#: ``reporter_stopped``. Shell builtins and coreutils ``sleep`` only: the
#: runtime image is ubi9 python-minimal.
STOP_REPORTER = (
    f"read -r pid < {REPORTER_PID_FILE} || exit 2\n"
    'kill -STOP "$pid" || exit 3\n'
    "state=\n"
    "for attempt in 1 2 3 4 5 6 7 8 9 10; do\n"
    "  state=$(while IFS= read -r line; do case $line in State:*) "
    'printf \'%s\' "${line#State:}";; esac; done < "/proc/$pid/status")\n'
    "  case $state in *stopped*) break;; esac\n"
    "  sleep 0.2\n"
    "done\n"
    'printf \'%s\\n%s\\n\' "$pid" "$state"\n'
)
#: ``old_attempt``: the claimed attempt -- the job's id with a number the product
#: never ran (no RUNNING/PENDING allocation) -- and the watch after the post
#: (three reporter intervals: the live attempt's next heartbeats still accepted).
FOREIGN_ATTEMPT_SUFFIX = "-a000"
FOREIGN_SETTLE_SECONDS = 45
#: Operations a training-progress chain must never contain: no hardware action,
#: no node isolation, no automatic workload restart. STOP_WORKLOADS is allowed
#: for the NaN/checkpoint branch only (and is asserted there explicitly).
FORBIDDEN = HARDWARE_OPERATIONS | {
    "MARK_UNSCHEDULABLE",
    "QUARANTINE",
    "RESTART_WORKLOAD",
}


def reporter_image(regional: Any) -> str:
    """The digest-pinned runtime image this GPU cluster's executor runs."""

    deployment = json.loads(
        regional.kubectl("gpu", "get", "deployment", RUNTIME_DEPLOYMENT, "-o", "json")
    )
    pod_spec: Any = deployment
    for key in ("spec", "template", "spec"):
        pod_spec = pod_spec.get(key) if isinstance(pod_spec, dict) else None
    containers = pod_spec.get("containers") if isinstance(pod_spec, dict) else None
    images = [
        str(item.get("image") or "")
        for item in (containers if isinstance(containers, list) else [])
        if isinstance(item, dict) and item.get("name") == RUNTIME_CONTAINER
    ]
    if len(images) != 1 or DIGEST_PINNED.fullmatch(images[0]) is None:
        raise RegionalFixtureError(
            f"the {RUNTIME_DEPLOYMENT} Deployment has no digest-pinned "
            f"{RUNTIME_CONTAINER} image for the progress reporter: {images}"
        )
    return images[0]


class ManagedProgressJob:
    """A managed single-node test job that runs the training-progress reporter.

    A thin wrapper over ``ManagedWorkloadFixture``: it renders a copy of the
    warm-spare training manifest with a shared ``emptyDir`` for the progress
    file and a ``gpu-fault-collector training-progress`` sidecar per reported
    rank (the reporter reports the training container's identity, not its own),
    from the cluster's own runtime image and connection Secret. The test
    application's progress is controlled by the runner, which writes the
    per-rank JSON into the shared volume with an atomic rename from inside the
    trainer container -- the enabled-reporter source path. The verdict tests
    monkeypatch this class; test_general_live_managed_job_start.py runs its
    start with the real fixtures and test_general_live_progress_job.py runs its
    Pod and progress write.
    """

    PROGRESS_DIR = PROGRESS_DIR

    def __init__(
        self,
        ctx: VariantContext,
        *,
        name: str,
        ranks: tuple[int, ...] = (0,),
        gpu_count: int = 8,
    ) -> None:
        from pathlib import Path

        from scripts.e2e.regional.managed_workload_fixture import (
            ImagePrewarmFixture,
            ManagedWorkloadFixture,
            ManagedWorkloadSettings,
            name_managed_workload,
            render_node_pinned_manifest,
        )

        yaml = __import__("yaml")
        if ctx.settings.site_file is None:
            raise VariantNotRun(
                "FIXTURE_NOT_READY", "a managed progress job needs --site-file"
            )
        self.ranks = tuple(sorted(set(ranks)))
        if self.ranks not in {(0,), (0, 1)}:
            raise ValueError(
                f"the progress job runs rank 0 or ranks 0 and 1, not {self.ranks}"
            )
        two_node = len(self.ranks) == 2
        if two_node and not ctx.settings.peer_node:
            raise VariantNotRun(
                "FIXTURE_NOT_READY",
                "a two-rank progress job needs --node-b: the product binds one "
                "rank per Pod and the fixture runs one Pod per node",
            )
        # Read before anything is rendered or journaled: a refusal here is a
        # preparation failure that leaves nothing to clean up.
        self.reporter_image = reporter_image(ctx.regional)
        manifests = Path(__file__).with_name("manifests") / "training"
        rendered = ctx.case_dir / f"{name}.source.yaml"
        if two_node:
            self.rank_nodes = {0: ctx.settings.node, 1: ctx.settings.peer_node}
            ranks_by_role = {role: (rank,) for role, rank in zip(RANK_ROLES, (0, 1))}
            document = yaml.safe_load(
                (manifests / TWO_NODE_MANIFEST).read_text(encoding="utf-8")
            )
            pin_replicas(
                document,
                {RANK_ROLES[rank]: node for rank, node in self.rank_nodes.items()},
            )
            rendered.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        else:
            self.rank_nodes = {0: ctx.settings.node}
            ranks_by_role = {RANK_ROLES[0]: self.ranks}
            render_node_pinned_manifest(
                manifests / SINGLE_NODE_MANIFEST, rendered, node=ctx.settings.node
            )
            document = yaml.safe_load(rendered.read_text(encoding="utf-8"))
        name_managed_workload(document, name)
        install_reporter(document, name, ranks_by_role, self.reporter_image)
        document.setdefault("metadata", {}).setdefault("labels", {}).update(
            {"gpu-fault.io/job-id": name, "gpu-fault.io/attempt-id": f"{name}-a001"}
        )
        rendered.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        rendered.chmod(0o600)
        self.ctx = ctx
        self.name = name
        self.attempt_id = f"{name}-a001"
        self.gpu_count = gpu_count
        self.prewarm = ImagePrewarmFixture(
            ctx.regional,
            case_id=ctx.spec.key,
            run_id=f"{ctx.run_id}-{name}",
            state_path=ctx.case_dir / f"{name}.prewarm.json",
        )
        self.fixture = ManagedWorkloadFixture(
            ctx.regional,
            ManagedWorkloadSettings(
                manifest=rendered,
                site_file=ctx.settings.site_file,
                job_id=name,
                attempt_id=self.attempt_id,
                restart_budget=0,
                expected_pods=len(self.rank_nodes),
                expected_gpu_count=gpu_count * len(self.rank_nodes),
                # The reporter sidecars make the template multi-container;
                # gpu-training-submit refuses one without its training container.
                training_container="pytorch",
            ),
            state_path=ctx.case_dir / f"{name}.ownership.json",
        )

    def start(self) -> dict[str, Any]:
        from scripts.e2e.regional.workload_restart_custody import (
            WorkloadOwnershipError,
        )

        self.prewarm.create([self.rank_nodes[rank] for rank in self.ranks])
        self.fixture.submit()
        try:
            # A two-node job brings up two Pods and a cross-node collective.
            return self.fixture.wait_running(
                timeout_seconds=900 if len(self.rank_nodes) == 1 else 1200
            )
        except WorkloadOwnershipError:
            raise
        except RegionalFixtureError as exc:
            # A Pod is Ready only when every container is; the wait's snapshot
            # says "ready: False" and nothing about which container or why.
            raise RegionalFixtureError(
                f"{exc}; not ready: {self.unready_containers()}"
            ) from exc

    def unready_containers(self) -> dict[str, Any]:
        """The kubelet's state for every container of the job that is not Ready."""

        try:
            pods = json.loads(
                self.ctx.regional.kubectl(
                    "gpu",
                    "get",
                    "pod",
                    "-l",
                    f"gpu-fault.io/job-id={self.name}",
                    "-o",
                    "json",
                )
            )["items"]
        except Exception as exc:  # noqa: BLE001 - a diagnostic, never the verdict
            return {"unreadable": f"{type(exc).__name__}: {exc}"}
        result: dict[str, Any] = {}
        for pod in pods if isinstance(pods, list) else []:
            if not isinstance(pod, dict):
                continue
            name = (pod.get("metadata") or {}).get("name")
            statuses = (pod.get("status") or {}).get("containerStatuses") or []
            for status in statuses:
                if not isinstance(status, dict) or status.get("ready") is True:
                    continue
                state = status.get("state") or {}
                kind = next(iter(state), "unknown")
                detail = state.get(kind) if isinstance(state.get(kind), dict) else {}
                result[f"{name}/{status.get('name')}"] = {
                    "state": kind,
                    "reason": detail.get("reason"),
                    "exit_code": detail.get("exitCode"),
                    "message": str(detail.get("message") or "")[:300],
                }
        return result

    def identity(self) -> dict[str, Any]:
        return {
            "job_id": self.name,
            "attempt_id": self.attempt_id,
            "ranks": list(self.ranks),
            "nodes": [self.rank_nodes[rank] for rank in self.ranks],
        }

    def pod_for(self, rank: int) -> str:
        """The name of the Pod that runs ``rank`` (and its reporter): the job's
        Pod on the node the rank was pinned to."""

        node = self.rank_nodes.get(rank)
        if node is None:
            raise RegionalFixtureError(f"rank {rank} is not a rank of {self.name}")
        names = [
            str(item["name"])
            for item in self.fixture.pods()
            if item.get("node") == node
        ]
        if len(names) != 1:
            raise RegionalFixtureError(
                f"{self.name} has {len(names)} Pods on {node}, rank {rank} needs one"
            )
        return names[0]

    def reporter_logs(self, *, tail: int = 50) -> dict[str, str]:
        """The last lines of every reporter sidecar -- a diagnostic for a
        heartbeat the store never showed (an ingest refusal is answered to the
        reporter, which logs it, not to the runner). Never the verdict."""

        logs: dict[str, str] = {}
        for rank in self.ranks:
            key = f"progress-reporter-{rank}"
            try:
                logs[key] = str(
                    self.ctx.regional.kubectl(
                        "gpu",
                        "logs",
                        self.pod_for(rank),
                        "-c",
                        key,
                        f"--tail={tail}",
                        timeout=60,
                    )
                )[-4000:]
            except Exception as exc:  # noqa: BLE001 - a diagnostic, never the verdict
                logs[key] = f"unreadable: {type(exc).__name__}: {exc}"
        return logs

    def write_progress(self, rank: int, **fields: Any) -> dict[str, Any]:
        """Write one rank's progress JSON atomically from the trainer container.

        The runner drives the test application's output; the in-Pod reporter
        reads it and posts the heartbeat. ``fields`` are the progress-file keys
        the reporter consumes (``step``, ``loss``, ``numerical_error``,
        ``checkpoint_ref`` and ``labels`` such as ``phase``/``progress_group``).
        """

        payload = json.dumps(
            {key: value for key, value in fields.items() if value is not None}
        )
        pod = self.pod_for(rank)
        path = f"{self.PROGRESS_DIR}/rank-{rank}.json"
        # ``-i``: kubectl forwards its stdin only when asked, and without it the
        # ``cat`` read end-of-file and renamed an empty progress file in place.
        return self.ctx.regional.kubectl(
            "gpu",
            "exec",
            "-i",
            pod,
            "-c",
            "pytorch",
            "--",
            "/bin/sh",
            "-c",
            f"cat > {path}.tmp && mv {path}.tmp {path}",
            input_text=payload,
        )

    def post_heartbeat(
        self, rank: int, attempt_id: str, **fields: Any
    ) -> dict[str, Any]:
        """POST one heartbeat claiming ``attempt_id`` from rank ``rank``'s
        reporter sidecar (``HEARTBEAT_POST``) and return what it printed: the
        heartbeat and the product's answer, or the client's error; other
        output is an undelivered answer, never an exception."""

        output = self.ctx.regional.kubectl(
            "gpu",
            "exec",
            self.pod_for(rank),
            "-c",
            f"progress-reporter-{rank}",
            "--",
            component_python("gpu"),
            "-c",
            HEARTBEAT_POST,
            attempt_id,
            str(rank),
            json.dumps(
                {key: value for key, value in fields.items() if value is not None}
            ),
            timeout=120,
        )
        lines = [line for line in str(output).splitlines() if line.strip()]
        try:
            answer = json.loads(lines[-1])
        except (IndexError, ValueError):
            answer = None
        if not isinstance(answer, dict):
            return {"delivered": False, "error": f"unparseable: {str(output)[-300:]}"}
        return answer

    def stop_reporter(self, rank: int) -> dict[str, Any]:
        pod = self.pod_for(rank)
        return self.ctx.regional.kubectl(
            "gpu",
            "exec",
            pod,
            "-c",
            f"progress-reporter-{rank}",
            "--",
            "/bin/sh",
            "-c",
            STOP_REPORTER,
        )

    def delete(self) -> dict[str, Any]:
        from scripts.e2e.regional.managed_workload_fixture import remove_managed_job

        return remove_managed_job(self.fixture, self.prewarm)


def pin_replicas(document: dict[str, Any], nodes_by_role: dict[str, str]) -> None:
    """Pin each ``pytorchReplicaSpecs`` role's single Pod to its node."""

    replicas = (document.get("spec") or {}).get("pytorchReplicaSpecs") or {}
    for role, node in nodes_by_role.items():
        replica = replicas.get(role)
        if not isinstance(replica, dict) or replica.get("replicas") != 1:
            raise ValueError(f"the two-node progress job needs exactly one {role}")
        pod_spec = (replica.get("template") or {}).get("spec")
        if not isinstance(pod_spec, dict):
            raise ValueError(f"the {role} replica has no Pod spec")
        pod_spec["nodeName"] = node


def install_reporter(
    document: dict[str, Any],
    name: str,
    ranks_by_role: dict[str, tuple[int, ...]],
    image: str,
) -> None:
    """Add the shared progress volume and, per replica role, one reporter
    sidecar for each of the ranks that role's Pod reports -- a reporter posts
    its Pod's identity, so a rank's reporter must run in the Pod the product
    allocated that rank to."""

    from scripts.e2e.regional.acceptance_gpu_load import set_gpu_duty_percent

    set_gpu_duty_percent(document, 20)
    replicas = (document.get("spec") or {}).get("pytorchReplicaSpecs") or {}
    for role, replica in replicas.items():
        ranks = tuple(ranks_by_role.get(role, ()))
        template = (replica.get("template") or {}).get("spec")
        if not isinstance(template, dict):
            continue
        volumes = template.setdefault("volumes", [])
        volumes.append({"name": "gfrs-progress", "emptyDir": {}})
        volumes.append(
            {
                "name": "gfrs-control-plane-ca",
                "secret": {
                    "secretName": CONNECTION_SECRET,
                    "items": [{"key": "ca.crt", "path": "ca.crt"}],
                },
            }
        )
        mount = {"name": "gfrs-progress", "mountPath": PROGRESS_DIR}
        containers = template.setdefault("containers", [])
        for container in containers:
            if isinstance(container, dict):
                container.setdefault("volumeMounts", []).append(dict(mount))
        for rank in ranks:
            # One outbox per reporter; a shared directory would
            # make two reporters replay each other's buffered heartbeats.
            for volume, limit in (("state", "64Mi"), ("tmp", "16Mi")):
                volumes.append(
                    {
                        "name": f"gfrs-reporter-{rank}-{volume}",
                        "emptyDir": {"sizeLimit": limit},
                    }
                )
            containers.append(_reporter_container(name, rank, mount, image))


def _reporter_container(
    name: str, rank: int, mount: dict[str, Any], image: str
) -> dict[str, Any]:
    progress_file = f"{PROGRESS_DIR}/rank-{rank}.json"
    return {
        "name": f"progress-reporter-{rank}",
        "image": image,
        "imagePullPolicy": "IfNotPresent",
        "command": [
            "/bin/sh",
            "-c",
            REPORTER_LAUNCHER,
            "gfrs-progress-reporter",
            REPORTER_BIN,
            "training-progress",
            "--attempt-id",
            f"{name}-a001",
            "--rank",
            str(rank),
            "--progress-file",
            progress_file,
        ],
        "env": [
            *(
                {
                    "name": variable,
                    "valueFrom": {
                        "secretKeyRef": {"name": CONNECTION_SECRET, "key": key}
                    },
                }
                for variable, key in CONNECTION_ENV
            ),
            # The collector's HTTP client verifies TLS through the default
            # context, which reads SSL_CERT_FILE (the private endpoint's CA).
            {"name": "SSL_CERT_FILE", "value": f"{CA_DIR}/ca.crt"},
            # The reporter reports the TRAINING container's identity so the
            # control plane's allocation match accepts it (watcher check).
            {"name": "CONTAINER_NAME", "value": "pytorch"},
            {
                "name": "NODE_NAME",
                "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}},
            },
            {
                "name": "POD_UID",
                "valueFrom": {"fieldRef": {"fieldPath": "metadata.uid"}},
            },
        ],
        "volumeMounts": [
            dict(mount),
            {"name": "gfrs-control-plane-ca", "mountPath": CA_DIR, "readOnly": True},
            {"name": f"gfrs-reporter-{rank}-state", "mountPath": STATE_DIR},
            # readOnlyRootFilesystem makes /tmp read-only too.
            {"name": f"gfrs-reporter-{rank}-tmp", "mountPath": "/tmp"},
        ],
        "resources": {
            "requests": {"cpu": "100m", "memory": "128Mi"},
            "limits": {"cpu": "500m", "memory": "256Mi"},
        },
        # The product's own in-cluster collector container, same image.
        "securityContext": {
            "allowPrivilegeEscalation": False,
            "readOnlyRootFilesystem": True,
            "runAsNonRoot": True,
            "runAsUser": 65532,
            "capabilities": {"drop": ["ALL"]},
        },
    }


def reporter_stopped(output: Any) -> dict[str, Any]:
    """The reporter's PID and process state ``STOP_REPORTER`` printed.

    ``kubectl exec`` returns 0 whether or not the reporter stopped (the live
    ``kill -STOP 1`` did); the injection is judged on the process state. A
    reporter still running -- or one that is PID 1, which no in-namespace
    SIGSTOP reaches -- injected nothing, so the variant is inconclusive
    rather than graded on whatever the product then does.
    """

    lines = [line.strip() for line in str(output).splitlines() if line.strip()]
    pid = int(lines[0]) if lines and lines[0].isdigit() else None
    state = lines[1] if len(lines) > 1 else ""
    if pid is None or pid == 1 or "(stopped)" not in state:
        raise VariantInconclusive(
            "the progress reporter did not stop, no reporter silence was "
            f"injected: pid={lines[0] if lines else None!r} state={state!r}"
        )
    return {"pid": pid, "state": state}


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def _job(
    ctx: VariantContext, suffix: str, *, ranks: tuple[int, ...] = (0,)
) -> ManagedProgressJob:
    name = ctx.identity(
        run_identity(CASE, suffix, ctx.attempt, ctx.case_dir).replace("gl027", "gl27")
    )
    job = ManagedProgressJob(ctx, name=name, ranks=ranks)
    ctx.cleanup.push(f"job-{name}", job.delete)
    job.start()
    return job


def _rank_node(ctx: VariantContext, job: Any, rank: int) -> str:
    """The node ``rank`` of ``job`` runs on; the lane node when the job does
    not say (a job without ``rank_nodes``, or none for the rank). A records
    read never raises over the job's shape."""

    rank_nodes = getattr(job, "rank_nodes", None)
    node = rank_nodes.get(rank) if isinstance(rank_nodes, dict) else None
    return str(node) if node else ctx.settings.node


def _job_nodes(ctx: VariantContext, job: Any) -> str:
    """Every node the job's ranks run on, comma-joined for ``TRAINING_PROBE``.

    A training incident names the graded rank's node alone (the finding's
    ``node_id`` is that rank's node), and rank 1 of the two-node job runs on
    ``--node-b``: a read scoped to the lane node never lists it (live
    2026-10-26 17:43Z, HyperPod ``straggler_group``). One node for a
    single-rank job, as before; the lane node only for a job that names none.
    """

    ranks = getattr(job, "ranks", None) or (0,)
    return ",".join(dict.fromkeys(_rank_node(ctx, job, rank) for rank in ranks))


def _records(
    ctx: VariantContext,
    job: ManagedProgressJob,
    since: Any,
    *,
    others: tuple[str, ...] = (),
) -> dict[str, Any]:
    """The job's training records (``TRAINING_PROBE``); ``others`` adds the
    stored progress states of further attempt ids."""

    arguments = [
        ctx.settings.regional.cluster_id,
        _job_nodes(ctx, job),
        job.name,
        job.attempt_id,
        since.isoformat(),
        *([",".join(others)] if others else []),
    ]
    document = ctx.regional.cpu_python(TRAINING_PROBE, *arguments)
    ctx.evidence.write("training-records", document)
    return document


def _scan(ctx: VariantContext) -> dict[str, Any]:
    result = ctx.regional.cpu_python(SCAN_POST, ctx.settings.regional.cluster_id)
    ctx.evidence.write("training-scan", result)
    return result


def _incident(records: dict[str, Any], kind: str) -> dict[str, Any] | None:
    prefix = f"training-{kind}-"
    found = [
        item
        for item in records.get("incidents") or []
        if str(item.get("event_id", "")).startswith(prefix)
    ]
    return found[0] if found else None


def _workflow(records: dict[str, Any], incident: dict[str, Any]) -> dict[str, Any]:
    return (records.get("workflows") or {}).get(
        str(incident.get("workflow_request_id"))
    ) or {}


def _wait_incident(
    ctx: VariantContext,
    job: ManagedProgressJob,
    kind: str,
    since: Any,
    *,
    name: str,
    scan: bool = False,
    before_poll: Callable[[], Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """The records once the ``kind`` incident exists, polled every 20 s within
    the detection budget. ``scan`` triggers the product's cluster scan before
    each read; ``before_poll`` runs first on every poll (a variant that must
    keep its ranks' progress moving while the product grades them)."""

    last: dict[str, Any] = {}

    def accept() -> dict[str, Any] | None:
        nonlocal last
        if before_poll is not None:
            before_poll()
        if scan:
            _scan(ctx)
        last = _records(ctx, job, since)
        return last if _incident(last, kind) else None

    value = ctx.wait_until(
        accept,
        timeout_seconds=HEARTBEAT_TIMEOUT + STARTUP_GRACE + 180,
        poll_seconds=20,
        name=name,
    )
    records = value if value is not None else last
    return records, _incident(records, kind)


def settle_budget_seconds(workflow: dict[str, Any]) -> int:
    """How long a chain may still take to settle once its incident is first read.

    One step may stay non-terminal for the product's per-step cap (the remote
    COLLECT_DIAGNOSTIC_BUNDLE took 22 s live, 2026-10-26), every planned step
    is dispatched one poll interval apart, and the chain margin covers ingest
    and the store read. A chain whose steps each run to the cap is not waited
    out: the budget bounds a product that is working, and a record still in
    flight past it is reported with the step still waiting.
    """

    steps = max(1, len(workflow.get("official_steps") or []))
    return WORKFLOW_STEP_CAP + steps * WORKFLOW_POLL_INTERVAL + CHAIN_MARGIN_SECONDS


def record_settled(incident: dict[str, Any], workflow: dict[str, Any]) -> bool:
    """Whether the incident's record has settled: its state out of the in-flight
    states and its workflow terminal. An incident that names no workflow (the
    plan compiled to nothing) is settled once its state is not in flight."""

    if incident.get("state") in IN_FLIGHT_STATES:
        return False
    if not incident.get("workflow_request_id"):
        return True
    return str(workflow.get("status")) in TERMINAL_WORKFLOWS


def _executed(workflow: dict[str, Any]) -> list[str]:
    """``OPERATION STATUS`` per recorded step execution, in step order."""

    return [
        f"{item.get('operation')} {item.get('status')}"
        for item in workflow.get("step_executions") or []
    ]


def _settle_record(
    incident: dict[str, Any] | None, workflow: dict[str, Any]
) -> dict[str, Any]:
    if incident is None:
        return {"incident_id": None, "settled": False}
    return {
        "incident_id": incident.get("incident_id"),
        "state": incident.get("state"),
        "workflow_request_id": incident.get("workflow_request_id"),
        "workflow_status": workflow.get("status"),
        "planned": operations(workflow),
        "executed": _executed(workflow),
        "settled": record_settled(incident, workflow),
    }


def _settle(
    ctx: VariantContext,
    job: ManagedProgressJob,
    kind: str,
    since: Any,
    records: dict[str, Any] | None,
    *,
    name: str,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    """The records once the ``kind`` incident has settled, and how long that took.

    Starts from ``records`` (the snapshot that first showed the incident; a
    fresh read when ``None``) and polls the store every ``STORE_POLL_SECONDS``
    until ``record_settled`` holds, within ``settle_budget_seconds`` of the planned
    chain. Returns the settled records, the incident and the settle record --
    the budget, the seconds waited, the polls, the state, the workflow status,
    the planned and the executed steps -- which is also written as
    ``<name>-settled``. No incident is nothing to settle (one read at most;
    the chain check reports the missing finding). A record still in flight at
    the end of the budget raises ``VariantInconclusive`` naming the observed
    state and the executed steps: the product was still executing, or stuck,
    and nothing about its terminal state was graded.
    """

    started = ctx.clock()
    polls = 0
    if records is None:
        records = _records(ctx, job, since)
        polls = 1
    incident = _incident(records, kind)
    budget = settle_budget_seconds(_workflow(records, incident or {}))
    if incident is not None and not record_settled(
        incident, _workflow(records, incident)
    ):
        last = records

        def accept() -> dict[str, Any] | None:
            nonlocal last, polls
            last = _records(ctx, job, since)
            polls += 1
            current = _incident(last, kind)
            if current is None:
                return None
            return last if record_settled(current, _workflow(last, current)) else None

        value = ctx.wait_until(
            accept,
            timeout_seconds=budget,
            poll_seconds=STORE_POLL_SECONDS,
            name=f"{name}-settle",
        )
        records = value if value is not None else last
        incident = _incident(records, kind)
    workflow = _workflow(records, incident or {})
    settle = {
        "budget_seconds": budget,
        "waited_seconds": round((ctx.clock() - started).total_seconds(), 1),
        "polls": polls,
        **_settle_record(incident, workflow),
    }
    ctx.evidence.write(f"{name}-settled", settle)
    if incident is not None and not settle["settled"]:
        executed = ", ".join(settle["executed"]) or "nothing"
        planned = ", ".join(settle["planned"]) or "nothing"
        raise VariantInconclusive(
            f"training_{kind} incident {settle['incident_id']} did not settle "
            f"within {budget} s ({settle['waited_seconds']} s waited, {polls} "
            f"reads): state {settle['state']}, workflow "
            f"{settle['workflow_request_id']} {settle['workflow_status']}; "
            f"executed {executed}; planned {planned}"
        )
    return records, incident, settle


def _settled_errors(settle: dict[str, Any], *, kind: str) -> list[str]:
    """The settled chain ran to its end: the workflow SUCCEEDED.

    An ESCALATED reached through a FAILED or BLOCKED step is the product's
    failure exit, not the planned ESCALATE_SUPPORT hand-off; a SUPERSEDED
    record was replaced by another chain. A missing incident is the chain
    check's error, not this one's.
    """

    if not settle.get("incident_id"):
        return []
    status = settle.get("workflow_status")
    if status == "SUCCEEDED":
        return []
    executed = ", ".join(settle.get("executed") or []) or "nothing"
    return [
        f"training_{kind} workflow settled {status}, not SUCCEEDED "
        f"(executed {executed})"
    ]


def _chain_errors(
    incident: dict[str, Any] | None,
    workflow: dict[str, Any],
    *,
    kind: str,
    required_ops: tuple[str, ...],
    allow_stop: bool,
    states: frozenset[str],
) -> list[str]:
    errors: list[str] = []
    if incident is None:
        return [f"no training_{kind} finding reached an incident"]
    ops = operations(workflow)
    missing = [item for item in required_ops if item not in ops]
    if missing:
        errors.append(f"training_{kind} workflow {ops} lacks {missing}")
    forbidden = FORBIDDEN if allow_stop else FORBIDDEN | {"STOP_WORKLOADS"}
    planned = sorted(set(ops) & forbidden)
    if planned:
        errors.append(f"training_{kind} workflow plans forbidden {planned}")
    if states and incident.get("state") not in states:
        errors.append(
            f"training_{kind} incident is {incident.get('state')}, expected {sorted(states)}"
        )
    return errors


# --------------------------------------------------------------------------- #
# variants
# --------------------------------------------------------------------------- #
def nan(ctx: VariantContext) -> dict[str, Any]:
    run_survey(ctx)
    job = _job(ctx, "nan")
    identity = job.identity()
    ctx.hold(30, name="nan-running")
    started = utc_now()
    job.write_progress(
        0, step=10, numerical_error=True, checkpoint_ref="s3://test/unchecked"
    )
    records, incident = _wait_incident(ctx, job, "nonfinite-loss", started, name="nan")
    # ESCALATED is the terminal write of the last step: judge the settled record.
    records, incident, settle = _settle(
        ctx, job, "nonfinite-loss", started, records, name="nan"
    )
    workflow = _workflow(records, incident or {})
    stages = {
        "chain": _chain_errors(
            incident,
            workflow,
            kind="nonfinite-loss",
            required_ops=("FREEZE_EVIDENCE", "STOP_WORKLOADS", "ESCALATE_SUPPORT"),
            allow_stop=True,
            states=frozenset({"ESCALATED"}),
        ),
        "settled": _settled_errors(settle, kind="nonfinite-loss"),
        "no_restart": _no_restart_errors(records),
        "checkpoint_not_trusted": _checkpoint_errors(workflow),
    }
    return _result(stages, extra={"attempt": identity, "settled": settle})


def checkpoint_unverified(ctx: VariantContext) -> dict[str, Any]:
    run_survey(ctx)
    job = _job(ctx, "ckpt")
    ctx.hold(30, name="ckpt-running")
    started = utc_now()
    job.write_progress(
        0, step=10, numerical_error=True, checkpoint_ref="s3://test/last"
    )
    records, incident = _wait_incident(
        ctx, job, "nonfinite-loss", started, name="checkpoint"
    )
    records, incident, settle = _settle(
        ctx, job, "nonfinite-loss", started, records, name="checkpoint"
    )
    workflow = _workflow(records, incident or {})
    stages = {
        "chain": _chain_errors(
            incident,
            workflow,
            kind="nonfinite-loss",
            required_ops=("FREEZE_EVIDENCE", "STOP_WORKLOADS"),
            allow_stop=True,
            states=frozenset({"ESCALATED"}),
        ),
        "settled": _settled_errors(settle, kind="nonfinite-loss"),
        "checkpoint_not_trusted": _checkpoint_errors(workflow),
        "no_restart": _no_restart_errors(records),
    }
    return _result(
        stages,
        notes=[
            "an unvalidated checkpoint is evidence, never auto-selected or restarted"
        ],
        extra={"settled": settle},
    )


def stalled_step(ctx: VariantContext) -> dict[str, Any]:
    run_survey(ctx)
    job = _job(ctx, "stall")
    started = utc_now()
    job.write_progress(0, step=10, loss=1.0)
    # The step stops advancing: resend the same step until the scan latches a
    # stall (last_progress_at stops moving).
    ctx.hold(
        HEARTBEAT_TIMEOUT + 30,
        name="stall-hold",
        check=lambda: job.write_progress(0, step=10, loss=1.0),
    )
    records, incident = _wait_incident(
        ctx, job, "hang", started, name="stall", scan=True
    )
    workflow = _workflow(records, incident or {})
    stages = {
        "chain": _chain_errors(
            incident,
            workflow,
            kind="hang",
            required_ops=("FREEZE_EVIDENCE", "VALIDATE_FABRIC"),
            allow_stop=False,
            states=frozenset(),
        ),
    }
    return _result(
        stages,
        notes=["a fresh heartbeat with a frozen step is a stall, not reporter silence"],
    )


def reporter_silent(ctx: VariantContext) -> dict[str, Any]:
    run_survey(ctx)
    job = _job(ctx, "silent")
    started = utc_now()
    job.write_progress(0, step=10, loss=1.0)
    ctx.hold(30, name="silent-warm")
    # SIGSTOP the reporter: heartbeats stop, the Pod lives. Proven by the
    # reporter's process state, never assumed from the exec's exit status.
    ctx.evidence.write("silent-stop", reporter_stopped(job.stop_reporter(0)))
    ctx.cleanup.push("reporter", lambda: None)
    ctx.hold(HEARTBEAT_TIMEOUT + 30, name="silent-hold")
    records, incident = _wait_incident(
        ctx, job, "reporter-silent", started, name="silent", scan=True
    )
    workflow = _workflow(records, incident or {})
    stages = {
        "chain": _chain_errors(
            incident,
            workflow,
            kind="reporter-silent",
            required_ops=("FREEZE_EVIDENCE", "VALIDATE_FABRIC"),
            allow_stop=False,
            states=frozenset(),
        ),
        "not_hung": []
        if _incident(records, "hang") is None
        else ["reporter silence produced an NCCL/hang finding instead"],
    }
    return _result(
        stages, notes=["a lost heartbeat is reporter silence, not an NCCL hang"]
    )


def step_regression(ctx: VariantContext) -> dict[str, Any]:
    run_survey(ctx)
    job = _job(ctx, "regress")
    started = utc_now()
    job.write_progress(0, step=100, loss=1.0)
    ctx.hold(20, name="regress-advance")
    job.write_progress(0, step=40, loss=1.0)  # a regression outside a restore phase
    records, incident = _wait_incident(
        ctx, job, "step-regression", started, name="regress"
    )
    workflow = _workflow(records, incident or {})
    stages = {
        "chain": _chain_errors(
            incident,
            workflow,
            kind="step-regression",
            required_ops=("FREEZE_EVIDENCE",),
            allow_stop=False,
            states=frozenset(),
        ),
    }
    return _result(stages)


def straggler_group(ctx: VariantContext) -> dict[str, Any]:
    run_survey(ctx)
    # Two ranks are two Pods on two nodes (``--node-b``): the product binds one
    # rank per Pod and compares only allocated ranks (see TWO_NODE_MANIFEST).
    job = _job(ctx, "strag", ranks=(0, 1))
    identity = job.identity()
    started = utc_now()
    labels = {"progress_group": "data-parallel"}
    steps = {0: 100, 1: 100 - STRAGGLER_LAG}
    writes: list[dict[str, Any]] = []

    def advance() -> None:
        # Both ranks keep training every poll, rank 1 STRAGGLER_LAG steps
        # behind: a lagging rank, never a frozen one (that would be a stall).
        for rank in (0, 1):
            job.write_progress(rank, step=steps[rank], loss=1.0, labels=labels)
        writes.append({"at": utc_now().isoformat(), "steps": dict(steps)})
        ctx.evidence.write("straggler-advance", {"writes": writes})
        for rank in steps:
            steps[rank] += STRAGGLER_ADVANCE

    records, incident = _wait_incident(
        ctx,
        job,
        "straggler",
        started,
        name="straggler",
        scan=True,
        before_poll=advance,
    )
    if incident is None:
        # An ingest refusal is answered to the reporter, not to the runner:
        # keep what each reporter logged next to the records that lack it.
        ctx.evidence.write("straggler-reporters", job.reporter_logs())
    workflow = _workflow(records, incident or {})
    stages = {
        "chain": _chain_errors(
            incident,
            workflow,
            kind="straggler",
            required_ops=("FREEZE_EVIDENCE", "VALIDATE_FABRIC"),
            allow_stop=False,
            states=frozenset(),
        ),
        "same_group_only": _straggler_scope(
            incident, records, reference_node=_rank_node(ctx, job, 0)
        ),
        "not_stalled": []
        if _incident(records, "hang") is None
        else ["a lagging rank that keeps advancing was graded as a stall"],
    }
    return _result(
        stages,
        notes=[
            "stragglers are compared only inside one progress_group",
            f"rank 1 runs {STRAGGLER_LAG} steps behind rank 0 (two Pods, two nodes)"
            " while both advance every poll",
        ],
        extra={"attempt": identity},
    )


def bounded_phase(ctx: VariantContext) -> dict[str, Any]:
    run_survey(ctx)
    job = _job(ctx, "phase")
    started = utc_now()
    # INITIALIZING defers stalled-step detection for up to 300 s: no step, in
    # phase, no hang while the budget holds.
    job.write_progress(0, phase="INITIALIZING", labels={"phase": "INITIALIZING"})
    ctx.hold(HEARTBEAT_TIMEOUT + 30, name="phase-hold", check=lambda: _scan(ctx))
    during = _records(ctx, job, started)
    stages = {
        "suppressed": []
        if _incident(during, "hang") is None
        and _incident(during, "progress-unavailable") is None
        else ["the bounded phase did not suppress stall detection within its budget"],
    }
    return _result(
        stages,
        notes=[
            "INITIALIZING defers stall detection <= 300 s; CHECKPOINT_* <= 1800 s;"
            " a stale heartbeat is still reporter-silent"
        ],
    )


def old_attempt(ctx: VariantContext) -> dict[str, Any]:
    """A heartbeat claiming an attempt the product does not run controls nothing.

    The live attempt's own heartbeat is in the store first (the positive
    control: the ingest accepts this job's reporter). Then rank 0's reporter
    sidecar POSTs one heartbeat claiming ``<job>-a000`` -- the job's id with an
    attempt number the product never ran, the live Pod's identity, and a NaN,
    the payload that stops an attempt once accepted -- through the product's
    ingest route. The product must answer ``accepted=false``, keep no progress
    for that attempt, raise no training finding, STOP or restart, and keep
    accepting the live attempt's own heartbeats. An answer that is not the
    ingest's grading (a transport or HTTP error) refused nothing: INCONCLUSIVE.
    """

    run_survey(ctx)
    job = _job(ctx, "old")
    started = utc_now()
    job.write_progress(0, step=10, loss=1.0)
    before: dict[str, Any] = {}

    def heard() -> dict[str, Any] | None:
        nonlocal before
        before = _records(ctx, job, started)
        return before if rank_states(before, 0) else None

    if (
        ctx.wait_until(
            heard, timeout_seconds=HEARTBEAT_TIMEOUT, poll_seconds=15, name="old-live"
        )
        is None
    ):
        raise VariantInconclusive(
            "the live attempt's own heartbeat never reached the store, so a "
            f"refused foreign one would prove nothing: {before.get('progress_states')}"
        )
    foreign = f"{job.name}{FOREIGN_ATTEMPT_SUFFIX}"
    answer = job.post_heartbeat(0, foreign, step=10, loss=1.0, numerical_error=True)
    ctx.evidence.write("old-attempt-foreign", answer)
    refused = foreign_refusal(answer, foreign)
    ctx.hold(FOREIGN_SETTLE_SECONDS, name="old-settle")
    records = _records(ctx, job, started, others=(foreign,))
    stages = {
        "refused": refused,
        "not_recorded": foreign_state_errors(records, foreign),
        "no_stop": []
        if not any(
            "STOP_WORKLOADS" in operations(_workflow(records, item))
            for item in records.get("incidents") or []
        )
        else ["a heartbeat of an attempt the product does not run stopped work"],
        "no_finding": []
        if not records.get("incidents")
        else [
            "training incidents were raised beside the healthy live attempt: "
            + ", ".join(str(item.get("event_id")) for item in records["incidents"])
        ],
        "no_restart": _no_restart_errors(records),
        "live_attempt_kept": live_kept_errors(records, answer),
    }
    return _result(
        stages,
        notes=[
            "a heartbeat for an attempt the product does not run is accepted=false "
            "and controls nothing, even when it carries a NaN"
        ],
        extra={"foreign_attempt": foreign, "foreign_answer": answer.get("body")},
    )


def resumed_progress(ctx: VariantContext) -> dict[str, Any]:
    run_survey(ctx)
    job = _job(ctx, "resume")
    started = utc_now()
    job.write_progress(0, step=STALLED_STEP, loss=1.0)
    ctx.hold(
        HEARTBEAT_TIMEOUT + 30,
        name="resume-stall",
        check=lambda: job.write_progress(0, step=STALLED_STEP, loss=1.0),
    )
    records, incident = _wait_incident(
        ctx, job, "hang", started, name="resume", scan=True
    )
    if incident is None:
        raise VariantInconclusive("no hang finding to recover from")
    # The original rank makes new valid progress while the stall chain is in
    # flight (its VALIDATE_FABRIC waits for evidence newer than itself); only
    # that clears the stall. RECOVERED is the chain's terminal write: judge the
    # record once it has settled, from a read after the new progress.
    job.write_progress(0, step=RESUMED_STEP, loss=1.0)
    records, recovered, settle = _settle(ctx, job, "hang", started, None, name="resume")
    state = (recovered or {}).get("state")
    stages = {
        "recovered": []
        if state == "RECOVERED"
        else [f"the stall incident is {state}, not RECOVERED"],
        "validated_by_progress": validated_by_progress(
            records, recovered or {}, job.attempt_id
        ),
    }
    return _result(
        stages,
        notes=[
            "recovery requires the original rank's newer finite progress, not healthy fabric"
        ],
        extra={"settled": settle},
    )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _no_restart_errors(records: dict[str, Any]) -> list[str]:
    decision = records.get("decision") or {}
    if (
        str(decision.get("status")) in {"PLAN_CREATED"}
        and "RESTART" in str(decision.get("reason", "")).upper()
    ):
        return ["the NaN stop recursively scheduled a restart"]
    for item in records.get("incidents") or []:
        if "RESTART_WORKLOAD" in operations(
            (records.get("workflows") or {}).get(str(item.get("workflow_request_id")))
            or {}
        ):
            return ["a training workflow planned RESTART_WORKLOAD"]
    return []


def _checkpoint_errors(workflow: dict[str, Any]) -> list[str]:
    for step in workflow.get("official_steps") or []:
        disposition = (step.get("parameters") or {}).get("checkpoint_disposition")
        if disposition not in (None, "REQUIRES_APPLICATION_VALIDATION"):
            return [
                f"checkpoint disposition is {disposition}, not application-validated"
            ]
    return []


def _straggler_scope(
    incident: dict[str, Any] | None,
    records: dict[str, Any],
    *,
    reference_node: str,
) -> list[str]:
    if incident is None:
        return []
    # Only rank 1 (the lagging rank) may carry a straggler finding; rank 0 is
    # the reference and must not. A training incident names the graded rank's
    # node alone (``node_ids``); its ``reasons`` carry the finding's text and
    # never the rank (live: "training rank is lagging peer progress"), so a
    # straggler incident on rank 0's node is the reference rank flagged.
    offenders = [
        item["event_id"]
        for item in records.get("incidents") or []
        if str(item.get("event_id", "")).startswith("training-straggler-")
        and reference_node in (item.get("node_ids") or [])
    ]
    return ["the reference rank was flagged as its own straggler"] if offenders else []


def _result(
    stages: dict[str, list[str]],
    *,
    notes: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "verdict": verdict(stages),
        "stages": stages,
        "errors": [
            f"{stage}: {error}" for stage, items in stages.items() for error in items
        ],
        "notes": notes or [],
        **(extra or {}),
    }


def _spec(
    variant: str,
    injection: str,
    expected: str,
    execute,
    *,
    risk: str = "M",
    max_minutes: int = 22,
    lanes: tuple[str, ...] = ("eks", "hp"),
) -> VariantSpec:
    return VariantSpec(
        case_id=CASE,
        variant=variant,
        proof_mode=NATIVE,
        lanes=lanes,
        risk=risk,
        flags="S",
        injection=injection,
        expected=expected,
        cleanup="delete the test Job and progress files; confirm no control job was stopped",
        max_minutes=max_minutes,
        execute=execute,
    )


VARIANTS = [
    _spec(
        "nan",
        "the enabled reporter posts numerical_error for the attempt's rank 0",
        "training_nonfinite-loss -> FREEZE_EVIDENCE/STOP_WORKLOADS/ESCALATE_SUPPORT, ESCALATED; no restart, no cordon",
        nan,
    ),
    _spec(
        "stalled_step",
        "a fresh heartbeat whose step stops advancing past 120 s",
        "a scan mints training_hang (stalled step) -> FREEZE_EVIDENCE/VALIDATE_FABRIC; no STOP/reset",
        stalled_step,
    ),
    _spec(
        "reporter_silent",
        "SIGSTOP the reporter so heartbeats stop while the Pod lives",
        "training_reporter-silent (not a hang) -> FREEZE_EVIDENCE/VALIDATE_FABRIC",
        reporter_silent,
    ),
    _spec(
        "step_regression",
        "post a step lower than the previous outside a restore phase",
        "training_step-regression -> FREEZE_EVIDENCE/VALIDATE_FABRIC; no STOP",
        step_regression,
        risk="L",
    ),
    _spec(
        "straggler_group",
        "two ranks (two Pods on the lane node and --node-b) in one progress_group,"
        " rank 1 60 steps behind rank 0 (30 behind the group median >= 20) while"
        " both keep advancing",
        "training_straggler bound to the lagging rank only; the reference rank is"
        " not flagged and neither rank is graded a stall",
        straggler_group,
        max_minutes=30,
    ),
    _spec(
        "bounded_phase",
        "declare phase INITIALIZING with no step for > 120 s (< the 300 s budget)",
        "stall detection is suppressed inside the phase budget; no hang/progress-unavailable",
        bounded_phase,
        risk="L",
    ),
    _spec(
        "old_attempt",
        "rank 0's reporter sidecar POSTs one heartbeat claiming the job's never-run "
        "attempt id <job>-a000 (the live Pod's identity, numerical_error=true) "
        "through the product's ingest, after the live attempt's own heartbeat is stored",
        "the ingest answers accepted=false and stores no progress for that attempt; "
        "no training finding, STOP or restart; the live attempt stays RUNNING and "
        "its own heartbeats are still accepted",
        old_attempt,
        risk="L",
    ),
    _spec(
        "checkpoint_unverified",
        "NaN with a checkpoint_ref present",
        "STOP + ESCALATE; checkpoint_disposition REQUIRES_APPLICATION_VALIDATION, no auto-restart",
        checkpoint_unverified,
    ),
    _spec(
        "resumed_progress",
        "after a stall, the original rank posts newer finite progress",
        "VALIDATE_FABRIC on the new application progress -> RECOVERED; healthy fabric alone is not enough",
        resumed_progress,
    ),
]
