from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, cast

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gpu_fault.admin.site import load_site  # noqa: E402
from gpu_fault.regional import cluster_token_sha256  # noqa: E402
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalLiveFixture,
    RegionalLiveSettings,
)

EXECUTOR_APP = "gpu-fault-cluster-executor"
CPU_APPS = ("gpu-fault-api-ha", "gpu-fault-control-worker")
REGISTRY_SECRET = "gpu-fault-regional-clusters"
REGISTRY_API_PREFIX = "/v1/regional/registry"
REGISTRY_READY_TIMEOUT_SECONDS = 180
# The execution owner every acceptance claim advertises. No command is ever
# created for it, so a claim that carries it authenticates exactly like the
# executor's own claim (same route, same token, same cluster binding) and
# returns 200 with zero commands: ``claim_remote_commands`` filters candidates
# by ``step.execution_owner`` in every store, and the route only substitutes
# the adapter owner for an *empty* list. The previous probes advertised the
# real ``gpu-fault-kubernetes-adapter`` owner with a 60 s lease, so an
# AUTH-016 sampler firing every 2 s for eight minutes leased -- and starved --
# the production executor's own work. Readiness is never probed with it: that
# route reports 503 for an owner set that leaves backlog unclaimed.
ACCEPTANCE_PROBE_OWNER = "gpu-fault-acceptance-probe"

# Runs inside a control-plane API Pod: the registry API is execution-token
# scoped and the token lives only in that Pod's environment. The body carries
# digests, never plaintext tokens (see durable_registry_payload).
REGISTRY_API_PROBE = r"""
import json
import os
import sys
import urllib.error
import urllib.request

method, path = sys.argv[1], sys.argv[2]
body = sys.argv[3] if len(sys.argv) > 3 else ""
request = urllib.request.Request(
    "http://127.0.0.1:8080" + path,
    data=body.encode() if body else None,
    method=method,
    headers={
        "X-GPU-Fault-Execution-Token": os.environ["GPU_FAULT_EXECUTION_TOKEN"],
        "Content-Type": "application/json",
    },
)
try:
    with urllib.request.urlopen(request, timeout=30) as response:
        status, text = response.status, response.read().decode()
except urllib.error.HTTPError as exc:
    status, text = exc.code, exc.read().decode()
try:
    value = json.loads(text)
except ValueError:
    value = {"raw": text[:2000]}
print(json.dumps({"status": status, "body": value}))
"""

REGISTRY_REVISION_PROBE = r"""
import json
from gpu_fault.app import ApplicationContext
store = ApplicationContext.from_environment().store
head = store.get_regional_registry_head()
revision = store.get_regional_registry_revision(head.generation)
print(json.dumps({
    "generation": head.generation,
    "registrations": [item.model_dump(mode="json") for item in revision.registrations],
}, default=str))
"""


def durable_registry_payload(
    entries: list[dict[str, Any]],
    *,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Registrations as the revision API takes them: digests, never plaintext.

    The Secret-era helpers hand the registry plaintext ``token`` /
    ``retiring_token`` fields (the control plane used to digest them at load).
    ``RegionalClusterRegistration`` is strict and stores only ``token_sha256`` /
    ``retiring_token_sha256``, so the digesting happens here, on the runner, and
    the plaintext never reaches argv, logs or evidence. A rotated entry gets a
    fresh ``updated_at``: the rotation window is measured from it.
    """

    observed = (now or datetime.now(timezone.utc)).isoformat().replace("+00:00", "Z")
    payload: list[dict[str, Any]] = []
    for entry in entries:
        item = dict(entry)
        rotated = False
        token = item.pop("token", None)
        if token:
            item["token_sha256"] = cluster_token_sha256(str(token))
            rotated = True
        retiring = item.pop("retiring_token", None)
        if retiring:
            item["retiring_token_sha256"] = cluster_token_sha256(str(retiring))
            rotated = True
        if rotated:
            item["updated_at"] = observed
        payload.append(item)
    return payload


CONNECTION_SECRET = "gpu-fault-regional-connection"
FORBIDDEN_NAMESPACE = "gf-forbidden-probe"
WRITE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


class IdentityAcceptanceError(RuntimeError):
    pass


class IdentityCaseFailure(IdentityAcceptanceError):
    """A case handler failed after it had already gathered partial evidence.

    ``details`` carries whatever checks, samples and restore state the handler
    had when the exception surfaced, so ``run_identity_acceptance`` writes them
    into the case evidence instead of only the exception text.
    """

    def __init__(self, message: str, *, details: dict[str, Any]) -> None:
        super().__init__(message)
        self.details = details


def run_cleanup_steps(
    steps: list[tuple[str, Callable[[], Any]]],
) -> tuple[dict[str, Any], list[str]]:
    """Run every restore step, whatever the earlier ones did.

    A ``finally`` block that calls its restore steps back to back stops at the
    first one that raises, leaves the later resources (a token Secret, a probe
    Pod, a namespace) in place and replaces the exception that ended the case
    with the cleanup's. Every step here runs; the ones that raised come back as
    ``cleanup_errors`` for the caller to record and fail on, and the caller's
    original exception, if any, is the one that propagates.
    """

    outcomes: dict[str, Any] = {}
    errors: list[str] = []
    for name, step in steps:
        try:
            outcomes[name] = step()
        except Exception as exc:  # noqa: BLE001 - every step must run
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    return outcomes, errors


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(
    command: list[str],
    *,
    input_text: str | None = None,
    check: bool = True,
    timeout: int = 300,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout,
        cwd=cwd,
        env=env,
    )
    if check and completed.returncode:
        raise IdentityAcceptanceError(
            f"command failed ({completed.returncode}): {' '.join(command[:5])}; "
            f"stderr={completed.stderr[-1000:]}"
        )
    return completed


@dataclass(frozen=True)
class ClusterTarget:
    cluster_id: str
    context: str
    region: str
    hyperpod_cluster_name: str
    eks_cluster_arn: str
    executor_role_arn: str
    control_plane_url: str
    ca_file: Path


class IdentitySite:
    def __init__(self, site_file: Path) -> None:
        self.site_file = site_file.resolve()
        self.site = load_site(self.site_file, repository_root=ROOT)
        self.config = self.site.release_config
        self.namespace = str(self.config["namespace"])
        self.region = str(self.config["aws_region"])
        self.cpu_kubeconfig = Path(str(self.config["cpu_kubeconfig"])).resolve()
        gpu_value = str(
            self.config.get("gpu_kubeconfig")
            or self.site.environment.get("KUBECONFIG")
            or ""
        )
        if not gpu_value:
            raise IdentityAcceptanceError("site does not resolve a GPU kubeconfig")
        self.gpu_kubeconfig = Path(gpu_value).resolve()
        self.targets = {
            str(item["cluster_id"]): ClusterTarget(
                cluster_id=str(item["cluster_id"]),
                context=str(item["context"]),
                region=str(item["region"]),
                hyperpod_cluster_name=str(item["hyperpod_cluster_name"]),
                eks_cluster_arn=str(item["eks_cluster_arn"]),
                executor_role_arn=str(item["executor_irsa_role_arn"]),
                control_plane_url=str(item["control_plane_url"]),
                ca_file=Path(str(item["ca_file"])).resolve(),
            )
            for item in self.config["clusters"]
        }
        if not self.targets:
            raise IdentityAcceptanceError("site contains no GPU clusters")
        self._api_pod_name: str | None = None
        # Seconds from the last revision POST until every required member had
        # acked it (``missing_member_ids`` empty): the isolation delay AUTH-007
        # and AUTH-016 record. ``rollout_control`` after ``write_registry`` used
        # to be measured instead, and it reads a head that is already applied:
        # one status call, ~1 s, whatever the propagation actually took.
        self.last_registry_ready_seconds: float | None = None

    def target(self, cluster_id: str) -> ClusterTarget:
        if not cluster_id:
            if len(self.targets) != 1:
                raise IdentityAcceptanceError(
                    "--cluster-id is required for a multi-cluster site"
                )
            return next(iter(self.targets.values()))
        try:
            return self.targets[cluster_id]
        except KeyError as exc:
            raise IdentityAcceptanceError(
                f"cluster is not present in the site: {cluster_id}"
            ) from exc

    def regional(self, target: ClusterTarget) -> RegionalLiveFixture:
        return RegionalLiveFixture(
            RegionalLiveSettings(
                cpu_kubeconfig=self.cpu_kubeconfig,
                gpu_kubeconfig=self.gpu_kubeconfig,
                gpu_context=target.context,
                namespace=self.namespace,
                cluster_id=target.cluster_id,
                region=self.region,
            )
        )

    def cpu(self, *arguments: str, **kwargs: Any) -> str:
        target = next(iter(self.targets.values()))
        return str(self.regional(target).kubectl("cpu", *arguments, **kwargs))

    def gpu(self, target: ClusterTarget, *arguments: str, **kwargs: Any) -> str:
        return str(self.regional(target).kubectl("gpu", *arguments, **kwargs))

    def ready_pods(self, plane: str, app: str, target: ClusterTarget) -> list[str]:
        value = json.loads(
            self.regional(target).kubectl(
                plane,
                "get",
                "pod",
                "-l",
                f"app={app}",
                "-o",
                "json",
            )
        )
        result = []
        for item in value.get("items", []):
            conditions = item.get("status", {}).get("conditions", [])
            if item.get("status", {}).get("phase") != "Running":
                continue
            if not any(
                condition.get("type") == "Ready" and condition.get("status") == "True"
                for condition in conditions
            ):
                continue
            result.append(str(item["metadata"]["name"]))
        return sorted(result)

    def any_executor_pod(self, target: ClusterTarget) -> str:
        value = json.loads(
            self.gpu(
                target,
                "get",
                "pod",
                "-l",
                f"app={EXECUTOR_APP}",
                "-o",
                "json",
            )
        )
        names = sorted(
            str(item["metadata"]["name"])
            for item in value.get("items", [])
            if item.get("status", {}).get("phase") == "Running"
        )
        if not names:
            raise IdentityAcceptanceError(
                f"no Running executor Pod for {target.cluster_id}"
            )
        return names[0]

    def pod_json(
        self,
        plane: str,
        target: ClusterTarget,
        pod: str,
        script: str,
        *arguments: str,
        timeout: int = 180,
    ) -> dict[str, Any]:
        output = self.regional(target).kubectl(
            plane,
            "exec",
            "-i",
            pod,
            "--",
            "python3",
            "-",
            *arguments,
            input_text=script,
            timeout=timeout,
        )
        value = json.loads(output.splitlines()[-1])
        if not isinstance(value, dict):
            raise IdentityAcceptanceError("Pod probe did not return a JSON object")
        return cast(dict[str, Any], value)

    # -- registry: durable revisions first, the bootstrap Secret only before any
    # revision exists. Once the control plane has published a revision (every
    # production site), ``clusters.json`` is a bootstrap copy that nothing
    # reads; editing it and rolling the control plane changes nothing, and
    # AUTH-016 lost its new token to a 403 that way (live, 2026-09-07).

    def _api_pod(self) -> str:
        """A Ready control-plane API Pod, selected once and reused.

        Every registry call used to list the API Pods first -- one extra
        ``kubectl get`` per call, and AUTH-016 makes hundreds of them. The
        name is cached; ``api_pod_json`` drops it when an exec against it fails
        so the next call selects again (the Pod may have rolled).
        """

        cached: str | None = getattr(self, "_api_pod_name", None)
        if cached is None:
            target = next(iter(self.targets.values()))
            pods = self.ready_pods("cpu", "gpu-fault-api-ha", target)
            if not pods:
                raise IdentityAcceptanceError("no Ready control-plane API Pod")
            cached = str(pods[0])
            self._api_pod_name = cached
        return cached

    def api_pod_json(self, script: str, *arguments: str) -> dict[str, Any]:
        """``pod_json`` against the cached API Pod; re-selects once on failure."""

        target = next(iter(self.targets.values()))
        try:
            return self.pod_json("cpu", target, self._api_pod(), script, *arguments)
        except Exception:
            self._api_pod_name = None
            return self.pod_json("cpu", target, self._api_pod(), script, *arguments)

    def registry_api(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        arguments = [method, path]
        if payload is not None:
            arguments.append(json.dumps(payload, separators=(",", ":")))
        return self.api_pod_json(REGISTRY_API_PROBE, *arguments)

    def registry_generation(self) -> int | None:
        """The durable registry head generation, or None while the site still
        runs from the bootstrap Secret."""

        response = self.registry_api("GET", f"{REGISTRY_API_PREFIX}/status")
        status = int(response.get("status") or 0)
        if status == 404:
            return None
        if status != 200:
            raise IdentityAcceptanceError(
                f"registry status returned {status}: {response.get('body')}"
            )
        return int(response["body"]["generation"])

    def wait_registry_ready(
        self, minimum_generation: int, *, timeout_seconds: int | None = None
    ) -> dict[str, Any]:
        deadline = time.monotonic() + (
            timeout_seconds or REGISTRY_READY_TIMEOUT_SECONDS
        )
        last: dict[str, Any] = {}
        while time.monotonic() < deadline:
            response = self.registry_api("GET", f"{REGISTRY_API_PREFIX}/status")
            last = response.get("body") or {}
            # RegionalRegistryStatus has no "ready" flag: the head is applied
            # when every required member has acked it.
            if (
                int(response.get("status") or 0) == 200
                and int(last.get("generation") or 0) >= minimum_generation
                and "missing_member_ids" in last
                and not last.get("missing_member_ids")
            ):
                return cast(dict[str, Any], last)
            time.sleep(2)
        raise IdentityAcceptanceError(
            f"registry generation {minimum_generation} did not become ready: {last}"
        )

    def registry(self) -> list[dict[str, Any]]:
        if self.registry_generation() is not None:
            revision = self.api_pod_json(REGISTRY_REVISION_PROBE)
            return cast(list[dict[str, Any]], revision["registrations"])
        value = json.loads(
            self.cpu(
                "get",
                "secret",
                REGISTRY_SECRET,
                "-o",
                "json",
            )
        )
        return cast(
            list[dict[str, Any]],
            json.loads(base64.b64decode(value["data"]["clusters.json"])),
        )

    def write_registry(
        self,
        entries: list[dict[str, Any]],
        *,
        reason: str = "regional acceptance registry change",
    ) -> None:
        generation = self.registry_generation()
        if generation is not None:
            posted = time.monotonic()
            response = self.registry_api(
                "POST",
                f"{REGISTRY_API_PREFIX}/revisions",
                {
                    "expected_generation": generation,
                    "registrations": durable_registry_payload(entries),
                    "reason": reason,
                },
            )
            if int(response.get("status") or 0) != 200:
                raise IdentityAcceptanceError(
                    f"registry revision publish returned {response.get('status')}: "
                    f"{response.get('body')}"
                )
            self.wait_registry_ready(generation + 1)
            self.last_registry_ready_seconds = time.monotonic() - posted
            return
        self.last_registry_ready_seconds = None
        patch = {
            "stringData": {"clusters.json": json.dumps(entries, separators=(",", ":"))}
        }
        self.cpu(
            "patch",
            "secret",
            REGISTRY_SECRET,
            "--type=merge",
            "--patch-file=/dev/stdin",
            input_text=json.dumps(patch, separators=(",", ":")),
        )

    def rollout_control(self) -> float:
        """Make a registry change effective on the control plane.

        On a durable-revision site this is a no-op: ``write_registry`` already
        waited for every member's ack, and the propagation time is in
        ``last_registry_ready_seconds``. Only a bootstrap-Secret site still
        needs the control-plane Deployments restarted.
        """

        started = time.monotonic()
        generation = self.registry_generation()
        if generation is not None:
            return time.monotonic() - started
        for deployment in CPU_APPS:
            self.cpu("rollout", "restart", f"deployment/{deployment}")
        for deployment in CPU_APPS:
            self.cpu(
                "rollout",
                "status",
                f"deployment/{deployment}",
                "--timeout=600s",
                timeout=660,
            )
        return time.monotonic() - started


CLAIM_PROBE = r"""
import json
import os
import ssl
import time
import urllib.error
import urllib.request

payload = {
    "executor_id": "regional-identity-acceptance",
    "executor_protocol_version": int(
        os.getenv("GPU_FAULT_EXECUTOR_PROTOCOL_VERSION", "2")
    ),
    "executor_artifact_sha256": os.environ[
        "GPU_FAULT_EXECUTOR_ARTIFACT_SHA256"
    ],
    "executor_compatibility_digest": os.environ[
        "GPU_FAULT_EXECUTOR_COMPATIBILITY_DIGEST"
    ],
    "execution_owners": ["__PROBE_OWNER__"],
    "max_commands": 1,
    "lease_seconds": 60,
}
request = urllib.request.Request(
    os.environ["GPU_FAULT_CONTROL_PLANE_URL"].rstrip("/")
    + "/v1/regional/executors/claim",
    data=json.dumps(payload, separators=(",", ":")).encode(),
    method="POST",
    headers={
        "Authorization": "Bearer " + os.environ["GPU_FAULT_CONTROL_PLANE_TOKEN"],
        "Content-Type": "application/json",
        "X-GPU-Fault-Cluster-ID": os.environ["GPU_FAULT_CLUSTER_ID"],
    },
)
context = ssl.create_default_context(
    cafile=os.environ["GPU_FAULT_CONTROL_PLANE_CA_FILE"]
)
started = time.monotonic()
try:
    with urllib.request.urlopen(request, context=context, timeout=20) as response:
        body = json.loads(response.read() or b"{}")
        print(json.dumps({
            "status": response.status,
            "latency_seconds": time.monotonic() - started,
            "command_count": len(body.get("commands") or []),
        }))
except urllib.error.HTTPError as exc:
    print(json.dumps({
        "status": exc.code,
        "latency_seconds": time.monotonic() - started,
        "detail": exc.read().decode()[:500],
    }))
except Exception as exc:
    # URLError, socket timeout, TLS failure: a transport-level outcome, which
    # is exactly what ISO-006 has to observe from the blocked cluster.
    print(json.dumps({
        "status": None,
        "transport_error": type(exc).__name__,
        "latency_seconds": time.monotonic() - started,
    }))
""".replace("__PROBE_OWNER__", ACCEPTANCE_PROBE_OWNER)


def claim(site: IdentitySite, target: ClusterTarget) -> dict[str, Any]:
    pod = site.any_executor_pod(target)
    return site.pod_json("gpu", target, pod, CLAIM_PROBE)


def claim_sample(fixture: RegionalLiveFixture) -> dict[str, Any]:
    """One probe-owner claim from a Ready executor Pod of ``fixture``'s cluster.

    ``attempts=1``: the claim is authenticated traffic against the control
    plane and its latency is the measurement; a kubectl retry would both
    double the request and report the retry's latency as the sample's.
    """

    return fixture.executor_python(CLAIM_PROBE, timeout=60, attempts=1)


def secret_digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def read_cluster_token(site: IdentitySite, target: ClusterTarget) -> str:
    value = json.loads(
        site.gpu(
            target,
            "get",
            "secret",
            CONNECTION_SECRET,
            "-o",
            "json",
        )
    )
    return base64.b64decode(value["data"]["cluster-token"]).decode().strip()


def write_cluster_token(
    site: IdentitySite,
    target: ClusterTarget,
    token: str,
) -> None:
    patch = {
        "stringData": {
            "cluster-token": token,
        },
    }
    site.gpu(
        target,
        "patch",
        "secret",
        CONNECTION_SECRET,
        "--type=merge",
        "--patch-file=/dev/stdin",
        input_text=json.dumps(patch, separators=(",", ":")),
    )


def rollout_executor(site: IdentitySite, target: ClusterTarget) -> float:
    started = time.monotonic()
    site.gpu(target, "rollout", "restart", f"deployment/{EXECUTOR_APP}")
    site.gpu(
        target,
        "rollout",
        "status",
        f"deployment/{EXECUTOR_APP}",
        "--timeout=600s",
        timeout=660,
    )
    return time.monotonic() - started
