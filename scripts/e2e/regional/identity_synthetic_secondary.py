"""A second logical cluster for AUTH-007/008 on a one-cluster site.

Both cases need a *registered* cluster B with a data plane of its own: AUTH-007
disables B's registration and watches B's claim turn 403 while A keeps 200;
AUTH-008 seeds a probe-owner command in B, proves neither header/token
cross-pairing can lease it and then lets B lease it through its own executor.
The live site has one physical GPU cluster and ``join-cluster`` refuses a
second ``cluster_id`` for the same EKS, so the spec's single-cluster recipe
(a second ``cluster_id`` registering the same EKS with its own token, its own
``allowed_namespaces`` and its own executor in another namespace) is realised
here by the runner itself, for the length of one case:

* the registry gets one more durable revision carrying B as a ``synthetic``
  registration -- minted by the perf suite's synthetic builder, so it has a
  fresh random token, a run id and an expiry the perf sweep and the registry's
  load-time hygiene both recognise -- and one more revision without it at the
  end;
* the GPU plane gets one namespace named after B holding B's own
  ``gpu-fault-regional-connection`` Secret (A's control-plane URL and CA
  copied, B's token and id), the hold script ConfigMap and one Pod labelled
  ``app=gpu-fault-cluster-executor`` that runs the deployed executor image with
  the executor's connection environment but never starts the claim loop
  (``probes/auth_logical_secondary_hold.py``); ``IdentitySite.regional`` maps
  B's target onto that namespace, so the case bodies exec their probes there
  exactly as they would in a second site cluster.

Everything created is labelled with the run id, journaled with its UID before
the create is acknowledged and deleted with UID/resourceVersion preconditions
(``seeded_command_fixture.delete_owned_resource``), whatever the case body did.
The evidence card records the revision generations before/during/after, the
Pod identity, every cleanup step and a proof that no registration, namespace or
Pod of B remains and that A's registration is byte-for-byte what it was.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from scripts.e2e.regional import seeded_command_fixture as seeded
from scripts.e2e.regional.acceptance_runner_common import write_json_atomic
from scripts.e2e.regional.identity_acceptance_common import (
    CONNECTION_SECRET,
    EXECUTOR_APP,
    ClusterTarget,
    IdentityAcceptanceError,
    IdentityCaseFailure,
    registry_content,
    run_cleanup_steps,
    utc_now,
)
from scripts.e2e.regional.identity_auth_probes import AUTH008_EXECUTOR_IDENTITY_PROBE
from scripts.perf import regional_capacity_registry as perf_registry

SITE_SECONDARY = "site"
SYNTHETIC_SECONDARY = "synthetic"
SECONDARY_REGISTRATION_MODES = (SITE_SECONDARY, SYNTHETIC_SECONDARY)
SYNTHETIC_SECONDARY_KIND = "synthetic-logical"
SYNTHETIC_SECONDARY_CASES = frozenset({"GF-REGIONAL-AUTH-007", "GF-REGIONAL-AUTH-008"})
SYNTHETIC_SECONDARY_PREFIX = "auth-logical-"
# The id doubles as B's namespace, so it is a DNS label; the prefix keeps it
# apart from every real cluster id (those come from site.yaml and never carry
# it) and from the perf suite's ``perf-cap-`` ids.
SYNTHETIC_SECONDARY_ID = re.compile(
    r"auth-logical-[a-z0-9](?:[a-z0-9-]{0,36}[a-z0-9])?"
)
# How long B may authenticate even if this process dies mid-case: the
# registration's ``synthetic_expires_at`` and the Pod's ``activeDeadlineSeconds``
# both come from it, so an orphaned B stops being a credential on its own.
SYNTHETIC_SECONDARY_LIFETIME = timedelta(minutes=30)
HOLD_PROBE = Path(__file__).with_name("probes") / "auth_logical_secondary_hold.py"
SECONDARY_POD = "gpu-fault-cluster-executor-synthetic"
SECONDARY_HOLD_CONFIGMAP = "gpu-fault-cluster-executor-hold"
CASE_LABEL = "gpu-fault.io/acceptance-case"
SECONDARY_LABEL = "gpu-fault.io/synthetic-secondary"
RUN_LABEL: str = seeded.RUN_LABEL
CA_MOUNT_PATH = "/etc/gpu-fault/tls"
# Copied from A's connection Secret into B's: where the control plane is and
# how to trust it. Never A's token, id or namespaces.
COPIED_CONNECTION_KEYS = ("control-plane-url", "ca.crt")
POD_READY_TIMEOUT_SECONDS = 180
JOURNAL_FILE = "synthetic-secondary.json"
HEX64 = re.compile(r"[0-9a-f]{64}")
SYNTHETIC_LIMITATION = (
    "Cluster B is a synthetic logical registration on the primary's EKS with its "
    "own token, namespace and executor Pod: the denial and recovery semantics are "
    "those of any registered cluster, but B has no nodes, agents or workloads, so "
    "this run does not show a second physical cluster's data plane being isolated "
    "(GF-REGIONAL-ISO-006 stays unverified)."
)


def validate_synthetic_secondary_id(
    cluster_id: str, *, site_cluster_ids: Iterable[str]
) -> str:
    if SYNTHETIC_SECONDARY_ID.fullmatch(cluster_id or "") is None:
        raise IdentityAcceptanceError(
            "a synthetic secondary id must match auth-logical-<dns-label> "
            "(lowercase letters, digits and hyphens; at most 50 characters)"
        )
    if cluster_id in set(site_cluster_ids):
        raise IdentityAcceptanceError(
            f"synthetic secondary id collides with a site cluster: {cluster_id}"
        )
    return cluster_id


def synthetic_secondary(primary: ClusterTarget, cluster_id: str) -> ClusterTarget:
    """Cluster B as a synthetic logical cluster on the primary's EKS.

    Same kube context, control plane URL and CA as the primary; its own
    namespace (named after it) for the executor Pod; no HyperPod or IRSA
    identity of its own, because it has no physical cluster. ``registered`` is
    False because site.yaml does not list it -- the runner registers it for
    the case and the execute evidence says so.
    """

    if cluster_id == primary.cluster_id:
        raise IdentityAcceptanceError("primary and secondary clusters must differ")
    return replace(
        primary,
        cluster_id=cluster_id,
        hyperpod_cluster_name=cluster_id,
        eks_cluster_arn="",
        executor_role_arn="",
        registered=False,
        kind=SYNTHETIC_SECONDARY_KIND,
        executor_namespace=cluster_id,
    )


def synthetic_secondary_from_arguments(
    arguments: Any, site: Any, primary: ClusterTarget
) -> ClusterTarget | None:
    """Cluster B for ``--secondary-registration synthetic``; None in site mode.

    Site mode is the default and is left exactly as it was; the synthetic mode
    needs the explicit allow switch, one of the two cases that disable or
    claim as B, and an id that cannot be a real cluster's.
    """

    mode = (
        getattr(arguments, "secondary_registration", SITE_SECONDARY) or SITE_SECONDARY
    )
    allowed = bool(getattr(arguments, "allow_synthetic_secondary", False))
    if mode == SITE_SECONDARY:
        if allowed:
            raise IdentityAcceptanceError(
                "--allow-synthetic-secondary requires --secondary-registration synthetic"
            )
        return None
    if mode != SYNTHETIC_SECONDARY:
        raise IdentityAcceptanceError(f"unknown secondary registration mode: {mode}")
    if arguments.case not in SYNTHETIC_SECONDARY_CASES:
        raise IdentityAcceptanceError(
            "a synthetic secondary is only for GF-REGIONAL-AUTH-007 and "
            "GF-REGIONAL-AUTH-008"
        )
    if not allowed:
        raise IdentityAcceptanceError(
            "--secondary-registration synthetic requires --allow-synthetic-secondary"
        )
    cluster_id = validate_synthetic_secondary_id(
        str(getattr(arguments, "secondary_cluster_id", "") or ""),
        site_cluster_ids=getattr(site, "targets", {}),
    )
    return synthetic_secondary(primary, cluster_id)


def synthetic_run_id(cluster_id: str, run_dir: Path, attempt: int) -> str:
    """The run identity every owned resource and the registration carry."""

    suffix = re.sub(r"[^a-z0-9]+", "-", run_dir.name.rsplit("-", 1)[-1].lower())
    suffix = suffix.strip("-") or "run"
    # A Kubernetes label value (63 characters): the run-directory suffix gives
    # way before the cluster id and attempt do.
    tail = f"-a{int(attempt)}"
    suffix = suffix[: max(4, 63 - len(cluster_id) - len(tail) - 1)]
    return f"{cluster_id}-{suffix}{tail}"


def synthetic_registration(
    cluster_id: str,
    *,
    run_id: str,
    expires_at: datetime,
    region: str,
    namespace: str,
    token: str | None = None,
) -> dict[str, Any]:
    """B's registry entry, minted by the perf suite's synthetic builder.

    Plaintext ``token`` included: ``IdentitySite.write_registry`` digests it
    on the way to the revision API and the evidence only ever sees the digest.
    ``allowed_namespaces`` is B's own namespace alone, as the spec's second
    logical cluster has.
    """

    return perf_registry.synthetic_cluster_entry(
        cluster_id,
        run_id=run_id,
        expires_at=expires_at,
        region=region,
        allowed_namespaces=[namespace],
        token=token,
    )


def redacted_registration(entry: dict[str, Any]) -> dict[str, Any]:
    return perf_registry.redacted_registry_entries([entry])[0]


def executor_template(site: Any, primary: ClusterTarget) -> dict[str, str]:
    """The deployed executor image and pins B's Pod must carry.

    Read from A's executor Deployment: the image is what B runs, the artifact
    and compatibility pins are what the control plane checks on a claim, and a
    B Pod pinned differently would read 503 instead of the 200/403 the cases
    judge.
    """

    deployment = json.loads(
        site.gpu(primary, "get", "deployment", EXECUTOR_APP, "-o", "json")
    )
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    environment = {
        item["name"]: item.get("value")
        for item in container.get("env", [])
        if "value" in item
    }
    template = {
        "image": str(container.get("image") or ""),
        "artifact": str(environment.get("GPU_FAULT_EXECUTOR_ARTIFACT_SHA256") or ""),
        "compatibility": str(
            environment.get("GPU_FAULT_EXECUTOR_COMPATIBILITY_DIGEST") or ""
        ),
    }
    if not template["image"] or any(
        HEX64.fullmatch(template[key]) is None for key in ("artifact", "compatibility")
    ):
        raise IdentityAcceptanceError(
            "the primary executor Deployment does not carry an image and 64-hex "
            "artifact/compatibility pins"
        )
    return template


def _owned_labels(*, run_id: str, case_id: str, cluster_id: str) -> dict[str, str]:
    return {
        RUN_LABEL: run_id,
        CASE_LABEL: case_id,
        SECONDARY_LABEL: cluster_id,
    }


def namespace_manifest(*, cluster_id: str, run_id: str, case_id: str) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "Namespace",
        "metadata": {
            "name": cluster_id,
            "labels": _owned_labels(
                run_id=run_id, case_id=case_id, cluster_id=cluster_id
            ),
        },
    }


def connection_secret_manifest(
    *,
    cluster_id: str,
    namespace: str,
    run_id: str,
    case_id: str,
    token: str,
    copied: dict[str, str],
) -> dict[str, Any]:
    """B's ``gpu-fault-regional-connection``: A's URL and CA, B's id and token.

    Same Secret name and keys the executor Deployment reads, so B's Pod is
    wired exactly like a second executor Deployment in another namespace
    would be. ``copied`` carries the base64 values as A's Secret holds them.
    """

    missing = [key for key in COPIED_CONNECTION_KEYS if not copied.get(key)]
    if missing:
        raise IdentityAcceptanceError(
            "primary connection Secret lacks keys to copy: " + ", ".join(missing)
        )

    def encode(value: str) -> str:
        return base64.b64encode(value.encode()).decode()

    return {
        "apiVersion": "v1",
        "kind": "Secret",
        "type": "Opaque",
        "metadata": {
            "name": CONNECTION_SECRET,
            "namespace": namespace,
            "labels": _owned_labels(
                run_id=run_id, case_id=case_id, cluster_id=cluster_id
            ),
        },
        "data": {
            **{key: copied[key] for key in COPIED_CONNECTION_KEYS},
            "cluster-id": encode(cluster_id),
            "cluster-token": encode(token),
            "allowed-namespaces": encode(namespace),
        },
    }


def hold_configmap_manifest(
    *, cluster_id: str, namespace: str, run_id: str, case_id: str
) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": SECONDARY_HOLD_CONFIGMAP,
            "namespace": namespace,
            "labels": _owned_labels(
                run_id=run_id, case_id=case_id, cluster_id=cluster_id
            ),
        },
        "data": {HOLD_PROBE.name: HOLD_PROBE.read_text(encoding="utf-8")},
    }


def secondary_pod_manifest(
    *,
    cluster_id: str,
    namespace: str,
    run_id: str,
    case_id: str,
    image: str,
    pins: dict[str, str],
    lifetime_seconds: int,
) -> dict[str, Any]:
    """B's executor Pod: the deployed image, the executor's connection env,
    the hold script instead of the claim loop.

    The recipe is the seeded-command fixture's -- one throwaway Pod,
    ``restartPolicy: Never``, a bounded ``activeDeadlineSeconds``, a short
    grace period, the probe script from a ConfigMap at ``/scripts``, the CA
    from the connection Secret's ``ca.crt``, an ``emptyDir`` at ``/state`` for
    ``ready.json`` -- with the connection environment the executor Deployment
    declares (URL, token, cluster id and allowed namespaces from
    ``gpu-fault-regional-connection``, CA at ``/etc/gpu-fault/tls/ca.crt``,
    the artifact and compatibility pins) so every claim probe the cases exec
    finds what it reads on a real executor. No ServiceAccount token: the Pod
    never touches the Kubernetes API.
    """

    def from_secret(name: str, key: str) -> dict[str, Any]:
        return {
            "name": name,
            "valueFrom": {"secretKeyRef": {"name": CONNECTION_SECRET, "key": key}},
        }

    environment: list[dict[str, Any]] = [
        {
            "name": "PATH",
            "value": (
                "/opt/gpu-fault/executor/bin:/usr/local/sbin:/usr/local/bin:"
                "/usr/sbin:/usr/bin:/sbin:/bin"
            ),
        },
        from_secret("GPU_FAULT_CONTROL_PLANE_URL", "control-plane-url"),
        from_secret("GPU_FAULT_CONTROL_PLANE_TOKEN", "cluster-token"),
        from_secret("GPU_FAULT_CLUSTER_ID", "cluster-id"),
        from_secret("GPU_FAULT_ALLOWED_WORKLOAD_NAMESPACES", "allowed-namespaces"),
        {"name": "GPU_FAULT_CONTROL_PLANE_CA_FILE", "value": f"{CA_MOUNT_PATH}/ca.crt"},
        {"name": "GPU_FAULT_EXECUTOR_ARTIFACT_SHA256", "value": pins["artifact"]},
        {
            "name": "GPU_FAULT_EXECUTOR_COMPATIBILITY_DIGEST",
            "value": pins["compatibility"],
        },
        {
            "name": "GPU_FAULT_CLUSTER_EXECUTOR_ID",
            "value": f"{cluster_id}/{SECONDARY_POD}",
        },
    ]
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": SECONDARY_POD,
            "namespace": namespace,
            "labels": {
                "app": EXECUTOR_APP,
                **_owned_labels(run_id=run_id, case_id=case_id, cluster_id=cluster_id),
            },
        },
        "spec": {
            "restartPolicy": "Never",
            "activeDeadlineSeconds": int(lifetime_seconds),
            "terminationGracePeriodSeconds": 5,
            "automountServiceAccountToken": False,
            # The executor Deployment's own tolerations: B's Pod may schedule
            # exactly where an executor may, and nowhere else.
            "tolerations": [
                {
                    "key": "gpu-fault.io/quarantined",
                    "operator": "Exists",
                    "effect": "NoSchedule",
                },
                {
                    "key": "node.kubernetes.io/unschedulable",
                    "operator": "Exists",
                    "effect": "NoSchedule",
                },
            ],
            "containers": [
                {
                    "name": "executor",
                    "image": image,
                    "command": [
                        "/opt/gpu-fault/executor/bin/python",
                        f"/scripts/{HOLD_PROBE.name}",
                    ],
                    "env": environment,
                    "readinessProbe": {
                        "exec": {
                            "command": ["/bin/sh", "-c", "test -f /state/ready.json"]
                        },
                        "periodSeconds": 2,
                        "timeoutSeconds": 2,
                        "failureThreshold": 3,
                    },
                    "resources": {
                        "requests": {"cpu": "50m", "memory": "128Mi"},
                        "limits": {"cpu": "500m", "memory": "512Mi"},
                    },
                    "securityContext": {
                        "allowPrivilegeEscalation": False,
                        "capabilities": {"drop": ["ALL"]},
                    },
                    "volumeMounts": [
                        {"name": "script", "mountPath": "/scripts", "readOnly": True},
                        {"name": "tls", "mountPath": CA_MOUNT_PATH, "readOnly": True},
                        {"name": "state", "mountPath": "/state"},
                    ],
                }
            ],
            "volumes": [
                {"name": "script", "configMap": {"name": SECONDARY_HOLD_CONFIGMAP}},
                {
                    "name": "tls",
                    "secret": {
                        "secretName": CONNECTION_SECRET,
                        "items": [{"key": "ca.crt", "path": "ca.crt"}],
                    },
                },
                {"name": "state", "emptyDir": {}},
            ],
        },
    }


@dataclass
class SyntheticSecondaryJournal:
    """What this run owns, written to the case directory before every step.

    A run that dies between two steps leaves the operator the exact resource
    identities to remove by hand; ``resources`` maps ``kind/name`` to the UID
    the create returned (None while the create is in flight).
    """

    run_id: str
    case_id: str
    cluster_id: str
    namespace: str
    physical_cluster_id: str
    registry_started: bool = False
    registry_generation_before: int | None = None
    registry_generation_with_secondary: int | None = None
    registration: dict[str, Any] | None = None
    primary_registration: dict[str, Any] | None = None
    expires_at: str | None = None
    resources: dict[str, str | None] = field(default_factory=dict)
    pod: dict[str, Any] | None = None
    preflight: dict[str, Any] | None = None
    updated_at: str = ""


class SyntheticSecondary:
    """Register, run and remove cluster B for one AUTH-007/008 execution."""

    def __init__(
        self,
        site: Any,
        primary: ClusterTarget,
        secondary: ClusterTarget,
        *,
        case_id: str,
        case_dir: Path,
        run_id: str,
        lifetime: timedelta = SYNTHETIC_SECONDARY_LIFETIME,
    ) -> None:
        if getattr(secondary, "kind", SITE_SECONDARY) != SYNTHETIC_SECONDARY_KIND:
            raise IdentityAcceptanceError("the secondary is not a synthetic target")
        if (
            not secondary.executor_namespace
            or secondary.cluster_id == primary.cluster_id
        ):
            raise IdentityAcceptanceError("synthetic secondary identity is incomplete")
        if not timedelta(minutes=5) <= lifetime <= timedelta(hours=2):
            raise IdentityAcceptanceError(
                "synthetic secondary lifetime is outside 5m..2h"
            )
        self.site = site
        self.primary = primary
        self.secondary = secondary
        self.case_id = case_id
        self.case_dir = case_dir
        self.lifetime = lifetime
        self.journal = SyntheticSecondaryJournal(
            run_id=run_id,
            case_id=case_id,
            cluster_id=secondary.cluster_id,
            namespace=secondary.executor_namespace,
            physical_cluster_id=primary.cluster_id,
        )

    # -- transport -----------------------------------------------------------

    def client(
        self,
        *arguments: str,
        stdin: bytes | None = None,
        check: bool = True,
        timeout: int = 300,
    ) -> str:
        """kubectl on the GPU plane, defaulting to B's namespace.

        The signature is the seeded fixture's ``dataplane`` client, so
        ``delete_owned_resource`` and ``resource_metadata`` drive it unchanged.
        """

        return str(
            self.site.regional(self.secondary).kubectl(
                "gpu",
                *arguments,
                input_text=stdin.decode() if stdin is not None else None,
                check=check,
                timeout=timeout,
            )
        )

    def _write_journal(self) -> None:
        self.journal.updated_at = utc_now()
        write_json_atomic(self.case_dir / JOURNAL_FILE, asdict(self.journal))

    def _connection_keys(self) -> dict[str, str]:
        document = json.loads(
            self.site.gpu(
                self.primary, "get", "secret", CONNECTION_SECRET, "-o", "json"
            )
        )
        data = document.get("data") or {}
        copied = {key: str(data.get(key) or "") for key in COPIED_CONNECTION_KEYS}
        missing = [key for key, value in copied.items() if not value]
        if missing:
            raise IdentityAcceptanceError(
                "primary connection Secret lacks keys to copy: " + ", ".join(missing)
            )
        return copied

    def _primary_row(self, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        for row in rows:
            if row.get("cluster_id") == self.primary.cluster_id:
                return registry_content([row])[0]
        return None

    # -- read-only -------------------------------------------------------------

    def read_only_preflight(self) -> dict[str, Any]:
        """Why B cannot be created right now, or the facts it will be built from."""

        errors: list[str] = []
        result: dict[str, Any] = {
            "cluster_id": self.secondary.cluster_id,
            "namespace": self.journal.namespace,
        }
        generation = self.site.registry_generation()
        result["registry_generation"] = generation
        if generation is None:
            errors.append("a synthetic secondary requires a durable registry")
        rows = self.site.registry()
        ids = {str(row.get("cluster_id") or "") for row in rows}
        result["registration_absent"] = self.secondary.cluster_id not in ids
        if not result["registration_absent"]:
            errors.append(
                f"a registration named {self.secondary.cluster_id} already exists"
            )
        primary_row = next(
            (row for row in rows if row.get("cluster_id") == self.primary.cluster_id),
            None,
        )
        result["primary_registration_enabled"] = bool(
            primary_row and primary_row.get("enabled") is True
        )
        if not result["primary_registration_enabled"]:
            errors.append("the primary registration is absent or disabled")
        if self.journal.primary_registration is None:
            self.journal.primary_registration = self._primary_row(rows)
        namespace = seeded.resource_metadata(
            "namespace", self.journal.namespace, client=self.client
        )
        result["namespace_absent"] = not namespace
        if namespace:
            errors.append(f"namespace {self.journal.namespace} already exists")
        try:
            result["executor"] = executor_template(self.site, self.primary)
        except Exception as exc:  # noqa: BLE001 - a refusal reason, not a crash
            errors.append(f"executor template: {type(exc).__name__}: {exc}")
        try:
            self._connection_keys()
            result["connection_keys_copied"] = list(COPIED_CONNECTION_KEYS)
        except Exception as exc:  # noqa: BLE001 - a refusal reason, not a crash
            errors.append(f"connection Secret: {type(exc).__name__}: {exc}")
        result["errors"] = errors
        return result

    def plan_details(self) -> dict[str, Any]:
        """What the execute phase will create; read-only."""

        preflight = self.read_only_preflight()
        intended = synthetic_registration(
            self.secondary.cluster_id,
            run_id=self.journal.run_id,
            expires_at=datetime.now(timezone.utc) + self.lifetime,
            region=str(getattr(self.site, "region", self.primary.region)),
            namespace=self.journal.namespace,
        )
        intended.pop("token")
        intended["synthetic_expires_at"] = None
        return {
            "kind": SYNTHETIC_SECONDARY_KIND,
            "in_site": False,
            "cluster_id": self.secondary.cluster_id,
            "namespace": self.journal.namespace,
            "run_id": self.journal.run_id,
            "physical_plane": {
                "cluster_id": self.primary.cluster_id,
                "context": self.primary.context,
            },
            "registration": {
                **intended,
                "token": "minted at execute; only its sha256 is recorded",
                "lifetime_seconds": int(self.lifetime.total_seconds()),
            },
            "pod": {
                "name": SECONDARY_POD,
                "namespace": self.journal.namespace,
                "app": EXECUTOR_APP,
                "image": (preflight.get("executor") or {}).get("image"),
                "probe": HOLD_PROBE.name,
                "connection_secret": CONNECTION_SECRET,
                "copied_from_primary": list(COPIED_CONNECTION_KEYS),
            },
            "cleanup": [
                "delete the Pod, then publish a durable revision without B",
                "delete B's connection Secret, hold ConfigMap and namespace with "
                "uid/resourceVersion preconditions",
                "prove no B registration, namespace or Pod remains and that the "
                "primary registration is unchanged",
            ],
            "preflight": preflight,
        }

    # -- execute ---------------------------------------------------------------

    def _create(self, manifest: dict[str, Any]) -> dict[str, Any]:
        kind = str(manifest["kind"]).lower()
        name = str(manifest["metadata"]["name"])
        key = f"{kind}/{name}"
        self.journal.resources[key] = None
        self._write_journal()
        try:
            metadata = json.loads(
                self.client(
                    "create",
                    "-f",
                    "-",
                    "-o",
                    "jsonpath={.metadata}",
                    stdin=json.dumps(manifest).encode(),
                )
            )
        except Exception:
            # A lost ACK may have committed the create: record it for cleanup
            # without replaying it, then fail.
            metadata = seeded.resource_metadata(kind, name, client=self.client)
            if (metadata.get("labels") or {}).get(RUN_LABEL) == self.journal.run_id:
                self.journal.resources[key] = metadata.get("uid")
                self._write_journal()
            raise
        if (
            not isinstance(metadata, dict)
            or not metadata.get("uid")
            or not metadata.get("resourceVersion")
            or (metadata.get("labels") or {}).get(RUN_LABEL) != self.journal.run_id
        ):
            raise IdentityAcceptanceError(f"{key} create identity was not confirmed")
        self.journal.resources[key] = str(metadata["uid"])
        self._write_journal()
        return metadata

    def _publish_registration(self, entry: dict[str, Any]) -> None:
        original = self.site.registry()
        self.journal.primary_registration = self._primary_row(original)
        self.journal.registry_generation_before = self.site.registry_generation()
        self.journal.registration = redacted_registration(entry)
        self.journal.expires_at = str(entry["synthetic_expires_at"])
        self.journal.registry_started = True
        self._write_journal()
        self.site.write_registry(
            [*original, entry],
            reason=(
                f"{self.case_id} synthetic secondary {self.journal.run_id}: "
                f"register {self.secondary.cluster_id}"
            ),
            expected_entries=original,
        )
        self.journal.registry_generation_with_secondary = (
            self.site.registry_generation()
        )
        self._write_journal()

    def _wait_pod_ready(self, expected_uid: str) -> dict[str, Any]:
        self.client(
            "wait",
            "--for=condition=Ready",
            f"pod/{SECONDARY_POD}",
            f"--timeout={POD_READY_TIMEOUT_SECONDS}s",
            timeout=POD_READY_TIMEOUT_SECONDS + 30,
        )
        if SECONDARY_POD not in self.site.ready_pods(
            "gpu", EXECUTOR_APP, self.secondary
        ):
            raise IdentityAcceptanceError(
                "synthetic secondary Pod is not Ready under the executor label"
            )
        metadata = seeded.resource_metadata("pod", SECONDARY_POD, client=self.client)
        if metadata.get("uid") != expected_uid:
            raise IdentityAcceptanceError(
                "synthetic secondary Pod was replaced before readiness"
            )
        node = self.client(
            "get", "pod", SECONDARY_POD, "-o", "jsonpath={.spec.nodeName}", timeout=60
        ).strip()
        return {"uid": str(metadata["uid"]), "node": node}

    def provision(self) -> None:
        """Publish B, then stand up B's namespace, Secret, ConfigMap and Pod."""

        preflight = self.read_only_preflight()
        self.journal.preflight = preflight
        self._write_journal()
        if preflight["errors"]:
            raise IdentityAcceptanceError(
                "synthetic secondary preflight refused: "
                + "; ".join(preflight["errors"])
            )
        template = preflight["executor"]
        copied = self._connection_keys()
        entry = synthetic_registration(
            self.secondary.cluster_id,
            run_id=self.journal.run_id,
            expires_at=datetime.now(timezone.utc) + self.lifetime,
            region=str(getattr(self.site, "region", self.primary.region)),
            namespace=self.journal.namespace,
        )
        self._publish_registration(entry)
        identity = {
            "cluster_id": self.secondary.cluster_id,
            "namespace": self.journal.namespace,
            "run_id": self.journal.run_id,
            "case_id": self.case_id,
        }
        self._create(
            namespace_manifest(
                cluster_id=identity["cluster_id"],
                run_id=identity["run_id"],
                case_id=identity["case_id"],
            )
        )
        self._create(
            connection_secret_manifest(**identity, token=entry["token"], copied=copied)
        )
        self._create(hold_configmap_manifest(**identity))
        pod_metadata = self._create(
            secondary_pod_manifest(
                **identity,
                image=template["image"],
                pins=template,
                lifetime_seconds=int(self.lifetime.total_seconds()),
            )
        )
        ready = self._wait_pod_ready(str(pod_metadata["uid"]))
        reported = self.site.pod_json(
            "gpu", self.secondary, SECONDARY_POD, AUTH008_EXECUTOR_IDENTITY_PROBE
        )
        if (
            reported.get("cluster_id") != self.secondary.cluster_id
            or reported.get("artifact") != template["artifact"]
            or reported.get("compatibility") != template["compatibility"]
        ):
            raise IdentityAcceptanceError(
                "synthetic secondary Pod does not carry B's identity and the "
                "deployed executor pins"
            )
        self.journal.pod = {
            "name": SECONDARY_POD,
            "namespace": self.journal.namespace,
            "uid": ready["uid"],
            "node": ready["node"],
            "image": template["image"],
            "executor_id": reported.get("executor_id"),
        }
        self._write_journal()

    # -- cleanup ---------------------------------------------------------------

    def _delete(self, kind: str, name: str) -> str:
        seeded.delete_owned_resource(
            kind,
            name,
            self.journal.run_id,
            client=self.client,
            namespace=self.journal.namespace,
            expected_uid=self.journal.resources.get(f"{kind}/{name}"),
            require_uid=True,
        )
        return "removed"

    def _remove_registration(self) -> dict[str, Any]:
        current = self.site.registry()
        if not any(
            row.get("cluster_id") == self.secondary.cluster_id for row in current
        ):
            return {"published": False, "reason": "registration already absent"}
        remaining = [
            row for row in current if row.get("cluster_id") != self.secondary.cluster_id
        ]
        # Only B leaves: whatever other callers published meanwhile stays.
        self.site.write_registry(
            remaining,
            reason=(
                f"{self.case_id} synthetic secondary {self.journal.run_id}: "
                f"remove {self.secondary.cluster_id}"
            ),
            expected_entries=current,
        )
        return {"published": True, "generation": self.site.registry_generation()}

    def proof(self) -> dict[str, Any]:
        """Nothing of B remains; A's registration is what it was."""

        rows = self.site.registry()
        primary_now = self._primary_row(rows)
        namespace_absent = not seeded.resource_metadata(
            "namespace", self.journal.namespace, client=self.client
        )
        result = {
            "registry_generation_after": self.site.registry_generation(),
            "registration_absent": not any(
                row.get("cluster_id") == self.secondary.cluster_id for row in rows
            ),
            "primary_registration_untouched": (
                self.journal.primary_registration is not None
                and primary_now == self.journal.primary_registration
            ),
            "namespace_absent": namespace_absent,
            "pod_absent": namespace_absent
            or not seeded.resource_metadata("pod", SECONDARY_POD, client=self.client),
        }
        result["residue_free"] = all(
            result[key] is True
            for key in (
                "registration_absent",
                "primary_registration_untouched",
                "namespace_absent",
                "pod_absent",
            )
        )
        return result

    def teardown(self) -> tuple[dict[str, Any], list[str]]:
        """Remove everything this run owns; every step runs, errors are kept.

        The Pod goes first so no exec can still be in flight, then B leaves
        the registry -- a passive Pod that never claims makes the registration
        the credential to revoke first, unlike the seeded fixture's real
        claimant -- then the namespaced resources and the namespace. The proof
        runs whatever happened before it.
        """

        steps: list[tuple[str, Callable[[], Any]]] = []
        if f"pod/{SECONDARY_POD}" in self.journal.resources:
            steps.append(("delete_pod", lambda: self._delete("pod", SECONDARY_POD)))
        if self.journal.registry_started:
            steps.append(("remove_registration", self._remove_registration))
        if f"secret/{CONNECTION_SECRET}" in self.journal.resources:
            steps.append(
                (
                    "delete_connection_secret",
                    lambda: self._delete("secret", CONNECTION_SECRET),
                )
            )
        if f"configmap/{SECONDARY_HOLD_CONFIGMAP}" in self.journal.resources:
            steps.append(
                (
                    "delete_hold_configmap",
                    lambda: self._delete("configmap", SECONDARY_HOLD_CONFIGMAP),
                )
            )
        if f"namespace/{self.journal.namespace}" in self.journal.resources:
            steps.append(
                (
                    "delete_namespace",
                    lambda: self._delete("namespace", self.journal.namespace),
                )
            )
        outcomes, errors = run_cleanup_steps(steps)
        if not steps:
            outcomes["skipped"] = "no owned mutation attempted"
        proof_outcome, proof_errors = run_cleanup_steps([("proof", self.proof)])
        outcomes.update(proof_outcome)
        errors.extend(proof_errors)
        self._write_journal()
        return outcomes, errors

    def evidence(
        self, *, cleanup: dict[str, Any], cleanup_errors: list[str]
    ) -> dict[str, Any]:
        journal = asdict(self.journal)
        proof = cleanup.get("proof")
        return {
            "kind": SYNTHETIC_SECONDARY_KIND,
            "cluster_id": journal["cluster_id"],
            "namespace": journal["namespace"],
            "run_id": journal["run_id"],
            "physical_plane": {
                "cluster_id": self.primary.cluster_id,
                "context": self.primary.context,
            },
            "registration": journal["registration"],
            "expires_at": journal["expires_at"],
            "registry_generation_before": journal["registry_generation_before"],
            "registry_generation_with_secondary": journal[
                "registry_generation_with_secondary"
            ],
            "registry_generation_after": (
                proof.get("registry_generation_after")
                if isinstance(proof, dict)
                else None
            ),
            "pod": journal["pod"],
            "resources": journal["resources"],
            "cleanup": cleanup,
            "cleanup_errors": list(cleanup_errors),
            "proof": proof if isinstance(proof, dict) else {"residue_free": False},
            "journal": JOURNAL_FILE,
        }


def synthetic_secondary_lifecycle(
    arguments: Any,
    site: Any,
    primary: ClusterTarget,
    secondary: ClusterTarget | None,
    *,
    case_dir: Path,
) -> SyntheticSecondary | None:
    """The lifecycle for a synthetic B, or None when B is a site cluster."""

    if secondary is None or getattr(secondary, "kind", SITE_SECONDARY) != (
        SYNTHETIC_SECONDARY_KIND
    ):
        return None
    return SyntheticSecondary(
        site,
        primary,
        secondary,
        case_id=str(arguments.case),
        case_dir=case_dir,
        run_id=synthetic_run_id(
            secondary.cluster_id, Path(arguments.run_dir), int(arguments.attempt)
        ),
    )


def merge_synthetic_outcome(
    outcome: dict[str, Any], card: dict[str, Any], cleanup_errors: list[str]
) -> dict[str, Any]:
    """The case body's outcome plus B's evidence card and cleanup verdict."""

    checks = dict(outcome.get("checks") or {})
    checks["synthetic_secondary_cleanup_completed"] = not cleanup_errors
    checks["synthetic_secondary_residue_free"] = (card.get("proof") or {}).get(
        "residue_free"
    ) is True
    passed = (
        outcome.get("verdict") == "PASS"
        and checks["synthetic_secondary_cleanup_completed"]
        and checks["synthetic_secondary_residue_free"]
    )
    return {
        **outcome,
        "verdict": "PASS" if passed else "FAIL",
        "checks": checks,
        "cleanup_errors": [*(outcome.get("cleanup_errors") or []), *cleanup_errors],
        "secondary_registered": True,
        "secondary_kind": SYNTHETIC_SECONDARY_KIND,
        "synthetic_secondary": card,
        "limitations": [*(outcome.get("limitations") or []), SYNTHETIC_LIMITATION],
    }


def run_with_synthetic_secondary(
    lifecycle: SyntheticSecondary, body: Callable[[], dict[str, Any]]
) -> dict[str, Any]:
    """Provision B, run the case body against it, always tear B down.

    A body failure -- an ``IdentityCaseFailure`` with partial checks or any
    other exception -- surfaces after the teardown as an ``IdentityCaseFailure``
    whose details carry B's evidence card and the cleanup errors, chained to
    the original so the recorded error names the real cause. An operator abort
    (a ``BaseException``) also runs the teardown and then propagates.
    """

    failure: Exception | None = None
    outcome: dict[str, Any] | None = None
    try:
        lifecycle.provision()
        outcome = body()
    except Exception as exc:  # noqa: BLE001 - recorded, re-raised after cleanup
        failure = exc
    finally:
        cleanup, cleanup_errors = lifecycle.teardown()
    card = lifecycle.evidence(cleanup=cleanup, cleanup_errors=cleanup_errors)
    if failure is not None or outcome is None:
        details = (
            dict(failure.details) if isinstance(failure, IdentityCaseFailure) else {}
        )
        details["cleanup_errors"] = [
            *(details.get("cleanup_errors") or []),
            *cleanup_errors,
        ]
        details["secondary_registered"] = True
        details["secondary_kind"] = SYNTHETIC_SECONDARY_KIND
        details["synthetic_secondary"] = card
        raise IdentityCaseFailure(
            str(failure) if failure is not None else "case body returned nothing",
            details=details,
        ) from failure
    return merge_synthetic_outcome(outcome, card, cleanup_errors)
