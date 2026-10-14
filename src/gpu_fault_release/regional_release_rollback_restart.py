"""Which CPU tiers a rollback stage restarts, decided from proof.

Both CPU stages of a rollback -- the controller stage, which puts the
previous pins in front of the data restore, and the CPU restore -- re-apply
the three role Deployments. They used to force-restart every tier, so an
Executor-only rollback paid two full CPU rolls for pins that every running
Pod hot-loads anyway: ``gpu-fault-release-metadata`` is polled by the fleet
pin runtime and the digest it serves is reported on ``/healthz``.

A tier is restarted only when the rollback changes something its containers
read once, at start-up, or when that cannot be proven otherwise:

* the cluster registry was restored, or the email Secret restore wrote
  different data (both reach every tier as ``secretKeyRef`` env);
* the tier's role ConfigMaps (``envFrom``) were changed by the restore, or
  its ``-config-core`` by the controller patch;
* its live Deployment spec is not the one captured before the forward
  transaction began, or one of its Pods or containers started after that
  capture and may have resolved the candidate's env;
* the snapshot carries no baseline, or a read the proof needs failed.

A tier the controller stage restarted has Pods newer than the capture by
construction, so the CPU restore judges it against that restart instead.
After its apply the controller stage records what each tier it restarted now
starts from (``restarted_onto``): the live spec digest, the restart token the
apply minted (copied onto every Pod the template creates), the runtime image,
the container env and the role ConfigMaps its Pods resolved. The CPU restore
keeps such a tier when its spec is still that one, every Pod carries the
token, and its own runtime image, container env and role ConfigMap snapshot
are the recorded ones; otherwise it restarts the tier. An Agent or FULL
rollback therefore rolls a tier once, not twice, and no Pod keeps a start-up
input the CPU restore would change.

Every other tier keeps its Pods and must prove the restored pins on
``/healthz``; the apply restarts one that cannot (the fallback). Which tiers
restarted, why, and the apply's verdicts are the stage's details in
``rollback_timing``. Secret data is compared, never recorded or printed.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime
from functools import partial
from pathlib import Path
from types import TracebackType
from typing import Any, TypeVar

from gpu_fault.admin.deadlines import DeploymentDeadlineExceeded
from gpu_fault.container_env_snapshot import pod_container_env
from gpu_fault_release import regional_deployment_inventory as inventory
from gpu_fault_release.regional_release_config import canonical_sha256
from gpu_fault_release.regional_release_narration import narrate_step

#: The apply script's tier names, in its canonical order.
ROLE_DEPLOYMENTS = {
    "spool": "gpu-fault-telemetry-spool-worker",
    "worker": "gpu-fault-control-worker",
    "ingress": "gpu-fault-api-ha",
}
ALL_ROLES = tuple(ROLE_DEPLOYMENTS)
BASELINE_KEY = "cpu_restart_baseline"
#: A Pod that started this close to the capture is not proven older than the
#: forward transaction: the deploy host and the API server keep separate clocks.
POD_START_MARGIN_SECONDS = 60.0
FORCE_ROLES_VARIABLE = "GPU_FAULT_FORCE_ROLE_RESTART_ROLES"
REPORT_VARIABLE = "GPU_FAULT_ROLE_RESTART_REPORT"
#: Set by a rollback stage only; never inherited from the caller's shell.
ROLE_RESTART_VARIABLES = (FORCE_ROLES_VARIABLE, REPORT_VARIABLE)
REPORT_FILE_NAME = "role-restart.json"
#: The apply's restart token and runtime image on a role Pod template; the
#: ReplicaSet copies the template's annotations onto every Pod it creates.
RESTARTED_AT_ANNOTATION = "kubectl.kubernetes.io/restartedAt"
RUNTIME_IMAGE_ANNOTATION = "gpu-fault.io/runtime-image"
#: The controller stage's ``rollback_timing`` phase and, in its
#: ``cpu_restart`` details, the record of each tier it restarted.
CONTROLLER_STAGE = "controller_stage"
RESTART_RECORD_KEY = "restarted_onto"
RESTART_RECORD_FIELDS = (
    "spec_sha256",
    "restarted_at",
    "runtime_image",
    "container_env_sha256",
    "config_maps_sha256",
)

_T = TypeVar("_T")


def deployment_spec_sha256(document: Mapping[str, Any]) -> str:
    """The digest of a Deployment's whole spec: template, replicas, strategy."""

    return canonical_sha256(document.get("spec") or {})


def cpu_restart_baseline(
    deployments: Mapping[str, Mapping[str, Any]], captured_at: float
) -> dict[str, Any]:
    """What a rollback proves an untouched tier against, kept in the snapshot.

    ``captured_at`` is taken before the capture's first read, so a Pod that
    started earlier ran before the forward transaction changed anything.
    """

    return {
        "captured_at": captured_at,
        "spec_sha256": {
            name: deployment_spec_sha256(document)
            for name, document in sorted(deployments.items())
        },
    }


def controller_stage_restarts(timing: object) -> dict[str, dict[str, str]]:
    """``restarted_onto`` of a completed controller stage, by tier.

    Read from the rollback's durable ``rollback_timing``, so a resumed rollback
    whose controller stage completed in an earlier run finds it too. A stage
    that did not complete records nothing; an entry missing a field is
    dropped, which judges its tier against the capture again.
    """

    phases = timing.get("phases") if isinstance(timing, Mapping) else None
    entry = phases.get(CONTROLLER_STAGE) if isinstance(phases, Mapping) else None
    if not isinstance(entry, Mapping) or entry.get("status") != "COMPLETED":
        return {}
    details = entry.get("details")
    restart = details.get("cpu_restart") if isinstance(details, Mapping) else None
    record = restart.get(RESTART_RECORD_KEY) if isinstance(restart, Mapping) else None
    return {
        role: {field: value[field] for field in RESTART_RECORD_FIELDS}
        for role, value in (record.items() if isinstance(record, Mapping) else ())
        if role in ROLE_DEPLOYMENTS
        and isinstance(value, Mapping)
        and all(
            isinstance(value.get(field), str) and value.get(field)
            for field in RESTART_RECORD_FIELDS
        )
    }


def _epoch(value: object) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def _container_reason(pod: Mapping[str, Any], cutoff: float | None) -> str | None:
    """A container that is not running or, with a cutoff, started after it."""

    statuses = {
        status.get("name"): status
        for status in (pod.get("status") or {}).get("containerStatuses") or []
        if isinstance(status, dict)
    }
    for container in (pod.get("spec") or {}).get("containers") or []:
        state = (statuses.get(container.get("name")) or {}).get("state") or {}
        started = _epoch((state.get("running") or {}).get("startedAt"))
        if started is None:
            return "container-not-running"
        if cutoff is not None and started >= cutoff:
            return "container-restarted-after-capture"
    return None


def _pod_reason(pod: Mapping[str, Any], cutoff: float) -> str | None:
    """Why this Pod may hold env the rollback does not mean, or None."""

    created = _epoch((pod.get("metadata") or {}).get("creationTimestamp"))
    if created is None or created >= cutoff:
        return "pod-started-after-capture"
    return _container_reason(pod, cutoff)


def _restarted_pod_reason(pod: Mapping[str, Any], token: str) -> str | None:
    """Why this Pod may not come from the controller stage's restart, or None.

    The stage minted the token after it restored what the Pods read at
    start-up, so a Pod carrying it -- and a container started since -- resolved
    the restored inputs.
    """

    annotations = (pod.get("metadata") or {}).get("annotations") or {}
    if annotations.get(RESTARTED_AT_ANNOTATION) != token:
        return "pod-not-from-controller-stage"
    return _container_reason(pod, None)


def _template_annotations(document: Mapping[str, Any] | None) -> Mapping[str, Any]:
    template = ((document or {}).get("spec") or {}).get("template") or {}
    return (template.get("metadata") or {}).get("annotations") or {}


def _restart_token(document: Mapping[str, Any] | None) -> str:
    return str(_template_annotations(document).get(RESTARTED_AT_ANNOTATION) or "")


def _role_config_maps(maps: object, deployment: str) -> dict[str, dict[str, str]]:
    """A tier's role ConfigMaps in ``maps``, valued as the restore writes them."""

    if not isinstance(maps, Mapping):
        return {}
    return {
        str(name): {str(key): str(value) for key, value in data.items()}
        for name, data in maps.items()
        if str(name).startswith(f"{deployment}-config-") and isinstance(data, Mapping)
    }


def _restart_record(
    document: Mapping[str, Any] | None,
    config_maps: Mapping[str, Mapping[str, str]],
    *,
    before: str,
) -> dict[str, str] | None:
    """What a tier restarted onto, read back after the apply; None unless its
    template carries a fresh restart token and names the runtime image."""

    token = _restart_token(document)
    image = str(_template_annotations(document).get(RUNTIME_IMAGE_ANNOTATION) or "")
    if document is None or not token or token == before or not image:
        return None
    return {
        "spec_sha256": deployment_spec_sha256(document),
        "restarted_at": token,
        "runtime_image": image,
        "container_env_sha256": canonical_sha256(pod_container_env(dict(document))),
        "config_maps_sha256": canonical_sha256(config_maps),
    }


def _selected(pod: Mapping[str, Any], selector: Mapping[str, Any]) -> bool:
    metadata = pod.get("metadata") or {}
    labels = metadata.get("labels") or {}
    return not metadata.get("deletionTimestamp") and all(
        labels.get(key) == value for key, value in selector.items()
    )


def _by_name(value: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    items = value.get("items") if "items" in value else [value]
    return {
        str((item.get("metadata") or {}).get("name")): item
        for item in items or []
        if isinstance(item, dict)
    }


class RollbackRestartPlan:
    """One rollback stage's restart decision: which tiers, and why.

    Used as a context manager around the stage: its directory holds the
    stage's rendered inputs and the apply's report until the apply returned.
    Reasons accumulate while the stage restores what the previous release ran
    with; ``environment_arguments`` adds the Deployment and Pod proof and hands
    the decision to ``build_rollback_environment``. The CPU restore passes the
    rollback's ``rollback_timing``: a tier the controller stage restarted is
    proven against that stage's ``restarted_onto`` instead of the capture.
    """

    def __init__(
        self,
        release: Any,
        previous: Mapping[str, Any],
        rollback_timing: Mapping[str, Any] | None = None,
    ) -> None:
        self.release = release
        self.previous = previous
        self.reasons: dict[str, set[str]] = {role: set() for role in ALL_ROLES}
        self.restarted_onto = controller_stage_restarts(rollback_timing)
        #: The tiers this stage proved against ``restarted_onto``.
        self.controller_stage_restarted: list[str] = []
        #: Each tier's restart token before the apply, once the proof read it.
        self._tokens_before: dict[str, str] | None = None
        self._record: dict[str, dict[str, str]] | None = None
        self._directory: tempfile.TemporaryDirectory[str] | None = None

    def __enter__(self) -> RollbackRestartPlan:
        self._directory = tempfile.TemporaryDirectory(
            prefix="gpu-fault-rollback-inputs-"
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._directory is not None:
            self._directory.cleanup()
            self._directory = None

    @property
    def directory(self) -> Path:
        if self._directory is None:
            raise RuntimeError("the rollback restart plan is used outside its stage")
        return Path(self._directory.name)

    @property
    def report_path(self) -> Path:
        return self.directory / REPORT_FILE_NAME

    def force(self, reason: str, roles: Iterable[str] = ALL_ROLES) -> None:
        for role in roles:
            self.reasons[role].add(reason)

    def forced_roles(self) -> list[str]:
        return [role for role in ALL_ROLES if self.reasons[role]]

    def _read(self, what: str, read: Callable[[], _T]) -> _T | None:
        """A read the proof needs; one that fails restarts every tier."""

        try:
            return read()
        except DeploymentDeadlineExceeded:
            raise
        except Exception as exc:
            narrate_step(
                "rollback-cpu-restart-proof", read=what, error=type(exc).__name__
            )
            self.force(f"{what}-unreadable")
            return None

    def _live_data(self, kind: str, name: str) -> dict[str, Any]:
        release = self.release
        return dict(
            release._get_json(
                release._cpu("-n", release.config.namespace, "get", kind, name)
            ).get("data")
            or {}
        )

    def _live_deployments(self) -> dict[str, Mapping[str, Any]]:
        release = self.release
        return _by_name(
            release._get_json(
                release._cpu(
                    "-n",
                    release.config.namespace,
                    "get",
                    "deployment",
                    *ROLE_DEPLOYMENTS.values(),
                )
            )
        )

    def restore_secret_and_registry(self) -> None:
        """Restore the email Secret and the cluster registry, as both stages do.

        Every tier reads both through ``secretKeyRef`` env at start-up, so a
        registry restore, or a Secret restore that writes different data,
        restarts every tier.
        """

        release = self.release
        backup = (self.previous.get("secret_backups") or {}).get("cpu") or {}
        if backup.get("backup"):
            source = str(backup["source"])
            live = self._read("secret", lambda: self._live_data("secret", source))
            restored = release._restore_secret(
                release._cpu(), source=source, backup=str(backup["backup"])
            )
            if live is not None and live != restored:
                self.force("secret-changed")
        if release._restore_registry_backup():
            release._publish_restored_registry()
            self.force("registry-restored")

    def patch_controller_config(self, config: Mapping[str, str]) -> None:
        """Patch every ``-config-core`` with the previous Agent identities.

        A tier whose ``-config-core`` the patch changes restarts: those values
        are ``envFrom`` env, read once.
        """

        release = self.release
        names = {
            role: f"{deployment}-config-core"
            for role, deployment in ROLE_DEPLOYMENTS.items()
        }
        live = self._read(
            "config-core", lambda: release._config_maps_data(sorted(names.values()))
        )
        patch = json.dumps({"data": dict(config)}, sort_keys=True)
        for deployment in inventory.CPU_RUNTIME_DEPLOYMENTS:
            release.runner.run(
                release._cpu(
                    "-n",
                    release.config.namespace,
                    "patch",
                    "configmap",
                    f"{deployment}-config-core",
                    "--type=merge",
                    "-p",
                    patch,
                )
            )
        for role, name in names.items():
            current = (live or {}).get(name) or {}
            if live is not None and any(
                current.get(key) != value for key, value in config.items()
            ):
                self.force("config-core-changed", (role,))

    def restore_role_config_maps(self, snapshots: object) -> bool:
        """Restore the role ConfigMaps; whether they were restored at all.

        A tier one of whose ConfigMaps the restore changes restarts. Without a
        snapshot the apply renders the current ones, so every tier restarts.
        """

        live = None
        if isinstance(snapshots, dict) and snapshots:
            live = self._read(
                "config-maps",
                lambda: self.release._config_maps_data(sorted(snapshots)),
            )
        restored = bool(self.release._restore_cpu_role_config_maps(snapshots))
        if not restored:
            self.force("no-config-map-snapshot")
        elif live is not None:
            for role, deployment in ROLE_DEPLOYMENTS.items():
                if any(
                    live.get(name) != data
                    for name, data in _role_config_maps(snapshots, deployment).items()
                ):
                    self.force("config-maps-changed", (role,))
        return restored

    def _prove_deployments(
        self, runtime_image: str | None, container_env: Mapping[str, Any] | None
    ) -> None:
        """Restart a tier whose Deployment or Pods are not the expected ones:
        the captured ones, or the controller stage's restart of the tier."""

        baseline = self.previous.get(BASELINE_KEY)
        if not isinstance(baseline, dict):
            baseline = {}
        captured_at = baseline.get("captured_at")
        specs = baseline.get("spec_sha256")
        if not isinstance(captured_at, (int, float)) or not isinstance(specs, dict):
            self.force("no-baseline")
            return
        deployments = self._read("deployments", self._live_deployments)
        release = self.release
        pods = self._read(
            "pods",
            lambda: list(
                release._get_json(
                    release._cpu("-n", release.config.namespace, "get", "pods")
                ).get("items")
                or []
            ),
        )
        if deployments is None or pods is None:
            return
        self._tokens_before = {
            role: _restart_token(deployments.get(name))
            for role, name in ROLE_DEPLOYMENTS.items()
        }
        cutoff = float(captured_at) - POD_START_MARGIN_SECONDS
        # Without the restore's own target nothing can be compared with the
        # controller stage's restart: such a stage proves against the capture.
        restarts = self.restarted_onto if runtime_image is not None else {}
        self.controller_stage_restarted = [
            role for role in ALL_ROLES if role in restarts
        ]
        for role, name in ROLE_DEPLOYMENTS.items():
            record = restarts.get(role)
            if record is None:
                expected: object = specs.get(name)
                pod_reason = partial(_pod_reason, cutoff=cutoff)
            else:
                self._prove_restart_target(role, record, runtime_image, container_env)
                expected = record["spec_sha256"]
                pod_reason = partial(
                    _restarted_pod_reason, token=record["restarted_at"]
                )
            self._prove_tier(role, deployments.get(name), pods, expected, pod_reason)

    def _prove_tier(
        self,
        role: str,
        document: Mapping[str, Any] | None,
        pods: list[Any],
        expected_spec: object,
        pod_reason: Callable[[Mapping[str, Any]], str | None],
    ) -> None:
        spec = (document or {}).get("spec") or {}
        selector = (spec.get("selector") or {}).get("matchLabels") or {}
        if not expected_spec:
            self.force("no-baseline", (role,))
        elif document is None:
            self.force("deployment-missing", (role,))
        elif expected_spec != deployment_spec_sha256(document):
            self.force("spec-changed", (role,))
        elif not selector:
            self.force("pod-selector-unsupported", (role,))
        for pod in pods if selector else ():
            reason = pod_reason(pod) if _selected(pod, selector) else None
            if reason is not None:
                self.force(reason, (role,))

    def _prove_restart_target(
        self,
        role: str,
        record: Mapping[str, str],
        runtime_image: str | None,
        container_env: Mapping[str, Any] | None,
    ) -> None:
        """Restart a tier the controller stage restarted onto other inputs
        than this stage's: runtime image, container env (without a snapshot
        both stages render the current template's) and role ConfigMaps."""

        name = ROLE_DEPLOYMENTS[role]
        env = None if container_env is None else container_env.get(name)
        maps = _role_config_maps(self.previous.get("cpu_role_config_maps"), name)
        if record["runtime_image"] != runtime_image:
            self.force("runtime-image-changed", (role,))
        if container_env is not None:
            if canonical_sha256(env) != record["container_env_sha256"]:
                self.force("container-env-changed", (role,))
        if canonical_sha256(maps) != record["config_maps_sha256"]:
            self.force("config-maps-changed", (role,))

    def environment_arguments(
        self,
        runtime_image: str | None = None,
        container_env: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Finish the proof; the decision as ``build_rollback_environment`` takes it.

        ``runtime_image`` and ``container_env`` (the verbatim env snapshot, or
        None for the current template) are what the CPU restore renders: the
        target a tier the controller stage restarted is compared against.
        """

        self._prove_deployments(runtime_image, container_env)
        forced = self.forced_roles()
        judged = self.controller_stage_restarted
        narrate_step(
            "rollback-cpu-restart",
            forced=",".join(forced) or "none",
            reasons=";".join(
                f"{role}:{','.join(sorted(self.reasons[role]))}" for role in forced
            )
            or "none",
            **({"controller_stage_restarted": ",".join(judged)} if judged else {}),
        )
        return {
            "force_restart_roles": forced,
            "role_restart_report": str(self.report_path),
        }

    def _report(self) -> dict[str, Any] | None:
        try:
            value = json.loads(self.report_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    def _restarted(self, report: Mapping[str, Any] | None) -> dict[str, list[str]]:
        """Each tier this stage restarted, with its reasons; the apply's
        fallback restarts are ``pin-proof-failed``."""

        restarted = {
            role: sorted(reasons) for role, reasons in self.reasons.items() if reasons
        }
        for role in (report or {}).get("fallback") or []:
            restarted.setdefault(str(role), []).append("pin-proof-failed")
        return restarted

    def record_restarts(self) -> None:
        """After the apply: what each tier this stage restarted now starts from.

        A tier is recorded only when its template carries a restart token
        other than the one the proof read before the apply -- the token this
        apply minted -- together with the role ConfigMaps its Pods resolved:
        the controller stage does not render them, and on a resumed rollback
        the CPU restore may already have rewritten them. A read that fails
        records nothing, so the CPU restore restarts those tiers again.
        """

        self._record = {}
        before = self._tokens_before
        snapshots = self.previous.get("cpu_role_config_maps")
        restarted = [
            role for role in self._restarted(self._report()) if role in ROLE_DEPLOYMENTS
        ]
        names = sorted(
            name
            for role in restarted
            for name in _role_config_maps(snapshots, ROLE_DEPLOYMENTS[role])
        )
        if not restarted:
            return
        if before is not None and names:
            try:
                deployments = self._live_deployments()
                maps = self.release._config_maps_data(names)
                for role in restarted:
                    record = _restart_record(
                        deployments.get(ROLE_DEPLOYMENTS[role]),
                        _role_config_maps(maps, ROLE_DEPLOYMENTS[role]),
                        before=before.get(role, ""),
                    )
                    if record is not None:
                        self._record[role] = record
            except DeploymentDeadlineExceeded:
                raise
            except Exception as exc:
                self._record = {}
                narrate_step("rollback-cpu-restart-record", error=type(exc).__name__)
        narrate_step(
            "rollback-cpu-restart-record", recorded=",".join(self._record) or "none"
        )

    def details(self) -> dict[str, Any]:
        """The stage's evidence: each restarted tier with its reasons, the kept
        tiers, the apply's ``/healthz`` verdicts, what the controller stage's
        restarts start from, and which tiers the CPU restore proved against
        them."""

        evidence: dict[str, Any] = {}
        report = self._report()
        restarted = self._restarted(report)
        if report is None:
            evidence["report"] = "missing"
        else:
            evidence["pin_wait"] = report.get("pin_wait") or {}
            evidence["restart_wait"] = report.get("restart_wait") or {}
        evidence["restarted"] = restarted
        evidence["kept"] = [role for role in ALL_ROLES if role not in restarted]
        if self._record is not None:
            evidence[RESTART_RECORD_KEY] = dict(self._record)
        if self.controller_stage_restarted:
            evidence["controller_stage_restarted"] = list(
                self.controller_stage_restarted
            )
        return {"cpu_restart": evidence}
