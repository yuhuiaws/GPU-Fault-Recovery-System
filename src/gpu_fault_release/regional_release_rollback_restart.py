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
from pathlib import Path
from types import TracebackType
from typing import Any, TypeVar

from gpu_fault.admin.deadlines import DeploymentDeadlineExceeded
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


def _epoch(value: object) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def _pod_reason(pod: Mapping[str, Any], cutoff: float) -> str | None:
    """Why this Pod may hold env the rollback does not mean, or None."""

    created = _epoch((pod.get("metadata") or {}).get("creationTimestamp"))
    if created is None or created >= cutoff:
        return "pod-started-after-capture"
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
        if started >= cutoff:
            return "container-restarted-after-capture"
    return None


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
    the decision to ``build_rollback_environment``.
    """

    def __init__(self, release: Any, previous: Mapping[str, Any]) -> None:
        self.release = release
        self.previous = previous
        self.reasons: dict[str, set[str]] = {role: set() for role in ALL_ROLES}
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
        elif live is not None and isinstance(snapshots, dict):
            for role, deployment in ROLE_DEPLOYMENTS.items():
                if any(
                    live.get(name) != {str(k): str(v) for k, v in data.items()}
                    for name, data in snapshots.items()
                    if str(name).startswith(f"{deployment}-config-")
                ):
                    self.force("config-maps-changed", (role,))
        return restored

    def _prove_deployments(self) -> None:
        """Restart a tier whose Deployment or Pods are not the captured ones."""

        baseline = self.previous.get(BASELINE_KEY)
        if not isinstance(baseline, dict):
            baseline = {}
        captured_at = baseline.get("captured_at")
        specs = baseline.get("spec_sha256")
        if not isinstance(captured_at, (int, float)) or not isinstance(specs, dict):
            self.force("no-baseline")
            return
        release = self.release
        namespace = release.config.namespace
        deployments = self._read(
            "deployments",
            lambda: _by_name(
                release._get_json(
                    release._cpu(
                        "-n", namespace, "get", "deployment", *ROLE_DEPLOYMENTS.values()
                    )
                )
            ),
        )
        pods = self._read(
            "pods",
            lambda: list(
                release._get_json(release._cpu("-n", namespace, "get", "pods")).get(
                    "items"
                )
                or []
            ),
        )
        if deployments is None or pods is None:
            return
        cutoff = float(captured_at) - POD_START_MARGIN_SECONDS
        for role, name in ROLE_DEPLOYMENTS.items():
            document = deployments.get(name)
            spec = (document or {}).get("spec") or {}
            selector = (spec.get("selector") or {}).get("matchLabels") or {}
            if not specs.get(name):
                self.force("no-baseline", (role,))
            elif document is None:
                self.force("deployment-missing", (role,))
            elif specs[name] != deployment_spec_sha256(document):
                self.force("spec-changed", (role,))
            elif not selector:
                self.force("pod-selector-unsupported", (role,))
            for pod in pods if selector else ():
                reason = _pod_reason(pod, cutoff) if _selected(pod, selector) else None
                if reason is not None:
                    self.force(reason, (role,))

    def environment_arguments(self) -> dict[str, Any]:
        """Finish the proof; the decision as ``build_rollback_environment`` takes it."""

        self._prove_deployments()
        forced = self.forced_roles()
        narrate_step(
            "rollback-cpu-restart",
            forced=",".join(forced) or "none",
            reasons=";".join(
                f"{role}:{','.join(sorted(self.reasons[role]))}" for role in forced
            )
            or "none",
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

    def details(self) -> dict[str, Any]:
        """The stage's evidence: each restarted tier with its reasons, the kept
        tiers, and the apply's ``/healthz`` verdicts."""

        restarted = {
            role: sorted(reasons) for role, reasons in self.reasons.items() if reasons
        }
        evidence: dict[str, Any] = {}
        report = self._report()
        if report is None:
            evidence["report"] = "missing"
        else:
            for role in report.get("fallback") or []:
                restarted.setdefault(str(role), []).append("pin-proof-failed")
            evidence["pin_wait"] = report.get("pin_wait") or {}
            evidence["restart_wait"] = report.get("restart_wait") or {}
        evidence["restarted"] = restarted
        evidence["kept"] = [role for role in ALL_ROLES if role not in restarted]
        return {"cpu_restart": evidence}
