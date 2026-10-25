"""The disposable single-node attribution target of GF-GENERAL-LIVE-017.

A managed PyTorchJob rendered from ``single-node-warm-spare-pytorchjob.yaml``
(``ManagedWorkloadFixture`` underneath, like 003's idle job and 028's failing
job) whose one ``pytorch`` container is what the product attributes a reviewed
CRI log line to. Three modes, each a real process with a genuine host PID:

* ``trainer`` -- the manifest's own torchrun trainer (GPU duty as shipped),
  long-running, heartbeating: the LIVE attribution target.
* ``hold`` -- ``python3`` as the container's init, waiting for the runner's
  exit file (``request_exit``) or a bounded deadline, then exiting 0: the
  attempt the watcher records SUCCEEDED, whose container is EXITED.
* ``fail_once`` -- the first attempt (``<name>-a001``) exits 1 after
  :data:`FAIL_ONCE_SECONDS`; the product's RESTART_WORKLOAD then runs the
  trainer under a new attempt id and a new Pod UID. The old Pod's UID is the
  one a late log line must not reach the new attempt through. The container
  learns its attempt id from the downward API (the Pod's attempt label).

The wrapper also declares, for its own Pod only, the ``gpu-fault.io/host-pid``
annotation the completion watcher reads (``completion_pod_parsing``): the value
comes from the node probe reading the container's cgroup, never a guess. The
variant tests monkeypatch this class; test_general_live_managed_job_start.py
runs the sibling wrappers' start with the real fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from scripts.e2e.regional.general_live_applog_fixture import HOST_PID_ANNOTATION
from scripts.e2e.regional.general_live_fixture import VariantContext, VariantNotRun

SINGLE_NODE_MANIFEST = "single-node-warm-spare-pytorchjob.yaml"
TRAINING_CONTAINER = "pytorch"
MODES = ("trainer", "hold", "fail_once")
HOLD_EXIT_FILE = "/tmp/gfrs-live-exit"
#: The hold process exits 0 on its own after this long: nothing outlives the run.
HOLD_MAX_SECONDS = 1800
FAIL_ONCE_SECONDS = 30
ATTEMPT_ENV = "GFRS_ATTEMPT_ID"
ATTEMPT_LABEL = "gpu-fault.io/attempt-id"
START_TIMEOUT_SECONDS = 900

HOLD_SCRIPT = """\
exec python3 - <<'PY'
import os
import sys
import time

# GF-GENERAL-LIVE-017 hold process (scripts/e2e/regional/general_live_applog_job.py):
# a real Python init process with a genuine host PID that exits 0 on the
# runner's exit file or at its own deadline.
deadline = time.monotonic() + %(max_seconds)d
print("HOLD pid=%%d" %% os.getpid(), flush=True)
while time.monotonic() < deadline and not os.path.exists(%(exit_file)r):
    time.sleep(1)
print("HOLD-EXIT", flush=True)
sys.exit(0)
PY
"""


def apply_mode(document: dict[str, Any], mode: str, *, first_attempt: str) -> None:
    """Render ``mode`` onto every ``pytorch`` container of the manifest, in place."""

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    containers = _training_containers(document)
    if not containers:
        raise ValueError(f"the manifest has no {TRAINING_CONTAINER} container")
    for container in containers:
        command = list(container.get("command") or [])
        if len(command) != 3 or command[:2] != ["/bin/bash", "-lc"]:
            raise ValueError("the training container command is not a bash -lc script")
        if mode == "hold":
            container["command"] = [
                "/bin/bash",
                "-lc",
                HOLD_SCRIPT
                % {"max_seconds": HOLD_MAX_SECONDS, "exit_file": HOLD_EXIT_FILE},
            ]
        elif mode == "fail_once":
            container["command"] = [
                "/bin/bash",
                "-lc",
                f'if [ "${{{ATTEMPT_ENV}}}" = "{first_attempt}" ]; then '
                f"sleep {FAIL_ONCE_SECONDS}; exit 1; fi\n" + command[2],
            ]
            container["env"] = [
                item
                for item in container.get("env") or []
                if not (isinstance(item, dict) and item.get("name") == ATTEMPT_ENV)
            ] + [
                {
                    "name": ATTEMPT_ENV,
                    "valueFrom": {
                        "fieldRef": {"fieldPath": f"metadata.labels['{ATTEMPT_LABEL}']"}
                    },
                }
            ]


def _training_containers(document: dict[str, Any]) -> list[dict[str, Any]]:
    found = []
    replicas = (document.get("spec") or {}).get("pytorchReplicaSpecs") or {}
    for replica in replicas.values():
        pod_spec = ((replica or {}).get("template") or {}).get("spec") or {}
        for container in pod_spec.get("containers") or []:
            if (
                isinstance(container, dict)
                and container.get("name") == TRAINING_CONTAINER
            ):
                found.append(container)
    return found


def container_status(pod: dict[str, Any], name: str) -> dict[str, Any]:
    """One container's identity and state from a Pod document."""

    statuses = (pod.get("status") or {}).get("containerStatuses") or []
    selected = next((item for item in statuses if item.get("name") == name), {})
    state = selected.get("state") or {}
    return {
        "name": name,
        "container_id": str(selected.get("containerID") or "").split("://")[-1] or None,
        "running": state.get("running") is not None,
        "terminated": state.get("terminated"),
        "restart_count": selected.get("restartCount", 0),
    }


class ManagedApplogJob:
    """The managed attribution target (``ManagedWorkloadFixture`` underneath)."""

    def __init__(
        self,
        ctx: VariantContext,
        *,
        name: str,
        mode: str = "trainer",
        restart_budget: int = 0,
        gpu_count: int = 8,
    ) -> None:
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
                "FIXTURE_NOT_READY", "a managed attribution job needs --site-file"
            )
        source = (
            Path(__file__).with_name("manifests") / "training" / SINGLE_NODE_MANIFEST
        )
        rendered = render_node_pinned_manifest(
            source, ctx.case_dir / f"{name}.source.yaml", node=ctx.settings.node
        )
        document = yaml.safe_load(rendered.read_text(encoding="utf-8"))
        name_managed_workload(document, name)
        self.attempt_id = f"{name}-a001"
        apply_mode(document, mode, first_attempt=self.attempt_id)
        document.setdefault("metadata", {}).setdefault("labels", {}).update(
            {"gpu-fault.io/job-id": name, ATTEMPT_LABEL: self.attempt_id}
        )
        rendered.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        self.ctx = ctx
        self.name = name
        self.mode = mode
        self.node = ctx.settings.node
        self.namespace = ctx.regional.settings.namespace
        self.container = TRAINING_CONTAINER
        self.restart_budget = restart_budget
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
                restart_budget=restart_budget,
                expected_pods=1,
                expected_gpu_count=gpu_count,
            ),
            state_path=ctx.case_dir / f"{name}.ownership.json",
        )

    # -- lifecycle ------------------------------------------------------- #
    def start(self) -> dict[str, Any]:
        self.prewarm.create([self.node])
        self.fixture.submit()
        if self.mode == "trainer":
            return self.fixture.wait_running(timeout_seconds=START_TIMEOUT_SECONDS)
        pods = self.wait_pods(
            lambda items: bool(items)
            and all(item["phase"] == "Running" for item in items),
            timeout_seconds=START_TIMEOUT_SECONDS,
            name=f"{self.name}-running",
        )
        return {"pods": pods}

    def delete(self) -> dict[str, Any]:
        from scripts.e2e.regional.managed_workload_fixture import remove_managed_job

        return remove_managed_job(self.fixture, self.prewarm)

    # -- reads ----------------------------------------------------------- #
    def pods(self) -> list[dict[str, Any]]:
        return self.fixture.pods()

    def pod_detail(self, pod_name: str) -> dict[str, Any]:
        """The Pod's UID, phase, attempt label and its training container's id."""

        document = json.loads(
            self.ctx.regional.kubectl("gpu", "get", "pod", pod_name, "-o", "json")
        )
        metadata = document.get("metadata") or {}
        return {
            "name": metadata.get("name"),
            "uid": metadata.get("uid"),
            "namespace": metadata.get("namespace"),
            "node": (document.get("spec") or {}).get("nodeName"),
            "phase": (document.get("status") or {}).get("phase"),
            "attempt_id": (metadata.get("labels") or {}).get(ATTEMPT_LABEL),
            "annotations": {
                key: value
                for key, value in (metadata.get("annotations") or {}).items()
                if key.startswith("gpu-fault.io/")
            },
            "container": container_status(document, self.container),
        }

    def wait_pods(
        self,
        accept: Callable[[list[dict[str, Any]]], bool],
        *,
        timeout_seconds: float,
        name: str,
        poll_seconds: float = 10,
    ) -> list[dict[str, Any]]:
        last: list[dict[str, Any]] = []

        def check() -> list[dict[str, Any]] | None:
            nonlocal last
            last = self.pods()
            return last if accept(last) else None

        value = self.ctx.wait_until(
            check, timeout_seconds=timeout_seconds, poll_seconds=poll_seconds, name=name
        )
        return value if value is not None else last

    def wait_replaced(
        self, old_uids: set[str], *, timeout_seconds: float, name: str
    ) -> list[dict[str, Any]]:
        """The product's restart: a Running Pod whose UID is none of ``old_uids``."""

        return self.wait_pods(
            lambda items: any(
                item["phase"] == "Running" and str(item["uid"]) not in old_uids
                for item in items
            ),
            timeout_seconds=timeout_seconds,
            name=name,
        )

    def alive(self) -> dict[str, Any]:
        pods = self.pods()
        return {"alive": self.fixture.pods_healthy(pods), "pods": pods}

    def identity(self) -> dict[str, Any]:
        return {
            "job_id": self.name,
            "attempt_id": self.attempt_id,
            "mode": self.mode,
            "node": self.node,
            "namespace": self.namespace,
            "container": self.container,
            "restart_budget": self.restart_budget,
        }

    # -- the workload's own declarations and actions ----------------------- #
    def annotate_host_pid(self, pod_name: str, pid: int) -> str:
        """Declare the training container's host PID on the runner's own Pod."""

        if int(pid) < 1:
            raise ValueError("a host PID is positive")
        return self.ctx.regional.kubectl(
            "gpu",
            "annotate",
            "pod",
            pod_name,
            f"{HOST_PID_ANNOTATION}={int(pid)}",
            "--overwrite",
        )

    def request_exit(self, pod_name: str) -> str:
        """Ask the hold process to exit 0 (its exit file)."""

        if self.mode != "hold":
            raise ValueError("only the hold mode exits on request")
        return self.ctx.regional.kubectl(
            "gpu",
            "exec",
            pod_name,
            "-c",
            self.container,
            "--",
            "/bin/sh",
            "-c",
            f"touch {HOLD_EXIT_FILE}",
        )


__all__ = [
    "ATTEMPT_ENV",
    "ATTEMPT_LABEL",
    "FAIL_ONCE_SECONDS",
    "HOLD_EXIT_FILE",
    "HOLD_MAX_SECONDS",
    "MODES",
    "ManagedApplogJob",
    "TRAINING_CONTAINER",
    "apply_mode",
    "container_status",
]
