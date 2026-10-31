"""GF-GENERAL-LIVE-028/029's managed failing job: the fixture, apart from the verdicts.

``ManagedFailJob`` renders the single-node warm-spare training manifest with a
bash command per replica that exits, stops, hangs, exits once (the product's
restart copy keeps running) or exits on a runner trigger, and submits it
through ``ManagedWorkloadFixture``. It lives apart from
``general_live_attempts`` (the architecture gate's 1500-line file cap), which
re-exports every name here; the verdict tests monkeypatch
``general_live_attempts.ManagedFailJob``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from scripts.e2e.regional.general_live_fixture import VariantContext, VariantNotRun

if TYPE_CHECKING:
    from scripts.e2e.regional.managed_workload_fixture import ManagedWorkloadFixture

SINGLE_NODE_MANIFEST = "single-node-warm-spare-pytorchjob.yaml"


class ManagedFailJob:
    """A managed single-node job whose critical container(s) can be made to
    exit, stop, or hang; a thin wrapper over ManagedWorkloadFixture.

    ``mode`` is baked into the rendered command: ``exit:<code>`` exits the code
    after a short run, ``stop`` waits for a delete/suspend, ``hang`` ignores the
    terminal signal so the cleanup window times out. One mode applies to every
    replica; a mapping gives each replica role its own (``{"Master": "exit:1",
    "Worker": "hang"}``). ``ranks`` 2 adds a Worker replica cloned from the
    manifest's Master, both pinned to the node, and ``gpu_count`` is then per
    Pod (two 4-GPU Pods fill one p5en node); the product infers the critical
    rank count from the replicas (``training_submit_cli._pytorch_templates``).
    The verdict tests monkeypatch this class;
    test_general_live_managed_job_start.py runs its start with the real
    fixtures.
    """

    def __init__(
        self,
        ctx: VariantContext,
        *,
        name: str,
        mode: str | Mapping[str, str],
        restart_budget: int = 1,
        gpu_count: int = 8,
        ranks: int = 1,
        copy_run_seconds: int | None = None,
    ) -> None:
        from pathlib import Path

        from scripts.e2e.regional.managed_workload_fixture import (
            ImagePrewarmFixture,
            ManagedWorkloadFixture,
            ManagedWorkloadSettings,
            name_managed_workload,
            render_node_pinned_manifest,
        )
        from scripts.e2e.regional.training_topology import fit_gpu_count

        yaml = __import__("yaml")
        if ctx.settings.site_file is None:
            raise VariantNotRun("FIXTURE_NOT_READY", "a managed job needs --site-file")
        source = (
            Path(__file__).with_name("manifests") / "training" / SINGLE_NODE_MANIFEST
        )
        rendered = render_node_pinned_manifest(
            source, ctx.case_dir / f"{name}.source.yaml", node=ctx.settings.node
        )
        document = yaml.safe_load(rendered.read_text(encoding="utf-8"))
        name_managed_workload(document, name)
        _add_worker_ranks(document, ranks)
        fit_gpu_count(document, gpu_count)
        _apply_mode(document, mode, copy_run_seconds=copy_run_seconds)
        document.setdefault("metadata", {}).setdefault("labels", {}).update(
            {"gpu-fault.io/job-id": name, "gpu-fault.io/attempt-id": f"{name}-a001"}
        )
        rendered.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        self.name = name
        self.node = ctx.settings.node
        self.attempt_id = f"{name}-a001"
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
                expected_pods=ranks,
                expected_gpu_count=gpu_count,
            ),
            state_path=ctx.case_dir / f"{name}.ownership.json",
        )

    def start(self) -> dict[str, Any]:
        # The node the manifest is pinned to, never the cluster id: a prewarm
        # Pod bound to a node that does not exist never runs.
        self.prewarm.create([self.node])
        return self.fixture.submit()

    def suspend(self) -> None:
        """The job controller's stop: suspend the workload, as Kueue would."""

        suspend_workload(self.fixture)

    def delete(self) -> dict[str, Any]:
        from scripts.e2e.regional.managed_workload_fixture import remove_managed_job

        return remove_managed_job(self.fixture, self.prewarm)


def suspend_workload(fixture: ManagedWorkloadFixture) -> None:
    """Suspend a managed workload fixture's own workload as its job controller would.

    ``spec.runPolicy.suspend`` (PyTorchJob) or ``spec.suspend`` (Job,
    JobSet): the toggle a scheduler such as Kueue applies, which the
    training operator answers by deleting the replica Pods. The product
    reads it as a stop by someone else (completion_observation: a suspend
    without its own initiator annotation), never as a failure. Fenced on
    the UID and resourceVersion the fixture just adopted, like
    ``ManagedWorkloadFixture.annotate_auto_resume``. Lives here, with its only
    caller (028 ``controller_stop``), rather than on the fixture class.
    """

    current = fixture.workload()
    fixture.adopt(current)
    spec = current.get("spec") or {}
    if fixture.resource == "pytorchjob":
        operation: dict[str, Any] = (
            {"op": "add", "path": "/spec/runPolicy/suspend", "value": True}
            if isinstance(spec.get("runPolicy"), dict)
            else {
                "op": "add",
                "path": "/spec/runPolicy",
                "value": {"suspend": True},
            }
        )
    else:
        operation = {"op": "add", "path": "/spec/suspend", "value": True}
    fixture.regional.kubectl(
        "gpu",
        "patch",
        fixture.resource,
        fixture.name,
        "--type=json",
        "--patch-file=/dev/stdin",
        input_text=json.dumps(
            [
                {
                    "op": "test",
                    "path": "/metadata/uid",
                    "value": current["metadata"]["uid"],
                },
                {
                    "op": "test",
                    "path": "/metadata/resourceVersion",
                    "value": current["metadata"]["resourceVersion"],
                },
                operation,
            ]
        ),
    )


#: The Pod's own attempt id, from its ``gpu-fault.io/attempt-id`` label through
#: the downward API: the product's restart copy relabels the template with the
#: ``-r-`` id it derives (``adapters/kubernetes/restart_operations.py``), so a
#: container can tell the first attempt from a restart.
ATTEMPT_ID_ENV = "GFRS_ATTEMPT_ID"
ATTEMPT_ID_LABEL = "gpu-fault.io/attempt-id"
#: ``exit_once``: the restart copy's run before it exits 0 (``copy_run_seconds``).
COPY_RUN_SECONDS_ENV = "GFRS_COPY_RUN_SECONDS"
#: ``trigger:<code>`` exits the code once the runner creates this file in the
#: container (028 api_unreachable: the exit must happen under the block).
EXIT_TRIGGER_PATH = "/tmp/gfrs-live-exit"
MODE_COMMANDS = {
    "exit": "sleep 30; exit ${CODE:-1}",
    "stop": "trap 'exit 143' TERM; sleep 3600",
    "hang": "trap '' TERM; sleep 3600",
    # The first attempt exits CODE; the restart copy runs until stopped (028
    # old_terminal) or, with COPY_RUN_SECONDS_ENV set, that long and exits 0
    # (029 worker_restart: the replacement watcher observes it, then its end).
    "exit_once": (
        f"case \"${{{ATTEMPT_ID_ENV}:-}}\" in *-r-*) trap 'exit 143' TERM; "
        f"sleep ${{{COPY_RUN_SECONDS_ENV}:-3600}}; exit 0;; "
        "*) sleep 30; exit ${CODE:-1};; esac"
    ),
    "trigger": f"while [ ! -e {EXIT_TRIGGER_PATH} ]; do sleep 1; done; exit ${{CODE:-1}}",
}
#: Modes whose ``:<code>`` suffix is the exit code.
EXIT_MODES = frozenset({"exit", "exit_once"})


def _add_worker_ranks(document: dict[str, Any], ranks: int) -> None:
    """Give the single-node manifest ``ranks`` critical ranks: its Master and,
    for a second rank, a Worker replica cloned from it (same node, image,
    devices); the product counts the ranks from the replicas."""

    import copy

    if ranks not in (1, 2):
        raise ValueError(f"a managed fail job runs one or two ranks, not {ranks}")
    replicas = (document.get("spec") or {}).get("pytorchReplicaSpecs") or {}
    if list(replicas) != ["Master"]:
        raise ValueError("the single-node manifest renders one Master replica")
    if ranks == 2:
        worker = copy.deepcopy(replicas["Master"])
        worker["replicas"] = 1
        replicas["Worker"] = worker


def _apply_mode(
    document: dict[str, Any],
    mode: str | Mapping[str, str],
    *,
    copy_run_seconds: int | None = None,
) -> None:
    replicas = document.get("spec", {}).get("pytorchReplicaSpecs") or {}
    modes: Mapping[str, str] = (
        mode if isinstance(mode, Mapping) else dict.fromkeys(replicas, mode)
    )
    unknown = sorted(set(modes) - set(replicas))
    if unknown:
        raise ValueError(f"no replica to apply a mode to: {', '.join(unknown)}")
    missing = sorted(set(replicas) - set(modes))
    if missing:
        raise ValueError(f"no mode for replica: {', '.join(missing)}")
    for role, replica in replicas.items():
        role_mode = modes[role]
        kind = role_mode.split(":", 1)[0]
        script = MODE_COMMANDS.get(kind, MODE_COMMANDS["exit"])
        code = (
            role_mode.split(":", 1)[1]
            if ":" in role_mode and (kind in EXIT_MODES or kind == "trigger")
            else "1"
        )
        identity = (
            [
                {
                    "name": ATTEMPT_ID_ENV,
                    "valueFrom": {
                        "fieldRef": {
                            "fieldPath": f"metadata.labels['{ATTEMPT_ID_LABEL}']"
                        }
                    },
                },
                *(
                    [{"name": COPY_RUN_SECONDS_ENV, "value": str(copy_run_seconds)}]
                    if copy_run_seconds is not None
                    else []
                ),
            ]
            if kind == "exit_once"
            else []
        )
        for container in ((replica.get("template") or {}).get("spec") or {}).get(
            "containers"
        ) or []:
            container["command"] = ["/bin/bash", "-c", script]
            container["env"] = [
                {"name": "CODE", "value": code},
                *identity,
                *(container.get("env") or []),
            ]


__all__ = [
    "ATTEMPT_ID_ENV",
    "ATTEMPT_ID_LABEL",
    "COPY_RUN_SECONDS_ENV",
    "EXIT_MODES",
    "EXIT_TRIGGER_PATH",
    "MODE_COMMANDS",
    "SINGLE_NODE_MANIFEST",
    "ManagedFailJob",
    "suspend_workload",
]
