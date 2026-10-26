"""GF-GENERAL-LIVE-027 ``reporter_silent``: the reporter really stops.

Live 2026-10-26 (plain-EKS lane eksb, node ip-10-0-32-162, release
37db33f254a1, attempt 7) ``reporter_silent`` FAILED "no training_reporter-silent
finding reached an incident; reporter silence produced an NCCL/hang finding
instead". The product's own records say why: at the last read the attempt's
heartbeat was still fresh (``observed_at`` 09:51:57, thirteen minutes after
the "stop") with ``last_progress_at`` frozen at 09:38:56 and ``step`` 10 -- a
stalled step, which ``TrainingHealthService.scan`` grades ``training_hang``
("training step stopped advancing"), exactly what the ``stalled_step`` variant
expects of it. The runner's injection, ``kubectl exec -c progress-reporter-0 --
/bin/sh -c 'kill -STOP 1 || true'``, had stopped nothing: the reporter was the
container's PID 1, and pid_namespaces(7) drops a SIGSTOP sent to a PID
namespace's init from inside that namespace while ``kill`` returns 0. The
product graded the stall it was shown.

The reporter container now runs a launcher shell as PID 1 with the reporter as
its child, and the stop is judged on the reporter's process state. Here they
run for real: the fake kubelet of ``test_general_live_progress_job`` starts the
rendered launcher with a stand-in program, ``kubectl exec`` runs the rendered
stop script against the container's ``/tmp`` emptyDir, and the proof is
``/proc/<pid>/status``.
"""

from __future__ import annotations

import dataclasses
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_progress as pr
from scripts.e2e.regional.general_live_fixture import VariantInconclusive
from tests.regional import test_general_live_progress_job as job_tests
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    make_context,
    make_settings,
    node_probe,
)
from tests.regional.test_general_live_progress_job import private_umask  # noqa: F401

pytestmark = pytest.mark.usefixtures("private_umask")

SPEC = {spec.variant: spec for spec in pr.VARIANTS}["reporter_silent"]
REPORTER = "progress-reporter-0"
NODE = "ip-10-0-43-5.us-west-2.compute.internal"
CLEAN_SURVEY = {
    "residue": {},
    "collector_env": {"GPU_FAULT_HOST_INTERVAL_SECONDS": "15"},
    "tools": {},
}
#: What the live stop printed in effect: the reporter, PID 1, still running.
LIVE_UNSTOPPED = "1\n\tS (sleeping)\n"
STOPPED = "4242\n\tT (stopped)\n"


class LauncherCluster(job_tests.ProgressCluster):
    """The fake kubelet, running the reporter's launcher with a stand-in.

    A container the kubelet starts through ``REPORTER_LAUNCHER`` gets a real
    local process tree -- the launcher shell where the container's PID 1 would
    be, ``sleep`` standing in for the reporter -- so the PID file the launcher
    writes, the stop script and ``/proc/<pid>/status`` are the real thing.
    """

    instances: list[LauncherCluster] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.launchers: dict[tuple[str, str], subprocess.Popen[bytes]] = {}
        LauncherCluster.instances.append(self)

    def start(self, pod: dict[str, Any], container: dict[str, Any]) -> dict[str, Any]:
        status = super().start(pod, container)
        command = job_tests.container_command(container)
        if "running" in status["state"] and job_tests.is_launcher(command):
            launcher = [self.remap(pod, container, item) for item in command[:4]]
            self.launchers[(pod["metadata"]["name"], container["name"])] = (
                subprocess.Popen(
                    [*launcher, "sleep", "600"],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            )
        return status

    def shutdown(self) -> None:
        for process in self.launchers.values():
            for signum in (signal.SIGCONT, signal.SIGKILL):
                try:
                    os.killpg(process.pid, signum)
                except ProcessLookupError:
                    break
            process.wait(timeout=10)


@pytest.fixture
def launcher_cluster(monkeypatch: pytest.MonkeyPatch) -> Any:
    LauncherCluster.instances = []
    monkeypatch.setattr(job_tests, "ProgressCluster", LauncherCluster)
    yield
    for cluster in LauncherCluster.instances:
        cluster.shutdown()


def status_field(pid: int, name: str) -> str:
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        key, _separator, value = line.partition(":")
        if key == name:
            return value.strip()
    raise AssertionError(f"/proc/{pid}/status has no {name}")


def reporter_pid(cluster: LauncherCluster, pod: str) -> int:
    """The PID the launcher recorded in the container's ``/tmp`` emptyDir."""

    path = cluster.local_path(pod, REPORTER, pr.REPORTER_PID_FILE)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if path.is_file() and path.read_text().strip().isdigit():
            return int(path.read_text())
        time.sleep(0.05)
    raise AssertionError(f"the launcher never wrote {path}")


def only_pod(cluster: job_tests.ProgressCluster) -> str:
    (pod,) = [name for kind, name in cluster.objects if kind == "pod"]
    return pod


# --------------------------------------------------------------------------- #
# the reporter container and the stop, over a real process
# --------------------------------------------------------------------------- #
def test_the_reporter_runs_as_a_child_of_its_containers_pid_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cluster, job = job_tests.progress_job(
        tmp_path, monkeypatch, variant="reporter_silent"
    )

    job.start()

    pod = only_pod(cluster)
    container = cluster.container(cluster.objects[("pod", pod)], REPORTER)
    command = job_tests.container_command(container)
    # PID 1 is the launcher shell; the reporter is the program it hands on.
    assert command[:4] == [
        "/bin/sh",
        "-c",
        pr.REPORTER_LAUNCHER,
        "gfrs-progress-reporter",
    ]
    assert command[4:6] == [pr.REPORTER_BIN, "training-progress"]
    assert (pod, REPORTER) in cluster.reporters, (
        "the product CLI accepted the argv the launcher hands on"
    )
    assert pr.REPORTER_PID_FILE.startswith("/tmp/"), (
        "the PID file lives on the reporter's writable /tmp emptyDir"
    )


def test_stop_reporter_stops_the_launched_reporter_and_proves_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, launcher_cluster: Any
) -> None:
    cluster, job = job_tests.progress_job(
        tmp_path, monkeypatch, variant="reporter_silent"
    )
    job.start()
    pod = only_pod(cluster)
    launcher = cluster.launchers[(pod, REPORTER)]
    pid = reporter_pid(cluster, pod)
    # The reporter is the launcher's child and the program itself, no shell in
    # between -- the process a SIGSTOP from inside the namespace does reach.
    assert pid not in {1, launcher.pid}
    assert status_field(pid, "PPid") == str(launcher.pid)
    assert Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")[0] == b"sleep"

    stop = pr.reporter_stopped(job.stop_reporter(0))

    assert stop == {"pid": pid, "state": "T (stopped)"}
    assert status_field(pid, "State") == "T (stopped)"
    (call,) = [item for item in cluster.execs if item["container"] == REPORTER]
    assert call["command"] == ["/bin/sh", "-c", pr.STOP_REPORTER]
    # A Pod deletion's SIGTERM ends the launcher at once -- as PID 1 a shell
    # without a trap would ignore it until the grace period -- whether or not
    # the reporter it forwards to is stopped.
    launcher.terminate()
    assert launcher.wait(timeout=10) == 143


@pytest.mark.parametrize(
    "output",
    ["", "garbage", LIVE_UNSTOPPED, "4242\n\tS (sleeping)\n", "1\n\tT (stopped)\n"],
    ids=["empty", "garbage", "pid1-running", "running", "pid1-stopped"],
)
def test_a_reporter_that_did_not_stop_is_inconclusive(output: str) -> None:
    with pytest.raises(VariantInconclusive, match="did not stop"):
        pr.reporter_stopped(output)


def test_reporter_stopped_reads_the_pid_and_the_state() -> None:
    assert pr.reporter_stopped(STOPPED) == {"pid": 4242, "state": "T (stopped)"}


# --------------------------------------------------------------------------- #
# the variant over the fakes
# --------------------------------------------------------------------------- #
class StoppedReporterJob:
    """``ManagedProgressJob`` stand-in whose stop prints a stopped reporter.

    Carries the real job's ``ranks``/``rank_nodes`` surface (rank 0 on the lane
    node, rank 1 on ``--node-b``): the records read names the job's nodes.
    """

    output = STOPPED

    def __init__(
        self, ctx: Any, *, name: str, ranks: tuple[int, ...] = (0,), gpu_count: int = 8
    ) -> None:
        self.name = name
        self.attempt_id = f"{name}-a001"
        self.ranks = tuple(sorted(set(ranks)))
        self.rank_nodes = {
            rank: ctx.settings.node if rank == 0 else ctx.settings.peer_node
            for rank in self.ranks
        }

    def start(self) -> dict[str, Any]:
        return {}

    def identity(self) -> dict[str, Any]:
        return {
            "job_id": self.name,
            "attempt_id": self.attempt_id,
            "ranks": list(self.ranks),
            "nodes": [self.rank_nodes[rank] for rank in self.ranks],
        }

    def write_progress(self, rank: int, **fields: Any) -> dict[str, Any]:
        return {}

    def stop_reporter(self, rank: int) -> str:
        return self.output

    def delete(self) -> None:
        pass


class UnstoppedReporterJob(StoppedReporterJob):
    output = LIVE_UNSTOPPED


def incident(kind: str) -> dict[str, Any]:
    return {
        "incident_id": f"inc-training-{kind}",
        "event_id": f"training-{kind}-deadbeef",
        "state": "ACTION_PENDING",
        "attempt_id": "a",
        "workflow_request_id": "wf",
        "reasons": [],
    }


def records(*kinds: str) -> dict[str, Any]:
    return {
        "progress_states": [],
        "incidents": [incident(kind) for kind in kinds],
        "workflows": {
            "wf": {
                "official_steps": [
                    {"operation": op, "parameters": {}}
                    for op in ("FREEZE_EVIDENCE", "VALIDATE_FABRIC")
                ],
                "step_executions": [],
            }
        },
        "decision": None,
        "notifications": [],
    }


def context(
    tmp_path: Path, doc: dict[str, Any], reads: list[tuple[str, ...]] | None = None
) -> Any:
    """``reads`` collects the arguments of every records read (the probe's argv)."""

    (tmp_path / "site.yaml").write_text("site: test\n")
    settings = dataclasses.replace(
        make_settings(SPEC, node=NODE), site_file=tmp_path / "site.yaml"
    )

    def cpu_python(script: str, *arguments: str) -> dict[str, Any]:
        if reads is not None and script is pr.TRAINING_PROBE:
            reads.append(arguments)
        return doc

    ctx = make_context(
        SPEC,
        tmp_path=tmp_path,
        regional=FakeRegional(cpu_python=cpu_python),
        probes={},
        settings=settings,
    )
    node_probe(ctx, FakeProbe({"survey": CLEAN_SURVEY}))
    return ctx


def test_reporter_silent_journals_the_proven_stop_and_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pr, "ManagedProgressJob", StoppedReporterJob)
    reads: list[tuple[str, ...]] = []
    ctx = context(tmp_path, records("reporter-silent"), reads)

    result = SPEC.execute(ctx)

    assert result["verdict"] == "PASS", result["errors"]
    assert json.loads((ctx.case_dir / "silent-stop.json").read_text()) == {
        "pid": 4242,
        "state": "T (stopped)",
    }
    # The single-rank job's records are read on the lane node, its only node.
    assert reads and {arguments[1] for arguments in reads} == {NODE}, reads


def test_reporter_silent_is_inconclusive_when_the_reporter_kept_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live run's shape: nothing stopped, the product graded a stall.

    An injection that never happened grades nothing about the product: the
    variant is INCONCLUSIVE at the stop, not FAIL on the hang it then reads.
    """

    monkeypatch.setattr(pr, "ManagedProgressJob", UnstoppedReporterJob)
    ctx = context(tmp_path, records("hang"))

    with pytest.raises(VariantInconclusive, match="did not stop"):
        SPEC.execute(ctx)
