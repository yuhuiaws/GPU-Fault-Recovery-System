"""The CRI Pod-log replay probe's fixture files, readers and CLI shell.

``create-log``/``seed-log``/``write``/``rotate`` refusals the runner branches on
(their reason tokens), the read-only ``inspect`` and ``preflight`` readers over
a fake kubelet tree, cgroup and ``/proc``, the collector-state oracle on odd
state files, and ``main``'s failure line for an exception that is not a
``ProbeError``. The host is ``_cov95_general_live_applog``'s fake node.
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional.probes import general_live_applog_probe as probe
from tests.regional._cov95_general_live_applog import (
    CONTAINER,
    NAMESPACE,
    OLD_UID,
    POD,
    RUN_ID,
    UID,
    fixture_name,
    install_host,
    kubelet_pod,
    record_flags,
)

Cli = Callable[..., tuple[int, dict[str, Any], str]]
CONTAINER_ID = "cd" * 32


@pytest.fixture
def host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    host = install_host(monkeypatch, tmp_path)
    probe.save_state(
        RUN_ID,
        {
            "run_id": RUN_ID,
            "container": CONTAINER,
            "glob": probe.source_glob(CONTAINER, RUN_ID),
            "logs": {},
            "dropin": str(probe.dropin_path(RUN_ID)),
            "enabled_log_collector": True,
        },
    )
    return host


@pytest.fixture
def cli(capsys: pytest.CaptureFixture[str]) -> Cli:
    def invoke(*argv: str) -> tuple[int, dict[str, Any], str]:
        code = probe.main(list(argv))
        captured = capsys.readouterr()
        lines = captured.out.strip().splitlines()
        assert lines, f"the probe prints one JSON object for {argv}: {captured}"
        return code, json.loads(lines[-1]), captured.err

    return invoke


def create_log(cli: Cli, pod_uid: str = UID) -> Any:
    return cli(
        "create-log",
        "--run-id",
        RUN_ID,
        "--pod-uid",
        pod_uid,
        "--container",
        CONTAINER,
        "--namespace",
        NAMESPACE,
        "--pod-name",
        POD,
    )


def write_args(records: Any, *, target: str = "current") -> argparse.Namespace:
    sha256, encoded = record_flags(records)
    return argparse.Namespace(
        run_id=RUN_ID,
        pod_uid=UID,
        container=CONTAINER,
        sha256=sha256,
        b64=encoded,
        target=target,
    )


def write_argv(records: Any) -> list[str]:
    sha256, encoded = record_flags(records)
    return [
        "write",
        "--run-id",
        RUN_ID,
        "--pod-uid",
        UID,
        "--container",
        CONTAINER,
        "--sha256",
        sha256,
        "--b64",
        encoded,
    ]


def seed_args(records: list[dict[str, str]]) -> argparse.Namespace:
    sha256, encoded = record_flags(records)
    return argparse.Namespace(
        run_id=RUN_ID,
        pod_uid=OLD_UID,
        container=CONTAINER,
        namespace=NAMESPACE,
        pod_name=POD,
        sha256=sha256,
        b64=encoded,
        allow_create_dir=True,
        await_gc_seconds=0,
    )


# --------------------------------------------------------------------------- #
# create-log / seed-log / write / rotate refusals
# --------------------------------------------------------------------------- #
def test_create_log_refuses_a_fixture_file_left_on_disk(
    host: SimpleNamespace, cli: Cli
) -> None:
    pod = kubelet_pod(host)
    leftover = pod / CONTAINER / fixture_name()
    leftover.write_text("an earlier round's line\n", encoding="utf-8")
    code, answer, _ = create_log(cli)
    assert code == 1 and answer["error_reason"] == "fixture_exists", answer
    assert leftover.read_text(encoding="utf-8") == "an earlier round's line\n", (
        "the file on disk is never truncated"
    )
    assert probe.load_state(RUN_ID)["logs"] == {}, "no record for a refused log"


def test_seed_log_passes_both_seams_and_writes_the_records(
    host: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    stages: list[str] = []
    probe.seed_log(
        seed_args([{"message": "NCCL WARN timeout"}]), interrupt=stages.append
    )
    answer = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert stages == ["directory", "container"], "the seam is called after each step"
    assert answer["records"] == 1 and answer["file"]["size"] > 0, answer
    assert answer["kubelet_directory"] is False, "the old Pod's directory was made"


def test_a_write_racing_kubelets_gc_names_the_file_gone(
    host: SimpleNamespace, cli: Cli, monkeypatch: pytest.MonkeyPatch
) -> None:
    pod = kubelet_pod(host)
    assert create_log(cli)[0] == 0, "fixture created"
    clock = probe.utc_now

    def gc_takes_the_directory() -> Any:
        shutil.rmtree(pod)  # between the existence check and the append
        return clock()

    monkeypatch.setattr(probe, "utc_now", gc_takes_the_directory)
    code, answer, _ = cli(*write_argv([{"message": "lost"}]))
    assert code == 1 and answer["error_reason"] == "fixture_gone", answer
    assert str(answer["error_site"]).startswith("_append_records:"), answer
    assert probe.load_state(RUN_ID)["logs"][f"{UID}/{CONTAINER}"]["writes"] == 0, (
        "a lost write is not counted"
    )


def test_write_refuses_an_unknown_target(host: SimpleNamespace, cli: Cli) -> None:
    kubelet_pod(host)
    assert create_log(cli)[0] == 0, "fixture created"
    with pytest.raises(probe.ProbeError, match="target must be one of") as refused:
        probe.write(write_args([{"message": "x"}], target="archive"))
    assert refused.value.reason == "input_shape", refused.value.reason


def test_rotate_refuses_to_overwrite_an_existing_rotation(
    host: SimpleNamespace, cli: Cli
) -> None:
    pod = kubelet_pod(host)
    assert create_log(cli)[0] == 0, "fixture created"
    taken = pod / CONTAINER / f"{fixture_name()}.1"
    taken.write_text("someone else's rotation\n", encoding="utf-8")
    code, answer, _ = cli(
        "rotate", "--run-id", RUN_ID, "--pod-uid", UID, "--container", CONTAINER
    )
    assert code == 1 and answer["error_reason"] == "rotation_exists", answer
    assert taken.read_text(encoding="utf-8") == "someone else's rotation\n", (
        "the existing file is left as it was"
    )


def test_validate_records_refuses_a_record_that_is_not_an_object() -> None:
    with pytest.raises(probe.ProbeError, match="must be an object") as refused:
        probe.validate_records(["step 1"])
    assert refused.value.reason == "input_shape", refused.value.reason


def test_an_unexpected_exception_is_reported_with_its_class_and_probe_site(
    host: SimpleNamespace, cli: Cli
) -> None:
    kubelet_pod(host)
    assert create_log(cli)[0] == 0, "fixture created"
    code, answer, err = cli(*write_argv(b"this is not JSON"))
    assert code == 1, answer
    assert answer["error_class"] == "JSONDecodeError", answer
    assert answer["error_reason"] is None, "only a ProbeError carries a reason token"
    assert str(answer["error_site"]).startswith("decode_records:"), (
        "the site is the probe's own innermost frame, not the json module's"
    )
    assert err.startswith("JSONDecodeError: "), err


# --------------------------------------------------------------------------- #
# inspect / preflight readers
# --------------------------------------------------------------------------- #
def test_inspect_lists_only_this_pods_real_log_directories(
    host: SimpleNamespace, cli: Cli
) -> None:
    pod = kubelet_pod(host)
    (pod / "pytorch.lock").write_text("", encoding="utf-8")  # not a container dir
    kubelet_pod(host, OLD_UID)  # another Pod
    (host.pods / f"README_{UID}").write_text("", encoding="utf-8")  # a file
    code, answer, _ = cli("inspect", "--run-id", RUN_ID, "--pod-uid", UID)
    assert code == 0, answer
    assert answer == {
        "run_id": RUN_ID,
        "pod_uid": UID,
        "directories": [{"path": str(pod), "containers": {CONTAINER: ["0.log"]}}],
    }, "without a container id no process is read"


def test_inspect_without_a_pod_log_root_finds_nothing(
    host: SimpleNamespace, cli: Cli, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(probe, "POD_LOG_ROOT", host.root / "no-pods")
    code, answer, _ = cli("inspect", "--run-id", RUN_ID, "--pod-uid", UID)
    assert code == 0 and answer["directories"] == [], answer


def proc_entry(root: Path, pid: int, status: str, *, stat: bool = True) -> None:
    directory = root / str(pid)
    directory.mkdir(parents=True)
    (directory / "status").write_text(status, encoding="ascii")
    if stat:
        fields = ["0"] * 52
        fields[21] = "4321"
        (directory / "stat").write_text(f"{pid} (python3) S " + " ".join(fields[3:]))
        (directory / "cmdline").write_bytes(b"python3\x00train.py\x00")


def container_tree(root: Path) -> tuple[Path, Path]:
    cgroup, proc = root / "cgroup", root / "proc"
    # cgroup v2 under the systemd driver: an unreadable procs entry ...
    scope = cgroup / "kubepods.slice" / "pod" / f"cri-containerd-{CONTAINER_ID}.scope"
    (scope / "cgroup.procs").mkdir(parents=True)
    # ... and the cgroupfs driver's readable one.
    procs = cgroup / "kubepods" / "burstable" / CONTAINER_ID / "cgroup.procs"
    procs.parent.mkdir(parents=True)
    procs.write_text("10\n11\n", encoding="ascii")
    proc_entry(proc, 10, "Name:\tpython3\nState:\tS (sleeping)\nNSpid:\t10\t1\n")
    proc_entry(proc, 11, "Name:\tpython3\nNSpid:\t11\t7\n", stat=False)
    return cgroup, proc


def test_container_init_pid_skips_unreadable_entries_and_picks_the_namespace_init(
    tmp_path: Path,
) -> None:
    cgroup, proc = container_tree(tmp_path)
    value = probe.container_init_pid(CONTAINER_ID, cgroup_root=cgroup, proc_root=proc)
    assert value["process_count"] == 2 and value["init_pid"] == 10, value
    assert value["init"]["nspid"] == [10, 1] and value["init"]["start_ticks"] == 4321
    assert value["init"]["cmdline"] == "python3 train.py", value["init"]
    assert len(value["cgroup_procs"]) == 2, "both drivers' entries were globbed"
    other = probe.container_init_pid(
        f"containerd://{CONTAINER_ID}", cgroup_root=cgroup, proc_root=proc
    )
    assert other["init_pid"] == 10, "a runtime-prefixed id names the same container"


def test_a_process_that_cannot_be_read_is_reported_not_raised(tmp_path: Path) -> None:
    cgroup, proc = container_tree(tmp_path)
    shutil.rmtree(proc / "10")  # exited between the cgroup read and /proc
    value = probe.container_init_pid(CONTAINER_ID, cgroup_root=cgroup, proc_root=proc)
    assert value["process_count"] == 2 and value["init_pid"] == 10, value
    assert value["init"] == {"pid": 10, "error": "process could not be read"}, (
        "without a readable namespace init the first process is reported as is"
    )


def test_inspect_with_a_container_id_reads_its_init_process(
    host: SimpleNamespace, cli: Cli, monkeypatch: pytest.MonkeyPatch
) -> None:
    cgroup, proc = container_tree(host.root)
    reader = probe.container_init_pid
    monkeypatch.setattr(
        probe,
        "container_init_pid",
        lambda container_id: reader(container_id, cgroup_root=cgroup, proc_root=proc),
    )
    code, answer, _ = cli(
        "inspect", "--run-id", RUN_ID, "--pod-uid", UID, "--container-id", CONTAINER_ID
    )
    assert code == 0 and answer["process"]["init_pid"] == 10, answer


def test_preflight_counts_kubelets_layout_past_an_unreadable_directory(
    host: SimpleNamespace, cli: Cli
) -> None:
    if os.geteuid() == 0:
        pytest.skip("root reads a mode-0000 directory")
    kubelet_pod(host)
    locked = kubelet_pod(host, OLD_UID)
    locked.chmod(0o000)
    try:
        code, answer, _ = cli("preflight")
    finally:
        locked.chmod(0o755)
    assert code == 0, answer
    assert answer["pod_log_layout"] == {
        "directories": 2,
        "kubelet_shape": 2,
        "with_container_log": 1,
    }, "a directory that cannot be listed is counted by shape only"
    assert answer["unit"]["LoadState"] == "loaded", answer["unit"]
    assert answer["collector_state"] is False, answer


# --------------------------------------------------------------------------- #
# the collector-state oracle on odd state files
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "content, readable",
    [
        ([1, 2], False),
        ({"training_log_files": ["x"], "training_log_pending": ["y"]}, True),
    ],
    ids=["not-an-object", "maps-of-the-wrong-type"],
)
def test_collector_state_view_tolerates_odd_state_files(
    host: SimpleNamespace, content: object, readable: bool
) -> None:
    probe.COLLECTOR_STATE_PATH.parent.mkdir(parents=True)
    probe.COLLECTOR_STATE_PATH.write_text(json.dumps(content), encoding="utf-8")
    view = probe.collector_state_view(RUN_ID)
    assert view["readable"] is readable, view
    assert (view["tracked"], view["pending"]) == ({}, {}), "nothing is attributed"


# --------------------------------------------------------------------------- #
# the CLI shell
# --------------------------------------------------------------------------- #
def test_a_successful_command_exits_zero(host: SimpleNamespace, cli: Cli) -> None:
    code, answer, err = cli("snapshot", "--run-id", RUN_ID)
    assert (code, err) == (0, ""), answer
    assert answer["window_open"] is True, answer


def test_running_the_file_as_a_script_exits_with_the_cli_status(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # An unsafe run ID is refused before the probe reads or writes the host.
    monkeypatch.setattr(sys, "argv", [probe.__file__, "snapshot", "--run-id", "BAD ID"])
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(probe.__file__, run_name="__main__")
    assert raised.value.code == 1, "a refused command exits 1"
    answer = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert (answer["error_reason"], answer["command"]) == ("input_shape", "snapshot")
    assert answer["error_message"] == "unsafe run ID", answer
