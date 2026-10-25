"""Contract tests for the CRI Pod-log host probe (GF-GENERAL-LIVE-017).

Nothing here runs systemd, kubectl or the node log collector: the stdlib-only
validators, the CRI syntax the probe writes, the cgroup/proc read of a
container's init PID, the collector-state oracle and the file handlers are
driven over temporary trees, and the lines the probe writes are decoded by the
product's own ``decode_record``/``pod_identity`` to prove they are the product's
input, not a look-alike.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from gpu_fault.collectors.logs.node_records import decode_record, pod_identity
from scripts.e2e.regional.probes import general_live_applog_probe as probe

RUN = "gl17-live-pid-a1-deadbeef"
UID = "0f3b7c2e-4d5a-4b6c-8d9e-0a1b2c3d4e5f"
CONTAINER_ID = "ab" * 32


class _Budget:
    max_bytes = 1 << 20
    bytes_read = 0

    def spent(self) -> str | None:
        return None

    def consume_bytes(self, raw_bytes: int) -> None:
        self.bytes_read += raw_bytes


# --------------------------------------------------------------------------- #
# identifiers and reviewed records
# --------------------------------------------------------------------------- #
def test_identifiers_are_checked_by_shape() -> None:
    assert probe.safe_id(RUN) == RUN
    with pytest.raises(probe.ProbeError, match="unsafe run ID"):
        probe.safe_id("bad id; rm -rf /")
    assert probe.checked_pod_uid(UID) == UID
    with pytest.raises(probe.ProbeError, match="UUID"):
        probe.checked_pod_uid("../etc")
    with pytest.raises(probe.ProbeError, match="DNS-1123"):
        probe.checked_name("Bad_Name", "Pod name")
    with pytest.raises(probe.ProbeError, match="label"):
        probe.checked_label("py.torch")
    assert probe.checked_container_id(f"containerd://{CONTAINER_ID}") == CONTAINER_ID
    with pytest.raises(probe.ProbeError, match="64 hex"):
        probe.checked_container_id("short")
    assert len(probe.digest(RUN)) == 16 and probe.digest(RUN) == probe.digest(RUN)
    assert probe.fixture_file_name(RUN) == f"gfrs-live-{probe.digest(RUN)}.log"
    assert probe.source_glob("pytorch", RUN) == (
        f"/var/log/pods/*/pytorch/{probe.fixture_file_name(RUN)}*"
    )


def test_validate_records_accepts_reviewed_set_and_rejects_bad_input() -> None:
    checked = probe.validate_records(
        [{"message": "a line"}, {"message": "part ", "flag": "P", "stream": "stderr"}]
    )
    assert checked == [
        {"message": "a line", "stream": "stdout", "flag": "F"},
        {"message": "part ", "stream": "stderr", "flag": "P"},
    ]
    for records, match in (
        ([], "non-empty list"),
        ([{"message": ""}], "non-empty message"),
        ([{"message": "two\nlines"}], "one line"),
        ([{"message": "x", "stream": "file"}], "stdout or stderr"),
        ([{"message": "x", "flag": "Q"}], "P or F"),
        ([{"message": "x"}] * (probe.MAX_RECORDS + 1), "at most"),
        ([{"message": "A" * (probe.MAX_MESSAGE_BYTES + 1)}], "too large"),
    ):
        with pytest.raises(probe.ProbeError, match=match):
            probe.validate_records(records)


def test_decode_records_requires_the_matching_digest() -> None:
    content = json.dumps([{"message": "x"}]).encode()
    good = hashlib.sha256(content).hexdigest()
    b64 = base64.b64encode(content).decode()
    assert probe.decode_records(good, b64) == [
        {"message": "x", "stream": "stdout", "flag": "F"}
    ]
    with pytest.raises(probe.ProbeError, match="does not match"):
        probe.decode_records("0" * 64, b64)
    with pytest.raises(probe.ProbeError, match="sha256 is required"):
        probe.decode_records("nope", b64)
    with pytest.raises(probe.ProbeError, match="base64"):
        probe.decode_records(good, "%%%")


# --------------------------------------------------------------------------- #
# the lines are the product's input
# --------------------------------------------------------------------------- #
def test_cri_line_and_pod_log_dir_are_decoded_by_the_product(tmp_path: Path) -> None:
    at = datetime(2026, 10, 25, 10, 0, 0, 123456, tzinfo=timezone.utc)
    line = probe.cri_line(
        {"message": "hello world", "stream": "stdout", "flag": "F"}, at
    )
    assert line == "2026-10-25T10:00:00.123456000Z stdout F hello world\n"
    decoded = decode_record(line.encode(), 0, datetime.now(timezone.utc), {}, 65536)
    assert decoded is not None
    _start, event_at, message, fields = decoded
    assert (event_at, message) == (at, "hello world")
    assert fields == {"timestamp_source": "cri", "stream": "stdout"}

    root = tmp_path / "pods"
    directory = root / probe.pod_log_dir("gpu-fault-system", "job-master-0", UID).name
    path = directory / "pytorch" / probe.fixture_file_name(RUN)
    path.parent.mkdir(parents=True)
    path.write_text(line)
    assert pod_identity(str(path), str(root)) == {
        "pod_uid": UID,
        "container_name": "pytorch",
    }


def test_a_partial_then_full_record_stitches_in_the_product_decoder() -> None:
    at = datetime(2026, 10, 25, 10, 0, 0, tzinfo=timezone.utc)
    fragments: dict[str, object] = {}
    head = probe.cri_line({"message": "col ", "stream": "stdout", "flag": "P"}, at)
    assert decode_record(head.encode(), 0, at, fragments, 65536) is None
    tail = probe.cri_line({"message": "timeout", "stream": "stdout", "flag": "F"}, at)
    decoded = decode_record(tail.encode(), 40, at, fragments, 65536)
    assert decoded is not None and decoded[2] == "col timeout"
    assert decoded[0] == 0, "the stitched record starts where the P fragment did"


def test_source_glob_matches_only_this_runs_files(tmp_path: Path) -> None:
    root = tmp_path / "var" / "log" / "pods"
    container = root / f"ns_pod_{UID}" / "pytorch"
    container.mkdir(parents=True)
    name = probe.fixture_file_name(RUN)
    for item in (
        "0.log",
        "0.log.20261025-100000",
        name,
        f"{name}.1",
        "gfrs-live-other.log",
    ):
        (container / item).write_text("")
    glob = probe.source_glob("pytorch", RUN).replace("/var/log/pods", str(root))
    found = sorted(path.name for path in Path("/").glob(glob.lstrip("/")))
    assert found == [name, f"{name}.1"], found


# --------------------------------------------------------------------------- #
# container init PID (cgroup + /proc)
# --------------------------------------------------------------------------- #
def _proc(root: Path, pid: int, *, nspid: str, comm: str, start: int) -> None:
    directory = root / str(pid)
    directory.mkdir(parents=True)
    (directory / "status").write_text(f"Name:\t{comm}\nNSpid:\t{nspid}\n")
    fields = ["0"] * 52
    fields[21] = str(start)
    (directory / "stat").write_text(f"{pid} ({comm}) S " + " ".join(fields[3:]) + "\n")
    (directory / "cmdline").write_bytes(b"python3\x00-u\x00train.py\x00")


def test_container_init_pid_reads_the_cgroup_and_picks_the_namespace_init(
    tmp_path: Path,
) -> None:
    cgroup = tmp_path / "cgroup"
    scope = (
        cgroup
        / "kubepods.slice"
        / "kubepods-burstable.slice"
        / f"kubepods-burstable-pod{UID.replace('-', '_')}.slice"
        / f"cri-containerd-{CONTAINER_ID}.scope"
    )
    scope.mkdir(parents=True)
    (scope / "cgroup.procs").write_text("4300\n4242\n")
    proc = tmp_path / "proc"
    _proc(proc, 4242, nspid="4242\t1", comm="python3", start=777)
    _proc(proc, 4300, nspid="4300\t17", comm="python3", start=900)
    value = probe.container_init_pid(
        f"containerd://{CONTAINER_ID}", cgroup_root=cgroup, proc_root=proc
    )
    assert value["init_pid"] == 4242, value
    assert value["init"]["start_ticks"] == 777 and value["init"]["comm"] == "python3"
    assert value["process_count"] == 2 and value["init"]["cmdline"].startswith(
        "python3"
    )
    with pytest.raises(probe.ProbeError, match="no process"):
        probe.container_init_pid("cd" * 32, cgroup_root=cgroup, proc_root=proc)


# --------------------------------------------------------------------------- #
# the collector-state oracle
# --------------------------------------------------------------------------- #
def test_collector_state_view_filters_this_runs_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = f"/var/log/pods/ns_pod_{UID}/pytorch/{probe.fixture_file_name(RUN)}"
    state = {
        "training_log_files": {
            path: {"device": 1, "inode": 2, "offset": 60},
            "/var/log/pods/ns_other_x/pytorch/0.log": {
                "device": 1,
                "inode": 9,
                "offset": 1,
            },
        },
        "training_log_pending": {
            path: {
                "cri": {
                    "stdout": {
                        "start": 0,
                        "at": "2026-10-25T10:00:00+00:00",
                        "message": "col ",
                    }
                }
            }
        },
    }
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state))
    monkeypatch.setattr(probe, "COLLECTOR_STATE_PATH", state_path)
    view = probe.collector_state_view(RUN)
    assert view["readable"] is True
    assert list(view["tracked"]) == [path] and view["tracked"][path]["offset"] == 60
    assert view["pending"][path]["cri"]["stdout"]["message"] == "col "
    monkeypatch.setattr(probe, "COLLECTOR_STATE_PATH", tmp_path / "missing.json")
    assert probe.collector_state_view(RUN)["readable"] is False


# --------------------------------------------------------------------------- #
# file handlers over a temporary tree (no systemd)
# --------------------------------------------------------------------------- #
@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    pods = tmp_path / "var" / "log" / "pods"
    pods.mkdir(parents=True)
    monkeypatch.setattr(probe, "POD_LOG_ROOT", pods)
    monkeypatch.setattr(probe, "STATE_ROOT", tmp_path / "run" / "applog")
    # No systemd here: the residue check reads the unit through this.
    monkeypatch.setattr(
        probe,
        "unit_state",
        lambda unit: {"ActiveState": "inactive", "UnitFileState": "disabled"},
    )
    probe.save_state(
        RUN,
        {
            "run_id": RUN,
            "container": "pytorch",
            "glob": probe.source_glob("pytorch", RUN),
            "logs": {},
            "dropin": str(tmp_path / "dropin.conf"),
            "enabled_log_collector": True,
        },
    )
    return {"pods": pods, "tmp": tmp_path}


def _emitted(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def _create(pod_uid: str, *, allow: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        run_id=RUN,
        pod_uid=pod_uid,
        container="pytorch",
        namespace="gpu-fault-system",
        pod_name="job-master-0",
        allow_create_dir=allow,
    )


def _write(
    records: list[dict[str, str]], *, target: str = "current"
) -> argparse.Namespace:
    content = json.dumps(records, sort_keys=True).encode()
    return argparse.Namespace(
        run_id=RUN,
        pod_uid=UID,
        container="pytorch",
        sha256=hashlib.sha256(content).hexdigest(),
        b64=base64.b64encode(content).decode(),
        target=target,
    )


def test_create_log_uses_the_real_pod_directory_and_never_touches_kubelets_files(
    tree: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    kubelet_dir = tree["pods"] / f"gpu-fault-system_job-master-0_{UID}" / "pytorch"
    kubelet_dir.mkdir(parents=True)
    (kubelet_dir / "0.log").write_text("2026-10-25T10:00:00Z stdout F real\n")
    probe.create_log(_create(UID))
    receipt = _emitted(capsys)
    assert receipt["created_dirs"] == [] and receipt["kubelet_directory"] is True
    path = Path(str(receipt["file"]["path"]))  # type: ignore[index]
    assert path.parent == kubelet_dir and path.stat().st_size == 0
    assert (kubelet_dir / "0.log").read_text().endswith("real\n"), (
        "kubelet's own log file is untouched"
    )
    with pytest.raises(probe.ProbeError, match="already exists"):
        probe.create_log(_create(UID))


def test_create_log_creates_an_old_pods_directory_only_when_allowed(
    tree: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(probe.ProbeError, match="does not exist"):
        probe.create_log(_create(UID))
    probe.create_log(_create(UID, allow=True))
    receipt = _emitted(capsys)
    directory = tree["pods"] / f"gpu-fault-system_job-master-0_{UID}"
    assert receipt["created_dirs"] == [str(directory), str(directory / "pytorch")]
    assert receipt["kubelet_directory"] is False
    with pytest.raises(probe.ProbeError, match="differs from the open window"):
        probe.create_log(
            argparse.Namespace(
                **{**vars(_create(UID, allow=True)), "container": "other"}
            )
        )


def test_create_log_may_start_over_only_after_kubelet_took_the_file(
    tree: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    probe.create_log(_create(UID, allow=True))
    first = _emitted(capsys)
    path = Path(str(first["file"]["path"]))  # type: ignore[index]
    with pytest.raises(probe.ProbeError, match="already exists"):
        probe.create_log(_create(UID, allow=True))
    # kubelet's GC removed the recreated directory (old_pod) before a read.
    path.unlink()
    path.parent.rmdir()
    path.parent.parent.rmdir()
    probe.create_log(_create(UID, allow=True))
    second = _emitted(capsys)
    assert second["recreated"] == 1 and Path(str(second["file"]["path"])).exists()  # type: ignore[index]
    entry = probe.load_state(RUN)["logs"][f"{UID}/pytorch"]
    assert entry["recreated"] == 1 and entry["rotated"] == [] and entry["writes"] == 0


def test_write_rotate_and_write_rotated_follow_the_inode(
    tree: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    probe.create_log(_create(UID, allow=True))
    path = Path(str(_emitted(capsys)["file"]["path"]))  # type: ignore[index]
    probe.write(_write([{"message": "col ", "flag": "P"}]))
    head = _emitted(capsys)
    assert head["records"] == 1 and head["flags"] == ["P"]
    text = path.read_text()
    assert text.endswith(" stdout P col \n") and text.startswith(str(head["stamps"][0]))  # type: ignore[index]
    inode = path.stat().st_ino
    probe.rotate(argparse.Namespace(run_id=RUN, pod_uid=UID, container="pytorch"))
    rotated = _emitted(capsys)
    rotated_path = Path(str(rotated["rotated"]["path"]))  # type: ignore[index]
    assert rotated_path == Path(str(path) + ".1")
    assert rotated_path.stat().st_ino == inode, "the rename keeps the inode"
    assert path.stat().st_size == 0 and path.stat().st_ino != inode
    probe.write(_write([{"message": "timeout", "flag": "F"}], target="rotated"))
    tail = _emitted(capsys)
    assert tail["target"] == "rotated"
    assert rotated_path.read_text().splitlines()[-1].endswith(" stdout F timeout"), (
        "the F fragment lands on the rotated inode"
    )
    state = probe.load_state(RUN)
    entry = state["logs"][f"{UID}/pytorch"]
    assert entry["rotated"] == [str(rotated_path)] and entry["writes"] == 2


def test_write_refuses_a_rotated_target_before_any_rotation(
    tree: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    probe.create_log(_create(UID, allow=True))
    _emitted(capsys)
    with pytest.raises(probe.ProbeError, match="never rotated"):
        probe.write(_write([{"message": "x"}], target="rotated"))
    with pytest.raises(probe.ProbeError, match="no fixture log"):
        probe.write(
            argparse.Namespace(
                **{
                    **vars(_write([{"message": "x"}])),
                    "pod_uid": UID.replace("0f3b", "1a2b"),
                }
            )
        )


def test_remove_fixture_files_takes_exactly_what_the_run_made(
    tree: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    probe.create_log(_create(UID, allow=True))
    path = Path(str(_emitted(capsys)["file"]["path"]))  # type: ignore[index]
    probe.write(_write([{"message": "x"}]))
    _emitted(capsys)
    probe.rotate(argparse.Namespace(run_id=RUN, pod_uid=UID, container="pytorch"))
    _emitted(capsys)
    state = probe.load_state(RUN)
    removed = probe.remove_fixture_files(state)
    assert sorted(removed["files"]) == sorted([str(path), str(path) + ".1"])
    directory = path.parent.parent
    assert removed["directories"] == [str(path.parent), str(directory)]
    assert not directory.exists(), "the recreated old-Pod directory is removed"
    residue = {
        key: value
        for key, value in probe.window_residue(RUN, state).items()
        if key in {"fixture_files", "created_directories"}
    }
    assert residue == {"fixture_files": False, "created_directories": False}


def test_remove_fixture_files_leaves_a_directory_it_no_longer_owns_alone(
    tree: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    probe.create_log(_create(UID, allow=True))
    path = Path(str(_emitted(capsys)["file"]["path"]))  # type: ignore[index]
    (path.parent / "0.log").write_text("kubelet came back\n")
    state = probe.load_state(RUN)
    removed = probe.remove_fixture_files(state)
    assert removed["files"] == [str(path)] and removed["directories"] == []
    assert probe.window_residue(RUN, state)["created_directories"] is True
    # A directory kubelet already removed is not residue either.
    (path.parent / "0.log").unlink()
    path.parent.rmdir()
    path.parent.parent.rmdir()
    assert probe.remove_fixture_files(state)["absent"], "nothing left to remove"
    assert probe.window_residue(RUN, state)["created_directories"] is False


def test_file_view_and_real_pod_log_dirs(tree: dict[str, Path]) -> None:
    directory = tree["pods"] / f"gpu-fault-system_job-master-0_{UID}" / "pytorch"
    directory.mkdir(parents=True)
    (directory / "0.log").write_text("x\n")
    found = probe.real_pod_log_dirs(UID)
    assert found == [
        {"path": str(directory.parent), "containers": {"pytorch": ["0.log"]}}
    ]
    assert probe.file_view(directory / "0.log")["size"] == 2
    assert probe.file_view(directory / "missing")["exists"] is False
