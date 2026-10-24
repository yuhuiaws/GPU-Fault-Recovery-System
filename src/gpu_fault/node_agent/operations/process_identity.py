"""Bounded process identity checks before diagnostic attachment."""

from pathlib import Path
from typing import Any, Callable

from gpu_fault.node_agent.protocol import NodeActionCommand


def process_start_ticks(proc_root: Path, pid: int) -> int | None:
    try:
        with (proc_root / str(pid) / "stat").open() as stream:
            text = stream.read(8192)
        fields = text[text.rfind(")") + 2 :].split()
        return None if fields[0] in {"Z", "X"} else int(fields[19])
    except (OSError, ValueError, IndexError):
        return None


def namespace_pids(proc_root: Path, pid: int) -> list[int]:
    values = {pid}
    try:
        with (proc_root / str(pid) / "status").open() as stream:
            for line in stream.read(65536).splitlines():
                if line.startswith("NSpid:"):
                    values.update(int(value) for value in line.split()[1:] if value.isdecimal())
    except OSError:
        pass
    return sorted(values)


def workload_process(proc_root: Path, pid: int, paths: list[str]) -> bool:
    expected = [
        path.rstrip("/")
        for path in paths
        if isinstance(path, str) and path.startswith("/") and path != "/"
    ]
    if not expected:
        return False
    try:
        with (proc_root / str(pid) / "cgroup").open() as stream:
            text = stream.read(65536)
    except OSError:
        return False
    return any(
        path == root or path.startswith(root + "/")
        for line in text.splitlines()
        if len(parts := line.split(":", 2)) == 3
        for path in [parts[2].rstrip("/")]
        for root in expected
    )


def target_identity_matches(proc_root: Path, pid: int, expected: Any) -> bool:
    return (
        type(expected) is int
        and expected > 0
        and process_start_ticks(proc_root, pid) == expected
    )


def scoped_processes(
    proc_root: Path,
    command: NodeActionCommand,
    rows: list[dict[str, Any]],
    errors: list[str],
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    rows = list({row["pid"]: row for row in rows}.values())
    paths = command.parameters.get("workload_cgroup_paths_by_node", {}).get(
        command.node_id, []
    )
    identities = {
        row["pid"]: {
            "verified": workload_process(proc_root, row["pid"], paths),
            "start_time_ticks": process_start_ticks(proc_root, row["pid"]),
        }
        for row in rows
    }
    if command.parameters.get("require_workload_identity"):
        selected = [
            row
            for row in rows
            if identities[row["pid"]]["verified"]
            and identities[row["pid"]]["start_time_ticks"] is not None
        ]
        if len(selected) != len(rows):
            errors.append(
                "processes skipped: workload cgroup/start identity not proven"
            )
        rows = selected
    return rows, identities


def stamp_process_identity(
    proc_root: Path,
    ranks: list[dict[str, Any]],
    identities: dict[int, dict[str, Any]],
    paths: list[str] | None = None,
) -> None:
    for row in ranks:
        identity = identities[row["pid"]]
        unchanged = identity["start_time_ticks"] == process_start_ticks(
            proc_root, row["pid"]
        )
        row["process_identity"] = {
            **identity,
            "verified": bool(
                identity["verified"]
                and unchanged
                and (row.get("process_identity") or {}).get("verified", True)
                and (paths is None or workload_process(proc_root, row["pid"], paths))
            ),
        }
        if not unchanged:
            row.update(
                flight_recorder={"status": "process_changed"},
                python_stack={},
                proc={},
                gpu={},
            )


def bind_process_ranks(
    command: NodeActionCommand,
    rows: list[dict[str, Any]],
    read_environment: Callable[[int], dict[str, str]],
) -> None:
    rank_mapping = command.parameters.get("rank_by_pid_by_node", {})
    mapping = (
        rank_mapping.get(command.node_id, {}) if isinstance(rank_mapping, dict) else {}
    )
    for row in rows:
        environment = read_environment(row["pid"])
        rank = environment.get("RANK")
        if rank is None or not rank.isdecimal():
            rank = mapping.get(str(row["pid"]))
        try:
            row["rank"] = int(rank) if rank is not None else None
        except (TypeError, ValueError):
            row["rank"] = None
        try:
            local = environment.get("LOCAL_RANK")
            row["local_rank"] = int(local) if local is not None else None
        except (TypeError, ValueError):
            row["local_rank"] = None
        row["_environment"] = environment
