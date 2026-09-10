from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from gpu_fault.node_agent.protocol import (
    NodeActionCommand,
)


class HungProcessOperationsMixin:
    # Attributes supplied by the composed concrete implementation.
    expand_python_cgroup_processes: Any
    now: Callable[..., Any]
    proc_root: Path
    python_stack_tool: Any
    runner: Callable[..., Any]
    sleep: Callable[..., Any]

    def _capture_hung_process_state(
        self,
        work_dir: Path,
        command: NodeActionCommand,
        manifest: dict[str, Any],
    ) -> None:
        settings = self._hung_capture_settings(command)
        process_rows = self._discover_hung_processes(
            work_dir,
            command,
            manifest,
            settings["max_processes"],
        )
        self._capture_system_context(work_dir, manifest)
        self._capture_python_processes(
            work_dir,
            manifest,
            process_rows,
            settings["python_sample_count"],
            settings["python_sample_interval"],
            settings["python_timeout"],
        )
        self._capture_proc_samples(
            work_dir,
            manifest,
            process_rows,
            settings["sample_count"],
            settings["duration"],
            settings["sample_interval"],
        )

    def _hung_capture_settings(self, command):
        sample_count = min(
            5,
            max(
                1,
                int(command.parameters.get("strace_sample_count", 3)),
            ),
        )
        duration = min(
            30,
            max(
                1,
                int(command.parameters.get("strace_duration_seconds", 3)),
            ),
        )
        sample_interval = min(
            30,
            max(
                0,
                int(command.parameters.get("strace_sample_interval_seconds", 2)),
            ),
        )
        python_sample_count = min(
            5,
            max(
                3,
                int(command.parameters.get("pyspy_sample_count", sample_count)),
            ),
        )
        python_sample_interval = min(
            30,
            max(
                0,
                int(
                    command.parameters.get(
                        "pyspy_sample_interval_seconds",
                        sample_interval,
                    )
                ),
            ),
        )
        python_timeout = min(
            60,
            max(
                1,
                int(command.parameters.get("pyspy_timeout_seconds", 10)),
            ),
        )
        max_processes = min(
            32,
            max(
                1,
                int(command.parameters.get("max_processes", 16)),
            ),
        )
        return {
            "sample_count": sample_count,
            "duration": duration,
            "sample_interval": sample_interval,
            "python_sample_count": python_sample_count,
            "python_sample_interval": python_sample_interval,
            "python_timeout": python_timeout,
            "max_processes": max_processes,
        }

    def _discover_hung_processes(self, work_dir, command, manifest, max_processes):
        query = [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,process_name",
            "--format=csv,noheader,nounits",
        ]
        process_rows = []
        raw_target_pids = command.parameters.get("target_pids_by_node", {})
        target_pids = (
            raw_target_pids.get(command.node_id, [])
            if isinstance(raw_target_pids, dict)
            else []
        )
        raw_gpu_by_pid = command.parameters.get("target_gpu_uuids_by_pid_by_node", {})
        gpu_by_pid = (
            raw_gpu_by_pid.get(command.node_id, {})
            if isinstance(raw_gpu_by_pid, dict)
            else {}
        )
        if target_pids:
            for raw_pid in target_pids:
                if not str(raw_pid).isdigit():
                    continue
                pid = int(raw_pid)
                try:
                    process_name = (
                        (self.proc_root / str(pid) / "comm").read_text().strip()
                    )
                except OSError:
                    process_name = "unknown"
                process_rows.append(
                    {
                        "pid": pid,
                        "gpu_uuid": str(gpu_by_pid.get(str(pid), "")),
                        "process_name": process_name,
                        "association": "hung_triage_target",
                    }
                )
        try:
            completed = None
            if not target_pids:
                completed = self.runner(
                    query,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if completed.returncode == 0:
                    for line in completed.stdout.splitlines():
                        fields = [item.strip() for item in line.split(",", 2)]
                        if len(fields) != 3 or not fields[0].isdigit():
                            continue
                        process_rows.append(
                            {
                                "pid": int(fields[0]),
                                "gpu_uuid": fields[1],
                                "process_name": fields[2],
                            }
                        )
            target_gpu_uuids = set(command.gpu_uuids)
            if target_gpu_uuids and not target_pids:
                process_rows = [
                    row for row in process_rows if row["gpu_uuid"] in target_gpu_uuids
                ]
            for row in process_rows:
                row.setdefault("association", "gpu_compute")
            expand_processes = bool(
                command.parameters.get(
                    "expand_python_cgroup_processes",
                    self.expand_python_cgroup_processes,
                )
            )
            if expand_processes and not target_pids:
                process_rows = self._expand_python_processes_in_cgroups(process_rows)
            process_rows = process_rows[:max_processes]
            process_output = "\n".join(
                (f"{row['pid']}, {row['gpu_uuid']}, {row['process_name']}")
                for row in process_rows
            )
            if process_output:
                process_output += "\n"
            if completed is not None and completed.stderr:
                process_output += "\n[stderr]\n" + completed.stderr
            (work_dir / "nvidia-compute-processes.csv").write_text(
                process_output,
                encoding="utf-8",
                errors="replace",
            )
            manifest["gpu_processes"] = process_rows
            manifest["target_gpu_uuids"] = sorted(target_gpu_uuids)
            manifest["captures"].append(
                {
                    "file": "nvidia-compute-processes.csv",
                    "command": query,
                    "returncode": (
                        completed.returncode if completed is not None else 0
                    ),
                }
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            manifest["captures"].append(
                {
                    "file": "nvidia-compute-processes.csv",
                    "command": query,
                    "returncode": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        return process_rows

    def _capture_system_context(self, work_dir, manifest):
        for filename, argv in (
            ("process-tree.txt", ["ps", "-eF"]),
            ("network-sockets.txt", ["ss", "-tpn"]),
        ):
            try:
                completed = self.runner(
                    argv,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                (work_dir / filename).write_text(
                    (completed.stdout or "")
                    + ("\n[stderr]\n" + completed.stderr if completed.stderr else ""),
                    encoding="utf-8",
                    errors="replace",
                )
                manifest["captures"].append(
                    {
                        "file": filename,
                        "command": argv,
                        "returncode": completed.returncode,
                    }
                )
            except (
                OSError,
                subprocess.TimeoutExpired,
            ) as exc:
                manifest["captures"].append(
                    {
                        "file": filename,
                        "command": argv,
                        "returncode": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    def _capture_python_processes(
        self,
        work_dir,
        manifest,
        process_rows,
        python_sample_count,
        python_sample_interval,
        python_timeout,
    ):
        python_process_rows = []
        for row in process_rows:
            proc_dir = self.proc_root / str(row["pid"])
            try:
                process_cmdline = (
                    (proc_dir / "cmdline")
                    .read_bytes()[:1_048_576]
                    .replace(b"\0", b" ")
                    .decode("utf-8", errors="replace")
                )
            except OSError:
                process_cmdline = ""
            if "python" in (f"{row['process_name']} {process_cmdline}".lower()):
                python_process_rows.append(row)
        manifest["python_stack_sampling"] = {
            "sample_count": python_sample_count,
            "interval_seconds": python_sample_interval,
            "command_timeout_seconds": python_timeout,
            "process_count": len(python_process_rows),
            "mode": "round_parallel",
        }
        if python_process_rows and self.python_stack_tool:
            with ThreadPoolExecutor(
                max_workers=len(python_process_rows),
                thread_name_prefix="gpu-fault-pyspy",
            ) as executor:
                for sample_index in range(1, python_sample_count + 1):
                    futures = [
                        executor.submit(
                            self._capture_python_stack_sample,
                            work_dir,
                            row["pid"],
                            sample_index,
                            python_sample_count,
                            python_timeout,
                        )
                        for row in python_process_rows
                    ]
                    manifest["captures"].extend(future.result() for future in futures)
                    if (
                        sample_index < python_sample_count
                        and python_sample_interval > 0
                    ):
                        self.sleep(python_sample_interval)

    def _capture_proc_samples(
        self,
        work_dir,
        manifest,
        process_rows,
        sample_count,
        duration,
        sample_interval,
    ):
        """Sample /proc and strace for every hung process, round by round.

        strace used to run the whole schedule for one process before starting
        the next, and slept the interval once per process on top: with the
        defaults on a 16-rank node that is 16 x (3 x 3 s + 2 x 2 s), about
        208 s of a bundle whose only purpose is to photograph the node *while*
        it is hung, against a step deadline that does not grow with the rank
        count. Rounds are the shape py-spy already used, so the total is now
        the schedule's own length (samples x duration plus the intervals) for
        any number of ranks -- strictly below the old serial worst case.
        """

        manifest["strace_sampling"] = {
            "sample_count": sample_count,
            "duration_seconds_per_sample": duration,
            "interval_seconds": sample_interval,
            "process_count": len(process_rows),
            "mode": "round_parallel",
        }
        for row in process_rows:
            self._capture_proc_metadata(work_dir, manifest, row["pid"])
        if not process_rows:
            return
        with ThreadPoolExecutor(
            max_workers=len(process_rows),
            thread_name_prefix="gpu-fault-strace",
        ) as executor:
            for sample_index in range(1, sample_count + 1):
                for row in process_rows:
                    self._capture_proc_stack_sample(
                        work_dir,
                        manifest,
                        row["pid"],
                        sample_index,
                        sample_count,
                    )
                futures = [
                    executor.submit(
                        self._capture_strace_sample,
                        work_dir,
                        row["pid"],
                        sample_index,
                        sample_count,
                        duration,
                        sample_interval,
                    )
                    for row in process_rows
                ]
                manifest["captures"].extend(future.result() for future in futures)
                if sample_index < sample_count and sample_interval > 0:
                    self.sleep(sample_interval)

    def _capture_proc_metadata(
        self, work_dir: Path, manifest: dict[str, Any], pid: int
    ) -> None:
        proc_dir = self.proc_root / str(pid)
        for name in ("status", "cmdline"):
            source = proc_dir / name
            target = work_dir / f"proc-{pid}-{name}.txt"
            try:
                data = source.read_bytes()[:1_048_576]
                target.write_bytes(data.replace(b"\0", b" "))
                manifest["captures"].append(
                    {
                        "file": target.name,
                        "source": str(source),
                        "returncode": 0,
                    }
                )
            except OSError as exc:
                manifest["captures"].append(
                    {
                        "file": target.name,
                        "source": str(source),
                        "returncode": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    def _capture_proc_stack_sample(
        self,
        work_dir: Path,
        manifest: dict[str, Any],
        pid: int,
        sample_index: int,
        sample_count: int,
    ) -> None:
        proc_dir = self.proc_root / str(pid)
        for name in ("stack", "wchan"):
            source = proc_dir / name
            target = work_dir / (f"proc-{pid}-{name}-sample-{sample_index:02d}.txt")
            try:
                data = source.read_bytes()[:1_048_576]
                target.write_bytes(data.replace(b"\0", b" "))
                manifest["captures"].append(
                    {
                        "file": target.name,
                        "source": str(source),
                        "returncode": 0,
                        "pid": pid,
                        "sample_index": sample_index,
                        "sample_count": sample_count,
                    }
                )
            except OSError as exc:
                error = f"{type(exc).__name__}: {exc}"
                target.write_text(error, encoding="utf-8")
                manifest["captures"].append(
                    {
                        "file": target.name,
                        "source": str(source),
                        "returncode": None,
                        "pid": pid,
                        "sample_index": sample_index,
                        "sample_count": sample_count,
                        "error": error,
                    }
                )

    def _capture_strace_sample(
        self,
        work_dir: Path,
        pid: int,
        sample_index: int,
        sample_count: int,
        duration: int,
        sample_interval: int,
    ) -> dict[str, Any]:
        trace_prefix = work_dir / (f"strace-{pid}-sample-{sample_index:02d}")
        argv = [
            "timeout",
            "--signal=INT",
            f"{duration}s",
            "strace",
            "-ff",
            "-tt",
            "-T",
            "-s",
            "256",
            "-p",
            str(pid),
            "-o",
            str(trace_prefix),
        ]
        started_at = self.now()
        capture: dict[str, Any] = {
            "file": (f"strace-{pid}-sample-{sample_index:02d}*"),
            "command": argv,
            "pid": pid,
            "sample_index": sample_index,
            "sample_count": sample_count,
            "duration_seconds": duration,
            "interval_seconds": sample_interval,
            "started_at": started_at.isoformat(),
        }
        try:
            completed = self.runner(
                argv,
                check=False,
                capture_output=True,
                text=True,
                timeout=duration + 5,
            )
            capture["returncode"] = completed.returncode
        except (
            OSError,
            subprocess.TimeoutExpired,
        ) as exc:
            capture.setdefault("returncode", None)
            capture["error"] = f"{type(exc).__name__}: {exc}"
        # A fixed-duration sample is *supposed* to end by being killed:
        # timeout(1) exits 124 and strace exits 130 on SIGINT. Reporting that
        # as a failed capture made every healthy bundle report one failure per
        # sample, so the summary's failed_capture_count was useless. Only count
        # it as a failure when no trace file was produced.
        #
        # Counted outside the guard because the files are on disk either way:
        # when the runner raises -- ``TimeoutExpired`` because the wall clock
        # beat our own timeout, which is exactly the deep hang worth tracing --
        # strace has usually been running for the whole sample and its output
        # is the only record of it. Globbing only on the happy path reported
        # ``trace_file_count: 0`` for those samples, so the bundle read as
        # "nothing captured" while the traces sat beside the manifest.
        trace_files = self._trace_files(work_dir, trace_prefix.name)
        capture["trace_file_count"] = len(trace_files)
        if capture["returncode"] in {124, 130} and trace_files:
            capture["terminated_by"] = "sample_duration"
            capture["raw_returncode"] = capture["returncode"]
            capture["returncode"] = 0
        elif capture["returncode"] not in {
            0,
            None,
        } and (not trace_files):
            capture["error"] = (
                f"strace produced no trace file (exit {capture['returncode']})"
            )
        capture["completed_at"] = self.now().isoformat()
        return capture

    @staticmethod
    def _trace_files(work_dir: Path, prefix: str) -> list[str]:
        """The trace files this sample wrote, whatever the runner reported.

        Guarded because it runs in one thread per rank: anything that escapes
        comes back out of ``future.result()`` and takes every other rank's
        sample with it.
        """

        try:
            return sorted(item.name for item in work_dir.glob(f"{prefix}*"))
        except OSError:
            return []

    def _capture_python_stack_sample(
        self,
        work_dir: Path,
        pid: int,
        sample_index: int,
        sample_count: int,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        target = work_dir / (f"python-stack-{pid}-sample-{sample_index:02d}.txt")
        argv = [
            self.python_stack_tool,
            "dump",
            "--nonblocking",
            "--pid",
            str(pid),
        ]
        capture = {
            "file": target.name,
            "command": argv,
            "pid": pid,
            "sample_index": sample_index,
            "sample_count": sample_count,
            "started_at": self.now().isoformat(),
            "tool": "py-spy",
        }
        try:
            completed = self.runner(
                argv,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            output = completed.stdout or ""
            if completed.stderr:
                output += "\n[stderr]\n" + completed.stderr
            target.write_text(
                output,
                encoding="utf-8",
                errors="replace",
            )
            capture["returncode"] = completed.returncode
        except (OSError, subprocess.TimeoutExpired) as exc:
            error = f"{type(exc).__name__}: {exc}"
            target.write_text(error, encoding="utf-8")
            capture["returncode"] = None
            capture["error"] = error
        capture["completed_at"] = self.now().isoformat()
        return capture

    def _expand_python_processes_in_cgroups(
        self, gpu_processes: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        cgroup_gpu_uuids: dict[str, set[str]] = {}
        cgroup_gpu_pids: dict[str, set[int]] = {}
        for row in gpu_processes:
            for cgroup in self._process_cgroups(row["pid"]):
                cgroup_gpu_uuids.setdefault(cgroup, set()).add(row["gpu_uuid"])
                cgroup_gpu_pids.setdefault(cgroup, set()).add(row["pid"])
        if not cgroup_gpu_uuids:
            return gpu_processes
        known_pids = {row["pid"] for row in gpu_processes}
        expanded = list(gpu_processes)
        try:
            candidates = list(self.proc_root.iterdir())
        except OSError:
            return expanded
        for proc_dir in candidates:
            if not proc_dir.name.isdigit():
                continue
            pid = int(proc_dir.name)
            if pid in known_pids:
                continue
            matched_cgroups = self._process_cgroups(pid) & cgroup_gpu_uuids.keys()
            if not matched_cgroups:
                continue
            try:
                cmdline = (
                    (proc_dir / "cmdline")
                    .read_bytes()[:1_048_576]
                    .replace(b"\0", b" ")
                    .decode("utf-8", errors="replace")
                    .strip()
                )
                comm = (
                    (proc_dir / "comm")
                    .read_text(encoding="utf-8", errors="replace")
                    .strip()
                )
            except OSError:
                continue
            if "python" not in f"{comm} {cmdline}".lower():
                continue
            gpu_uuids = sorted(
                {
                    gpu_uuid
                    for cgroup in matched_cgroups
                    for gpu_uuid in cgroup_gpu_uuids[cgroup]
                }
            )
            source_gpu_pids = sorted(
                {
                    source_pid
                    for cgroup in matched_cgroups
                    for source_pid in cgroup_gpu_pids[cgroup]
                }
            )
            expanded.append(
                {
                    "pid": pid,
                    "gpu_uuid": ",".join(gpu_uuids),
                    "process_name": comm or cmdline.split(" ", 1)[0],
                    "association": "training_container_cgroup",
                    "source_gpu_pids": source_gpu_pids,
                }
            )
            known_pids.add(pid)
        return expanded

    def _process_cgroups(self, pid: int) -> set[str]:
        try:
            lines = (
                (self.proc_root / str(pid) / "cgroup")
                .read_text(encoding="utf-8", errors="replace")
                .splitlines()
            )
        except OSError:
            return set()
        paths = set()
        for line in lines:
            fields = line.split(":", 2)
            if len(fields) == 3 and fields[2] not in {
                "",
                "/",
            }:
                paths.add(fields[2])
        return paths
