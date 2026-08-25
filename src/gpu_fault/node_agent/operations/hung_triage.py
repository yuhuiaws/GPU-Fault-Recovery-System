from __future__ import annotations

import hashlib
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable


from gpu_fault.node_agent.protocol import (
    NodeActionCommand,
)


class HungTriageOperationsMixin:
    # Attributes supplied by the composed concrete implementation.
    _await_flight_dumps: Callable[..., Any]
    _capture_python_stack_sample: Any
    _counter_delta: Callable[..., Any]
    _efa_counter_snapshot: Callable[..., Any]
    _finish_flight_recorder: Callable[..., Any]
    _process_environment: Callable[..., Any]
    _trigger_flight_recorder: Callable[..., Any]
    proc_root: Any
    python_stack_tool: Any
    runner: Callable[..., Any]
    sleep: Callable[..., Any]

    _PYSPY_HEADER_RE = re.compile(
        r"^(?:process\s+\d+\s*:|python\s+v\d)",
        re.IGNORECASE,
    )
    _COLLECTIVE_FRAME_RE = re.compile(
        r"ProcessGroupNCCL|distributed_c10d|torch/distributed|"
        r"all_reduce|all_gather|reduce_scatter|broadcast|"
        r"\ballreduce\b|\bbarrier\b|\bwait\b|nccl|"
        r"torch/cuda|\bsynchronize\b|cudaStreamSynchronize|"
        r"\bcurrent_stream\b",
        re.IGNORECASE,
    )

    def _collect_hung_triage(self, command: NodeActionCommand) -> dict[str, Any]:
        """Collect a bounded, read-only NCCL hang signal snapshot."""

        started = time.monotonic()
        timeout_seconds = min(
            10.0,
            max(
                2.0,
                float(command.parameters.get("triage_timeout_seconds", 10)),
            ),
        )
        errors: list[str] = []
        process_rows: list[dict[str, Any]] = []
        query = [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,process_name",
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = self.runner(
                query,
                check=False,
                capture_output=True,
                text=True,
                timeout=min(3.0, timeout_seconds),
            )
            if completed.returncode != 0:
                errors.append(
                    "nvidia-smi compute query failed: "
                    + (completed.stderr or "").strip()
                )
            else:
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
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(f"compute query: {type(exc).__name__}: {exc}")

        target_gpu_uuids = set(command.gpu_uuids)
        if target_gpu_uuids:
            process_rows = [
                row for row in process_rows if row["gpu_uuid"] in target_gpu_uuids
            ]
        rank_mapping = command.parameters.get("rank_by_pid_by_node", {})
        node_mapping = (
            rank_mapping.get(command.node_id, {})
            if isinstance(rank_mapping, dict)
            else {}
        )
        for row in process_rows:
            environment = self._process_environment(row["pid"])
            rank = node_mapping.get(str(row["pid"]))
            if rank is None:
                rank = environment.get("RANK")
            try:
                row["rank"] = int(rank) if rank is not None else None
            except (TypeError, ValueError):
                row["rank"] = None
            local_rank = environment.get("LOCAL_RANK")
            try:
                row["local_rank"] = int(local_rank) if local_rank is not None else None
            except ValueError:
                row["local_rank"] = None
            row["_environment"] = environment

        max_processes = min(
            32,
            max(
                1,
                int(command.parameters.get("triage_max_processes", 16)),
            ),
        )
        process_rows = process_rows[:max_processes]
        flight_requests = {
            row["pid"]: self._trigger_flight_recorder(row) for row in process_rows
        }
        proc_before = {
            row["pid"]: self._proc_triage_sample(row["pid"]) for row in process_rows
        }
        efa_before = self._efa_counter_snapshot()

        with tempfile.TemporaryDirectory(prefix="gpu-fault-hung-triage-") as temporary:
            output_dir = Path(temporary)
            stack_futures = {}
            if process_rows and self.python_stack_tool:
                stack_executor = ThreadPoolExecutor(
                    max_workers=len(process_rows),
                    thread_name_prefix="gpu-fault-triage-pyspy",
                )
                try:
                    stack_futures = {
                        row["pid"]: stack_executor.submit(
                            self._capture_python_stack_sample,
                            output_dir,
                            row["pid"],
                            1,
                            1,
                            min(
                                2,
                                max(
                                    1,
                                    int(timeout_seconds - 1),
                                ),
                            ),
                        )
                        for row in process_rows
                    }
                    if time.monotonic() - started < timeout_seconds:
                        self.sleep(1)
                    proc_after = {
                        row["pid"]: self._proc_triage_sample(row["pid"])
                        for row in process_rows
                    }
                    efa_after = self._efa_counter_snapshot()
                    ranks = []
                    gpu_stats = self._gpu_triage_snapshot()
                    for row in process_rows:
                        pid = row["pid"]
                        stack = {}
                        future = stack_futures.get(pid)
                        if future is not None:
                            try:
                                capture = future.result(
                                    timeout=max(
                                        0.1,
                                        timeout_seconds - (time.monotonic() - started),
                                    )
                                )
                                path = output_dir / str(capture.get("file") or "")
                                text = (
                                    path.read_text(
                                        encoding="utf-8",
                                        errors="replace",
                                    )
                                    if path.is_file()
                                    else ""
                                )
                                stack = self._stack_summary(text)
                                if capture.get("error"):
                                    stack["error"] = capture["error"]
                            except Exception as exc:
                                stack = {"error": (f"{type(exc).__name__}: {exc}")}
                        before = proc_before.get(pid, {})
                        after = proc_after.get(pid, {})
                        ranks.append(
                            {
                                "pid": pid,
                                "rank": row.get("rank"),
                                "local_rank": row.get("local_rank"),
                                "gpu_uuid": row["gpu_uuid"],
                                "process_name": row["process_name"],
                                "flight_recorder": (
                                    self._finish_flight_recorder(
                                        flight_requests.get(pid, {})
                                    )
                                ),
                                "python_stack": stack,
                                "proc": self._proc_triage_delta(before, after),
                                "gpu": gpu_stats.get(row["gpu_uuid"], {}),
                            }
                        )
                finally:
                    stack_executor.shutdown(wait=True, cancel_futures=True)
            else:
                if time.monotonic() - started < timeout_seconds:
                    self.sleep(1)
                proc_after = {
                    row["pid"]: self._proc_triage_sample(row["pid"])
                    for row in process_rows
                }
                efa_after = self._efa_counter_snapshot()
                gpu_stats = self._gpu_triage_snapshot()
                ranks = [
                    {
                        "pid": row["pid"],
                        "rank": row.get("rank"),
                        "local_rank": row.get("local_rank"),
                        "gpu_uuid": row["gpu_uuid"],
                        "process_name": row["process_name"],
                        "flight_recorder": (
                            self._finish_flight_recorder(
                                flight_requests.get(row["pid"], {})
                            )
                        ),
                        "python_stack": {},
                        "proc": self._proc_triage_delta(
                            proc_before.get(row["pid"], {}),
                            proc_after.get(row["pid"], {}),
                        ),
                        "gpu": gpu_stats.get(row["gpu_uuid"], {}),
                    }
                    for row in process_rows
                ]

        flight_wait_seconds = self._await_flight_dumps(
            flight_requests,
            ranks,
            started + timeout_seconds,
        )

        return {
            "triage_version": "nccl-hung-triage/v1",
            "node_id": command.node_id,
            "ranks": ranks,
            "efa": self._counter_delta(efa_before, efa_after),
            "errors": errors,
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "flight_recorder_wait_seconds": flight_wait_seconds,
            "read_only": True,
        }

    def _proc_triage_sample(self, pid: int) -> dict[str, Any]:
        proc_dir = self.proc_root / str(pid)
        result: dict[str, Any] = {}
        try:
            stat_text = (proc_dir / "stat").read_text()
            rest = stat_text[stat_text.rfind(")") + 2 :].split()
            result["state"] = rest[0]
            result["cpu_ticks"] = int(rest[11]) + int(rest[12])
        except (OSError, ValueError, IndexError):
            pass
        try:
            result["wchan"] = (proc_dir / "wchan").read_text().strip()
        except OSError:
            pass
        try:
            status = (proc_dir / "status").read_text()
            match = re.search(
                r"^voluntary_ctxt_switches:\s+(\d+)$",
                status,
                re.MULTILINE,
            )
            if match:
                result["voluntary_ctxt_switches"] = int(match.group(1))
        except OSError:
            pass
        states = {"R": 0, "S": 0, "D": 0}
        try:
            for task in (proc_dir / "task").iterdir():
                try:
                    text = (task / "stat").read_text()
                    state = text[text.rfind(")") + 2 :].split()[0]
                except (OSError, IndexError):
                    continue
                if state in states:
                    states[state] += 1
        except OSError:
            pass
        result["thread_states"] = states
        return result

    @staticmethod
    def _proc_triage_delta(
        before: dict[str, Any], after: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "cpu_ticks_delta": max(
                0,
                int(after.get("cpu_ticks", 0)) - int(before.get("cpu_ticks", 0)),
            ),
            "voluntary_ctxt_switches_delta": max(
                0,
                int(after.get("voluntary_ctxt_switches", 0))
                - int(before.get("voluntary_ctxt_switches", 0)),
            ),
            "wchan_before": before.get("wchan"),
            "wchan_after": after.get("wchan"),
            "wchan_unchanged": bool(
                before.get("wchan") and before.get("wchan") == after.get("wchan")
            ),
            "thread_states": after.get("thread_states", {}),
        }

    def _gpu_triage_snapshot(
        self,
    ) -> dict[str, dict[str, Any]]:
        argv = [
            "nvidia-smi",
            (
                "--query-gpu=uuid,utilization.gpu,"
                "utilization.memory,clocks_throttle_reasons.active"
            ),
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = self.runner(
                argv,
                check=False,
                capture_output=True,
                text=True,
                timeout=3,
            )
        except (OSError, subprocess.TimeoutExpired):
            return {}
        if completed.returncode != 0:
            return {}
        result = {}
        for line in completed.stdout.splitlines():
            fields = [item.strip() for item in line.split(",")]
            if len(fields) < 4:
                continue
            try:
                result[fields[0]] = {
                    "utilization_gpu_percent": float(fields[1]),
                    "utilization_memory_percent": float(fields[2]),
                    "clocks_throttle_reasons_active": fields[3],
                }
            except ValueError:
                continue
        return result

    @staticmethod
    def _stack_summary(text: str) -> dict[str, Any]:
        frames = []
        collective = False
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.lower().startswith("thread"):
                continue
            if HungTriageOperationsMixin._PYSPY_HEADER_RE.match(line):
                # ``Process <pid>: <argv>`` and ``Python v3.12.13
                # (/usr/bin/python3.12)`` are py-spy banners, not frames.
                # Keeping them made every rank's signature unique (the pid
                # differs), which silently disabled the classifier's
                # stack-mode comparison across ranks.
                continue
            match = re.match(
                r"^(?P<function>.+?)\s+\((?P<file>.+?)(?::\d+)?\)$",
                line,
            )
            if match:
                function = re.sub(
                    r"\([^)]*\)$",
                    "",
                    match.group("function"),
                ).strip()
                filename = re.sub(r":\d+$", "", match.group("file")).strip()
                normalized = f"{filename}:{function}"
            else:
                normalized = re.sub(r":\d+(?=\)?$)", "", line)
                normalized = re.sub(r"\([^)]*\)$", "", normalized).strip()
            if not normalized:
                continue
            frames.append(normalized)
            if HungTriageOperationsMixin._COLLECTIVE_FRAME_RE.search(normalized):
                collective = True
            if len(frames) >= 12:
                break
        joined = "|".join(frames)
        return {
            "signature": (
                hashlib.sha256(joined.encode()).hexdigest()[:24] if joined else None
            ),
            "collective_frames": collective,
            "frames": frames,
        }
