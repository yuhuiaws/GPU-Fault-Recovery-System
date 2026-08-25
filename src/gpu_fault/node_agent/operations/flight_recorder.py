from __future__ import annotations

import io
import json
import os
import pickle
import stat
import time
from pathlib import Path
from typing import Any, Callable


class FlightRecorderUnpickler(pickle.Unpickler):
    """Unpickler that refuses to import anything.

    PyTorch writes the NCCL flight recorder dump as ``pickle.dumps`` of a
    plain dict of builtins, so no global lookup is ever legitimate here.
    Refusing them keeps a compromised training container from getting
    code execution inside the root-owned Agent by planting a crafted
    dump at ``TORCH_NCCL_DEBUG_INFO_TEMP_FILE``.
    """

    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(
            f"refusing to resolve {module}.{name} from a flight recorder dump"
        )


class FlightRecorderOperationsMixin:
    # Attributes supplied by the composed concrete implementation.
    proc_root: Any
    sleep: Callable[..., Any]

    def _await_flight_dumps(
        self,
        requests: dict[int, dict[str, Any]],
        ranks: list[dict[str, Any]],
        deadline: float,
        *,
        interval: float = 0.25,
    ) -> float:
        """Re-check flight recorder dumps until the triage budget ends.

        PyTorch writes the dump from its own monitor thread once that
        thread notices the pipe write, which on a real 24-rank p5en hang
        took 3.4s. The single check done while assembling the ranks
        therefore always reported ``dump_missing`` and the classifier
        lost the only signal that can yield a CONFIRMED verdict, so keep
        polling the candidates with whatever budget is left.
        """

        pending = [
            row
            for row in ranks
            if (row.get("flight_recorder") or {}).get("status") == "dump_missing"
            and (requests.get(row.get("pid")) or {}).get("status") == "triggered"
        ]
        waiting_since = time.monotonic()
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            self.sleep(min(interval, remaining))
            still_pending = []
            for row in pending:
                finished = self._finish_flight_recorder(requests.get(row["pid"]) or {})
                if finished.get("status") == "dump_missing":
                    still_pending.append(row)
                else:
                    row["flight_recorder"] = finished
            pending = still_pending
        return round(time.monotonic() - waiting_since, 6)

    def _process_environment(self, pid: int) -> dict[str, str]:
        try:
            raw = (self.proc_root / str(pid) / "environ").read_bytes()[:1_048_576]
        except OSError:
            return {}
        values = {}
        for entry in raw.split(b"\0"):
            if b"=" not in entry:
                continue
            key, value = entry.split(b"=", 1)
            values[key.decode(errors="replace")] = value.decode(errors="replace")
        return values

    def _trigger_flight_recorder(self, process: dict[str, Any]) -> dict[str, Any]:
        environment = process.get("_environment") or {}
        base = environment.get("TORCH_NCCL_DEBUG_INFO_PIPE_FILE")
        rank = process.get("rank")
        pid = process["pid"]
        result: dict[str, Any] = {"status": "not_configured"}
        if not base:
            return result
        raw_candidates = []
        if "{rank}" in base and rank is not None:
            raw_candidates.append(Path(base.format(rank=rank)))
        if rank is not None:
            raw_candidates.append(Path(f"{base}{rank}.pipe"))
            raw_candidates.append(Path(f"{base}{rank}"))
        raw_candidates.append(Path(base))
        candidates = [
            self._process_namespace_path(pid, path) for path in raw_candidates
        ]
        pipe = next(
            (
                path
                for path in candidates
                if path.exists() and stat.S_ISFIFO(path.stat().st_mode)
            ),
            None,
        )
        if pipe is None:
            return {
                "status": "pipe_missing",
                "pipe_candidates": [str(path) for path in candidates],
            }
        dump_candidates = self._flight_dump_candidates(
            environment, rank, pid=pid, pipe=pipe
        )
        previous = {
            str(path): path.stat().st_mtime_ns
            for path in dump_candidates
            if path.exists() and path.is_file()
        }
        try:
            descriptor = os.open(pipe, os.O_WRONLY | os.O_NONBLOCK)
            try:
                os.write(descriptor, b"1\n")
            finally:
                os.close(descriptor)
        except OSError as exc:
            return {
                "status": "pipe_write_failed",
                "pipe": str(pipe),
                "error": f"{type(exc).__name__}: {exc}",
            }
        return {
            "status": "triggered",
            "pipe": str(pipe),
            "dump_candidates": [str(path) for path in dump_candidates],
            "previous_mtimes": previous,
        }

    @staticmethod
    def _process_namespace_path_for_root(proc_root: Path, pid: int, path: Path) -> Path:
        relative = Path(*path.parts[1:]) if path.is_absolute() else path
        return (
            proc_root / str(pid) / ("root" if path.is_absolute() else "cwd") / relative
        )

    def _process_namespace_path(self, pid: int, path: Path) -> Path:
        return self._process_namespace_path_for_root(self.proc_root, pid, path)

    def _flight_dump_candidates(
        self,
        environment: dict[str, str],
        rank: int | None,
        *,
        pid: int,
        pipe: Path | None = None,
    ) -> list[Path]:
        base = environment.get("TORCH_NCCL_DEBUG_INFO_TEMP_FILE")
        raw_candidates = [Path(base)] if base else []
        if base and rank is not None:
            raw_candidates.extend(
                [
                    Path(f"{base}{rank}"),
                    Path(f"{base}{rank}.json"),
                    Path(f"{base}.{rank}"),
                    Path(f"{base}.{rank}.json"),
                ]
            )
        if base:
            raw_candidates.append(Path(f"{base}.json"))
        candidates = [
            self._process_namespace_path(pid, path) for path in raw_candidates
        ]
        if pipe is not None:
            candidates.extend(
                [
                    pipe.with_suffix(""),
                    pipe.with_suffix(".json"),
                    Path(f"{pipe}.json"),
                ]
            )
        if rank is not None:
            candidates.extend(
                [
                    FlightRecorderOperationsMixin._process_namespace_path_for_root(
                        self.proc_root,
                        pid,
                        Path(f"/tmp/nccl_trace_rank_{rank}"),
                    ),
                    FlightRecorderOperationsMixin._process_namespace_path_for_root(
                        self.proc_root,
                        pid,
                        Path(f"/tmp/nccl_trace_rank_{rank}.json"),
                    ),
                ]
            )
        return list(dict.fromkeys(candidates))

    def _finish_flight_recorder(self, request: dict[str, Any]) -> dict[str, Any]:
        if request.get("status") != "triggered":
            return request
        previous = request.get("previous_mtimes") or {}
        files = [Path(value) for value in request.get("dump_candidates") or []]
        updated = [
            path
            for path in files
            if path.is_file()
            and path.stat().st_mtime_ns > int(previous.get(str(path), -1))
        ]
        if not updated:
            return {
                "status": "dump_missing",
                "pipe": request.get("pipe"),
            }
        dump_path = max(
            updated,
            key=lambda path: path.stat().st_mtime_ns,
        )
        summary = {
            "status": "dumped",
            "pipe": request.get("pipe"),
            "dump_file": str(dump_path),
            "dump_size_bytes": dump_path.stat().st_size,
        }
        try:
            summary.update(
                self._flight_recorder_summary(self._load_flight_dump(dump_path))
            )
        except (OSError, ValueError) as exc:
            summary["parse_error"] = f"{type(exc).__name__}: {exc}"
        return summary

    @staticmethod
    def _load_flight_dump(path: Path) -> Any:
        """Read a flight recorder dump in either on-disk format.

        ``torch._C._distributed_c10d._dump_nccl_trace()`` returns a
        pickled dict, and that is what PyTorch writes to
        ``TORCH_NCCL_DEBUG_INFO_TEMP_FILE`` - a real p5en dump starts
        with ``\\x80\\x02}``, not ``{``. Parsing JSON only meant every
        production dump ended as ``parse_error`` and the flight-recorder
        evidence (the only path to a CONFIRMED verdict) was thrown away.
        """

        raw = path.read_bytes()[: 64 * 1024 * 1024]
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
        try:
            return FlightRecorderUnpickler(io.BytesIO(raw)).load()
        except Exception as exc:  # unpickling raises many types
            raise ValueError(
                f"unparsable flight recorder dump: {type(exc).__name__}: {exc}"
            ) from exc

    @classmethod
    def _flight_recorder_summary(cls, payload: Any) -> dict[str, Any]:
        entries = cls._flight_entries(payload)
        summaries = [cls._flight_entry_summary(item) for item in entries]
        summaries = [item for item in summaries if item]
        if not summaries:
            return {"entry_count": 0}
        unfinished = [
            item for item in summaries if not item.get("time_discovered_completed")
        ]
        last = max(
            summaries,
            key=lambda item: (
                item.get("collective_seq_id", -1),
                str(item.get("time_created") or ""),
            ),
        )
        selected = [last]
        selected.extend(item for item in unfinished if item is not last)
        return {
            "entry_count": len(summaries),
            "last_entry": last,
            "unfinished_entry_count": len(unfinished),
            "unfinished_entries": selected[1:65],
            "unfinished_entries_truncated": len(selected) > 65,
        }

    @classmethod
    def _flight_entries(cls, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            direct = [item for item in value if isinstance(item, dict)]
            if direct and any(
                "collective_seq_id" in item or "collective_seq_id_" in item
                for item in direct
            ):
                return direct
            result = []
            for item in value:
                result.extend(cls._flight_entries(item))
            return result
        if isinstance(value, dict):
            for key in (
                "entries",
                "trace",
                "records",
                "flight_recorder",
            ):
                if key in value:
                    result = cls._flight_entries(value[key])
                    if result:
                        return result
            result = []
            for item in value.values():
                result.extend(cls._flight_entries(item))
            return result
        return []

    @staticmethod
    def _flight_entry_summary(
        entry: dict[str, Any],
    ) -> dict[str, Any]:
        def value(*names):
            for name in names:
                if name in entry:
                    return entry[name]
            return None

        seq = value("collective_seq_id", "collective_seq_id_")
        if seq is None:
            return {}
        try:
            seq = int(seq)
        except (TypeError, ValueError):
            return {}
        group_name = value("group_name", "group_name_")
        group_desc = value("group_desc", "group_desc_")
        pg_name = value("pg_name", "pg_name_")
        if pg_name is None:
            # torch 2.10 records the group as
            # ``process_group: (uid, desc)`` and has neither pg_name nor
            # group_name, so without this every process group collapsed
            # into one ``":"`` bucket and sequence ids from different
            # collectives were compared against each other.
            group = value("process_group", "process_group_")
            if isinstance(group, (list, tuple)) and group:
                parts = [str(part) for part in group[:2]]
                group_name = group_name or parts[0]
                if len(parts) > 1:
                    group_desc = group_desc or parts[1]
                pg_name = ":".join(parts)
        if pg_name is None:
            pg_name = f"{group_name or ''}:{group_desc or ''}"
        return {
            "pg_name": str(pg_name),
            "collective_seq_id": seq,
            "state": value("state", "state_"),
            "p2p_seq_id": value("p2p_seq_id", "p2p_seq_id_"),
            "op_id": value("op_id", "op_id_"),
            "profiling_name": value("profiling_name", "profiling_name_"),
            "retired": value("retired", "retired_"),
            "time_discovered_started": value(
                "time_discovered_started",
                "time_discovered_started_",
                "time_discovered_started_ns",
            ),
            "time_discovered_completed": value(
                "time_discovered_completed",
                "time_discovered_completed_",
                "time_discovered_completed_ns",
            ),
            "input_dims": value("input_dims", "input_dims_", "input_sizes"),
            "output_dims": value(
                "output_dims",
                "output_dims_",
                "output_sizes",
            ),
            "sizes": value("sizes", "sizes_"),
            "input_dtypes": value("input_dtypes", "input_dtypes_"),
            "timeout_ms": value("timeout_ms", "timeout_ms_"),
            "time_created": value(
                "time_created",
                "time_created_",
                "time_created_ns",
            ),
        }
