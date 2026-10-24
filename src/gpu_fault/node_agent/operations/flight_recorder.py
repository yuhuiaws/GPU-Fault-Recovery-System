from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
import stat
import time
from pathlib import Path
from typing import Any, Callable

from gpu_fault.flight_progress import apply_pg_status, completed, group_progress
from gpu_fault.node_agent.operations.collection_budget import bounded_read
from gpu_fault.node_agent.operations.flight_evidence import preserve_dump


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


def decode_flight_dump(raw: bytes) -> Any:
    if len(raw) > 64 * 1024 * 1024:
        raise ValueError("flight recorder dump exceeds 64 MiB")
    try:
        return json.loads(raw.decode("utf-8"))
    except RecursionError as exc:
        raise ValueError("flight recorder nesting exceeds parser limit") from exc
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass
    try:
        return FlightRecorderUnpickler(io.BytesIO(raw)).load()
    except Exception as exc:
        raise ValueError(
            f"unparsable flight recorder dump: {type(exc).__name__}: {exc}"
        ) from exc


class FlightRecorderOperationsMixin:
    # Attributes supplied by the composed concrete implementation.
    proc_root: Path
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
        therefore reported ``dump_missing``. Retry missing or partially written
        files within the budget; neither absence nor a snapshot confirms a cause.
        """

        pending = [
            row
            for row in ranks
            if (
                (row.get("flight_recorder") or {}).get("status") == "dump_missing"
                or (row.get("flight_recorder") or {}).get("parse_error")
            )
            and (requests.get(row["pid"]) or {}).get("status") == "triggered"
        ]
        waiting_since = time.monotonic()
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            self.sleep(min(interval, remaining))
            still_pending = []
            for row in pending:
                if time.monotonic() >= deadline:
                    return round(time.monotonic() - waiting_since, 6)
                finished = self._finish_flight_recorder(requests.get(row["pid"]) or {})
                row["flight_recorder"] = finished
                if finished.get("status") == "dump_missing" or finished.get(
                    "parse_error"
                ):
                    still_pending.append(row)
            pending = still_pending
        return round(time.monotonic() - waiting_since, 6)

    def _process_environment(self, pid: int) -> dict[str, str]:
        try:
            with (self.proc_root / str(pid) / "environ").open("rb") as stream:
                raw = stream.read(1_048_576)
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
        candidates = self._process_namespace_paths(pid, raw_candidates)
        pipe = next(
            (path for path in candidates if self._is_flight_pipe(path)),
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
        previous = {}
        for candidate in dump_candidates:
            candidate_stat = self._dump_file_stat(candidate)
            if candidate_stat is not None:
                previous[str(candidate)] = candidate_stat.st_mtime_ns
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
    def process_namespace_path_for_root(proc_root: Path, pid: int, path: Path) -> Path:
        # ``path`` is read from the training container's environment
        # (TORCH_NCCL_DEBUG_INFO_* variables), so it is attacker-influenced.
        # Map it into the container's mount namespace as the root-owned Agent
        # sees it (``/proc/<pid>/root`` for absolute paths, ``/cwd`` for
        # relative ones) and refuse anything that would escape that base via
        # ``..`` segments or an absolute anchor. Stripping ``parts[1:]`` alone
        # does not stop traversal, so a crafted value could otherwise steer a
        # privileged read or FIFO write outside the base (security review M-8).
        is_absolute = path.is_absolute()
        base = proc_root / str(pid) / ("root" if is_absolute else "cwd")
        parts = path.parts[1:] if is_absolute else path.parts
        if any(part == ".." for part in parts):
            raise ValueError(f"refusing path traversal in namespace path: {path}")
        candidate = base.joinpath(*parts)
        if not candidate.resolve().is_relative_to(base.resolve()):
            raise ValueError(f"namespace path escapes container base: {path}")
        return candidate

    def _process_namespace_path(self, pid: int, path: Path) -> Path:
        return self.process_namespace_path_for_root(self.proc_root, pid, path)

    def _process_namespace_paths(
        self, pid: int, raw_candidates: list[Path]
    ) -> list[Path]:
        """Map candidates into the namespace, dropping any that escape it."""
        mapped = []
        for path in raw_candidates:
            try:
                mapped.append(self._process_namespace_path(pid, path))
            except ValueError:
                # A traversal attempt is contained by simply never producing
                # a usable candidate for it; legitimate candidates still map.
                continue
        return mapped

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
        candidates = self._process_namespace_paths(pid, raw_candidates)
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
                    FlightRecorderOperationsMixin.process_namespace_path_for_root(
                        self.proc_root,
                        pid,
                        Path(f"/tmp/nccl_trace_rank_{rank}"),
                    ),
                    FlightRecorderOperationsMixin.process_namespace_path_for_root(
                        self.proc_root,
                        pid,
                        Path(f"/tmp/nccl_trace_rank_{rank}.json"),
                    ),
                ]
            )
        return list(dict.fromkeys(candidates))

    @staticmethod
    def _is_flight_pipe(path: Path) -> bool:
        """Whether the candidate is a FIFO, without raising if it cannot say."""

        try:
            return stat.S_ISFIFO(path.stat().st_mode)
        except OSError:
            return False

    @staticmethod
    def _dump_file_stat(path: Path) -> os.stat_result | None:
        """The candidate's ``stat``, or None when it is not a readable file.

        Every candidate path here is built from the training container's own
        environment and lives in the container's own directories, so a
        ``stat`` can fail for reasons that are nobody's fault and nobody's
        emergency: the directory mode changed, the file was rotated away, the
        env var held a component longer than NAME_MAX. ``Path.is_file`` only
        swallows the ENOENT class, so those raised straight out of the triage
        step and cost every rank's evidence -- see the tests in
        ``test_diagnostics.py``. One unreadable candidate is a candidate that
        did not answer, nothing more.
        """

        try:
            info = path.stat()
        except OSError:
            return None
        return info if stat.S_ISREG(info.st_mode) else None

    def _finish_flight_recorder(self, request: dict[str, Any]) -> dict[str, Any]:
        if request.get("status") != "triggered":
            return request
        previous = request.get("previous_mtimes") or {}
        files = [Path(value) for value in request.get("dump_candidates") or []]
        # One stat per candidate, kept: the winner used to be stat-ed twice
        # more outside the parse_error guard, so a dump the container rotated
        # in that window raised FileNotFoundError out of the whole command.
        updated: list[tuple[Path, os.stat_result]] = []
        for path in files:
            info = self._dump_file_stat(path)
            if info is not None and info.st_mtime_ns > int(previous.get(str(path), -1)):
                updated.append((path, info))
        if not updated:
            return {
                "status": "dump_missing",
                "pipe": request.get("pipe"),
            }
        dump_path, dump_stat = max(
            updated,
            key=lambda item: item[1].st_mtime_ns,
        )
        summary = {
            "status": "dumped",
            "pipe": request.get("pipe"),
            "dump_file": str(dump_path),
            "dump_size_bytes": dump_stat.st_size,
        }
        try:
            summary.update(
                bounded_read(
                    lambda: self._flight_recorder_summary(
                        self._load_flight_dump(dump_path)
                    ),
                    2,
                )
            )
            after = self._dump_file_stat(dump_path)
            if after is None or (after.st_ino, after.st_size, after.st_mtime_ns) != (
                dump_stat.st_ino,
                dump_stat.st_size,
                dump_stat.st_mtime_ns,
            ):
                summary["parse_error"] = "dump changed while being read"
            elif request.get("evidence_path"):
                summary["raw_evidence"] = bounded_read(
                    lambda: preserve_dump(
                        dump_path, Path(request["evidence_path"]), dump_stat
                    )
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
        evidence was thrown away.
        """

        limit = 64 * 1024 * 1024
        with path.open("rb") as stream:
            raw = stream.read(limit + 1)
        return decode_flight_dump(raw)

    @classmethod
    def _flight_recorder_summary(cls, payload: Any) -> dict[str, Any]:
        entries = cls._flight_entries(payload)
        summaries = [cls._flight_entry_summary(item) for item in entries]
        summaries = [item for item in summaries if item]
        if not summaries:
            return {"entry_count": 0}
        config = payload.get("pg_config", {}) if isinstance(payload, dict) else {}
        by_pg: dict[str, dict[str, Any]] = {}
        entries_by_pg: dict[str, list[dict[str, Any]]] = {}
        for item in summaries:
            name = item["pg_name"]
            if not name or name == ":":
                continue
            entries_by_pg.setdefault(name, []).append(item)
            current = by_pg.get(name)
            if current is None or (
                item["collective_seq_id"],
                str(item.get("time_created") or ""),
            ) > (current["collective_seq_id"], str(current.get("time_created") or "")):
                by_pg[name] = item
        for name, item in by_pg.items():
            semantic_entries = [
                dict(entry)
                for entry in sorted(
                    entries_by_pg[name],
                    key=lambda entry: (completed(entry), entry["collective_seq_id"]),
                )[:64]
            ]
            item["progress"] = apply_pg_status(
                group_progress(entries_by_pg[name]), payload, item.get("pg_id")
            )
            item["semantic_entries"] = semantic_entries
            group = (
                config.get(item.get("pg_config_key", name), {})
                if isinstance(config, dict)
                else {}
            )
            members = group.get("ranks") if isinstance(group, dict) else None
            if isinstance(members, str):
                try:
                    members = json.loads(members)
                except ValueError:
                    members = None
            if (
                isinstance(members, list)
                and members
                and all(type(rank) is int and rank >= 0 for rank in members)
            ):
                ordered = sorted(set(members))
                item["pg_member_count"] = len(ordered)
                item["pg_membership_sha256"] = hashlib.sha256(
                    json.dumps(ordered, separators=(",", ":")).encode()
                ).hexdigest()
                if len(ordered) <= 256:
                    item["pg_members"] = ordered
                    item["pg_rank_order"] = members
                else:
                    item["pg_members_compacted"] = True
        unfinished = [item for item in summaries if not completed(item)]
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
            "last_entries_by_pg": [by_pg[name] for name in sorted(by_pg)[:64]],
            "process_groups_truncated": len(by_pg) > 64,
            "unfinished_entry_count": len(unfinished),
            "unfinished_entries": selected[1:65],
            "unfinished_entries_truncated": len(selected) > 65,
        }

    @classmethod
    def _flight_entries(cls, value: Any) -> list[dict[str, Any]]:
        pending = [value]
        visited: set[int] = set()
        result = []
        while pending:
            item = pending.pop()
            if not isinstance(item, (dict, list, tuple)) or id(item) in visited:
                continue
            visited.add(id(item))
            if len(visited) > 100_000:
                raise ValueError("flight recorder structure exceeds traversal limit")
            if isinstance(item, dict) and (
                "collective_seq_id" in item or "collective_seq_id_" in item
            ):
                result.append(item)
            else:
                pending.extend(item.values() if isinstance(item, dict) else item)
                if len(pending) > 100_000:
                    raise ValueError(
                        "flight recorder structure exceeds traversal limit"
                    )
        return result

    @staticmethod
    def _flight_entry_summary(
        entry: dict[str, Any],
    ) -> dict[str, Any]:
        def value(*names: str) -> Any:
            for name in names:
                if name in entry:
                    return entry[name]
            return None

        seq = value("collective_seq_id", "collective_seq_id_")
        if seq is None or isinstance(seq, bool):
            return {}
        try:
            seq = int(seq)
        except (TypeError, ValueError):
            return {}
        group_name = value("group_name", "group_name_")
        group_desc = value("group_desc", "group_desc_")
        pg_name = value("pg_name", "pg_name_")
        if isinstance(pg_name, (list, tuple)):
            group_name = str(pg_name[0]) if pg_name else None
            pg_name = ":".join(str(part) for part in pg_name[:2])
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
            "pg_id": value("pg_id", "pg_id_"),
            "pg_config_key": str(group_name if group_name is not None else pg_name),
            "collective_seq_id": seq,
            "record_id": value("record_id", "record_id_"),
            "is_p2p": value("is_p2p", "is_p2p_"),
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
