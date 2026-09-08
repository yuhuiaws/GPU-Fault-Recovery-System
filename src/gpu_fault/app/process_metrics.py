"""Pod-coherent ``/metrics`` for a multi-process uvicorn Pod.

The ingress and control-worker containers run ``uvicorn --workers 4`` behind
one port. Every worker process builds its own ``ApplicationContext``, so every
in-memory counter, gauge and timestamp rendered on ``/metrics`` is
process-local, and a scrape is answered by whichever process accepted that
connection. Observed live (GF-REGIONAL-DESTR-018): the lease holder failed a
workflow for its lifetime and incremented ``lifetime_exceeded_total``; sixteen
scrapes of the same Pod returned seven distinct process fingerprints and the
counter read 0 on all of them. An alert written against such a series fires
by luck.

Every process publishes its *complete* rendered sample set to
``<dir>/<pid>.json`` -- on every render it answers and, from a small
background thread, every :data:`PUBLISH_INTERVAL_SECONDS` -- so the process
that counted does not have to be the one that answers the scrape. The
answering process merges its own fresh render with every other live process's
file and aggregates each family by the strategy registered for it in
:mod:`gpu_fault.app.metric_aggregation` (SUM, MAX, MIN, ANY or PER_PROCESS).
A file whose PID is gone is removed on sight, so a restarted process's old
counts leave the sum with it (Prometheus reads that as a counter reset, which
it is). The directory defaults to a per-Pod path under ``/dev/shm`` when
``POD_UID`` is set (the Deployment injects it) and is otherwise off, so a
single-process run and the unit tests aggregate a single source -- byte for
byte the plain render, plus the two merger gauges below.
``GPU_FAULT_PROCESS_METRICS_DIR`` overrides the path or disables the
mechanism with ``off``.

Two gauges belong to the merger itself and are never aggregated:
``gpu_fault_metrics_aggregation_processes`` (live processes merged, this one
included) and ``gpu_fault_metrics_aggregation_degraded`` (1 when the shared
directory was configured but unusable, so the sample is one process's view).

PER_PROCESS families carry a ``process="<slot>"`` label. The slot is a stable
0..N-1 number claimed with an ``flock`` on ``<dir>/slot-<n>.lock`` for the
process's lifetime; the kernel releases it when the process dies, so a
restarted worker takes the freed slot rather than a fresh number.
"""

from __future__ import annotations

import fcntl
import json
import logging
import math
import os
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock
from typing import Any

from gpu_fault.app.metric_aggregation import (
    DEGRADED_METRIC,
    PROCESS_LABEL,
    PROCESSES_METRIC,
    Strategy,
    strategy_for,
)

LOGGER = logging.getLogger(__name__)

ENVIRONMENT = "GPU_FAULT_PROCESS_METRICS_DIR"
DISABLED_TOKENS = frozenset({"off", "0", "false", "none", "disabled"})
DEFAULT_ROOT = Path("/dev/shm/gpu-fault-process-metrics")
PUBLISH_INTERVAL_SECONDS = 5.0
MAX_SLOTS = 16
FILE_FORMAT = 1

# Sample-name suffixes of the summary/histogram families. ``_max`` is a
# high-water mark and takes the maximum under SUM; the rest are additive.
SUMMARY_SUFFIXES = ("_bucket", "_count", "_sum", "_max")
MAXIMUM_SUFFIXES = ("_max",)

Labels = tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Sample:
    name: str
    labels: Labels
    value: str

    def line(self) -> str:
        if not self.labels:
            return f"{self.name} {self.value}"
        rendered = ",".join(f'{key}="{value}"' for key, value in self.labels)
        return f"{self.name}{{{rendered}}} {self.value}"


@dataclass
class Rendered:
    """One process's render, split into families and samples."""

    # family name -> ("# HELP ..." line or None, "# TYPE ..." line or None)
    families: dict[str, tuple[str | None, str | None]] = field(default_factory=dict)
    # family name -> samples in render order
    samples: dict[str, list[Sample]] = field(default_factory=dict)
    slot: int | None = None
    pid: int | None = None

    def family_type(self, family: str) -> str | None:
        type_line = self.families.get(family, (None, None))[1]
        if type_line is None:
            return None
        return type_line.split()[-1]


# --------------------------------------------------------------------------
# Exposition text <-> samples
# --------------------------------------------------------------------------


def _parse_labels(text: str) -> Labels:
    labels: list[tuple[str, str]] = []
    index = 0
    length = len(text)
    while index < length:
        while index < length and text[index] in ", ":
            index += 1
        if index >= length:
            break
        equals = text.index("=", index)
        key = text[index:equals].strip()
        index = equals + 1
        if index >= length or text[index] != '"':
            raise ValueError(f"unquoted label value in {text!r}")
        index += 1
        value_chars: list[str] = []
        while index < length:
            char = text[index]
            if char == "\\" and index + 1 < length:
                value_chars.append(text[index : index + 2])
                index += 2
                continue
            if char == '"':
                break
            value_chars.append(char)
            index += 1
        index += 1  # closing quote
        labels.append((key, "".join(value_chars)))
    return tuple(labels)


def _split_sample(line: str) -> tuple[str, Labels, str]:
    brace = line.find("{")
    if brace == -1:
        name, _, value = line.partition(" ")
        return name, (), value.strip()
    close = line.rfind("}")
    name = line[:brace]
    labels = _parse_labels(line[brace + 1 : close])
    return name, labels, line[close + 1 :].strip()


def _family_of(sample_name: str, families: Mapping[str, Any]) -> str:
    if sample_name in families:
        return sample_name
    for suffix in SUMMARY_SUFFIXES:
        if sample_name.endswith(suffix):
            candidate = sample_name[: -len(suffix)]
            if candidate in families or strategy_for(candidate) is not None:
                return candidate
    return sample_name


def parse_lines(lines: Iterable[str]) -> Rendered:
    """Split exposition text into families and samples, keeping render order.

    ``# HELP``/``# TYPE`` lines open a family; other comments and blank lines
    are dropped (Prometheus ignores them). A sample whose name is not a
    declared family is attached to the family its summary/histogram suffix
    points at, or opens an undeclared family of its own.
    """

    rendered = Rendered()
    for raw in lines:
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        if line.startswith("#"):
            parts = line.split(None, 3)
            if len(parts) >= 3 and parts[1] in ("HELP", "TYPE"):
                family = parts[2]
                help_line, type_line = rendered.families.get(family, (None, None))
                if parts[1] == "HELP":
                    help_line = help_line or line
                else:
                    type_line = type_line or line
                rendered.families[family] = (help_line, type_line)
                rendered.samples.setdefault(family, [])
            continue
        name, labels, value = _split_sample(line)
        family = _family_of(name, rendered.families)
        rendered.families.setdefault(family, (None, None))
        rendered.samples.setdefault(family, []).append(Sample(name, labels, value))
    return rendered


def _emit(
    rendered_families: Sequence[
        tuple[str, tuple[str | None, str | None], list[Sample]]
    ],
) -> list[str]:
    lines: list[str] = []
    for _family, (help_line, type_line), samples in rendered_families:
        if help_line:
            lines.append(help_line)
        if type_line:
            lines.append(type_line)
        lines.extend(sample.line() for sample in samples)
    return lines


# --------------------------------------------------------------------------
# Numeric helpers
# --------------------------------------------------------------------------


def _number(text: str) -> float:
    lowered = text.strip().lower()
    if lowered in ("+inf", "inf"):
        return math.inf
    if lowered == "-inf":
        return -math.inf
    if lowered == "nan":
        return math.nan
    return float(text)


def _decimals(source: str) -> int | None:
    lowered = source.lower()
    if "e" in lowered or "inf" in lowered or "nan" in lowered:
        return None
    _, dot, fraction = source.partition(".")
    return len(fraction) if dot else 0


def _format(value: float, sources: Sequence[str]) -> str:
    """Render a combined value in the decimal shape of its inputs, so a sum of
    ``0.250000`` and ``0.500000`` reads ``0.750000`` and a sum of integers stays
    an integer."""

    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    decimals = [_decimals(source) for source in sources]
    if any(places is None for places in decimals):
        return repr(value)
    places = max(decimals)  # type: ignore[type-var]
    if places == 0:
        return str(int(value))
    return f"{value:.{places}f}"


def _combine(values: Sequence[str], how: str) -> str:
    numbers = [_number(value) for value in values]
    if how == "sum":
        return _format(sum(numbers), values)
    if how == "max":
        return values[numbers.index(max(numbers))]
    return values[numbers.index(min(numbers))]


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def _fallback_strategy(family: str, family_type: str | None) -> Strategy:
    """For a family nobody registered (a plugin contributor, or drift the
    registry test did not run against): counters and additive families add
    up, timestamps and everything else take the maximum."""

    if family_type in ("counter", "summary", "histogram"):
        return Strategy.SUM
    if family.endswith(("_total", "_bucket", "_count", "_sum")):
        return Strategy.SUM
    return Strategy.MAX


_UNREGISTERED_REPORTED: set[str] = set()


def _resolve_strategy(family: str, family_type: str | None) -> Strategy:
    strategy = strategy_for(family)
    if strategy is not None:
        return strategy
    if family not in _UNREGISTERED_REPORTED:
        _UNREGISTERED_REPORTED.add(family)
        LOGGER.warning(
            "metric family %s has no aggregation strategy; using %s",
            family,
            _fallback_strategy(family, family_type).value,
        )
    return _fallback_strategy(family, family_type)


def _grouped(
    sources: Sequence[Rendered], family: str
) -> dict[tuple[str, Labels], list[str]]:
    groups: dict[tuple[str, Labels], list[str]] = {}
    for source in sources:
        for sample in source.samples.get(family, ()):
            groups.setdefault((sample.name, sample.labels), []).append(sample.value)
    return groups


def _aggregate_family(
    family: str,
    strategy: Strategy,
    sources: Sequence[Rendered],
) -> list[Sample]:
    """``sources`` are ordered: the local render first, then the other live
    processes by slot. Sample order follows the first source that has each
    series, so a single-source render is emitted unchanged."""

    if strategy is Strategy.ANY:
        for source in sources:
            if family in source.samples:
                return list(source.samples[family])
        return []
    if strategy is Strategy.PER_PROCESS:
        out: list[Sample] = []
        for source in sources:
            slot = "0" if source.slot is None else str(source.slot)
            for sample in source.samples.get(family, ()):
                labels = tuple(
                    (key, value) for key, value in sample.labels if key != PROCESS_LABEL
                ) + ((PROCESS_LABEL, slot),)
                out.append(Sample(sample.name, labels, sample.value))
        return out
    groups = _grouped(sources, family)
    out = []
    for (name, labels), values in groups.items():
        if len(values) == 1:
            out.append(Sample(name, labels, values[0]))
            continue
        if strategy is Strategy.SUM:
            high_water = name.endswith(MAXIMUM_SUFFIXES) and name != family
            quantile = any(key == "quantile" for key, _ in labels)
            how = "max" if (high_water or quantile) else "sum"
        elif strategy is Strategy.MAX:
            how = "max"
        else:
            how = "min"
        try:
            out.append(Sample(name, labels, _combine(values, how)))
        except ValueError:
            # A non-numeric value somewhere: keep the local process's reading.
            out.append(Sample(name, labels, values[0]))
    return out


def aggregate(local: Rendered, others: Sequence[Rendered]) -> list[str]:
    """Merge ``local`` with the other processes' renders into exposition text.

    Families keep the local render's order; families only another process has
    (a labelled reason this process never saw, a stamp only the lease holder
    moves) follow in slot order with that process's HELP/TYPE lines.
    """

    sources: list[Rendered] = [
        local,
        *sorted(others, key=lambda r: (r.slot is None, r.slot or 0, r.pid or 0)),
    ]
    order: list[str] = list(local.families)
    seen = set(order)
    for source in sources[1:]:
        for family in source.families:
            if family not in seen:
                seen.add(family)
                order.append(family)
    emitted: list[tuple[str, tuple[str | None, str | None], list[Sample]]] = []
    for family in order:
        help_line = type_line = None
        for source in sources:
            declared = source.families.get(family)
            if declared is None:
                continue
            help_line = help_line or declared[0]
            type_line = type_line or declared[1]
        family_type = type_line.split()[-1] if type_line else None
        strategy = _resolve_strategy(family, family_type)
        emitted.append(
            (
                family,
                (help_line, type_line),
                _aggregate_family(family, strategy, sources),
            )
        )
    return _emit(emitted)


def merger_lines(processes: int, degraded: bool) -> list[str]:
    return [
        f"# HELP {PROCESSES_METRIC} Live processes of this Pod whose samples were merged into this scrape, the answering process included.",
        f"# TYPE {PROCESSES_METRIC} gauge",
        f"{PROCESSES_METRIC} {processes}",
        f"# HELP {DEGRADED_METRIC} 1 when the Pod's processes could not share their samples (shared directory unusable), so this scrape is one process's view rather than the Pod's.",
        f"# TYPE {DEGRADED_METRIC} gauge",
        f"{DEGRADED_METRIC} {int(degraded)}",
    ]


# --------------------------------------------------------------------------
# Shared directory
# --------------------------------------------------------------------------


def metrics_directory(environ: Mapping[str, str] | None = None) -> Path | None:
    """Where this process shares its samples, or ``None`` when sharing is off."""

    values = os.environ if environ is None else environ
    explicit = values.get(ENVIRONMENT, "").strip()
    if explicit:
        if explicit.lower() in DISABLED_TOKENS:
            return None
        return Path(explicit)
    pod_uid = values.get("POD_UID", "").strip()
    if not pod_uid:
        return None
    return DEFAULT_ROOT / pod_uid


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class SlotRegistry:
    """Claims and keeps this process's stable slot in a shared directory."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._directory: Path | None = None
        self._slot: int | None = None
        self._handle: Any = None

    def slot(self, directory: Path) -> int | None:
        with self._lock:
            if self._directory == directory and self._slot is not None:
                return self._slot
            self._release()
            directory.mkdir(parents=True, exist_ok=True)
            for candidate in range(MAX_SLOTS):
                path = directory / f"slot-{candidate}.lock"
                handle = open(path, "a+", encoding="utf-8")  # noqa: SIM115 - held for life
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    handle.close()
                    continue
                self._handle = handle
                self._directory = directory
                self._slot = candidate
                return candidate
            return None

    def _release(self) -> None:
        if self._handle is not None:
            try:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
                self._handle.close()
            except OSError:
                pass
        self._handle = None
        self._slot = None
        self._directory = None

    def release(self) -> None:
        with self._lock:
            self._release()


SLOTS = SlotRegistry()


def publish(
    directory: Path,
    rendered: Rendered,
    *,
    pid: int | None = None,
    slot: int | None = None,
) -> None:
    """Write this process's render atomically; a partial file is never read."""

    directory.mkdir(parents=True, exist_ok=True)
    own = os.getpid() if pid is None else pid
    payload = {
        "format": FILE_FORMAT,
        "pid": own,
        "slot": slot,
        "published_at": time.time(),
        "families": {
            name: [help_line, type_line]
            for name, (help_line, type_line) in rendered.families.items()
        },
        "samples": [
            [sample.name, [list(pair) for pair in sample.labels], sample.value]
            for samples in rendered.samples.values()
            for sample in samples
        ],
    }
    handle, temporary = tempfile.mkstemp(
        prefix=f".{own}-", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, separators=(",", ":"))
        os.replace(temporary, directory / f"{own}.json")
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _load(path: Path) -> Rendered | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # Being rewritten, or never completed: the next scrape sees it.
        return None
    if not isinstance(payload, dict) or payload.get("format") != FILE_FORMAT:
        return None
    rendered = Rendered(pid=payload.get("pid"), slot=payload.get("slot"))
    families = payload.get("families")
    if isinstance(families, dict):
        for name, declared in families.items():
            if isinstance(declared, list) and len(declared) == 2:
                rendered.families[name] = (declared[0], declared[1])
                rendered.samples.setdefault(name, [])
    samples = payload.get("samples")
    if isinstance(samples, list):
        for item in samples:
            if not (isinstance(item, list) and len(item) == 3):
                continue
            name, labels, value = item
            pairs = tuple(
                (str(k), str(v)) for k, v in labels if isinstance(labels, list)
            )
            family = _family_of(str(name), rendered.families)
            rendered.families.setdefault(family, (None, None))
            rendered.samples.setdefault(family, []).append(
                Sample(str(name), pairs, str(value))
            )
    return rendered


def live_siblings(directory: Path, *, pid: int | None = None) -> list[Rendered]:
    """Every other live process's published render; a dead PID's file is
    removed rather than counted again after a worker restart."""

    own = os.getpid() if pid is None else pid
    others: list[Rendered] = []
    if not directory.is_dir():
        return others
    for path in sorted(directory.glob("*.json")):
        try:
            other = int(path.stem)
        except ValueError:
            continue
        if other == own:
            continue
        if not _alive(other):
            path.unlink(missing_ok=True)
            continue
        loaded = _load(path)
        if loaded is not None:
            others.append(loaded)
    return others


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------


def pod_coherent_lines(
    lines: Sequence[str],
    *,
    directory: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> list[str]:
    """Publish this process's render and return the Pod-wide exposition.

    Falls back to the local render (flagged degraded) when the shared
    directory is configured but unusable; a scrape must never fail because
    of its own bookkeeping.
    """

    local = parse_lines(lines)
    target = metrics_directory(environ) if directory is None else directory
    if target is None:
        local.slot = 0
        return aggregate(local, []) + merger_lines(1, False)
    try:
        local.slot = SLOTS.slot(target)
        local.pid = os.getpid()
        publish(target, local, slot=local.slot)
        others = live_siblings(target)
    except OSError as error:
        LOGGER.warning(
            "process metrics could not be shared through %s: %s", target, error
        )
        local.slot = 0
        return aggregate(local, []) + merger_lines(1, True)
    return aggregate(local, others) + merger_lines(1 + len(others), False)


def publish_forever(
    render: Callable[[], Sequence[str]],
    stop: Event,
    *,
    directory: Path | None = None,
    interval: float = PUBLISH_INTERVAL_SECONDS,
) -> None:
    """Publish this process's render until ``stop`` is set.

    Without this, a process publishes only when it answers a scrape, and the
    process that counted may never be asked. Runs in every worker process; it
    returns at once when sharing is off.
    """

    target = metrics_directory() if directory is None else directory
    if target is None:
        return
    while True:
        try:
            rendered = parse_lines(render())
            rendered.slot = SLOTS.slot(target)
            rendered.pid = os.getpid()
            publish(target, rendered, slot=rendered.slot)
        except Exception:  # noqa: BLE001 - a publisher must not die on one write
            LOGGER.exception("process metrics could not be published to %s", target)
        if stop.wait(interval):
            return
