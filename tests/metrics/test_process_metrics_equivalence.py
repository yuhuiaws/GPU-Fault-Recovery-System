"""Old and new Pod-coherent merges emit the same scrape for the same inputs.

The merge got cheaper -- the C JSON encoder, sibling samples built only for
the families the merge reads, a bulk well-formedness check in place of a
Python loop per entry -- and none of that may change a byte. Every case here
feeds the frozen pre-change module (``_process_metrics_reference``) and
``gpu_fault.app.process_metrics`` the same local render and the same sibling
publications, each in its own copy of one shared directory, and demands the
same exposition, the same bytes in the answering process's own publication,
and the same files left behind (dead PIDs pruned, unknown evidence kept).
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import math
import os
import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.app import ApplicationContext, create_app
from gpu_fault.app import process_metrics as new
from gpu_fault.app.metric_aggregation import MERGER_FAMILIES, STRATEGIES, Strategy
from tests._builders import asgi_client, build_store
from tests.metrics import _process_metrics_reference as old

CLOCK = 1_700_000_000.0
SIBLING_PID = 3_900_000
SUFFIXES = ("_bucket", "_count", "_sum", "_max")
LABEL_KEYS = ("cluster_id", "reason", "status", "gpu_node", "process")
LABEL_VALUES = (
    "a",
    "perf-cap-001",
    'quote \\" inside',
    "brace } inside",
    "comma, inside",
    "eq = inside",
    " spaced ",
    "é-ünicode",
    "back\\\\slash",
    "",
    "new\\nline",
)
VALUES = (
    "0",
    "1",
    "3",
    "42",
    "-2",
    "0.250000",
    "0.5",
    "1.000000",
    "1e3",
    "2.5e-3",
    "+Inf",
    "-Inf",
    "NaN",
    "1700000000.000",
)
UNREGISTERED = (
    ("unit_equiv_custom_total", "counter"),
    ("unit_equiv_gauge", "gauge"),
    ("unit_equiv_untyped", None),
    ("unit_equiv_latency", "summary"),
    ("unit_equiv_hist", "histogram"),
)


def _registered(strategy: Strategy, count: int = 3) -> list[str]:
    return sorted(
        name
        for name, value in STRATEGIES.items()
        if value is strategy and name not in MERGER_FAMILIES
    )[:count]


# ---------------------------------------------------------------------------
# One Pod, run twice
# ---------------------------------------------------------------------------


@dataclass
class Sibling:
    pid: int
    payload: Any
    # File text instead of the JSON payload (a torn or foreign file).
    raw: str | None = None


@dataclass
class Case:
    local: list[str]
    siblings: list[Sibling] = field(default_factory=list)
    dead: set[int] = field(default_factory=set)
    # Slots a live process holds without a publication.
    held: set[int] = field(default_factory=set)
    garbage: dict[str, str] = field(default_factory=dict)


class Pod:
    def __init__(self, root: Path, dead: set[int]) -> None:
        self.root = root
        self.dead = dead

    def run(
        self, implementation: Any, case: Case
    ) -> tuple[list[str], dict[str, bytes]]:
        directory = self.root / implementation.__name__.rsplit(".", 1)[-1]
        directory.mkdir()
        for sibling in case.siblings:
            text = (
                sibling.raw
                if sibling.raw is not None
                else json.dumps(sibling.payload, separators=(",", ":"))
            )
            (directory / f"{sibling.pid}.json").write_text(text, encoding="utf-8")
        for name, text in case.garbage.items():
            (directory / name).write_text(text, encoding="utf-8")
        self.dead.clear()
        self.dead.update(case.dead)
        handles = []
        for slot in sorted(case.held):
            handle = open(directory / f"slot-{slot}.lock", "a+")  # noqa: SIM115
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            handles.append(handle)
        try:
            lines = implementation.pod_coherent_lines(case.local, directory=directory)
        finally:
            for handle in handles:
                handle.close()
            implementation.SLOTS.release()
        left = {
            path.name: path.read_bytes()
            for path in sorted(directory.iterdir())
            if not path.name.endswith(".lock")
        }
        return lines, left

    def assert_same(self, case: Case) -> list[str]:
        expected, expected_left = self.run(old, case)
        actual, actual_left = self.run(new, case)
        assert actual == expected, _first_difference(expected, actual)
        assert actual_left == expected_left, "the shared directory diverged"
        return actual


def _first_difference(expected: list[str], actual: list[str]) -> str:
    for index, (left, right) in enumerate(zip(expected, actual)):
        if left != right:
            return f"line {index}: expected {left!r}, got {right!r}"
    return f"length {len(expected)} expected, {len(actual)} got"


@pytest.fixture
def pod(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Pod]:
    dead: set[int] = set()

    def kill(pid: int, signal: int) -> None:
        assert signal == 0, "liveness probes only"
        if pid in dead:
            raise ProcessLookupError(pid)

    # time and os are the shared modules: both implementations see the patch.
    monkeypatch.setattr(new.time, "time", lambda: CLOCK)
    monkeypatch.setattr(new.os, "kill", kill)
    monkeypatch.setattr(old, "SLOTS", old.SlotRegistry())
    monkeypatch.setattr(new, "SLOTS", new.SlotRegistry())
    yield Pod(tmp_path, dead)
    old.SLOTS.release()
    new.SLOTS.release()


# ---------------------------------------------------------------------------
# Renders and publications
# ---------------------------------------------------------------------------


def _payload(
    rendered: old.Rendered, pid: int, slot: Any, stamp: Any = CLOCK
) -> dict[str, Any]:
    """The document the pre-change ``publish`` wrote for ``rendered``."""

    return {
        "format": 1,
        "pid": pid,
        "slot": slot,
        "published_at": stamp,
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


def _pool(rng: random.Random) -> list[tuple[str, str | None]]:
    """A Pod's families: registered ones of every strategy, registered names
    that end in a summary suffix, and unregistered ones (type fallbacks)."""

    families: list[tuple[str, str | None]] = []
    for strategy in Strategy:
        for name in rng.sample(_registered(strategy), k=rng.randint(1, 2)):
            kind = (
                rng.choice(("counter", "gauge", "summary", "histogram"))
                if strategy is Strategy.PER_PROCESS
                else "gauge"
            )
            families.append((name, kind))
    suffixed = sorted(name for name in STRATEGIES if name.endswith(SUFFIXES))[:4]
    families += [(name, rng.choice(("gauge", None))) for name in suffixed]
    families += rng.sample(UNREGISTERED, k=rng.randint(1, len(UNREGISTERED)))
    rng.shuffle(families)
    return families


def _labels(
    rng: random.Random, odd: bool, extra: tuple[tuple[str, str], ...] = ()
) -> str:
    keys = rng.sample(LABEL_KEYS, k=rng.choice((0, 0, 1, 1, 2)))
    pairs = [(key, rng.choice(LABEL_VALUES)) for key in keys] + list(extra)
    if odd and pairs and rng.random() < 0.2:
        pairs.append(pairs[0])  # a repeated label name
    if not pairs:
        return ""
    body = ",".join(f'{key}="{value}"' for key, value in pairs)
    if rng.random() < 0.1:
        body = " , " + body.replace(",", " , ") + ", "
    return "{" + body + "}"


def _value(rng: random.Random, odd: bool) -> str:
    if odd and rng.random() < 0.2:
        # Not a number, or a sample with a timestamp: kept verbatim by the
        # local merge, and a sibling publishing one is degraded.
        return rng.choice(("not-a-number", "7 1700000000000"))
    return rng.choice(VALUES)


def _sample_lines(
    rng: random.Random, family: str, kind: str | None, count: int, odd: bool
) -> list[str]:
    lines = []
    for _ in range(count):
        if kind == "summary" and rng.random() < 0.7:
            suffix = rng.choice(("_sum", "_count", "_max", ""))
            extra = (("quantile", rng.choice(("0.5", "0.99"))),) if not suffix else ()
            lines.append(
                f"{family}{suffix}{_labels(rng, odd, extra)} {_value(rng, odd)}"
            )
        elif kind == "histogram" and rng.random() < 0.7:
            suffix = rng.choice(("_bucket", "_bucket", "_count", "_sum"))
            extra = (
                (("le", rng.choice(("1", "5", "+Inf"))),) if suffix == "_bucket" else ()
            )
            lines.append(
                f"{family}{suffix}{_labels(rng, odd, extra)} {_value(rng, odd)}"
            )
        else:
            lines.append(f"{family}{_labels(rng, odd)} {_value(rng, odd)}")
    return lines


def _render(
    rng: random.Random, pool: list[tuple[str, str | None]], *, skip: float = 0.15
) -> list[str]:
    # One render in ten carries entries a sibling cannot publish cleanly
    # (read entry by entry); the rest read through the per-family path.
    odd = rng.random() < 0.1
    lines: list[str] = []
    for family, kind in pool:
        if rng.random() < skip:
            continue  # this process never rendered the family
        style = rng.choice(
            ("both", "both", "both", "type", "help", "none", "none", "twice")
        )
        if style in ("both", "help", "twice"):
            lines.append(f"# HELP {family} Help for {family}.")
        if style == "twice":
            lines.append(f"# HELP {family} A second HELP line is ignored.")
        if style in ("both", "type", "twice") and kind is not None:
            lines.append(f"# TYPE {family} {kind}")
        count = rng.choice((0, 1, 1, 2, 3, 4))
        lines += _sample_lines(rng, family, kind, count, odd)
    if rng.random() < 0.2:
        lines.insert(rng.randrange(len(lines) + 1), "# an ordinary comment")
    if rng.random() < 0.2:
        lines.insert(rng.randrange(len(lines) + 1), "")
    if rng.random() < 0.1:
        rng.shuffle(lines)  # samples ahead of their declarations
    return lines


FILE_DEFECTS = (
    "families-entry",
    "families-type",
    "samples-type",
    "pid-type",
    "pid-other",
    "slot",
    "stamp",
    "format",
    "root",
    "torn",
)
ENTRY_DEFECTS = (
    "entry-type",
    "entry-length",
    "labels-type",
    "pair-length",
    "pair-part",
    "duplicate-label",
    "value-text",
    "value-number",
    "value-float",
    "value-bool",
    "value-null",
    "name-type",
)


def _corrupt(rng: random.Random, sibling: Sibling, defect: str | None = None) -> None:
    """One defect a live sibling's file can carry."""

    payload = sibling.payload
    samples = payload["samples"]
    if defect is None:
        defect = rng.choice(FILE_DEFECTS + (ENTRY_DEFECTS if samples else ()))
    if defect == "families-entry" and payload["families"]:
        payload["families"][rng.choice(sorted(payload["families"]))] = [42, {}]
    elif defect == "families-type":
        payload["families"] = []
    elif defect == "samples-type":
        payload["samples"] = {}
    elif defect == "pid-type":
        payload["pid"] = str(sibling.pid)
    elif defect == "pid-other":
        payload["pid"] = sibling.pid + 1
    elif defect == "slot":
        payload["slot"] = rng.choice((None, True, -1, 16, "1"))
    elif defect == "stamp":
        payload["published_at"] = rng.choice(
            (None, True, "1", math.nan, math.inf, CLOCK - 61, CLOCK - 60, CLOCK + 1)
        )
    elif defect == "format":
        payload["format"] = rng.choice((2, 1.0, True, None))
    elif defect == "root":
        sibling.payload = [payload]
    elif defect == "torn":
        sibling.raw = rng.choice(("", "{", '{"format": 1, "samples": ['))
    elif defect in ENTRY_DEFECTS and samples:
        _corrupt_entry(samples, rng.randrange(len(samples)), defect)


def _corrupt_entry(samples: list[Any], index: int, defect: str) -> None:
    item = samples[index]
    if defect == "entry-type":
        samples[index] = "not-an-entry"
    elif defect == "entry-length":
        samples[index] = item[:2]
    elif defect == "labels-type":
        item[1] = {"reason": "not a list"}
    elif defect == "pair-length":
        item[1] = [["reason"]]
    elif defect == "pair-part":
        item[1] = [["reason", 5]]
    elif defect == "duplicate-label":
        item[1] = [["reason", "a"], ["reason", "b"]]
    elif defect == "value-text":
        item[2] = "not-a-number"
    elif defect == "value-number":
        item[2] = 7  # read as "7" before and after
    elif defect == "value-float":
        item[2] = 2.5
    elif defect == "value-bool":
        item[2] = True
    elif defect == "value-null":
        item[2] = None
    elif defect == "name-type":
        item[0] = 5


def _random_case(seed: int) -> Case:
    rng = random.Random(seed)
    pool = _pool(rng)
    case = Case(local=_render(rng, pool, skip=0.3))
    for index in range(rng.choice((0, 1, 2, 3, 3, 4))):
        pid = SIBLING_PID + index
        slot = rng.choice((1, 2, 3, 4, 5, 1))
        stamp = CLOCK - rng.choice((0.0, 0.0, 4.5, 59.0))
        sibling = Sibling(
            pid, _payload(old.parse_lines(_render(rng, pool)), pid, slot, stamp)
        )
        if rng.random() < 0.3:
            _corrupt(rng, sibling)
        if rng.random() < 0.1:
            case.dead.add(pid)
        case.siblings.append(sibling)
    if rng.random() < 0.15:
        case.held.add(rng.choice((0, 7, 14)))
    if rng.random() < 0.1:
        case.garbage = {"not-a-pid.json": "{}", "123abc.json": "{"}
    return case


@pytest.mark.parametrize("seed", range(160))
def test_random_pods_merge_to_the_same_scrape(pod: Pod, seed: int) -> None:
    pod.assert_same(_random_case(seed))


@pytest.mark.parametrize("variant", range(3))
@pytest.mark.parametrize("defect", FILE_DEFECTS + ENTRY_DEFECTS)
def test_every_publication_defect_reads_as_before(
    pod: Pod, defect: str, variant: int
) -> None:
    """Each defect once per variant, in a Pod whose other siblings are clean."""

    rng = random.Random(f"{defect}-{variant}")
    pool = _pool(rng)
    case = Case(local=_render(rng, pool))
    for index in range(3):
        pid = SIBLING_PID + index
        lines = _render(rng, pool, skip=0.0)
        case.siblings.append(
            Sibling(pid, _payload(old.parse_lines(lines), pid, index + 1))
        )
    _corrupt(rng, case.siblings[1], defect)
    pod.assert_same(case)


@pytest.mark.parametrize("table", ["none", "some", "reordered"])
def test_families_missing_from_the_families_table_read_as_before(
    pod: Pod, table: str
) -> None:
    """``publish`` lists every family it writes samples for; a file that does
    not opens the missing ones in the order its entries name them."""

    names = (
        "unit_equiv_z_total",
        "unit_equiv_a",
        "unit_equiv_m_count",
        "unit_equiv_a_sum",
    )
    samples = [
        [name, [["reason", str(index)]], str(index)] for index, name in enumerate(names)
    ]
    samples.append([names[0], [], "9"])
    families = {
        "none": {},
        "some": {"unit_equiv_a": ["# HELP unit_equiv_a Declared.", None]},
        "reordered": {name: [None, None] for name in reversed(names)},
    }[table]
    payload = {
        "format": 1,
        "pid": SIBLING_PID,
        "slot": 1,
        "published_at": CLOCK,
        "families": families,
        "samples": samples,
    }
    local = ["# TYPE unit_equiv_local gauge", "unit_equiv_local 1"]
    merged = pod.assert_same(
        Case(local=local, siblings=[Sibling(SIBLING_PID, payload)])
    )
    assert f"{new.DEGRADED_METRIC} 0" in merged, (
        "a sparse families table is not a defect"
    )


# ---------------------------------------------------------------------------
# Real renders
# ---------------------------------------------------------------------------


def _scrape_role(monkeypatch: pytest.MonkeyPatch, role: str) -> list[str]:
    monkeypatch.setenv("GPU_FAULT_SERVICE_ROLE", role)
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_WORKERS", "24")
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("POD_UID", f"pod-process-metrics-equivalence-{role}")
    monkeypatch.delenv("GPU_FAULT_TELEMETRY_SPOOL", raising=False)
    app = create_app(
        ApplicationContext(
            store=build_store(),
            execution_token="process-metrics-equivalence-token-" + "x" * 32,
        )
    )

    async def run() -> str:
        async with asgi_client(app) as client:
            response = await client.get("/metrics")
            assert response.status_code == 200, response.text
            return response.text

    merger = tuple(MERGER_FAMILIES)
    return [
        line
        for line in asyncio.run(run()).splitlines()
        if not any(family in line for family in merger)
    ]


@pytest.fixture(scope="module")
def role_renders(request: pytest.FixtureRequest) -> dict[str, list[str]]:
    """A real single-process scrape per role, merger gauges removed."""

    monkeypatch = pytest.MonkeyPatch()
    request.addfinalizer(monkeypatch.undo)
    monkeypatch.setenv("GPU_FAULT_PROCESS_METRICS_DIR", "off")
    monkeypatch.setenv(
        "GPU_FAULT_PROCESSOR_REPLAY_SECRET", "test-processor-replay-secret-" + "r" * 32
    )
    return {role: _scrape_role(monkeypatch, role) for role in ("ingress", "worker")}


def _perturbed(rng: random.Random, lines: list[str]) -> list[str]:
    """Another process's render of the same role: other values, a series it
    never saw, one it alone saw."""

    out = []
    for line in lines:
        if line.startswith("#") or not line.strip():
            out.append(line)
            continue
        head, _, value = line.rpartition(" ")
        roll = rng.random()
        if roll < 0.05:
            continue
        if roll < 0.55:
            if "." in value and value.replace(".", "", 1).isdigit():
                places = len(value.partition(".")[2])
                value = f"{float(value) + rng.randint(0, 9) / 4:.{places}f}"
            elif value.isdigit():
                value = str(int(value) + rng.randint(0, 5))
        out.append(f"{head} {value}")
        if roll > 0.97 and "{" not in head:
            out.append(f'{head}{{reason="only-here"}} {rng.randint(0, 3)}')
    return out


@pytest.mark.parametrize("role", ["ingress", "worker"])
@pytest.mark.parametrize("seed", range(4))
def test_real_role_renders_merge_to_the_same_scrape(
    pod: Pod, role_renders: dict[str, list[str]], role: str, seed: int
) -> None:
    rng = random.Random(seed)
    local = role_renders[role]
    assert len(local) > 200, "the fixture did not render a full app"
    case = Case(local=local)
    for index, slot in enumerate(rng.sample(range(1, 8), k=3)):
        pid = SIBLING_PID + index
        rendered = old.parse_lines(_perturbed(rng, local))
        case.siblings.append(Sibling(pid, _payload(rendered, pid, slot)))
    if seed % 2:
        _corrupt_entry(case.siblings[0].payload["samples"], 3, "value-text")
    pod.assert_same(case)


# ---------------------------------------------------------------------------
# Fleet-sized ANY families
# ---------------------------------------------------------------------------

PER_NODE = tuple(_registered(Strategy.ANY, count=2))
SAMPLE_COUNTER = _registered(Strategy.PER_PROCESS, count=1)[0]


def _per_node(lines: list[str]) -> list[str]:
    return [line for line in lines if line.split("{", 1)[0] in PER_NODE]


def _fleet_render(clusters: int, nodes: int, *, sparse: bool) -> list[str]:
    """Two per-node ANY families, either one zero sample per node (the render
    the perf fleet produced) or declared with no samples at all (the sparse
    render that drops zero-valued series)."""

    lines = [f"# TYPE {SAMPLE_COUNTER} counter", f"{SAMPLE_COUNTER} 3"]
    for family in PER_NODE:
        lines += [f"# HELP {family} Per-node family.", f"# TYPE {family} gauge"]
        if sparse:
            continue
        lines += [
            f'{family}{{cluster_id="perf-cap-{cluster:03d}",'
            f'gpu_node="fleet-node-c{cluster:03d}-n{node:04d}"}} 0'
            for cluster in range(clusters)
            for node in range(nodes)
        ]
    return lines


@pytest.mark.parametrize(
    "local_sparse", [False, True], ids=["local-full", "local-sparse"]
)
def test_fleet_sized_any_families_merge_to_the_same_scrape(
    pod: Pod, local_sparse: bool
) -> None:
    case = Case(local=_fleet_render(4, 64, sparse=local_sparse))
    for index in range(3):
        pid = SIBLING_PID + index
        rendered = old.parse_lines(_fleet_render(4, 64, sparse=False))
        case.siblings.append(Sibling(pid, _payload(rendered, pid, index + 1)))
    merged = pod.assert_same(case)
    assert len(_per_node(merged)) == (0 if local_sparse else 2 * 4 * 64), (
        "a family the answering process declares is the one the scrape reads"
    )


# ---------------------------------------------------------------------------
# Targeted cases
# ---------------------------------------------------------------------------


def test_a_malformed_entry_in_a_family_the_merge_never_reads_still_degrades(
    pod: Pod,
) -> None:
    """The sibling's per-node entries are never built (the local render
    declares the ANY family), yet a malformed one still marks the sibling's
    publication incomplete exactly as before: the scrape reports degraded
    coverage, does not count the sibling as a complete process, and still
    merges the sibling's well-formed families."""

    local = _fleet_render(2, 8, sparse=True)
    sibling = old.parse_lines(_fleet_render(2, 8, sparse=False))
    payload = _payload(sibling, SIBLING_PID, 1)
    family = PER_NODE[0]
    index = next(
        position
        for position, entry in enumerate(payload["samples"])
        if entry[0] == family
    )
    payload["samples"][index][2] = "not-a-number"
    merged = pod.assert_same(
        Case(local=local, siblings=[Sibling(SIBLING_PID, payload)])
    )

    assert f"{new.DEGRADED_METRIC} 1" in merged, "coverage must stay degraded"
    assert f"{new.PROCESSES_METRIC} 1" in merged, "an incomplete sibling is not counted"
    assert f'{SAMPLE_COUNTER}{{process="1"}} 3' in merged, (
        "the sibling's well-formed families are still merged"
    )
    assert not _per_node(merged), "the declared, empty local family still wins"


def test_a_publication_whose_families_depend_on_entry_order_reads_as_before(
    pod: Pod, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``unit_equiv_order_sum`` first joins its declared summary base, then a
    ``unit_equiv_order_sum_count`` entry opens it as a registered family of
    its own, and its next entry stays there. Resolving each name once would
    misfile that entry; the reader falls back to reading entry by entry."""

    monkeypatch.setitem(STRATEGIES, "unit_equiv_order_sum", Strategy.PER_PROCESS)
    base, name = "unit_equiv_order", "unit_equiv_order_sum"
    payload = {
        "format": 1,
        "pid": SIBLING_PID,
        "slot": 1,
        "published_at": CLOCK,
        "families": {base: [None, f"# TYPE {base} summary"]},
        "samples": [
            [name, [], "1"],
            [f"{name}_count", [], "2"],
            [name, [["reason", "later"]], "3"],
        ],
    }
    local = [f"# TYPE {base} summary", f"{base}_sum 5"]
    merged = pod.assert_same(
        Case(local=local, siblings=[Sibling(SIBLING_PID, payload)])
    )
    assert f'{name}{{reason="later",process="1"}} 3' in merged, (
        "the later entry belongs to the family it opened"
    )


@pytest.mark.parametrize("seed", range(12))
def test_published_bytes_are_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seed: int
) -> None:
    """json.dumps (C encoder) writes what json.dump (pure Python) wrote."""

    monkeypatch.setattr(new.time, "time", lambda: CLOCK + 0.123456789)
    rng = random.Random(seed)
    lines = _render(rng, _pool(rng))
    old.publish(tmp_path / "old", old.parse_lines(lines), pid=4242, slot=seed % 3)
    new.publish(tmp_path / "new", new.parse_lines(lines), pid=4242, slot=seed % 3)
    expected = (tmp_path / "old" / "4242.json").read_bytes()
    assert (tmp_path / "new" / "4242.json").read_bytes() == expected, (
        "the publication format must not change"
    )


def test_samples_of_a_never_read_sibling_family_are_never_built(
    pod: Pod, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The merge builds a sibling's samples only for the families it reads:
    a sibling's per-node ANY family the local render declares costs no
    samples however large it is, a PER_PROCESS family costs one per entry."""

    built: list[str] = []

    class CountedSample(new.Sample):
        def __init__(self, name: str, labels: Any, value: str) -> None:
            built.append(name)
            super().__init__(name, labels, value)

    monkeypatch.setattr(new, "Sample", CountedSample)

    def samples_built(sibling_nodes: int) -> dict[str, int]:
        built.clear()
        local = _fleet_render(2, 4, sparse=False)
        sibling = old.parse_lines(_fleet_render(2, sibling_nodes, sparse=False))
        directory = pod.root / f"lazy-{sibling_nodes}"
        directory.mkdir()
        (directory / f"{SIBLING_PID}.json").write_text(
            json.dumps(_payload(sibling, SIBLING_PID, 1)), encoding="utf-8"
        )
        try:
            new.pod_coherent_lines(local, directory=directory)
        finally:
            new.SLOTS.release()
        return {name: built.count(name) for name in set(built)}

    small, large = samples_built(4), samples_built(400)
    assert small == large, "a sibling's unread ANY family must not be built"
    assert small[PER_NODE[0]] == 2 * 4, "only the local render's per-node samples"
    assert small[SAMPLE_COUNTER] >= 2, "the PER_PROCESS family is read from both"


def test_pending_families_build_on_first_read_in_publication_order() -> None:
    rendered = new.Rendered(
        families={"unit_equiv_gauge": (None, None)},
        samples={"unit_equiv_gauge": []},
        pending={
            "unit_equiv_gauge": [
                ["unit_equiv_gauge", [["reason", "b"]], "2"],
                ["unit_equiv_gauge", [], "1"],
            ]
        },
    )

    first = rendered.family_samples("unit_equiv_gauge")
    assert [sample.line() for sample in first] == [
        'unit_equiv_gauge{reason="b"} 2',
        "unit_equiv_gauge 1",
    ]
    assert rendered.pending == {}, "a built family is no longer pending"
    assert rendered.family_samples("unit_equiv_gauge") is first, "built once"
    assert rendered.family_samples("unit_equiv_absent") == (), "absent family"
    assert rendered.materialise() is rendered, "materialise returns the render"


def test_live_siblings_still_returns_complete_publications(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(new.os, "kill", lambda pid, signal: None)
    monkeypatch.setattr(new.time, "time", lambda: CLOCK)
    rendered = old.parse_lines(_fleet_render(1, 3, sparse=False))
    (tmp_path / f"{SIBLING_PID}.json").write_text(
        json.dumps(_payload(rendered, SIBLING_PID, 2)), encoding="utf-8"
    )

    (sibling,) = new.live_siblings(tmp_path, pid=os.getpid())

    assert sibling.pending == {}, "the public reader hands out built samples"
    assert [sample.line() for sample in sibling.samples[PER_NODE[0]]] == [
        sample.line() for sample in rendered.samples[PER_NODE[0]]
    ]
