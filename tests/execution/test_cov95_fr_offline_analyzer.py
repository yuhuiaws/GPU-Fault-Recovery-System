"""The offline FR analyzer adapter and its CPU-only acceptance, without PyTorch.

``tools.analyze_nccl_flights`` hands archived rank dumps to the official
PyTorch 2.10 flight-recorder builder; ``tools.verify_fr_analyzer`` checks the
product's adapted matcher (``gpu_fault.pytorch_fr_matcher``) against the real
``Op.match`` and the builder. Neither CI nor a trainer installs torch, so the
torch pieces both tools reach for -- the ``torch`` distribution metadata, the
``torch`` module's version, the FR ``builder`` and ``types`` modules -- are
stand-ins installed into ``importlib.metadata`` and ``sys.modules`` for one
test. What is asserted is the tools' own contract: which manifests are refused
and why, what the builder is handed, how its database is reported and bounded,
how the acceptance enumerates its cases and names the first disagreement, and
what both command lines print, write and exit with.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import resource
import runpy
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.pytorch_fr_matcher import COLLECTIVES, match_collective
from tools import analyze_nccl_flights as flights
from tools import verify_fr_analyzer as acceptance

TORCH_VERSION = "2.10.0+cpu"
BUILDER_MODULE = "torch.distributed.flight_recorder.components.builder"
TYPES_MODULE = "torch.distributed.flight_recorder.components.types"
MIB = 1024 * 1024
#: Every collective kind the acceptance matches, times its six mutations.
MATCHER_KINDS = sorted(COLLECTIVES | {"coalesced"})
MUTATIONS = ("none", "input", "output", "dtype", "state", "type")
BUILDER_KINDS = (
    "all_reduce",
    "broadcast",
    "all_gather",
    "_reduce_scatter_base",
    "allreduce_coalesced",
)


# --------------------------------------------------------------------------- #
# torch stand-ins
# --------------------------------------------------------------------------- #
class OracleBuilder:
    """``builder.build_db`` stand-in: one collective per sequence id.

    Its verdict is computed independently of the product matcher: a collective
    passes when every rank recorded the same operation, and each other rank is
    reported as a type mismatch against rank 0.
    """

    def __init__(self, collectives: list[Any] | None = None) -> None:
        self.collectives = collectives
        self.calls: list[dict[str, Any]] = []

    def build_db(self, details: dict[str, Any], args: Any, version: str) -> Any:
        self.calls.append(
            {"details": copy.deepcopy(details), "args": args, "version": version}
        )
        if self.collectives is not None:
            return SimpleNamespace(collectives=self.collectives)
        by_sequence: dict[int, dict[int, dict[str, Any]]] = {}
        for rank_text, payload in details.items():
            for operation in payload["entries"]:
                by_sequence.setdefault(operation["collective_seq_id"], {})[
                    int(rank_text)
                ] = operation
        collectives = []
        for sequence, by_rank in sorted(by_sequence.items()):
            first = by_rank[min(by_rank)]
            mismatches = {
                rank: SimpleNamespace(
                    collective_name=operation["profiling_name"],
                    type_of_mismatch="COLLECTIVE_TYPE_MISMATCH",
                    input_sizes=operation["input_sizes"],
                    output_sizes=operation["output_sizes"],
                )
                for rank, operation in by_rank.items()
                if operation["profiling_name"] != first["profiling_name"]
            }
            collectives.append(
                collective(
                    sequence,
                    first["profiling_name"],
                    pass_check=not mismatches,
                    expected_ranks=set(by_rank),
                    mismatch_collectives=mismatches or None,
                    type_of_mismatch="COLLECTIVE_TYPE_MISMATCH" if mismatches else None,
                )
            )
        return SimpleNamespace(collectives=collectives)


class FakeEngine:
    """``types.Op`` stand-in recording every match the acceptance asks for.

    ``verdict(left, right, pg_size)`` names the native ``MatchState``.
    """

    def __init__(self, verdict: Any) -> None:
        self.matches: list[dict[str, Any]] = []
        engine = self

        class Op:
            def __init__(
                self, event: dict[str, Any], memberships: dict[str, Any], pg_name: str
            ) -> None:
                self.event = event
                self.memberships = memberships
                self.pg_name = pg_name

            def match(self, other: Any) -> Any:
                size = len(self.memberships[self.pg_name])
                engine.matches.append(
                    {
                        "left": copy.deepcopy(self.event),
                        "right": copy.deepcopy(other.event),
                        "pg_name": self.pg_name,
                        "size": size,
                    }
                )
                name = verdict(self.event, other.event, size)
                return SimpleNamespace(state=SimpleNamespace(name=name))

        self.Op = Op


def agreeing(left: dict[str, Any], right: dict[str, Any], size: int) -> str:
    return match_collective(left, right, size)


def collective(
    sequence: int,
    name: str,
    *,
    pass_check: bool = True,
    expected_ranks: Any = (0, 1),
    missing_ranks: Any = None,
    mismatch_collectives: Any = None,
    type_of_mismatch: Any = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        collective_seq_id=sequence,
        group_id="0",
        collective_name=name,
        pass_check=pass_check,
        expected_ranks=expected_ranks,
        missing_ranks=missing_ranks,
        mismatch_collectives=mismatch_collectives,
        type_of_mismatch=type_of_mismatch,
    )


def install_torch(
    monkeypatch: pytest.MonkeyPatch,
    *,
    version: str | None = TORCH_VERSION,
    module_version: str | None = None,
    builder: OracleBuilder | None = None,
    engine: FakeEngine | None = None,
) -> None:
    """Make ``torch`` look installed at ``version`` (None: not installed).

    ``module_version`` is what ``torch.__version__`` says, by default the same.
    """

    real_version = importlib.metadata.version

    def version_of(name: str) -> str:
        if name != "torch":
            return real_version(name)
        if version is None:
            raise importlib.metadata.PackageNotFoundError(name)
        return version

    monkeypatch.setattr(importlib.metadata, "version", version_of)
    torch = types.ModuleType("torch")
    torch.__version__ = module_version or version or "0.0"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    if builder is not None:
        module = types.ModuleType(BUILDER_MODULE)
        module.build_db = builder.build_db  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, BUILDER_MODULE, module)
    if engine is not None:
        module = types.ModuleType(TYPES_MODULE)
        module.Op = engine.Op  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, TYPES_MODULE, module)


# --------------------------------------------------------------------------- #
# archived dumps
# --------------------------------------------------------------------------- #
def payload(name: str = "nccl:all_reduce", *, version: str = "2.10") -> dict:
    return {
        "version": version,
        "pg_config": {"0": {"name": "0", "desc": "default_pg", "ranks": "[0,1]"}},
        "entries": [acceptance.entry(name.split(":", 1)[1])],
    }


def dump(root: Path, number: int, raw: bytes, **overrides: Any) -> dict[str, Any]:
    """The manifest record of rank ``number``'s dump; ``overrides`` replace fields."""

    path = root / f"rank-{number}.dump"
    path.write_bytes(raw)
    record = {
        "rank": number,
        "pid": 100 + number,
        "process_start_ticks": 5000 + number,
        "node_id": f"node-{number}",
        "path": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    record.update(overrides)
    return record


def json_dump(root: Path, number: int, document: Any, **overrides: Any) -> dict:
    return dump(root, number, json.dumps(document).encode(), **overrides)


def sparse(root: Path, name: str, size: int) -> Path:
    path = root / name
    with path.open("wb") as handle:
        handle.truncate(size)
    return path


# --------------------------------------------------------------------------- #
# analyze(): refusals
# --------------------------------------------------------------------------- #
def test_a_torch_outside_2_10_is_unavailable_and_never_reaches_the_builder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builder = OracleBuilder()
    install_torch(monkeypatch, version="2.9.1", builder=builder)
    record = json_dump(tmp_path, 0, payload())
    assert flights.analyze({"records": [record]}, tmp_path) == {
        "status": "UNAVAILABLE",
        "reason": "unsupported analyzer version",
        "version": "2.9.1",
    }, "only the PyTorch 2.10 analyzer is trusted"
    assert builder.calls == [], "an unsupported analyzer never sees a dump"


@pytest.mark.parametrize(
    "manifest",
    [{}, {"records": {"rank": 0}}, {"records": []}, {"records": [{}] * 65537}],
    ids=["absent", "not-a-list", "empty", "over-65536"],
)
def test_the_manifest_must_bind_one_to_65536_ranks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, manifest: dict
) -> None:
    install_torch(monkeypatch, builder=OracleBuilder())
    with pytest.raises(ValueError, match="nonempty process-bound rank manifest"):
        flights.analyze(manifest, tmp_path)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"rank": "0"}, "unknown or duplicate rank"),
        ({"rank": -1}, "unknown or duplicate rank"),
        ({"rank": True}, "unknown or duplicate rank"),
        ({"pid": 0}, "process generation metadata is required"),
        ({"pid": "101"}, "process generation metadata is required"),
        ({"process_start_ticks": 0}, "process generation metadata is required"),
        ({"process_start_ticks": 1.5}, "process generation metadata is required"),
        ({"node_id": ""}, "process generation metadata is required"),
        ({"path": "../outside.dump"}, "archive path escapes manifest root"),
        ({"sha256": "0" * 64}, "raw evidence digest mismatch"),
    ],
)
def test_each_rank_record_is_bound_to_its_process_and_its_raw_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, Any],
    message: str,
) -> None:
    builder = OracleBuilder()
    install_torch(monkeypatch, builder=builder)
    root = tmp_path / "archive"
    root.mkdir()
    (tmp_path / "outside.dump").write_bytes(json.dumps(payload()).encode())
    record = json_dump(root, 1, payload(), **overrides)
    with pytest.raises(ValueError, match=message):
        flights.analyze({"records": [record]}, root)
    assert builder.calls == [], "a refused manifest never reaches the builder"


def test_a_rank_listed_twice_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_torch(monkeypatch, builder=OracleBuilder())
    first = json_dump(tmp_path, 0, payload())
    with pytest.raises(ValueError, match="unknown or duplicate rank"):
        flights.analyze({"records": [first, dict(first)]}, tmp_path)


@pytest.mark.parametrize(
    "document",
    [["not", "an", "object"], payload(version="2.9")],
    ids=["list", "older-dump"],
)
def test_a_dump_that_is_not_a_2_10_object_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, document: Any
) -> None:
    install_torch(monkeypatch, builder=OracleBuilder())
    records = [
        json_dump(tmp_path, 0, payload()),
        dump(tmp_path, 1, json.dumps(document).encode()),
    ]
    with pytest.raises(ValueError, match="mixed or unsupported dump versions"):
        flights.analyze({"records": records}, tmp_path)


def test_the_archive_is_bounded_in_total_and_per_dump_before_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builder = OracleBuilder()
    install_torch(monkeypatch, builder=builder)
    huge = sparse(tmp_path, "huge.dump", 256 * MIB + 1)
    record = {
        "rank": 0,
        "pid": 7,
        "process_start_ticks": 9,
        "node_id": "node-0",
        "path": huge.name,
        "sha256": "unused",
    }
    with pytest.raises(ValueError, match="offline input exceeds 256 MiB"):
        flights.analyze({"records": [record]}, tmp_path)
    large = sparse(tmp_path, "large.dump", 64 * MIB + 1)
    with pytest.raises(ValueError, match="individual dump exceeds 64 MiB"):
        flights.analyze({"records": [dict(record, path=large.name)]}, tmp_path)
    assert builder.calls == [], "an oversized archive is refused before the builder"


# --------------------------------------------------------------------------- #
# analyze(): the builder and its report
# --------------------------------------------------------------------------- #
def test_the_builder_gets_every_rank_by_number_and_its_database_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builder = OracleBuilder()
    install_torch(monkeypatch, builder=builder)
    records = [
        json_dump(tmp_path, 0, payload()),
        json_dump(tmp_path, 1, payload("nccl:reduce")),
    ]
    result = flights.analyze({"records": records}, tmp_path)
    assert len(builder.calls) == 1, builder.calls
    call = builder.calls[0]
    assert call["version"] == "2.10" and set(call["details"]) == {"0", "1"}, call
    assert call["details"]["1"]["rank"] == 1 and call["details"]["1"]["entries"][0][
        "profiling_name"
    ] == ("nccl:reduce"), "each rank's decoded dump is handed over with its rank"
    assert call["args"] == argparse.Namespace(
        verbose=False,
        just_print_entries=False,
        allow_incomplete_ranks=False,
        mismatch_cap=100,
    ), call["args"]
    assert result == {
        "status": "ANALYZED",
        "analyzer": f"pytorch/{TORCH_VERSION}",
        "rank_count": 2,
        "root_cause_confirmed": False,
        "collective_count": 1,
        "collectives_truncated": False,
        "collectives": [
            {
                "sequence": 1,
                "group": "0",
                "operation": "nccl:all_reduce",
                "pass_check": False,
                "expected_rank_count": 2,
                "missing_ranks": [],
                "missing_rank_count": 0,
                "mismatches": {
                    "1": {
                        "operation": "nccl:reduce",
                        "reason": "COLLECTIVE_TYPE_MISMATCH",
                        "input_sizes": [[4]],
                        "output_sizes": [[4]],
                    }
                },
                "mismatched_rank_count": 1,
                "mismatch": "COLLECTIVE_TYPE_MISMATCH",
            }
        ],
        "scope": "provided_archived_dump_set_only",
    }, result


def test_the_report_bounds_collectives_missing_ranks_and_mismatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mismatches = {
        rank: SimpleNamespace(
            collective_name="nccl:broadcast",
            type_of_mismatch="SIZE_OR_SYNTAX_MISMATCH",
            input_sizes=[[rank]],
            output_sizes=[[rank]],
        )
        for rank in range(130)
    }
    first = collective(
        1,
        "nccl:all_reduce",
        pass_check=False,
        expected_ranks=list(range(300)),
        missing_ranks=set(range(299, 99, -1)),
        mismatch_collectives=mismatches,
        type_of_mismatch="SIZE_OR_SYNTAX_MISMATCH",
    )
    rest = [collective(sequence, "nccl:all_reduce") for sequence in range(2, 4098)]
    install_torch(monkeypatch, builder=OracleBuilder([first, *rest]))
    record = json_dump(tmp_path, 0, payload())
    result = flights.analyze({"records": [record]}, tmp_path)
    assert result["collective_count"] == 4097 and result["collectives_truncated"], (
        "a database over 4096 collectives says it was truncated"
    )
    assert len(result["collectives"]) == 4096, len(result["collectives"])
    head = result["collectives"][0]
    assert head["missing_ranks"] == list(range(100, 228)), "sorted, first 128 only"
    assert head["missing_rank_count"] == 200, head["missing_rank_count"]
    assert sorted(head["mismatches"], key=int) == [str(r) for r in range(128)], (
        "at most 128 mismatched ranks are itemised"
    )
    assert head["mismatched_rank_count"] == 130, head["mismatched_rank_count"]
    assert head["expected_rank_count"] == 300, head
    tail = result["collectives"][-1]
    assert (tail["sequence"], tail["mismatches"], tail["mismatch"]) == (
        4096,
        {},
        "None",
    ), "a passing collective reports no mismatch"


# --------------------------------------------------------------------------- #
# the analyzer command line
# --------------------------------------------------------------------------- #
def limit_recorder(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, Any]]:
    """Keep the CLI's rlimits off the test process and record them instead."""

    requested: list[tuple[int, Any]] = []
    monkeypatch.setattr(
        resource,
        "setrlimit",
        lambda kind, limits: requested.append((kind, tuple(limits))),
    )
    monkeypatch.setenv("OMP_NUM_THREADS", "unset-by-test")
    monkeypatch.setenv("MKL_NUM_THREADS", "unset-by-test")
    return requested


def manifest_file(root: Path, records: list[dict[str, Any]]) -> Path:
    path = root / "manifest.json"
    path.write_text(json.dumps({"records": records}), encoding="utf-8")
    return path


def test_analyzer_cli_confines_itself_and_prints_the_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    requested = limit_recorder(monkeypatch)
    install_torch(monkeypatch, builder=OracleBuilder())
    manifest = manifest_file(tmp_path, [json_dump(tmp_path, 0, payload())])
    monkeypatch.setattr(sys, "argv", ["analyze_nccl_flights", str(manifest)])
    flights.main()
    assert requested == [
        (resource.RLIMIT_CORE, (0, 0)),
        (resource.RLIMIT_CPU, (60, 61)),
        (resource.RLIMIT_AS, (8 * 1024 * MIB, 8 * 1024 * MIB)),
    ], "no core dumps, one CPU minute, 8 GiB of address space"
    assert os.environ["OMP_NUM_THREADS"] == "1", "OpenMP pinned to one thread"
    assert os.environ["MKL_NUM_THREADS"] == "1", "MKL pinned to one thread"
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "ANALYZED" and printed["rank_count"] == 1, printed


@pytest.mark.parametrize("defect", ["digest", "unreadable", "torch-absent"])
def test_analyzer_cli_answers_undetermined_or_unavailable_and_exits_2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    defect: str,
) -> None:
    limit_recorder(monkeypatch)
    install_torch(
        monkeypatch,
        version=None if defect == "torch-absent" else TORCH_VERSION,
        builder=OracleBuilder(),
    )
    record = json_dump(tmp_path, 0, payload())
    if defect == "digest":
        record["sha256"] = "f" * 64
    manifest = manifest_file(tmp_path, [record])
    if defect == "unreadable":
        manifest = tmp_path / "absent.json"
    monkeypatch.setattr(sys, "argv", ["analyze_nccl_flights", str(manifest)])
    with pytest.raises(SystemExit) as stopped:
        flights.main()
    assert stopped.value.code == 2, "anything but ANALYZED exits 2"
    printed = json.loads(capsys.readouterr().out)
    expected = {
        "digest": {"status": "UNDETERMINED", "error": "raw evidence digest mismatch"},
        "torch-absent": {
            "status": "UNAVAILABLE",
            "reason": "PyTorch 2.10 analyzer is not installed",
        },
    }
    if defect == "unreadable":
        assert (
            printed["status"] == "UNDETERMINED" and "absent.json" in (printed["error"])
        ), printed
    else:
        assert printed == expected[defect], printed


def test_analyzer_cli_never_prints_more_than_8_mib(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    limit_recorder(monkeypatch)
    oversized = collective(
        1,
        "nccl:all_reduce",
        pass_check=False,
        mismatch_collectives={
            1: SimpleNamespace(
                collective_name="nccl:all_reduce",
                type_of_mismatch="SIZE_OR_SYNTAX_MISMATCH",
                input_sizes="x" * (8 * MIB),
                output_sizes=[[4]],
            )
        },
    )
    install_torch(monkeypatch, builder=OracleBuilder([oversized]))
    manifest = manifest_file(tmp_path, [json_dump(tmp_path, 0, payload())])
    monkeypatch.setattr(sys, "argv", ["analyze_nccl_flights", str(manifest)])
    with pytest.raises(SystemExit) as stopped:
        flights.main()
    assert stopped.value.code == 2, "an oversized answer is not an analysis"
    assert json.loads(capsys.readouterr().out) == {
        "status": "UNDETERMINED",
        "error": "analyzer output exceeds 8 MiB",
    }, "the oversized result is replaced, never printed"


def test_analyzer_module_entrypoint_runs_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    requested = limit_recorder(monkeypatch)
    install_torch(monkeypatch, version="2.11.0", builder=OracleBuilder())
    manifest = manifest_file(tmp_path, [json_dump(tmp_path, 0, payload())])
    monkeypatch.setattr(sys, "argv", ["analyze_nccl_flights", str(manifest)])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(Path(flights.__file__)), run_name="__main__")
    assert stopped.value.code == 2, "the script entry point exits like main()"
    assert len(requested) == 3, requested
    assert json.loads(capsys.readouterr().out)["version"] == "2.11.0", (
        "a newer analyzer is reported unavailable with its version"
    )


# --------------------------------------------------------------------------- #
# verify_fr_analyzer
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kind, inputs, outputs",
    [
        ("all_reduce", [[4]], [[4]]),
        ("all_gather", [[4]], [[8]]),
        ("_all_gather_base", [[4]], [[8]]),
        ("all_gather_into_tensor_coalesced", [[4]], [[8]]),
        ("reduce_scatter", [[8]], [[4]]),
        ("_reduce_scatter_base", [[8]], [[4]]),
        ("reduce_scatter_tensor_coalesced", [[8]], [[4]]),
        ("coalesced", [[4]], [[4]]),
    ],
)
def test_acceptance_entries_are_shaped_for_a_two_rank_group(
    kind: str, inputs: list, outputs: list
) -> None:
    operation = acceptance.entry(kind)
    assert (operation["input_sizes"], operation["output_sizes"]) == (inputs, outputs), (
        "gathers double the output, scatters the input, for two ranks"
    )
    assert operation["profiling_name"] == f"nccl:{kind}", operation
    assert (operation["state"], operation["process_group"]) == (
        "completed",
        ["0", "default_pg"],
    ), operation


def test_acceptance_refuses_anything_but_the_real_2_10_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = FakeEngine(agreeing)
    install_torch(monkeypatch, version="2.9.0", engine=engine)
    with pytest.raises(ValueError, match="requires the real PyTorch 2.10 engine"):
        acceptance.verify(tmp_path)
    assert engine.matches == [], "nothing is compared against another engine"


def test_acceptance_compares_every_kind_and_mutation_and_both_builder_verdicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, builder = FakeEngine(agreeing), OracleBuilder()
    install_torch(monkeypatch, builder=builder, engine=engine)
    report = acceptance.verify(tmp_path)
    assert report == {
        "status": "PASS",
        "torch_version": TORCH_VERSION,
        "native_matcher_cases": len(MATCHER_KINDS) * len(MUTATIONS),
        "official_builder_cases": 2 * len(BUILDER_KINDS),
        "real_nccl_execution": False,
        "physical_acceptance": "NOT_RUN",
        "pp_in_scope": False,
    }, report
    seen = [
        (match["left"]["profiling_name"].split(":", 1)[1], match)
        for match in engine.matches
    ]
    assert [kind for kind, _ in seen] == [
        kind for kind in MATCHER_KINDS for _ in MUTATIONS
    ], "every collective kind is matched once per mutation, in sorted order"
    assert all(
        match["pg_name"] == "group" and match["size"] == 2 for _, match in seen
    ), "the native engine compares within one two-rank group"
    changed = [
        sorted(
            key for key in match["left"] if match["left"][key] != match["right"][key]
        )
        for _, match in seen[: len(MUTATIONS)]
    ]
    assert changed == [
        [],
        ["input_sizes"],
        ["output_sizes"],
        ["input_dtypes", "output_dtypes"],
        ["state"],
        ["profiling_name"],
    ], "each mutation changes exactly its field on the right-hand entry"
    assert [len(call["details"]) for call in builder.calls] == [2] * 10, (
        "every builder case is a two-rank dump set"
    )
    names = [
        [rank["entries"][0]["profiling_name"] for rank in call["details"].values()]
        for call in builder.calls
    ]
    assert names[:2] == [
        ["nccl:all_reduce", "nccl:all_reduce"],
        ["nccl:all_reduce", "nccl:reduce"],
    ], "the mismatching case changes rank 1's operation only"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["0.dump", "1.dump"], (
        "the dumps are written under the given root"
    )


def test_acceptance_names_the_first_case_the_engines_disagree_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def blind_to_state(left: dict, right: dict, size: int) -> str:
        if (
            left["profiling_name"] == "nccl:all_reduce"
            and right["state"] != "completed"
        ):
            return "FULLY_MATCHED"
        return match_collective(left, right, size)

    builder = OracleBuilder()
    install_torch(monkeypatch, builder=builder, engine=FakeEngine(blind_to_state))
    with pytest.raises(
        AssertionError,
        match="matcher differs for all_reduce/state: FULLY_MATCHED != "
        "COLLECTIVE_STATE_MISMATCH",
    ):
        acceptance.verify(tmp_path)
    assert builder.calls == [], "the builder phase never runs after a disagreement"


@pytest.mark.parametrize(
    "distribution, collectives, message",
    [
        (TORCH_VERSION, [], "official builder did not analyze all_reduce"),
        ("2.9.0", None, "official builder did not analyze all_reduce"),
        (
            TORCH_VERSION,
            [collective(1, "nccl:all_reduce", pass_check=True)],
            "wrong official builder verdict for all_reduce/True",
        ),
    ],
    ids=["no-collectives", "distribution-not-2.10", "never-flags"],
)
def test_acceptance_fails_on_a_builder_that_misses_or_misjudges_a_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    distribution: str,
    collectives: list | None,
    message: str,
) -> None:
    # ``distribution-not-2.10``: the torch module says 2.10 but the installed
    # distribution does not, so the adapter answers UNAVAILABLE -- no analysis.
    install_torch(
        monkeypatch,
        version=distribution,
        module_version=TORCH_VERSION,
        builder=OracleBuilder(collectives),
        engine=FakeEngine(agreeing),
    )
    with pytest.raises(AssertionError, match=message):
        acceptance.verify(tmp_path)


@pytest.mark.parametrize("report", [True, False])
def test_acceptance_cli_prints_and_optionally_writes_the_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    report: bool,
) -> None:
    install_torch(monkeypatch, builder=OracleBuilder(), engine=FakeEngine(agreeing))
    target = tmp_path / "report.json"
    argv = ["verify_fr_analyzer", *(["--report", str(target)] if report else [])]
    monkeypatch.setattr(sys, "argv", argv)
    acceptance.main()
    printed = capsys.readouterr().out
    assert json.loads(printed)["status"] == "PASS", printed
    if report:
        assert target.read_text() == printed, "the report file holds the printed JSON"
    else:
        assert not target.exists(), "no report file without --report"


def test_acceptance_module_entrypoint_runs_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    install_torch(monkeypatch, builder=OracleBuilder(), engine=FakeEngine(agreeing))
    target = tmp_path / "entry.json"
    monkeypatch.setattr(sys, "argv", ["verify_fr_analyzer", "--report", str(target)])
    runpy.run_path(str(Path(acceptance.__file__)), run_name="__main__")
    printed = json.loads(capsys.readouterr().out)
    assert printed == json.loads(target.read_text()), "printed and written agree"
    assert printed["native_matcher_cases"] == len(MATCHER_KINDS) * len(MUTATIONS), (
        "the script entry point runs the whole acceptance"
    )
