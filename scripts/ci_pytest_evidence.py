"""Complete CI receipts, content-bound selection and lossless merge provenance."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

from scripts.ci_coverage_config import (
    RUNTIME_SHARDS,
    SHARDS,
    parse_pytest_workers,
    pytest_targets,
)
from scripts.ci_gate_artifacts import CoverageGateError, sha256
from tools.pytest_result_identity import (
    PytestReceipt,
    parse_pytest_receipt,
    partition_for_nodeid,
    source_identity,
)

AGGREGATE_SCHEMA_VERSION = 2
IdentityResolver = Callable[[Path, str, Mapping[str, Any]], Mapping[str, Any]]


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def ci_context(identity: Mapping[str, Any], suite: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "shard": identity["shard"],
        "identity_sha256": identity["sha256"],
        "suite": suite,
    }


def _session_times(session: Mapping[str, Any]) -> None:
    try:
        started = datetime.fromisoformat(session["started_at"])
        finished = datetime.fromisoformat(session["finished_at"])
        valid = (
            started.tzinfo is not None
            and finished.tzinfo is not None
            and started <= finished
        )
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("pytest session time range is invalid")


def _validate_workers(
    selection: Mapping[str, Any], protocol: Mapping[str, Any], shard: str
) -> None:
    try:
        expected = parse_pytest_workers(protocol.get("pytest_workers"))
    except CoverageGateError as exc:
        raise ValueError(str(exc)) from exc
    actual = selection.get("numprocesses")
    # Legacy numeric receipts name the count; automatic modes need their request.
    requested = selection.get("requested_numprocesses", actual)
    if shard == "postgres":
        if (
            type(actual) is not int
            or actual != 0
            or type(requested) is not int
            or requested != 0
        ):
            raise ValueError("PostgreSQL pytest must run serially")
    elif (
        type(actual) is not int
        or actual < 0
        or (
            isinstance(expected, int)
            and (
                actual != expected
                or type(requested) is not int
                or requested != expected
            )
        )
        or (
            isinstance(expected, str)
            and (type(requested) is not str or requested != expected or actual == 0)
        )
    ):
        raise ValueError("pytest workers do not match the shard content identity")


def validate_shard_receipt(
    value: object,
    *,
    root: Path,
    identity: Mapping[str, Any],
    suite: str = "pytest",
) -> PytestReceipt:
    if not isinstance(value, dict) or not is_sha256(value.get("source_identity")):
        raise ValueError("pytest receipt source provenance is invalid")
    receipt = parse_pytest_receipt(
        value,
        root=root,
        expected_identity=value["source_identity"],
        require_session=True,
        require_passed=True,
    )
    session = value["session"]
    _session_times(session)
    if session.get("collection_errors") != [] or session.get("collection_skips") != []:
        raise ValueError("pytest requires explicit clean collection without skips")
    if session.get("ci_context") != ci_context(identity, suite):
        raise ValueError("pytest receipt does not match its shard content identity")
    shard = identity["shard"]
    targets = list(pytest_targets(root, shard))
    selection = session.get("selection")
    partition = (
        [len(RUNTIME_SHARDS), RUNTIME_SHARDS.index(shard)]
        if shard in RUNTIME_SHARDS
        else None
    )
    if (
        not isinstance(selection, dict)
        or selection.get("targets") != targets
        or selection.get("partition") != partition
        or selection.get("keyword") != ""
        or selection.get("markexpr") != ""
        or selection.get("deselect") != []
        or session.get("collected_files") != targets
    ):
        raise ValueError("pytest selection does not cover the expected shard tests")
    protocol = identity["protocol"]
    _validate_workers(selection, protocol, shard)
    stress = suite == "postgres_stress"
    if suite not in {"pytest", "postgres_stress"} or (stress and shard != "postgres"):
        raise ValueError("pytest receipt suite is invalid")
    for field in ("workers", "rounds"):
        expected = str(protocol[f"postgres_stress_{field}"]) if stress else ""
        if selection.get(f"stress_{field}") != expected:
            raise ValueError("pytest PostgreSQL stress parameters do not match")
    discovered = session["discovered_nodeids"]
    if any(nodeid.partition("::")[0] not in targets for nodeid in discovered):
        raise ValueError("pytest discovery contains tests outside the shard")
    expected_nodeids = {
        nodeid
        for nodeid in discovered
        if partition is None
        or partition_for_nodeid(nodeid, partition[0]) == partition[1]
    }
    if set(value["records"]) != expected_nodeids:
        raise ValueError("pytest did not execute the complete shard selection")
    for record in value["records"].values():
        duration = record.get("duration_seconds")
        if (
            type(duration) not in (int, float)
            or not math.isfinite(duration)
            or duration < 0
            or not isinstance(record.get("output"), str)
        ):
            raise ValueError("pytest result duration or diagnostics are invalid")
    return receipt


def validate_execution_provenance(execution: object, *, original_identity: str) -> None:
    if (
        not isinstance(execution, dict)
        or execution.get("source_identity") != original_identity
        or not isinstance(execution.get("producer"), dict)
    ):
        raise ValueError("pytest execution provenance is invalid")
    producer = execution["producer"]
    repository = producer.get("repository")
    if (
        not isinstance(repository, str)
        or not repository
        or not str(producer.get("git_commit") or "")
        or not str(producer.get("git_tree") or "")
        or not str(producer.get("run_id") or "").isdecimal()
        or not str(producer.get("workflow_ref") or "").startswith(
            f"{repository}/.github/workflows/ci.yml@"
        )
    ):
        raise ValueError("pytest original producer provenance is invalid")


def _load_receipt(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("pytest receipt must be an object")
    return value


def aggregate_pytest_results(
    root: Path,
    gates: Mapping[str, tuple[Path, dict[str, Any]]],
    *,
    resolve_identity: IdentityResolver,
) -> dict[str, Any]:
    if set(gates) != set(SHARDS):
        raise ValueError("pytest aggregate requires every coverage shard")
    records: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    for shard, (path, gate) in sorted(gates.items()):
        evidence = gate["evidence"]["pytest_results"]
        value = _load_receipt(path.parent / evidence["path"])
        validate_shard_receipt(value, root=root, identity=gate["identity"])
        overlap = records.keys() & value["records"].keys()
        if overlap:
            raise ValueError(
                "pytest tests occur in multiple shards: " + str(sorted(overlap))
            )
        records.update(value["records"])
        stress = gate["evidence"].get("postgres_stress_results")
        provenance[shard] = {
            "identity": gate["identity"],
            "gate_sha256": sha256(path),
            "producer": gate["producer"],
            "execution": gate["execution"],
            "reused_from": gate.get("reused_from"),
            "receipt_sha256": evidence["sha256"],
            "source_identity": value["source_identity"],
            "session": value["session"],
            "postgres_stress": (
                {
                    "receipt_sha256": stress["sha256"],
                    "receipt": _load_receipt(path.parent / stress["path"]),
                }
                if stress is not None
                else None
            ),
        }
    result: dict[str, Any] = {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "domain": "ci-pytest-aggregate",
        "validated_source_identity": source_identity(root),
        "shards": provenance,
        "records": dict(sorted(records.items())),
    }
    parse_aggregate_receipt(
        result,
        root=root,
        expected_identity=result["validated_source_identity"],
        resolve_identity=resolve_identity,
    )
    return result


def parse_aggregate_receipt(
    value: Mapping[str, Any],
    *,
    root: Path,
    expected_identity: str,
    resolve_identity: IdentityResolver,
) -> PytestReceipt:
    shards = value.get("shards")
    records = value.get("records")
    if (
        type(value.get("schema_version")) is not int
        or value["schema_version"] != AGGREGATE_SCHEMA_VERSION
        or value.get("domain") != "ci-pytest-aggregate"
        or value.get("validated_source_identity") != expected_identity
        or "source_identity" in value
        or "session" in value
        or not isinstance(shards, dict)
        or set(shards) != set(SHARDS)
        or not isinstance(records, dict)
        or not records
    ):
        raise ValueError("pytest aggregate identity or shard inventory is invalid")
    merged: dict[str, object] = {}
    discovered: set[str] = set()
    runtime_discovery: frozenset[str] | None = None
    try:
        for shard, provenance in shards.items():
            identity = provenance["identity"]
            current = resolve_identity(root, shard, identity)
            if identity != current:
                raise ValueError("pytest aggregate shard content identity has changed")
            if not is_sha256(provenance["gate_sha256"]) or not is_sha256(
                provenance["receipt_sha256"]
            ):
                raise ValueError("pytest aggregate evidence digest is invalid")
            validate_execution_provenance(
                provenance["execution"],
                original_identity=provenance["source_identity"],
            )
            session = provenance["session"]
            receipt_value = {
                "schema_version": 1,
                "source_identity": provenance["source_identity"],
                "session": session,
                "records": {
                    nodeid: records[nodeid] for nodeid in session["collected_nodeids"]
                },
            }
            receipt = validate_shard_receipt(
                receipt_value, root=root, identity=identity
            )
            if merged.keys() & receipt.records.keys():
                raise ValueError("pytest aggregate has overlapping shard selections")
            merged.update(receipt.records)
            assert receipt.discovered_nodeids is not None
            discovered.update(receipt.discovered_nodeids)
            if shard in RUNTIME_SHARDS:
                if (
                    runtime_discovery is not None
                    and runtime_discovery != receipt.discovered_nodeids
                ):
                    raise ValueError("runtime shards disagree on pytest discovery")
                runtime_discovery = receipt.discovered_nodeids
            stress = provenance["postgres_stress"]
            if shard == "postgres":
                if not is_sha256(stress["receipt_sha256"]):
                    raise ValueError("PostgreSQL stress evidence digest is invalid")
                stress_receipt = validate_shard_receipt(
                    stress["receipt"],
                    root=root,
                    identity=identity,
                    suite="postgres_stress",
                )
                if (
                    stress["receipt"]["source_identity"]
                    != provenance["source_identity"]
                    or stress_receipt.discovered_nodeids != receipt.discovered_nodeids
                ):
                    raise ValueError(
                        "PostgreSQL stress does not cover the original suite"
                    )
            elif stress is not None:
                raise ValueError("non-PostgreSQL shard contains stress evidence")
    except (KeyError, TypeError, AttributeError, CoverageGateError) as exc:
        raise ValueError("pytest aggregate provenance is incomplete") from exc
    if len(merged) != len(records) or set(merged) != discovered:
        raise ValueError("pytest aggregate does not cover complete discovery")
    return PytestReceipt(merged, frozenset(discovered))
