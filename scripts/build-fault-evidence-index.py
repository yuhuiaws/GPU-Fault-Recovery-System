from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped,unused-ignore]

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / "docs/evidence/fault"
MANIFEST = EVIDENCE_DIR / "manifest.yaml"
INDEX = EVIDENCE_DIR / "index.json"
CATALOG = ROOT / "testcases/fault-scenarios.yaml"
# Report verdict -> allowed catalog case evidence.verdict values.
REPORT_TO_CASE_VERDICTS = {
    "PASS": {"PASS"},
    "PASS_WITH_LIMITATIONS": {"PASS"},
    "FAIL": {"BLOCKED", "NOT_RUN"},
    "PARTIAL": {"PASS", "BLOCKED", "NOT_RUN"},
    "INVALID": {"BLOCKED", "NOT_RUN", "SUPERSEDED"},
}
VERDICTS = frozenset(REPORT_TO_CASE_VERDICTS)
TIME_FIELDS = (
    "executed_at",
    "executed_at_utc",
    "generated_at",
    "finished_at",
    "completed_at",
)
AWS_ACCOUNT = re.compile(r"arn:aws:[^:\s]+:[^:\s]*:\d{12}:")
S3_BUCKET = re.compile(r"s3://([^/\s\"']+)")


class EvidenceIndexError(ValueError):
    pass


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvidenceIndexError(f"{path} must contain a mapping")
    return value


def _validate_time(value: str, context: str) -> None:
    try:
        if "T" in value:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        else:
            date.fromisoformat(value)
    except ValueError as exc:
        raise EvidenceIndexError(f"{context} must be an ISO date or timestamp") from exc


def _raw_verdict(document: dict[str, Any]) -> Any:
    for field in ("verdict", "result", "status", "summary"):
        if field in document:
            return {"field": field, "value": document[field]}
    return None


def _raw_time(document: dict[str, Any]) -> dict[str, str] | None:
    for field in TIME_FIELDS:
        value = document.get(field)
        if isinstance(value, str):
            return {"field": field, "value": value}
    return None


def _catalog_report_cases() -> tuple[
    dict[str, list[str]],
    dict[str, str],
    set[str],
    dict[str, str],
]:
    """Return report links, owners, all ids and case evidence verdicts.

    The id set makes the manifest->catalog direction checkable.
    Only the catalog->manifest direction was enforced, so a ``case_ids`` entry
    naming a case the catalog does not have -- a retired id, a renamed id, a
    label that was never a case id -- read as curated coverage and printed
    into ``index.json`` as if a real case were backed by evidence.
    """
    catalog = _load_yaml(CATALOG)
    result: dict[str, list[str]] = defaultdict(list)
    owner: dict[str, str] = {}
    case_ids: set[str] = set()
    case_verdict: dict[str, str] = {}
    for case in catalog.get("test_cases", []):
        if not isinstance(case, dict):
            continue
        case_id = str(case["id"])
        case_ids.add(case_id)
        evidence = case.get("evidence")
        if isinstance(evidence, dict) and isinstance(evidence.get("verdict"), str):
            case_verdict[case_id] = evidence["verdict"]
        if not isinstance(evidence, dict) or "report" not in evidence:
            continue
        path = ROOT / str(evidence["report"])
        if not path.is_file():
            raise EvidenceIndexError(f"{case.get('id')} report does not exist: {path}")
        try:
            relative = path.relative_to(EVIDENCE_DIR)
        except ValueError as exc:
            raise EvidenceIndexError(
                f"{case.get('id')} report is outside {EVIDENCE_DIR.relative_to(ROOT)}"
            ) from exc
        result[relative.name].append(case_id)
        owner[case_id] = relative.name
    return result, owner, case_ids, case_verdict


def _pass_bindings() -> dict[str, dict[str, str] | None]:
    """Return every ``PASS`` case id mapped to its recorded component digests.

    ``None`` marks a case whose ``PASS`` predates the binding, so no digest was
    ever recorded. The distinction matters operationally: a stale binding says
    "this was true for other code", an absent one says "nobody knows".
    """
    catalog = _load_yaml(CATALOG)
    bindings: dict[str, dict[str, str] | None] = {}
    for case in catalog.get("test_cases", []):
        if not isinstance(case, dict):
            continue
        evidence = case.get("evidence")
        if not isinstance(evidence, dict) or evidence.get("verdict") != "PASS":
            continue
        verified = evidence.get("verified")
        components = verified.get("components") if isinstance(verified, dict) else None
        bindings[str(case["id"])] = components if isinstance(components, dict) else None
    return bindings


def live_component_digests() -> dict[str, str]:
    """Digest the component sources as they are in this checkout.

    Imported by path rather than as ``scripts.component_wheels`` so the
    documented ``python3 scripts/build-fault-evidence-index.py`` invocation keeps
    working from any working directory.
    """
    for candidate in (ROOT / "src", ROOT / "scripts"):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    from component_wheels import (  # noqa: PLC0415
        APPLICATION_COMPONENT_NAMES,
        component_source_digest,
    )

    return {name: component_source_digest(name) for name in APPLICATION_COMPONENT_NAMES}


def evidence_freshness(
    bindings: dict[str, dict[str, str] | None],
    live: dict[str, str],
) -> dict[str, list[str]]:
    """Classify every ``PASS`` verdict against the code now in the tree.

    ``STALE`` is derived here and never stored, so a verdict cannot be kept alive
    by typing a word into the catalog. It is also deliberately not a failure:
    these cases only run against real GPUs in a maintenance window, so blocking
    the build on every release would just pressure people into inventing
    evidence. Reporting is the useful half — an administrator planning the next
    live window learns which verdicts no longer cover the shipping code.
    """
    result: dict[str, list[str]] = {"fresh": [], "stale": [], "unbound": []}
    for case_id, components in sorted(bindings.items()):
        if components is None:
            result["unbound"].append(case_id)
        elif components == live:
            result["fresh"].append(case_id)
        else:
            result["stale"].append(case_id)
    return result


def _print_freshness() -> None:
    freshness = evidence_freshness(_pass_bindings(), live_component_digests())
    print(
        "PASS_EVIDENCE_FRESH= {fresh} STALE= {stale} UNBOUND= {unbound}".format(
            **{key: len(value) for key, value in freshness.items()}
        )
    )
    for state in ("stale", "unbound"):
        for case_id in freshness[state]:
            print(f"{state.upper()}: {case_id} must be re-verified in a live window")


def build_index() -> dict[str, Any]:
    manifest = _load_yaml(MANIFEST)
    if manifest.get("schema_version") != 1:
        raise EvidenceIndexError("manifest schema_version must be 1")
    entries = manifest.get("reports")
    if not isinstance(entries, list):
        raise EvidenceIndexError("manifest reports must be a list")

    files = {
        path.name for path in EVIDENCE_DIR.glob("*.json") if path.name != INDEX.name
    }
    declared = [str(item.get("file")) for item in entries if isinstance(item, dict)]
    if len(declared) != len(set(declared)):
        raise EvidenceIndexError("manifest contains duplicate report files")
    if files != set(declared):
        raise EvidenceIndexError(
            "manifest/file mismatch: "
            f"missing={sorted(files - set(declared))} "
            f"extra={sorted(set(declared) - files)}"
        )

    (
        catalog_cases,
        catalog_owner,
        catalog_ids,
        catalog_case_verdict,
    ) = _catalog_report_cases()
    output = []
    case_owner: dict[str, str] = {}
    verdict_counts: Counter[str] = Counter()
    for entry in entries:
        if not isinstance(entry, dict):
            raise EvidenceIndexError("manifest report entries must be mappings")
        name = str(entry["file"])
        path = EVIDENCE_DIR / name
        case_ids = entry.get("case_ids")
        legacy_labels = entry.get("legacy_case_labels", [])
        limitations = entry.get("limitations")
        verdict = entry.get("verdict")
        executed_at = entry.get("executed_at")
        if not isinstance(case_ids, list) or not all(
            isinstance(item, str) and item for item in case_ids
        ):
            raise EvidenceIndexError(f"{name} case_ids must be non-empty strings")
        if not isinstance(legacy_labels, list) or not all(
            isinstance(item, str) and item for item in legacy_labels
        ):
            raise EvidenceIndexError(
                f"{name} legacy_case_labels must be non-empty strings"
            )
        if not case_ids and not legacy_labels:
            raise EvidenceIndexError(
                f"{name} must declare case_ids or legacy_case_labels"
            )
        unknown = sorted(set(case_ids) - catalog_ids)
        if unknown:
            raise EvidenceIndexError(
                f"{name} case_ids are not in the catalog: {unknown}; "
                "rename them, or move retired identifiers to legacy_case_labels"
            )
        # legacy_case_labels 是给「本仓 catalog 从来没有/已经不再有这个编号」
        # 留的显式出口，所以它不许装真编号——否则它会变成绕过上面那条检查
        # 的后门，而这条检查正是为了不让 index.json 印出不存在的用例。
        catalogued_labels = sorted(set(legacy_labels) & catalog_ids)
        if catalogued_labels:
            raise EvidenceIndexError(
                f"{name} legacy_case_labels name live catalog cases: "
                f"{catalogued_labels}; put them in case_ids"
            )
        wrong_owner = sorted(
            f"{case_id}->{catalog_owner[case_id]}"
            for case_id in case_ids
            if case_id in catalog_owner and catalog_owner[case_id] != name
        )
        if wrong_owner:
            raise EvidenceIndexError(
                f"{name} claims cases whose catalog report is another file: "
                f"{wrong_owner}"
            )
        superseded = entry.get("superseded_by_case")
        if superseded is not None and superseded not in catalog_ids:
            raise EvidenceIndexError(
                f"{name} superseded_by_case is not a catalog case: {superseded!r}"
            )
        if verdict not in VERDICTS:
            raise EvidenceIndexError(f"{name} has invalid verdict {verdict!r}")
        allowed_case_verdicts = REPORT_TO_CASE_VERDICTS[verdict]
        conflicting = sorted(
            f"{case_id}={catalog_case_verdict.get(case_id)}"
            for case_id in case_ids
            if catalog_case_verdict.get(case_id) not in allowed_case_verdicts
        )
        if conflicting:
            raise EvidenceIndexError(
                f"{name} verdict {verdict!r} conflicts with catalog "
                f"evidence.verdict: {conflicting}; expected one of "
                f"{sorted(allowed_case_verdicts)}"
            )
        if not isinstance(executed_at, str):
            raise EvidenceIndexError(f"{name} executed_at must be a string")
        _validate_time(executed_at, f"{name} executed_at")
        if (
            not isinstance(limitations, list)
            or not limitations
            or not all(isinstance(item, str) and item for item in limitations)
        ):
            raise EvidenceIndexError(f"{name} limitations must be non-empty strings")
        for case_id in case_ids:
            previous = case_owner.setdefault(case_id, name)
            if previous != name:
                raise EvidenceIndexError(
                    f"case {case_id} is declared by both {previous} and {name}"
                )
        missing_cases = set(catalog_cases.get(name, ())) - set(case_ids)
        if missing_cases:
            raise EvidenceIndexError(
                f"{name} omits catalog cases: {sorted(missing_cases)}"
            )

        raw = path.read_text(encoding="utf-8")
        if AWS_ACCOUNT.search(raw):
            raise EvidenceIndexError(f"{name} contains an unredacted AWS account ARN")
        for bucket in S3_BUCKET.findall(raw):
            if bucket != "<EVIDENCE_BUCKET>":
                raise EvidenceIndexError(
                    f"{name} contains an unredacted S3 bucket: {bucket}"
                )
        document = json.loads(raw)
        if not isinstance(document, dict):
            raise EvidenceIndexError(f"{name} must contain a JSON object")
        raw_time = _raw_time(document)
        if raw_time is not None and raw_time["value"] != executed_at:
            raise EvidenceIndexError(
                f"{name} manifest executed_at {executed_at!r} "
                f"does not match raw {raw_time['field']} "
                f"{raw_time['value']!r}"
            )
        digest = hashlib.sha256(raw.encode()).hexdigest()
        verdict_counts[verdict] += 1
        item = {
            "file": name,
            "case_ids": case_ids,
            "verdict": verdict,
            "executed_at": executed_at,
            "limitations": limitations,
            "provenance": manifest["provenance"],
            "sha256": digest,
            "size_bytes": path.stat().st_size,
            "raw_schema_version": document.get("schema_version"),
            "raw_verdict": _raw_verdict(document),
            "raw_time": raw_time,
        }
        if legacy_labels:
            item["legacy_case_labels"] = legacy_labels
        if entry.get("superseded_by_case"):
            item["superseded_by_case"] = entry["superseded_by_case"]
        output.append(item)

    # Which PASS verdicts are bound to a code state at all. The digests
    # themselves stay out of the index: they would move with every source change
    # and turn `--check` into a permanent diff. Whether a binding exists is a
    # property of the catalog alone, so it is stable. Without this, the index
    # reports an empty public evidence set while the catalog carries ten PASS
    # verdicts, and nothing tells a reader those verdicts are unanchored.
    bindings = _pass_bindings()
    return {
        "schema_version": 1,
        "generated_from": "docs/evidence/fault/manifest.yaml",
        "curated_at": manifest["curated_at"],
        "provenance": manifest["provenance"],
        "provenance_note": manifest["provenance_note"],
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "pass_case_bindings": {
            "bound": sorted(
                case_id
                for case_id, components in bindings.items()
                if components is not None
            ),
            "unbound": sorted(
                case_id
                for case_id, components in bindings.items()
                if components is None
            ),
        },
        "reports": output,
    }


def render_index() -> str:
    return (
        json.dumps(
            build_index(),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the normalized curated fault-evidence index"
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = render_index()
    if args.check:
        current = INDEX.read_text(encoding="utf-8") if INDEX.exists() else ""
        if current != rendered:
            raise SystemExit(
                "fault evidence index is stale; run "
                "python3 scripts/build-fault-evidence-index.py"
            )
        print("Fault evidence index is current.")
        _print_freshness()
        return
    INDEX.write_text(rendered, encoding="utf-8")
    print(f"Wrote {INDEX.relative_to(ROOT)}")
    _print_freshness()


if __name__ == "__main__":
    main()
