"""Independent before/after proof for approved schema changes and keep reinstalls."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.bootstrap_common import CommandRunner
from gpu_fault.admin.installation_lifecycle import (
    INSTALLATION_FILE,
    completed_retained_uninstall,
    read_record,
    retained_database_identity,
)
from gpu_fault.admin.site import load_site
from gpu_fault.schema_migrations import POSTGRES_SCHEMA_MIGRATIONS
from scripts.e2e.regional.boot_acceptance_common import SiteFixture

# The baseline CPU release need not contain newly added deploy-host helpers.
STORE_PROBE = r"""
import hashlib, json, os
from pathlib import Path
import psycopg

parameters = json.loads(INPUT)
path = os.getenv("GPU_FAULT_STORE_URL_FILE", "").strip()
try:
    dsn = (
        Path(path).read_text(encoding="utf-8").strip()
        if path
        else os.getenv("GPU_FAULT_STORE_URL", "").strip()
    )
except (OSError, UnicodeError):
    raise RuntimeError("current database credentials unavailable") from None
if not dsn:
    raise RuntimeError("current database credentials unavailable")
with psycopg.connect(dsn, autocommit=True, connect_timeout=10,
    options="-c default_transaction_read_only=on -c statement_timeout=30000 -c lock_timeout=5000") as connection:
    with connection.transaction():
        connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        if connection.execute("SELECT pg_is_in_recovery()").fetchone()[0]:
            raise RuntimeError("lifecycle evidence requires the writer")
        version = connection.execute("SELECT version FROM gpu_fault_schema_version WHERE singleton").fetchone()[0]
        history = connection.execute("SELECT version,name,checksum FROM gpu_fault_schema_migrations ORDER BY version").fetchall()
        if parameters["keys"]:
            rows = connection.execute(
                "SELECT kind,key,payload FROM gpu_fault_control_records WHERE kind=ANY(%s) AND key=ANY(%s)",
                (["workflow","remote_command"], parameters["keys"])).fetchall()
        else:
            rows = connection.execute(
                "SELECT kind,key,payload FROM gpu_fault_control_records WHERE kind=ANY(%s) "
                "AND payload->>'status'=ANY(%s) ORDER BY kind,key LIMIT 32",
                (["workflow","remote_command"], ["SUCCEEDED","FAILED","SUPERSEDED","CANCELLED","TIMED_OUT"])).fetchall()
        hashes = {kind + "/" + key: hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            for kind,key,payload in rows}
print(json.dumps({"schema_version": version, "migrations": history, "records": hashes}))
"""


def observe(state_dir: Path, *, keys: list[str] | None = None) -> dict[str, Any]:
    site = load_site(state_dir / "site.yaml")
    clusters = site.release_config["clusters"]
    if not clusters:
        raise ValueError("lifecycle witness needs an installed baseline GPU")
    fixture = SiteFixture(site.source, clusters[0]["cluster_id"])
    identity = fixture.regional.evidence_identity()
    script = "INPUT = " + repr(json.dumps({"keys": keys or []})) + "\n" + STORE_PROBE
    store = fixture.regional.cpu_python(script)
    state_map = json.loads(
        fixture.regional.kubectl(
            "cpu", "get", "configmap", "gpu-fault-regional-release-state", "-o", "json"
        )
    )
    raw_release = json.loads(state_map["data"]["state.json"])
    release = {
        key: raw_release[key]
        for key in (
            "release_id",
            "phase",
            "transaction_committed",
            "completed_phases",
            "schema_change_acceptance",
            "database_schema_version",
            "rollback_completed",
            "release_lifecycle",
            "retained_database_origin",
            "bootstrap_database_origin",
            "bootstrap_store_safety",
        )
        if key in raw_release
    }
    jobs_record = (raw_release.get("aurora_prerequisite_repair") or {}).get(
        "jobs"
    ) or {}
    if not jobs_record:
        jobs_record = (raw_release.get("bootstrap_store_safety") or {}).get(
            "proof_jobs"
        ) or {}
    release["aurora_prerequisite_repair"] = {
        "jobs": {
            name: {
                key: value.get(key)
                for key in (
                    "name",
                    "uid",
                    "owner_uid",
                    "run_id",
                    "spec_sha256",
                    "status",
                )
            }
            for name, value in jobs_record.items()
        }
    }
    namespace = json.loads(
        fixture.regional.kubectl(
            "cpu",
            "get",
            "namespace",
            str(site.release_config["namespace"]),
            "-o",
            "json",
        )
    )
    jobs = json.loads(fixture.regional.kubectl("cpu", "get", "jobs", "-o", "json"))
    installation = (
        read_record(state_dir / INSTALLATION_FILE)
        if (state_dir / INSTALLATION_FILE).exists()
        else {}
    )
    return {
        "schema_version": 1,
        "report_type": "boot-store-lifecycle-observation",
        "observed_at": datetime.now(UTC).isoformat(),
        "state_dir_sha256": hashlib.sha256(
            str(state_dir.resolve()).encode()
        ).hexdigest(),
        "cpu_eks_arn": site.release_config["cpu_eks_arn"],
        "namespace_uid": namespace["metadata"]["uid"],
        "identity": identity,
        "database": retained_database_identity(site, CommandRunner()),
        "store": store,
        "release_state": release,
        "installation": installation,
        "live_jobs": [
            {"name": item["metadata"]["name"], "uid": item["metadata"]["uid"]}
            for item in jobs["items"]
        ],
    }


def transition_errors(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    mode: str,
    uninstall: dict[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    started = datetime.fromisoformat(before["observed_at"])
    finished = datetime.fromisoformat(after["observed_at"])
    if started.tzinfo is None or finished.tzinfo is None or finished < started:
        errors.append("lifecycle observation timestamps are unordered or unbound")
    if (
        before.get("report_type") != "boot-store-lifecycle-observation"
        or after.get("report_type") != before.get("report_type")
        or before.get("state_dir_sha256") != after.get("state_dir_sha256")
        or before.get("cpu_eks_arn") != after.get("cpu_eks_arn")
        or before.get("database") != after.get("database")
    ):
        errors.append("state directory, CPU or Aurora incarnation changed")
    old, new = before["store"], after["store"]
    for report in (old, new):
        version = report["schema_version"]
        if (
            type(version) is not int
            or not 1 <= version <= POSTGRES_SCHEMA_MIGRATIONS[-1].version
            or report["migrations"]
            != [
                [item.version, item.name, item.checksum]
                for item in POSTGRES_SCHEMA_MIGRATIONS
                if item.version <= version
            ]
        ):
            errors.append("schema version or complete migration identity is unverified")
    if any(new["records"].get(key) != value for key, value in old["records"].items()):
        errors.append("retained workflow/command history changed or disappeared")
    if new["migrations"][: len(old["migrations"])] != old["migrations"]:
        errors.append("historical schema migrations changed")
    release = after["release_state"]
    if mode in {"schema", "fail-forward"}:
        if before["namespace_uid"] != after["namespace_uid"]:
            errors.append("schema upgrade replaced the original namespace")
        acceptance = release.get("schema_change_acceptance") or {}
        if (
            type(old["schema_version"]) is not int
            or type(new["schema_version"]) is not int
            or new["schema_version"] <= old["schema_version"]
            or "schema-ready" not in release.get("completed_phases", [])
            or acceptance.get("database_schema_version") != new["schema_version"]
            or acceptance.get("mode") != "snapshot"
            or acceptance.get("snapshot_status") != "available"
            or not acceptance.get("snapshot_id")
        ):
            errors.append("real schema advance lacks the approved pre-schema snapshot")
        if mode == "fail-forward":
            if (
                release.get("phase") not in {"failed", "partial-convergence"}
                or release.get("transaction_committed") is True
            ):
                errors.append("schema failure was not retained for fail-forward")
            if release.get("rollback_completed") or str(
                release.get("release_lifecycle", "")
            ).startswith("ROLLED_BACK"):
                errors.append("schema failure attempted a runtime rollback")
        elif (
            release.get("phase") != "complete"
            or release.get("transaction_committed") is not True
        ):
            errors.append("schema candidate did not commit")
    elif mode == "reinstall":
        origin = release.get("retained_database_origin") or {}
        installation = after.get("installation") or {}
        handoff = installation.get("retained_uninstall") or {}
        if (
            uninstall is None
            or uninstall.get("phase") != "COMPLETED"
            or uninstall.get("reset_database") is not False
            or uninstall.get("cpu_disposition") != "keep"
            or uninstall.get("retained_database") != before.get("database")
            or handoff.get("previous_installation_id")
            != uninstall.get("installation_id")
            or installation.get("installation_id") == uninstall.get("installation_id")
            or before.get("namespace_uid") == after.get("namespace_uid")
            or origin.get("database_state") != "initialized"
            or origin.get("safe") is not True
            or "bootstrap_database_origin" in release
            or (origin.get("retained_database_handoff") or {}).get(
                "previous_installation_id"
            )
            != uninstall.get("installation_id")
            or release.get("phase") != "complete"
            or release.get("transaction_committed") is not True
        ):
            errors.append(
                "same-state keep reinstall lacks a fresh retained-database handoff"
            )
        proof = release.get("bootstrap_store_safety") or {}
        if (
            proof.get("safe") is not True
            or not (
                proof.get("schema_version") == new["schema_version"]
                or (proof.get("schema_version"), new["schema_version"]) == (17, 18)
                and proof.get("schema_ensure_required") is True
            )
            or release.get("database_schema_version") != new["schema_version"]
        ):
            errors.append(
                "new installation lacks safe pre-ensure proof and matching candidate schema"
            )
        if new["schema_version"] < old["schema_version"]:
            errors.append("retained schema moved backwards")
        if old["schema_version"] != new["schema_version"] and (
            (old["schema_version"], new["schema_version"]) != (17, 18)
            or origin.get("schema_ensure_required") is not True
        ):
            errors.append(
                "retained schema upgrade was not the reviewed v17-to-v18 transition"
            )
    else:
        raise ValueError("unknown lifecycle proof mode")
    records = (release.get("aurora_prerequisite_repair") or {}).get("jobs") or {}
    if mode == "reinstall" and not records:
        errors.append("reinstall lacks proof Job lifecycle records")
    live = {item["name"] for item in after["live_jobs"]}
    if any(
        not item.get("uid")
        or not item.get("owner_uid")
        or item.get("status") != "REMOVED"
        or item.get("name") in live
        for item in records.values()
    ):
        errors.append("proof Job ownership or explicit cleanup is incomplete")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("capture", "verify"))
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--before", type=Path)
    parser.add_argument("--mode", choices=("schema", "fail-forward", "reinstall"))
    args = parser.parse_args()
    if args.action == "capture":
        if args.output.exists() or args.output.is_symlink():
            parser.error(
                "before observation already exists; do not replace an old witness"
            )
        write_json_atomic(args.output, observe(args.state_dir))
        return
    if args.before is None or args.mode is None:
        parser.error("verify requires before and mode")
    if args.before.resolve() == args.output.resolve():
        parser.error("verification output must not overwrite its before observation")
    before = json.loads(args.before.read_text(encoding="utf-8"))
    after = observe(
        args.state_dir,
        keys=[key.split("/", 1)[1] for key in before["store"]["records"]],
    )
    uninstall = None
    if args.mode == "reinstall":
        installation = after["installation"]
        archive = (installation.get("retained_uninstall") or {}).get("archive")
        if not isinstance(archive, str) or Path(archive).name != archive:
            raise ValueError("retained installation archive is missing")
        uninstall = completed_retained_uninstall(args.state_dir / archive / "uninstall")
    errors = transition_errors(before, after, mode=args.mode, uninstall=uninstall)
    report = {
        "schema_version": 1,
        "report_type": "boot-store-lifecycle-proof",
        "mode": args.mode,
        "verdict": "FAIL" if errors else "PASS",
        "errors": errors,
        "before_sha256": hashlib.sha256(args.before.read_bytes()).hexdigest(),
        "after": after,
        "historical_records_checked": len(before["store"]["records"]),
        "scope": "observed lifecycle transition only; not the whole regional case",
    }
    write_json_atomic(args.output, report)
    print(json.dumps({"verdict": report["verdict"], "errors": errors}))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
