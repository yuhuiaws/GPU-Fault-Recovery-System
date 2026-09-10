from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import psycopg
from psycopg import sql

from gpu_fault.gpu_metrics import (
    GpuHealthFinding,
    GpuHealthSeverity,
)
from gpu_fault.models import Environment
from gpu_fault.store import PostgresStore
from gpu_fault.watcher import AttemptObservation, WorkloadPhase


def store_url_for_schema(url: str, schema: str) -> str:
    parsed = urlsplit(url)
    query = parse_qsl(parsed.query, keep_blank_values=True)
    query.append(("options", f"-csearch_path={schema}"))
    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            urlencode(query),
            parsed.fragment,
        )
    )


def finding(finding_id: str, observed_at: datetime) -> GpuHealthFinding:
    return GpuHealthFinding(
        finding_id=finding_id,
        cluster_id="cluster-a",
        node_id="node-a",
        observed_at=observed_at,
        severity=GpuHealthSeverity.WARNING,
        reason=finding_id,
        canonical_name="gpu_temperature_c",
        value=86,
    )


def store_dsn() -> str:
    path = (
        os.environ.get("GPU_FAULT_STORE_URL_FILE")
        or "/etc/gpu-fault/aurora/postgres-url"
    )
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return os.environ["GPU_FAULT_STORE_URL"]


def main() -> None:
    url = store_dsn()
    schema = f"gpu_fault_hot_state_guard_{uuid4().hex[:12]}"
    with psycopg.connect(url, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    isolated_url = store_url_for_schema(url, schema)
    now = datetime.now(timezone.utc)
    try:
        legacy = PostgresStore(isolated_url, hot_state_mode="legacy")
        try:
            legacy.save_attempt_observation(
                AttemptObservation(
                    cluster_id="cluster-a",
                    environment=Environment.HYPERPOD_EKS,
                    job_id="job-a",
                    attempt_id="attempt-a",
                    workload_phase=WorkloadPhase.RUNNING,
                    observed_at=now,
                    expected_critical_ranks=1,
                    runtime_profile_version="hyperpod-v1",
                )
            )
        finally:
            legacy.close()

        guard_error = None
        try:
            PostgresStore(isolated_url)
        except RuntimeError as exc:
            guard_error = str(exc)
        if not guard_error or "backfill is incomplete" not in guard_error:
            raise AssertionError(
                "dedicated mode accepted an incomplete legacy backfill"
            )

        dual = PostgresStore(isolated_url, hot_state_mode="dual")
        try:
            backfill = dual.backfill_hot_state_tables()
            status = dual.hot_state_migration_status()
        finally:
            dual.close()
        if any(item["missing_or_mismatched"] for item in status.values()):
            raise AssertionError(status)

        dedicated = PostgresStore(isolated_url)
        try:
            old = finding("finding-old", now - timedelta(days=31))
            recent = finding("finding-recent", now - timedelta(days=1))
            dedicated.update_gpu_finding(
                ("cluster-a", "node-a", "old"),
                old,
                old.observed_at,
            )
            dedicated.update_gpu_finding(
                ("cluster-a", "node-a", "recent"),
                recent,
                recent.observed_at,
            )
            deleted = dedicated.cleanup_hot_state(
                now=now,
                finding_history_retention=timedelta(days=30),
                limit=100,
            )
            history = dedicated.list_gpu_findings(
                "cluster-a", "node-a", active_only=False
            )
            active = dedicated.list_gpu_findings(
                "cluster-a", "node-a", active_only=True
            )
        finally:
            dedicated.close()

        if deleted["gpu_finding_history"] != 1:
            raise AssertionError(deleted)
        if [item.finding_id for item in history] != ["finding-recent"]:
            raise AssertionError(history)
        if {item.finding_id for item in active} != {
            "finding-old",
            "finding-recent",
        }:
            raise AssertionError(active)
        print(
            json.dumps(
                {
                    "guard_error": guard_error,
                    "backfill": backfill,
                    "status": status,
                    "finding_history_deleted": (deleted["gpu_finding_history"]),
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        with psycopg.connect(url, autocommit=True) as connection:
            connection.execute(
                sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema))
            )


if __name__ == "__main__":
    main()
