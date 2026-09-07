"""The legacy single-cluster deploy attaches a diagnostics parameter group.

Store review 2026-09-07, item K1. ``ensure_aurora`` created the Serverless v2
cluster on the engine-default parameter group, so ``log_lock_waits`` was off
(the deadlocks the store retries after 40P01 were invisible) and
``pg_stat_statements`` was not loaded (no per-statement view of ACU spend).
``deploy/hyperpod/deploy.sh`` has no runtime caller in the regional
architecture; this pins parity, not deployment.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _ensure_aurora() -> str:
    script = (ROOT / "deploy/hyperpod/deploy.sh").read_text(encoding="utf-8")
    start = script.index("ensure_aurora() {")
    end = script.index("\n}\n", start)
    return script[start:end]


def test_the_parameter_group_is_created_from_the_engine_family() -> None:
    body = _ensure_aurora()

    assert 'parameter_group="${AURORA_CLUSTER_ID}-pg"' in body
    assert "describe-db-cluster-parameter-groups" in body
    assert "--query 'DBEngineVersions[0].DBParameterGroupFamily'" in body, (
        "the family must follow AURORA_ENGINE_VERSION, not a hard-coded string"
    )
    assert "create-db-cluster-parameter-group" in body


def test_the_diagnostic_parameters_and_apply_methods_are_pinned() -> None:
    body = _ensure_aurora()

    assert "ParameterName=log_lock_waits,ParameterValue=1,ApplyMethod=immediate" in body
    assert (
        "ParameterName=shared_preload_libraries,ParameterValue=pg_stat_statements,"
        "ApplyMethod=pending-reboot"
    ) in body
    assert (
        "ParameterName=pg_stat_statements.track,ParameterValue=all,ApplyMethod=immediate"
        in body
    )
    assert (
        "ParameterName=log_min_duration_statement,ParameterValue=1000,"
        "ApplyMethod=immediate"
    ) in body


def test_the_group_is_attached_on_create_and_by_a_guarded_modify() -> None:
    body = _ensure_aurora()

    create = body.index("aws rds create-db-cluster \\")
    create_end = body.index("--region", body.index("--tags", create))
    assert (
        '--db-cluster-parameter-group-name "${parameter_group}"'
        in body[create:create_end]
    )
    assert "--query 'DBClusters[0].DBClusterParameterGroup'" in body
    guard = body.index('[[ "${current_parameter_group}" != "${parameter_group}" ]]')
    modify = body.index("aws rds modify-db-cluster \\", guard)
    # ``fi`` alone would match inside ``--db-cluster-identifier``.
    assert body.index("--apply-immediately", modify) < body.index(
        "\n        fi\n", modify
    )
    # The parameter group exists before the cluster that references it.
    assert body.index("modify-db-cluster-parameter-group") < create


def test_the_extension_itself_is_left_to_the_migration_tool() -> None:
    body = _ensure_aurora()

    statements = [
        line
        for line in body.splitlines()
        if "CREATE EXTENSION" in line and not line.lstrip().startswith("#")
    ]
    assert statements == [], statements
    assert "gpu-fault-store-migrate --ensure-diagnostics" in body
