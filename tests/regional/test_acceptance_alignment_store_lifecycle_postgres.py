"""Actual projected-DSN schema CLI and same-state retained bootstrap, on owned PG."""

from __future__ import annotations

import os
import sys
from contextlib import closing

import pytest

from gpu_fault import store_migrate
from gpu_fault_release import regional_release_prerequisite_repair as repair
from gpu_fault_release import rollout
from gpu_fault_release.regional_release_store_probe import inspect_database
from tests.admin.test_admin_reinstall_handoff import retained_release
from tests.regional.test_retained_store_schema_postgres import (
    database as database_fixture,
)
from tests.regional.test_retained_store_schema_postgres import (
    install_v17,
    migration_history,
    retained_rows,
    seed_records,
)

database = database_fixture

pytestmark = [
    pytest.mark.skipif(
        not os.getenv("GPU_FAULT_TEST_POSTGRES_URL"),
        reason="requires parent's explicitly owned serial local PostgreSQL grant",
    ),
    pytest.mark.allows_cluster_binaries("docker"),
]


def test_real_schema_cli_uses_projected_dsn_and_preserves_v17_history(
    database, tmp_path, monkeypatch
):
    connection = database.connection
    seed_records(connection)
    install_v17(connection)
    before, history = retained_rows(connection), migration_history(connection)
    dsn = tmp_path / "projected-dsn"
    dsn.write_text(connection.info.dsn)
    dsn.chmod(0o600)
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", str(dsn))
    monkeypatch.setenv(
        "GPU_FAULT_STORE_URL", "postgresql://stale.invalid/not-the-grant"
    )
    monkeypatch.setattr(sys, "argv", ["store-migrate", "--ensure-schema"])
    assert inspect_database(connection, 18, allow_schema_upgrade=True)[
        "schema_ensure_required"
    ]
    store_migrate.main()
    assert migration_history(connection)[:-1] == history
    assert retained_rows(connection) == before
    assert inspect_database(connection, 18)["schema_ensure_required"] is False
    monkeypatch.setattr(
        sys,
        "argv",
        ["store-migrate", "--state-table-status", "--state-table-kind", "workflow"],
    )
    store_migrate.main()


def test_same_state_keep_handoff_runs_native_ensure_before_any_business_cpu(
    database, tmp_path, monkeypatch
):
    connection = database.connection
    seed_records(connection)
    install_v17(connection)
    before = retained_rows(connection)
    _harness, _archive, instance = retained_release(tmp_path, monkeypatch)
    instance.runner.proof_schema_version = 17
    events = instance.runner.events
    original = instance.runner.run

    def run(arguments, **options):
        if "apply" in arguments and "Namespace" in str(options.get("input_text")):
            events.append("namespace-prerequisites")
            return ""
        value = original(arguments, **options)
        if any(str(item).endswith("/wait-for-kubernetes-job.sh") for item in arguments):
            name = arguments[3]
            if name in instance.runner.outputs:
                import json

                output = json.loads(instance.runner.outputs[name])
                output.update(
                    inspect_database(connection, 18, allow_schema_upgrade=True)
                )
                instance.runner.outputs[name] = json.dumps(output)
        return value

    monkeypatch.setattr(instance.runner, "run", run)
    monkeypatch.setattr(rollout, "ensure_runtime_profile", lambda *_args: None)
    monkeypatch.setattr(rollout, "bootstrap_gpu_clusters", lambda *_args: None)
    for name in (
        "_initialize_registry",
        "_prepare_nlb",
        "_wait_nlb",
        "_validate_release",
    ):
        monkeypatch.setattr(instance, name, lambda: None, raising=False)
    monkeypatch.setattr(instance, "_require_cpu_secrets", lambda **_kwargs: None)
    monkeypatch.setattr(instance, "_upload_release", lambda _diff=None: None)
    monkeypatch.setattr(
        instance,
        "_prepare_bootstrap_workflows",
        lambda: repair.prepare_bootstrap_workflows(instance),
        raising=False,
    )

    def ensure():
        assert instance.state["bootstrap_store_safety"]["schema_version"] == 17
        with closing(database.open_store(True)):
            pass
        instance.runner.proof_schema_version = 18
        events.append("native-ensure-v18")

    def cpu(**_kwargs):
        assert inspect_database(connection, 18)["safe"]
        assert retained_rows(connection) == before
        events.append("business-cpu")

    monkeypatch.setattr(instance, "_ensure_schema", ensure)
    monkeypatch.setattr(instance, "_apply_cpu", cpu)
    rollout.RegionalRelease.bootstrap(instance)
    assert (
        events.index("store-job")
        < events.index("native-ensure-v18")
        < events.index("business-cpu")
    )
    assert instance.state["retained_database_origin"]["schema_version"] == 17
    assert instance.state["bootstrap_store_safety"]["schema_version"] == 17
    assert all(
        item["status"] == "REMOVED" and item["uid"] and item["owner_uid"]
        for item in instance.state["bootstrap_store_safety"]["proof_jobs"].values()
    ), "schema bootstrap proof receipts must identify removed Jobs and their owners"
    assert repair.BOOTSTRAP_ORIGIN_KEY not in instance.state
    assert not instance.runner.jobs, (
        "schema bootstrap must not leave a proof Job running"
    )
