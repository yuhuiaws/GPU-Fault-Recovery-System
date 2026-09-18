from __future__ import annotations

import base64
import copy
import json
import sys
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from gpu_fault import store
from scripts.e2e.regional import boot_guard_isolation as guard
from tests.regional.test_boot_guard_isolation import Database, pod_spec


class GuardDatabase(Database):
    def __init__(self, name, calls, *, exists=False, schema_present=True):
        super().__init__(name, calls, schema_present=schema_present)
        self.exists = exists

    def execute(self, statement, *args):
        result = super().execute(statement, *args)
        if isinstance(statement, str) and "FROM pg_database" in statement:
            self.row = (1,) if self.exists else None
        return result


def fake_database(monkeypatch, *, exists=False, verification_failure=""):
    calls = []
    monkeypatch.delenv(guard.STORE_URL_FILE, raising=False)
    monkeypatch.setenv(guard.STORE_URL, "postgresql://fixture.invalid/gpu_fault")
    monkeypatch.setenv("PGSERVICE", "fixture-service")

    def connect(url, **kwargs):
        assert kwargs == {"autocommit": True, "connect_timeout": 10}, (
            "database identity probes must have a finite connect deadline"
        )
        assert "PGSERVICE" not in guard.os.environ, "ambient routing must be isolated"
        name = urlsplit(url).path.lstrip("/")
        verifying = "store-close" in calls
        return GuardDatabase(
            guard.PRODUCTION_DATABASE
            if verifying and verification_failure == "identity"
            else name,
            calls,
            exists=exists,
            schema_present=not (verifying and verification_failure == "schema"),
        )

    def open_store(url, **kwargs):
        assert urlsplit(url).path == "/" + guard.PROBE_DATABASE, (
            "never initialize production"
        )
        assert kwargs["initialize_schema"] is True, (
            "use the public isolated schema path"
        )
        return SimpleNamespace(close=lambda: calls.append("store-close"))

    monkeypatch.setattr("psycopg.connect", connect)
    monkeypatch.setattr(store, "PostgresStore", open_store)
    return calls


def test_existing_disposable_database_is_verified_without_recreation(
    monkeypatch,
) -> None:
    calls = fake_database(monkeypatch, exists=True)
    guard.initialize_database()
    assert not any(call.startswith("CREATE DATABASE") for call in calls), (
        "an existing isolated database must not be recreated"
    )
    assert calls == [
        "SELECT current_database()",
        "SELECT 1 FROM pg_database WHERE datname=%s",
        "SELECT current_database()",
        "store-close",
        "SELECT current_database()",
        "SELECT to_regclass('gpu_fault_schema_version')",
    ], (
        "verify admin/probe identity, close Store, then independently verify probe schema"
    )
    assert guard.os.environ["PGSERVICE"] == "fixture-service", (
        "restore the enclosing environment"
    )


@pytest.mark.parametrize("failure", ["identity", "schema"])
def test_existing_database_requires_independent_post_initialization_verification(
    failure, monkeypatch
) -> None:
    calls = fake_database(monkeypatch, exists=True, verification_failure=failure)
    with pytest.raises(
        guard.BootGuardError, match="identity differs|initialization was not verified"
    ):
        guard.initialize_database()
    assert "store-close" in calls, (
        "post-initialization refusal must not leave the Store pool open"
    )
    assert calls[-1] == (
        "SELECT current_database()"
        if failure == "identity"
        else "SELECT to_regclass('gpu_fault_schema_version')"
    ), "verification must stop at the first unproved database boundary"
    assert not any(call.startswith(("CREATE", "DROP")) for call in calls), (
        "verification failure must not recreate or reset the existing database"
    )
    assert guard.os.environ["PGSERVICE"] == "fixture-service", (
        "verification failure must restore the caller's routing environment"
    )


@pytest.mark.parametrize("residual", [False, True])
def test_drop_requires_a_confirmed_absence_and_restores_environment(
    residual, monkeypatch
) -> None:
    calls = fake_database(monkeypatch, exists=residual)
    if residual:
        with pytest.raises(guard.BootGuardError, match="remains after cleanup"):
            guard.drop_database()
    else:
        guard.drop_database()
    assert calls[0] == "SELECT current_database()", (
        "drop must follow database identity proof"
    )
    assert 'DROP DATABASE IF EXISTS "gpu_fault_guardprobe"' in calls, (
        "the only drop target is the fixed disposable database"
    )
    assert guard.os.environ["PGSERVICE"] == "fixture-service", (
        "failure must not leak routing changes"
    )


@pytest.mark.parametrize(
    "value",
    [
        "postgresql://[malformed/gpu_fault",
        "postgresql://fixture.invalid/gpu_fault#fragment",
    ],
)
def test_invalid_reference_is_rejected_without_reading_any_database(
    value, monkeypatch
) -> None:
    calls = fake_database(monkeypatch)
    monkeypatch.setenv(guard.STORE_URL, value)
    with pytest.raises(
        guard.BootGuardError, match="explicit production database reference"
    ):
        guard.database_urls()
    assert calls == [], "invalid URL shape must fail before connection"


@pytest.mark.parametrize("collision", ["variable", "key", "volume", "mount"])
def test_pod_isolation_refuses_ambiguous_or_colliding_store_routes(collision) -> None:
    spec = pod_spec()
    container = spec["containers"][0]
    if collision == "variable":
        container["env"].append(copy.deepcopy(container["env"][0]))
    elif collision == "key":
        container["env"][0]["valueFrom"]["secretKeyRef"]["key"] = "wrong-key"
    elif collision == "volume":
        spec["volumes"].append({"name": guard.PROBE_VOLUME, "emptyDir": {}})
    else:
        container["volumeMounts"].append({"name": "ca", "mountPath": guard.PROBE_MOUNT})
    before = copy.deepcopy(spec)
    with pytest.raises(guard.BootGuardError, match="repeats|declared DSN|collides"):
        guard.isolate_probe_store(spec)
    assert spec == before, "invalid isolation must not partially rewrite the pod"


def test_unrelated_config_and_secret_imports_survive_isolation() -> None:
    spec = pod_spec()
    spec["containers"][0]["envFrom"] = [{"secretRef": {"name": "unrelated"}}]
    guard.isolate_probe_store(spec)
    assert spec["containers"][0]["envFrom"] == [{"secretRef": {"name": "unrelated"}}], (
        "only production Store credential routes should be removed"
    )


@pytest.mark.parametrize(
    "operation", ["initialize", "reset", "drop", "secret", "unknown"]
)
def test_public_entry_uses_only_fake_database_boundaries(
    operation, tmp_path, monkeypatch, capsys
) -> None:
    calls = fake_database(monkeypatch)
    monkeypatch.setattr(
        guard,
        "sys",
        SimpleNamespace(argv=["boot-database", operation], stderr=sys.stderr),
    )
    assert guard.main() == int(operation == "unknown"), "unknown operations must fail"
    output = capsys.readouterr()
    if operation == "secret":
        document = json.loads(output.out)
        assert document["metadata"]["name"] == guard.PROBE_SECRET, (
            "render only the disposable reference"
        )
        url = base64.b64decode(document["data"]["postgres-url"]).decode()
        assert urlsplit(url).path == "/" + guard.PROBE_DATABASE, (
            "the rendered reference cannot target production"
        )
        assert calls == [], "rendering a reference does not connect to a database"
    elif operation == "unknown":
        assert output.out == "", "invalid operation must not claim success"
        assert output.err.strip() == "BOOT database operation failed: BootGuardError"
    else:
        assert calls, "initialize/reset/drop must use the recording database transport"
        assert guard.os.environ["PGSERVICE"] == "fixture-service", (
            "entrypoints restore inherited routing"
        )
