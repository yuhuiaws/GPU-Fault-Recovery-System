from __future__ import annotations

import json
import os
from contextlib import contextmanager
from types import SimpleNamespace

from scripts.e2e.regional.notify008_bundle import source_bundle
from scripts.e2e.regional.notify008_resources import manifests
from scripts.e2e.regional.probes import notify008_probe as probe
from tests.regional._cov95_notify008_lifecycle import target, uid
from tests.regional._cov95_notify008_support import RUN_ID

FACTS = {
    "postgres_major": 16,
    "database_is_unix_socket": True,
    "production_credentials_loaded": False,
}


def probe_environment(tmp_path, monkeypatch):
    value = target()
    case, control, work = (tmp_path / name for name in ("case", "control", "work"))
    for path in (case, control, work):
        path.mkdir()
    bundle = manifests(value, uid(10), source_bundle())["configmap"]["data"]
    for name, text in bundle.items():
        path = case / (
            f"scripts/e2e/regional/probes/{name}" if name.endswith(".py") else name
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    for name in list(os.environ):
        if name.startswith(("PG", "AWS_", "KUBE", "GPU_FAULT_")):
            monkeypatch.delenv(name)
    for name, text in probe.SAFE_ENVIRONMENT.items():
        monkeypatch.setenv(name, text)
    monkeypatch.setenv("NOTIFY008_POD_UID", uid(30))
    monkeypatch.setenv("NOTIFY008_RUN_ID", RUN_ID)
    monkeypatch.setenv("NOTIFY008_SECONDS", "540")
    monkeypatch.setattr(probe, "CASE_ROOT", case)
    monkeypatch.setattr(probe, "CONTROL", control)
    monkeypatch.setattr(probe, "WORK", work)
    monkeypatch.setattr(probe.os, "umask", lambda _mode: None)
    identity = {
        "distribution": "gpu-fault-control-plane",
        "version": value.runtime_version,
        "module_digest": value.runtime_module_digest,
    }
    monkeypatch.setattr(probe, "runtime_identity", lambda: identity)
    config = json.loads(bundle["config.json"])
    arguments = SimpleNamespace(
        expected_pod_uid=uid(30),
        expected_namespace_uid=uid(10),
        bundle_sha256=config["source_sha256"],
    )
    return value, arguments, identity


def argv(command, arguments):
    return [
        command,
        "--expected-pod-uid",
        arguments.expected_pod_uid,
        "--expected-namespace-uid",
        arguments.expected_namespace_uid,
        "--bundle-sha256",
        arguments.bundle_sha256,
    ]


@contextmanager
def fake_connection():
    yield object()
