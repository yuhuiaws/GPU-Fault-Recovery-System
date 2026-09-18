from __future__ import annotations

import base64
from types import SimpleNamespace

from gpu_fault import aurora_credential_refresh as aurora
from tests.regional.test_aurora_credential_refresh import FakeApps, FakeCore

SECRET_REFERENCE = "arn:aws:secretsmanager:us-west-2:123456789012:secret:local-only"
SYNTHETIC_DSN = (
    "postgresql://local:unit-old@database.invalid:5432/local?sslmode=verify-full"
)


def encode(value):
    return base64.b64encode(value.encode()).decode()


def core():
    result = FakeCore(
        {
            "postgres-url": encode(SYNTHETIC_DSN),
            "master-secret-arn": encode(SECRET_REFERENCE),
        }
    )
    result.events = []
    result.create_namespaced_event = lambda namespace, body: result.events.append(
        (namespace, body)
    )
    return result


def refresh(target=None, apps=None, **values):
    options = {
        "namespace": "gpu-fault-system",
        "secret_name": "gpu-fault-aurora",
        "secret_key": "postgres-url",
        "deployment": "gpu-fault-api-ha",
        "region_name": "us-west-2",
        "timestamp": "2030-01-01T00:00:00+00:00",
        "fetch_password": lambda arn, region: "unit-new",
        "verify": lambda dsn: None,
    }
    options.update(values)
    return aurora.refresh_once(
        core() if target is None else target,
        FakeApps() if apps is None else apps,
        **options,
    )


def main_fakes(monkeypatch):
    target, apps, sleeps = core(), FakeApps(), []
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    monkeypatch.setattr(aurora, "configure_logging", lambda: None)
    monkeypatch.setattr(aurora, "load_kubernetes_clients", lambda: (target, apps))
    monkeypatch.setattr(aurora, "read_current_password", lambda arn, region: "unit-new")
    monkeypatch.setattr(aurora, "verify_dsn", lambda dsn: None)
    monkeypatch.setattr(aurora, "time", SimpleNamespace(sleep=sleeps.append))
    return target, apps, sleeps
