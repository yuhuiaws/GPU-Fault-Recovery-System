"""Control-record retention is a site setting, off by default.

``GPU_FAULT_CONTROL_RECORD_RETENTION_DAYS`` defaults to ``0`` in the runtime,
so incident and workflow rows never leave the store unless the operator
declares ``spec.retention`` in ``site.yaml`` and reruns ``deploy``. The block
travels site -> release config -> control-plane apply environment -> worker
container, and the archive bucket grant is added to the control-plane role
only when an archive URI is declared.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault.admin import bootstrap_services
from gpu_fault.admin.bootstrap_site import preserve_existing_site_contract
from gpu_fault.admin.site import (
    RegionalSite,
    RetentionSiteConfig,
    SiteConfigError,
    load_site,
    site_retention,
)
from gpu_fault_release import regional_release_config as release_config_module
from gpu_fault_release import regional_release_rendering as rendering
from tests.admin.test_admin_site import site_file as write_site_file
from tests.regional._release_orchestrator_support import config_file

ROOT = Path(__file__).resolve().parents[2]
RENDER_SCRIPT = ROOT / "deploy/control-plane/tools/render_control_plane_role_split.py"
ARCHIVE_URI = "s3://gpu-fault-archive/prod/site-a"
RETENTION_ENV = (
    "GPU_FAULT_CONTROL_RECORD_RETENTION_DAYS",
    "GPU_FAULT_CONTROL_RECORD_ARCHIVE_S3_URI",
    "GPU_FAULT_CONTROL_RECORD_ARCHIVE_INTERVAL_SECONDS",
)


@pytest.fixture
def site_file(tmp_path: Path) -> Path:
    return write_site_file(tmp_path)


def _site_document(site_file: Path, retention: dict[str, Any] | None) -> Path:
    document = yaml.safe_load(site_file.read_text(encoding="utf-8"))
    if retention is not None:
        document["spec"]["retention"] = retention
    site_file.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return site_file


def test_retention_is_off_when_the_site_declares_nothing(site_file: Path) -> None:
    site = RegionalSite.from_value(yaml.safe_load(site_file.read_text()))

    assert site.spec.retention == RetentionSiteConfig()
    assert site.spec.retention.enabled is False
    assert site.spec.retention.environment() == {}


def test_retention_block_is_parsed_and_rendered_into_the_release_config(
    site_file: Path,
) -> None:
    rendered = load_site(
        _site_document(
            site_file,
            {
                "controlRecordRetentionDays": 365,
                "archiveS3Uri": ARCHIVE_URI,
                "archiveIntervalSeconds": 1800,
            },
        )
    )

    assert rendered.release_config["retention"] == {
        "control_record_retention_days": 365,
        "archive_s3_uri": ARCHIVE_URI,
        "archive_interval_seconds": 1800,
    }
    assert site_retention(yaml.safe_load(site_file.read_text())).archive_s3_uri == (
        ARCHIVE_URI
    )


@pytest.mark.parametrize(
    ("retention", "message"),
    [
        ({"controlRecordRetentionDays": -1}, "must be within 0"),
        ({"controlRecordRetentionDays": "30"}, "must be an integer"),
        (
            {"controlRecordRetentionDays": 30, "archiveS3Uri": "https://x/y"},
            "must be an s3://",
        ),
        (
            {"controlRecordRetentionDays": 30, "archiveS3Uri": "s3://"},
            "must be an s3://",
        ),
        (
            {
                "controlRecordRetentionDays": 30,
                "archiveS3Uri": ARCHIVE_URI,
                "archiveIntervalSeconds": 0,
            },
            "archiveIntervalSeconds must be within",
        ),
        ({"days": 30}, "unknown"),
    ],
)
def test_retention_block_is_validated(
    site_file: Path, retention: dict[str, Any], message: str
) -> None:
    with pytest.raises(SiteConfigError, match=message):
        load_site(_site_document(site_file, retention))


def test_release_config_threads_retention_and_defaults_off(tmp_path: Path) -> None:
    path = config_file(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    off = release_config_module.ReleaseConfig.load(path)

    assert off.retention.enabled is False
    assert off.retention.environment() == {}

    value["retention"] = {
        "control_record_retention_days": 90,
        "archive_s3_uri": ARCHIVE_URI,
        "archive_interval_seconds": 600,
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    on = release_config_module.ReleaseConfig.load(path)

    assert on.retention.control_record_retention_days == 90
    assert on.retention.environment() == {
        "GPU_FAULT_CONTROL_RECORD_RETENTION_DAYS": "90",
        "GPU_FAULT_CONTROL_RECORD_ARCHIVE_S3_URI": ARCHIVE_URI,
        "GPU_FAULT_CONTROL_RECORD_ARCHIVE_INTERVAL_SECONDS": "600",
    }

    value["retention"] = {"control_record_retention_days": 7}
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(release_config_module.ReleaseError, match="archive_s3_uri"):
        release_config_module.ReleaseConfig.load(path)


def test_cpu_apply_environment_carries_retention_only_when_declared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in RETENTION_ENV:
        monkeypatch.delenv(name, raising=False)

    class Release:
        def __init__(self, config: Any) -> None:
            self.config = config
            self.wheel_cm = "wheel-cm"
            self.wheel_sha = "a" * 64
            self.runtime_image = "registry.example/runtime@sha256:" + "b" * 64
            self.node_wheel_sha = "c" * 64
            self.executor_wheel_sha = "d" * 64
            self.admin_config_digest = "e" * 64
            self.admin_config_role_digests = {
                "ingress": "f" * 64,
                "worker": "1" * 64,
                "spool": "2" * 64,
            }
            self.release_id = "release-a"

    path = config_file(tmp_path)
    off = rendering.build_cpu_apply_environment(
        Release(release_config_module.ReleaseConfig.load(path)), finalize=False
    )
    assert not set(RETENTION_ENV) & set(off)

    value = json.loads(path.read_text(encoding="utf-8"))
    value["retention"] = {
        "control_record_retention_days": 30,
        "archive_s3_uri": ARCHIVE_URI,
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    on = rendering.build_cpu_apply_environment(
        Release(release_config_module.ReleaseConfig.load(path)), finalize=False
    )
    assert on["GPU_FAULT_CONTROL_RECORD_RETENTION_DAYS"] == "30"
    assert on["GPU_FAULT_CONTROL_RECORD_ARCHIVE_S3_URI"] == ARCHIVE_URI
    assert "GPU_FAULT_CONTROL_RECORD_ARCHIVE_INTERVAL_SECONDS" not in on, (
        "the interval keeps the runtime default unless the site declares one"
    )


def _source_deployment() -> dict[str, Any]:
    """The smallest gpu-fault-api-ha Deployment the renderer accepts."""

    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": "gpu-fault-api-ha"},
        "spec": {
            "replicas": 3,
            "selector": {"matchLabels": {"app": "gpu-fault-api-ha"}},
            "template": {
                "metadata": {
                    "labels": {"app": "gpu-fault-api-ha"},
                    "annotations": {"prometheus.io/port": "8080"},
                },
                "spec": {
                    "terminationGracePeriodSeconds": 120,
                    "containers": [
                        {
                            "name": "api",
                            "args": [
                                'exec "$@" gpu_fault.app:create_app --factory '
                                "--host 0.0.0.0 --port 8080"
                            ],
                            "env": [
                                {"name": "GPU_FAULT_PROCESSOR_WORKERS", "value": "24"}
                            ],
                            "readinessProbe": {"httpGet": {"port": 8080}},
                            "livenessProbe": {
                                "httpGet": {"path": "/healthz", "port": 8080}
                            },
                            "resources": {
                                "requests": {"cpu": "4", "memory": "4Gi"},
                                "limits": {"cpu": "8", "memory": "8Gi"},
                            },
                        }
                    ],
                },
            },
        },
    }


def _render_worker_env(extra_env: dict[str, str]) -> dict[str, dict[str, str]]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": str(ROOT / "src"),
        "GPU_FAULT_CONTROL_WORKER_REPLICAS": "6",
        **extra_env,
    }
    completed = subprocess.run(
        [sys.executable, str(RENDER_SCRIPT), "--json"],
        input=json.dumps(_source_deployment()),
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    items = json.loads(completed.stdout)["items"]
    config_maps = {
        item["metadata"]["name"]: item.get("data", {})
        for item in items
        if item["kind"] == "ConfigMap"
    }
    result: dict[str, dict[str, str]] = {}
    for item in items:
        if item["kind"] != "Deployment":
            continue
        container = item["spec"]["template"]["spec"]["containers"][0]
        values: dict[str, str] = {}
        # Literal values are externalised into per-role ConfigMaps (envFrom).
        for source in container.get("envFrom", []):
            values.update(config_maps.get(source["configMapRef"]["name"], {}))
        for entry in container.get("env", []):
            if "value" in entry:
                values[entry["name"]] = entry["value"]
            else:
                ref = entry["valueFrom"].get("configMapKeyRef")
                if ref and ref["name"] in config_maps:
                    values[entry["name"]] = config_maps[ref["name"]].get(ref["key"], "")
        result[item["metadata"]["name"]] = values
    return result


def test_render_script_forwards_retention_to_the_worker_only_when_set() -> None:
    off = _render_worker_env({})
    for deployment in off.values():
        assert not set(RETENTION_ENV) & set(deployment)

    on = _render_worker_env(
        {
            "GPU_FAULT_CONTROL_RECORD_RETENTION_DAYS": "30",
            "GPU_FAULT_CONTROL_RECORD_ARCHIVE_S3_URI": ARCHIVE_URI,
            "GPU_FAULT_CONTROL_RECORD_ARCHIVE_INTERVAL_SECONDS": "900",
        }
    )
    worker = on["gpu-fault-control-worker"]
    assert worker["GPU_FAULT_CONTROL_RECORD_RETENTION_DAYS"] == "30"
    assert worker["GPU_FAULT_CONTROL_RECORD_ARCHIVE_S3_URI"] == ARCHIVE_URI
    assert worker["GPU_FAULT_CONTROL_RECORD_ARCHIVE_INTERVAL_SECONDS"] == "900"
    for name, deployment in on.items():
        if name != "gpu-fault-control-worker":
            assert not set(RETENTION_ENV) & set(deployment), (
                f"{name} runs no archiver and must not validate a retention it "
                "cannot honour"
            )


def test_control_plane_policy_grants_the_archive_prefix_only_when_declared() -> None:
    without = bootstrap_services.control_plane_policy_document(
        region="us-east-1", account_id="123456789012"
    )
    assert [item for item in without["Statement"] if "s3" in str(item)] == []

    with_archive = bootstrap_services.control_plane_policy_document(
        region="us-east-1", account_id="123456789012", archive_s3_uri=ARCHIVE_URI
    )
    archive = next(
        item
        for item in with_archive["Statement"]
        if item.get("Sid") == "ControlRecordArchive"
    )
    assert archive["Action"] == "s3:PutObject", (
        "the archiver only ever writes bundles; it neither reads nor deletes them"
    )
    assert archive["Resource"] == "arn:aws:s3:::gpu-fault-archive/prod/site-a/*"

    bucket_only = bootstrap_services.control_plane_policy_document(
        region="us-east-1",
        account_id="123456789012",
        archive_s3_uri="s3://gpu-fault-archive",
    )
    assert bucket_only["Statement"][-1]["Resource"] == (
        "arn:aws:s3:::gpu-fault-archive/*"
    )


def test_regenerating_the_site_preserves_the_declared_retention() -> None:
    existing = {
        "spec": {
            "retention": {
                "controlRecordRetentionDays": 30,
                "archiveS3Uri": ARCHIVE_URI,
            },
            "clusters": [],
        }
    }
    generated = {"spec": {"clusters": []}}

    preserved = preserve_existing_site_contract(copy.deepcopy(generated), existing)

    assert preserved["spec"]["retention"] == existing["spec"]["retention"]
    assert preserve_existing_site_contract(copy.deepcopy(generated), {"spec": {}}) == (
        generated
    )


def test_retention_without_an_archive_uri_derives_the_site_bucket(
    site_file: Path,
) -> None:
    """Turning retention on is one line: the bucket name follows the account,
    Region and site so bootstrap can create it and the role can be granted."""

    rendered = load_site(_site_document(site_file, {"controlRecordRetentionDays": 30}))

    retention = rendered.release_config["retention"]
    account = rendered.release_config["cpu_eks_arn"].split(":")[4]
    region = rendered.release_config["aws_region"]
    site_name = rendered.release_config["site_name"]
    assert retention["control_record_retention_days"] == 30
    assert retention["archive_s3_uri"] == (
        f"s3://gpu-fault-control-records-{account}-{region}/{site_name}/control-record-archive"
    )
    assert (
        RetentionSiteConfig.from_value(
            {"controlRecordRetentionDays": 30}
        ).archive_s3_uri
        is None
    ), "the raw block keeps None; resolution happens per site"
