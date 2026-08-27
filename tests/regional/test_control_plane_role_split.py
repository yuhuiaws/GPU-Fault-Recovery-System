from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from gpu_fault.app import ApplicationContext, create_app
from tests._builders import asgi_client, build_context, build_store

ROOT = Path(__file__).resolve().parents[2]


def _effective_env(items: dict[str, dict], deployment: dict) -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    for source in container.get("envFrom", []):
        reference = source.get("configMapRef")
        if reference:
            values.update(items[reference["name"]].get("data", {}))
    for item in container.get("env", []):
        values[item["name"]] = item.get("value")
    return values


def _deployment() -> dict:
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
                                "python -m pip install --no-cache-dir "
                                "'/artifact/"
                                "gpu_fault_control_plane-0.10.0-"
                                "py3-none-any.whl"
                                "[collectors,postgres]' && "
                                "exec uvicorn "
                                "gpu_fault.app:create_app --factory "
                                "--host 0.0.0.0 --port 8080"
                            ],
                            "env": [
                                {
                                    "name": "GPU_FAULT_POSTGRES_POOL_MAX_SIZE",
                                    "value": "48",
                                },
                                {"name": "GPU_FAULT_PROCESSOR_WORKERS", "value": "24"},
                                {
                                    "name": (
                                        "GPU_FAULT_PROCESSOR_GPU_TELEMETRY_WORKERS"
                                    ),
                                    "value": "8",
                                },
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


def test_role_split_renders_ingress_and_scalable_workers() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "deploy/control-plane/tools/render_control_plane_role_split.py"),
            "--json",
        ],
        input=json.dumps(_deployment()),
        text=True,
        capture_output=True,
        check=True,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "GPU_FAULT_CONTROL_WORKER_REPLICAS": "6",
        },
    )
    items = {
        item["metadata"]["name"]: item for item in json.loads(result.stdout)["items"]
    }
    ingress = items["gpu-fault-api-ha"]
    ingress_pdb = next(
        item
        for item in json.loads(result.stdout)["items"]
        if item["kind"] == "PodDisruptionBudget"
        and item["metadata"]["name"] == "gpu-fault-api-ha-pdb"
    )
    worker = items["gpu-fault-control-worker"]
    worker_pdb = items["gpu-fault-control-worker-pdb"]
    spool = items["gpu-fault-telemetry-spool-worker"]
    spool_pdb = items["gpu-fault-telemetry-spool-worker-pdb"]

    assert ingress["spec"]["replicas"] == 3
    assert ingress_pdb["spec"]["minAvailable"] == 2, "ingress PDB lost quorum"
    assert ingress_pdb["spec"]["selector"]["matchLabels"] == {
        "app": "gpu-fault-api-ha"
    }, "ingress PDB selects the wrong pods"
    ingress_env = _effective_env(items, ingress)
    assert ingress_env["GPU_FAULT_TELEMETRY_SPOOL"] == "false"
    assert ingress_env["GPU_FAULT_POSTGRES_POOL_MAX_SIZE"] == "40"
    assert ingress_env["GPU_FAULT_POSTGRES_POOL_MIN_SIZE"] == "2"
    assert ingress_env["GPU_FAULT_STORE_IO_WORKERS"] == "28"
    assert ingress_env["GPU_FAULT_TELEMETRY_SPOOL_STORE_IO_WORKERS"] == "8"
    assert ingress_env["GPU_FAULT_TELEMETRY_SPOOL_ADMISSION_PARTITIONS"] == "8"
    assert ingress_env["GPU_FAULT_TELEMETRY_SPOOL_BATCH_GROUPS"] == "8"
    assert ingress_env["GPU_FAULT_FAULT_STORE_IO_WORKERS"] == "8"
    assert ingress_env["GPU_FAULT_EVIDENCE_STORE_IO_WORKERS"] == "4"
    assert ingress_env["GPU_FAULT_LIFESPAN_SHUTDOWN_MAX_SECONDS"] == "20"
    assert ingress_env["GPU_FAULT_PROCESSOR_FAULT_ADMISSION_BATCH_SIZE"] == "64"
    assert ingress_env["GPU_FAULT_PROCESSOR_FAULT_ADMISSION_BATCH_GROUPS"] == "8"
    assert ingress_env["GPU_FAULT_PROCESSOR_FAULT_ADMISSION_PROJECTION_MARGIN"] == "0"
    assert ingress_env["GPU_FAULT_PROCESSOR_EVIDENCE_ADMISSION_BATCH_SIZE"] == "32"
    assert ingress_env["GPU_FAULT_PROCESSOR_EVIDENCE_ADMISSION_BATCH_GROUPS"] == "4"
    assert ingress_env["GPU_FAULT_PROCESSOR_RETRY_BACKOFF_SECONDS"] == "1"
    assert ingress_env["GPU_FAULT_PROCESSOR_RETRY_BACKOFF_MAX_SECONDS"] == "30"
    assert ingress_env["GPU_FAULT_TELEMETRY_REQUEST_BUDGET_SECONDS"] == "30"
    assert (
        "--workers 4"
        in (ingress["spec"]["template"]["spec"]["containers"][0]["args"][0])
    )
    assert (
        "--loop uvloop --http httptools"
        in (ingress["spec"]["template"]["spec"]["containers"][0]["args"][0])
    )
    assert (
        "--limit-concurrency 4096"
        in (ingress["spec"]["template"]["spec"]["containers"][0]["args"][0])
    )
    assert (
        "[collectors,postgres,performance]"
        in (ingress["spec"]["template"]["spec"]["containers"][0]["args"][0])
    )
    assert (
        ingress["spec"]["template"]["spec"].get("securityContext", {}).get("sysctls")
        is None
    )
    assert worker["spec"]["replicas"] == 6
    assert worker["spec"]["template"]["spec"]["terminationGracePeriodSeconds"] == 240
    container = worker["spec"]["template"]["spec"]["containers"][0]
    assert container["name"] == "control-worker"
    assert "--port 8081" in container["args"][0]
    assert "--limit-max-requests" not in container["args"][0]
    assert (
        worker["spec"]["template"]["spec"]["topologySpreadConstraints"][0][
            "whenUnsatisfiable"
        ]
        == "DoNotSchedule"
    )
    assert worker_pdb["spec"]["maxUnavailable"] == 1
    assert worker_pdb["spec"]["selector"]["matchLabels"] == {
        "app": "gpu-fault-control-worker"
    }
    assert spool["spec"]["replicas"] == 0
    spool_container = spool["spec"]["template"]["spec"]["containers"][0]
    assert spool_container["name"] == "telemetry-spool-worker"
    assert "--port 8082" in spool_container["args"][0]
    assert "--workers 1" in spool_container["args"][0]
    spool_env = _effective_env(items, spool)
    assert spool_env["GPU_FAULT_SERVICE_ROLE"] == "spool-worker"
    assert spool["spec"]["template"]["spec"]["terminationGracePeriodSeconds"] == 130
    assert spool_env["GPU_FAULT_LIFESPAN_SHUTDOWN_MAX_SECONDS"] == "30"
    assert spool_env["GPU_FAULT_TELEMETRY_SPOOL"] == "true"
    assert spool_env["GPU_FAULT_TELEMETRY_SPOOL_MAX_ITEM_BYTES"] == str(4 * 1024 * 1024)
    assert spool_env["GPU_FAULT_TELEMETRY_SPOOL_REPLAY_BATCH_MAX_BYTES"] == str(
        8 * 1024 * 1024
    )
    assert spool_env["GPU_FAULT_TELEMETRY_SPOOL_MAX_IN_FLIGHT_BYTES"] == str(
        64 * 1024 * 1024
    )
    assert spool_env["GPU_FAULT_TELEMETRY_SPOOL_WORKERS"] == "8"
    assert spool_env["GPU_FAULT_TELEMETRY_SPOOL_FAULT_PRESSURE_WORKERS"] == "1"
    assert spool_env["GPU_FAULT_POSTGRES_POOL_MAX_SIZE"] == "12"
    worker_env = _effective_env(items, worker)
    assert worker_env["GPU_FAULT_PROCESSOR_FAULT_WORKERS"] == "4"
    assert worker_env["GPU_FAULT_LIFESPAN_SHUTDOWN_MAX_SECONDS"] == "130"
    assert worker_env["GPU_FAULT_PROCESSOR_FAULT_IDLE_BACKOFF_MAX_SECONDS"] == "0.5"
    assert worker_env["GPU_FAULT_PROCESSOR_FAULT_BUSY_BACKOFF_MAX_SECONDS"] == "0.1"
    assert worker_env["GPU_FAULT_PROCESSOR_NOTIFICATION_FALLBACK_SECONDS"] == "5"
    assert worker_env["GPU_FAULT_PROCESSOR_NOTIFICATION_SHARDS"] == "24"
    assert worker_env["GPU_FAULT_PROCESSOR_COMPLETION_CLUSTER_CONCURRENCY"] == "1"
    assert worker_env["GPU_FAULT_PROCESSOR_RETRY_BACKOFF_SECONDS"] == "1"
    assert worker_env["GPU_FAULT_PROCESSOR_RETRY_BACKOFF_MAX_SECONDS"] == "30"
    assert worker_env["GPU_FAULT_PROCESSOR_ROUTINE_STARVATION_SECONDS"] == "30"
    assert worker_env["GPU_FAULT_PROCESSOR_FAULT_PRESSURE_EVIDENCE_WORKERS"] == "1"
    assert worker_env["GPU_FAULT_PROCESSOR_THREAD_DUMP_SIGNAL"] == "SIGUSR2"
    assert spool_env["GPU_FAULT_POSTGRES_STATEMENT_TIMEOUT_SECONDS"] == "20"
    assert spool_env["GPU_FAULT_TELEMETRY_SPOOL_NOTIFICATION_FALLBACK_SECONDS"] == "5"
    assert spool_env["GPU_FAULT_TELEMETRY_SPOOL_REPLAY_BATCH_MAX_ITEMS"] == "64"
    assert spool_pdb["spec"]["maxUnavailable"] == 1
    for deployment in (ingress, worker, spool):
        pod_spec = deployment["spec"]["template"]["spec"]
        assert pod_spec["enableServiceLinks"] is False
    ingress_pod_spec = ingress["spec"]["template"]["spec"]
    assert ingress_pod_spec["terminationGracePeriodSeconds"] == 120
    assert (
        ingress_pod_spec["containers"][0]["livenessProbe"]["httpGet"]["path"]
        == "/healthz"
    )
    assert (
        "--timeout-graceful-shutdown 60"
        in (ingress_pod_spec["containers"][0]["args"][0])
    )


POOL_ENV = (
    "GPU_FAULT_PROCESSOR_WORKERS",
    "GPU_FAULT_PROCESSOR_FAULT_WORKERS",
    "GPU_FAULT_PROCESSOR_OBSERVATION_WORKERS",
    "GPU_FAULT_PROCESSOR_GPU_TELEMETRY_WORKERS",
    "GPU_FAULT_PROCESSOR_HOST_TELEMETRY_WORKERS",
)


def _env_names(items: dict[str, dict], deployment: dict) -> set[str]:
    return set(_effective_env(items, deployment))


def test_processor_pool_sizing_is_worker_tier_only() -> None:
    """The ingress tier must not carry processor pool sizing.

    ``run_processor`` creates those thread pools and it never starts
    under ``GPU_FAULT_SERVICE_ROLE=ingress``, so a value on the ingress
    container changes no behaviour - it only gets counted. It put 24
    lanes plus four pools into the ingress thread budget and made the
    ingress replicas export ``gpu_fault_processor_workers`` for pools
    they never create.
    """

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "deploy/control-plane/tools/render_control_plane_role_split.py"),
            "--json",
        ],
        input=json.dumps(_deployment()),
        text=True,
        capture_output=True,
        check=True,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "GPU_FAULT_CONTROL_WORKER_REPLICAS": "6",
        },
    )
    items = {
        item["metadata"]["name"]: item for item in json.loads(result.stdout)["items"]
    }
    ingress = items["gpu-fault-api-ha"]
    worker = items["gpu-fault-control-worker"]
    assert _env_names(items, ingress).isdisjoint(POOL_ENV)
    # Dropping them from ingress must not drop the tier that uses them,
    # which would quietly fall back to the 4-lane code default.
    assert set(POOL_ENV) <= _env_names(items, worker)


GENERATED = ROOT / "deploy/control-plane/regional/generated"


def test_generated_role_split_manifests_are_current(tmp_path: Path) -> None:
    """The checked-in manifests must match a fresh render.

    They are what deploy.sh applies, so a change to
    control-plane-deployment.yaml or the regional patch that is not
    re-rendered ships the old control plane while the review shows the
    new one. Re-render with
    deploy/control-plane/tools/render-control-plane-role-split.sh.
    """

    if shutil.which("kubectl") is None:
        pytest.skip("kubectl is unavailable")
    rendered_dir = tmp_path / "generated"
    rendered = subprocess.run(
        [
            "bash",
            str(ROOT / "deploy/control-plane/tools/render-control-plane-role-split.sh"),
        ],
        text=True,
        capture_output=True,
        check=False,
        env={
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "GPU_FAULT_CONTROL_WORKER_REPLICAS": "6",
            "GPU_FAULT_ROLE_SPLIT_OUT_DIR": str(rendered_dir),
        },
    )
    assert rendered.returncode == 0, rendered.stderr

    expected = sorted(path.name for path in GENERATED.glob("gpu-fault-*.yaml"))
    assert expected == sorted(
        path.name for path in rendered_dir.glob("gpu-fault-*.yaml")
    )
    for filename in expected:
        assert yaml.safe_load(
            (GENERATED / filename).read_text(encoding="utf-8")
        ) == yaml.safe_load((rendered_dir / filename).read_text(encoding="utf-8"))
    assert (GENERATED / "manifest-list.txt").read_text() == (
        rendered_dir / "manifest-list.txt"
    ).read_text()


def test_role_split_verifier_rejects_a_missing_worker_tier(tmp_path) -> None:
    """The self-check has to fail on the split that looks healthy.

    An ingress-only control plane passes every probe and answers 202 to
    everything; the requests just sit in the queue. The verifier is the
    only thing in the deploy that notices.
    """

    stub = tmp_path / "kubectl"
    stub.write_text(
        "#!/bin/sh\n"
        'case "$*" in\n'
        "*gpu-fault-control-worker*) exit 1 ;;\n"
        "esac\n"
        "cat <<'JSON'\n"
        + json.dumps(
            {
                "metadata": {"name": "gpu-fault-api-ha"},
                "spec": {
                    "replicas": 3,
                    "template": {
                        "spec": {
                            "containers": [
                                {
                                    "name": "api",
                                    "args": ["--port 8080 --workers 4"],
                                    "env": [
                                        {
                                            "name": "GPU_FAULT_SERVICE_ROLE",
                                            "value": "ingress",
                                        }
                                    ],
                                }
                            ]
                        }
                    },
                },
            }
        )
        + "\nJSON\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "deploy/control-plane/tools/verify_control_plane_role_split.py"),
        ],
        text=True,
        capture_output=True,
        check=False,
        env={
            "PATH": f"{tmp_path}:/usr/bin:/bin",
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        },
    )
    assert result.returncode == 1
    assert "gpu-fault-control-worker is missing" in result.stdout


def test_role_split_verifier_rejects_pool_env_on_ingress(tmp_path) -> None:
    """A stale `set env` on the ingress tier has to fail the deploy.

    `kubectl set env` values are not in the last-applied annotation, so
    applying the generated manifest that no longer carries them does not
    remove them. Nothing breaks - the pools are never created on this
    tier - which is why only the verifier can catch it before the number
    is read back as capacity.
    """

    ingress = {
        "metadata": {"name": "gpu-fault-api-ha"},
        "spec": {
            "replicas": 3,
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": "api",
                            "args": ["--port 8080 --workers 4"],
                            "env": [
                                {"name": "GPU_FAULT_SERVICE_ROLE", "value": "ingress"},
                                {"name": "GPU_FAULT_PROCESSOR_WORKERS", "value": "24"},
                            ],
                        }
                    ]
                }
            },
        },
    }
    worker = {
        "metadata": {"name": "gpu-fault-control-worker"},
        "spec": {
            "replicas": 6,
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": "control-worker",
                            "args": ["--port 8081 --workers 4"],
                            "env": [
                                {"name": "GPU_FAULT_SERVICE_ROLE", "value": "worker"},
                                {"name": "GPU_FAULT_PROCESSOR_WORKERS", "value": "24"},
                            ],
                        }
                    ]
                }
            },
        },
        "status": {"readyReplicas": 6},
    }
    spool = {
        "metadata": {"name": "gpu-fault-telemetry-spool-worker"},
        "spec": {
            "replicas": 3,
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": "telemetry-spool-worker",
                            "args": ["--port 8082 --workers 1"],
                            "env": [
                                {
                                    "name": "GPU_FAULT_SERVICE_ROLE",
                                    "value": "spool-worker",
                                },
                                {"name": "GPU_FAULT_TELEMETRY_SPOOL", "value": "true"},
                            ],
                        }
                    ]
                }
            },
        },
        "status": {"readyReplicas": 3},
    }
    stub = tmp_path / "kubectl"
    stub.write_text(
        "#!/bin/sh\n"
        'case "$*" in\n'
        "*gpu-fault-telemetry-spool-worker*)\n"
        "cat <<'SPOOL'\n" + json.dumps(spool) + "\nSPOOL\n"
        ";;\n"
        "*gpu-fault-control-worker*)\n"
        "cat <<'WORKER'\n" + json.dumps(worker) + "\nWORKER\n"
        ";;\n"
        "*)\n"
        "cat <<'INGRESS'\n" + json.dumps(ingress) + "\nINGRESS\n"
        ";;\n"
        "esac\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "deploy/control-plane/tools/verify_control_plane_role_split.py"),
        ],
        text=True,
        capture_output=True,
        check=False,
        env={
            "PATH": f"{tmp_path}:/usr/bin:/bin",
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        },
    )
    assert result.returncode == 1
    assert "GPU_FAULT_PROCESSOR_WORKERS" in result.stdout
    assert "capacity that does not exist" in result.stdout


def test_application_rejects_unknown_service_role(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_SERVICE_ROLE", "unknown")
    with pytest.raises(ValueError, match="GPU_FAULT_SERVICE_ROLE"):
        create_app(build_context())


def test_unknown_regional_route_has_no_implicit_bucket() -> None:
    app = create_app(build_context())

    assert (
        app.state.regional_authorization_bucket("/v1/new-domain/forgotten-route")
        is None
    )


def _metrics(monkeypatch, role: str) -> str:
    monkeypatch.setenv("GPU_FAULT_SERVICE_ROLE", role)
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_WORKERS", "24")
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("POD_UID", f"pod-role-split-{role}")
    if role == "spool-worker":
        monkeypatch.setenv("GPU_FAULT_TELEMETRY_SPOOL", "true")
    app = create_app(
        ApplicationContext(
            store=build_store(),
            execution_token="control-plane-role-split-token-" + "x" * 32,
        )
    )

    async def run() -> str:
        async with asgi_client(app) as client:
            response = await client.get("/metrics")
            assert response.status_code == 200
            return response.text

    return asyncio.run(run())


def test_ingress_reports_no_processor_worker_threads(monkeypatch) -> None:
    """The gauge has to count threads, not environment.

    A Grafana sum of ``gpu_fault_processor_workers`` over replicas is how
    consumer concurrency gets read off the fleet. The ingress replicas
    have the env - the shared ``kubectl set env`` put it there - but
    ``run_processor`` never starts for them, so exporting 24 added three
    replicas x 24 lanes of capacity that does not exist.
    """

    metrics = _metrics(monkeypatch, "ingress")

    assert "gpu_fault_processor_workers 0" in metrics
    assert "gpu_fault_processor_active_consumer 0" in metrics


def test_consuming_tier_still_reports_its_worker_threads(monkeypatch) -> None:
    """Zeroing the ingress tier must not zero the tier that consumes."""

    metrics = _metrics(monkeypatch, "worker")

    assert "gpu_fault_processor_workers 24" in metrics
    assert "gpu_fault_processor_active_consumer 1" in metrics


def test_spool_tier_reports_no_main_processor_threads(monkeypatch) -> None:
    metrics = _metrics(monkeypatch, "spool-worker")

    assert "gpu_fault_processor_workers 0" in metrics
    assert "gpu_fault_processor_active_consumer 0" in metrics
    assert "gpu_fault_telemetry_spool_enabled 1" in metrics


def test_ingress_health_reports_inactive_processor(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_SERVICE_ROLE", "ingress")
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("POD_UID", "pod-role-split-ingress-health")
    app = create_app(
        ApplicationContext(
            store=build_store(),
            execution_token="control-plane-role-split-token-" + "x" * 32,
        )
    )

    async def run() -> dict:
        async with asgi_client(app) as client:
            response = await client.get("/healthz")
            assert response.status_code == 200
            return response.json()

    health = asyncio.run(run())

    assert health["service_role"] == "ingress"
    assert health["processor_role"] == "inactive"


def test_legacy_release_filter_removes_only_new_component_pins() -> None:
    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": "legacy"},
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": "api",
                            "env": [
                                {
                                    "name": (
                                        "GPU_FAULT_REQUIRED_AGENT_COMPATIBILITY_DIGEST"
                                    ),
                                    "value": "a" * 64,
                                },
                                {
                                    "name": "GPU_FAULT_REQUIRED_AGENT_ARTIFACT_SHA256",
                                    "value": "b" * 64,
                                },
                            ],
                        }
                    ]
                }
            }
        },
    }
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "deploy/control-plane/tools/filter_legacy_release_env.py"),
        ],
        input=yaml.safe_dump(deployment),
        text=True,
        capture_output=True,
        check=True,
    )
    rendered = yaml.safe_load(result.stdout)
    names = {
        item["name"]
        for item in rendered["spec"]["template"]["spec"]["containers"][0]["env"]
    }

    assert names == {"GPU_FAULT_REQUIRED_AGENT_ARTIFACT_SHA256"}
