from __future__ import annotations

from pathlib import Path

import yaml

from gpu_fault.admin.config import ProcessorConfig, default_admin_config

ROOT = Path(__file__).resolve().parents[2]
GENERATED = ROOT / "deploy/control-plane/regional/generated"


def _environment(path: Path) -> dict[str, str]:
    environment = {}
    for document in yaml.safe_load_all(path.read_text(encoding="utf-8")):
        if not isinstance(document, dict):
            continue
        template = document.get("spec", {}).get("template", {})
        containers = template.get("spec", {}).get("containers", [])
        for container in containers:
            for item in container.get("env", []):
                if "value" in item:
                    environment[item["name"]] = str(item["value"])
    return environment


def test_control_plane_manifests_default_to_dedicated_hot_state() -> None:
    for relative in (
        "deploy/control-plane/base/control-plane-deployment.yaml",
        "deploy/control-plane/regional/regional-control-plane-patch.yaml",
    ):
        environment = _environment(ROOT / relative)
        assert environment["GPU_FAULT_POSTGRES_HOT_STATE_MODE"] == "dedicated"
        assert environment["GPU_FAULT_PROCESSOR_QUEUE_STATE_MODE"] == "dedicated"


def test_hyperpod_deploy_keeps_dedicated_and_role_split_defaults() -> None:
    script = (ROOT / "deploy/hyperpod/deploy.sh").read_text(encoding="utf-8")

    assert "GPU_FAULT_POSTGRES_HOT_STATE_MODE:-dedicated" in script
    assert "GPU_FAULT_PROCESSOR_QUEUE_STATE_MODE:-dedicated" in script
    # The split is applied from the checked-in manifests before the
    # environment below is set on both tiers, and the deploy fails if
    # the worker tier did not come up.
    assert "control-plane/regional/generated/${manifest}.yaml" in script
    assert "verify_control_plane_role_split.py" in script
    assert "deployment/gpu-fault-control-worker" in script
    assert "enable-control-plane-role-split" not in script


def test_capacity_defaults_follow_the_perf_evidence() -> None:
    config = default_admin_config()

    # 性能压测验收方案 §2 + product requirement: one whole-cluster correlated
    # fault is >= largest_cluster_node_count fault-priority requests, and the
    # reserve is what admits them once routine traffic fills the lane.
    assert (
        config.fault_reserved_cluster_depth()
        >= config.capacity.largest_cluster_node_count
    )
    # §13.4: 1000-node cluster, depth 1024 -> 243 HTTP 429; depth 4096 -> 0.
    assert ProcessorConfig().max_cluster_queue_depth >= 4096
    # §13.4 again, for the legacy imperative path that still sets these.
    script = (ROOT / "deploy/hyperpod/deploy.sh").read_text(encoding="utf-8")
    assert "GPU_FAULT_PROCESSOR_MAX_CLUSTER_QUEUE_DEPTH:-4096" in script
    assert "GPU_FAULT_PROCESSOR_FAULT_RESERVED_CLUSTER_DEPTH:-512" in script
    # §13.4 / §2: the shipped ConfigMaps are what production actually runs.
    for name in (
        "gpu-fault-control-worker-config-processor.yaml",
        "gpu-fault-api-ha-config-processor.yaml",
    ):
        data = yaml.safe_load((GENERATED / name).read_text(encoding="utf-8"))["data"]
        assert data["GPU_FAULT_PROCESSOR_FAULT_RESERVED_CLUSTER_DEPTH"] == "512"
        assert data["GPU_FAULT_PROCESSOR_MAX_CLUSTER_QUEUE_DEPTH"] == "4096"
    # §13.3: node counts ship to the runtime so the reserve can be checked
    # against the wave at start-up and read back from the live state.
    for name in (
        "gpu-fault-control-worker-config-core.yaml",
        "gpu-fault-api-ha-config-core.yaml",
    ):
        data = yaml.safe_load((GENERATED / name).read_text(encoding="utf-8"))["data"]
        assert data["GPU_FAULT_CAPACITY_LARGEST_CLUSTER_NODE_COUNT"] == "512"
        assert data["GPU_FAULT_CAPACITY_MANAGED_NODE_COUNT"] == "512"


def test_hyperpod_deploy_runs_schema_ddl_in_a_job() -> None:
    """DDL must not run inside an API process.

    The in-process bootstrap takes SHARE ROW EXCLUSIVE on
    gpu_fault_processor_queue, which blocks every admission, claim and
    completion while it holds. Pod replacement makes process startup a
    recurring event, so a `true` default turns a one-off into a recurring
    stall. The manifests say false; this asserts deploy.sh's imperative
    `set env` does not put it back to true, and that the migration Job it
    replaces the bootstrap with actually runs.
    """

    script = (ROOT / "deploy/hyperpod/deploy.sh").read_text(encoding="utf-8")

    assert "GPU_FAULT_POSTGRES_AUTO_SCHEMA_INIT:-false" in script
    assert "postgres-schema-ensure-job.yaml" in script
    # The three Jobs run in order through one helper: online index build
    # (F-J3), transactional ensure, read-only preflight gate.
    assert "run_postgres_job gpu-fault-postgres-index-build" in script
    assert "run_postgres_job gpu-fault-postgres-schema-ensure" in script
    assert "run_postgres_job gpu-fault-postgres-schema-preflight" in script
    assert 'postgres-schema-ensure-job.yaml" 12m' in script
    assert (
        script.index("postgres-index-build-job.yaml")
        < script.index('postgres-schema-ensure-job.yaml" 12m')
        < script.index("postgres-schema-preflight-job.yaml")
    )
    for relative in (
        "deploy/control-plane/base/control-plane-deployment.yaml",
        "deploy/control-plane/regional/regional-control-plane-patch.yaml",
    ):
        environment = _environment(ROOT / relative)
        assert environment["GPU_FAULT_POSTGRES_AUTO_SCHEMA_INIT"] == "false"
