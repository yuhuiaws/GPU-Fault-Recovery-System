from __future__ import annotations

import argparse
import base64
import json
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin import failure_domain_map as domains
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.failure_domains import (
    FAILURE_DOMAIN_FILE,
    FAILURE_DOMAIN_MAP_ANNOTATION,
    failure_domain_map,
    failure_domain_map_sha256,
)
from gpu_fault_release.regional_deployment_inventory import CPU_RUNTIME_DEPLOYMENTS
from scripts.e2e.regional import boot_acceptance_common as boot
from scripts.e2e.regional import boot_membership_observation as membership
from scripts.e2e.regional import capacity_acceptance_base as capacity
from scripts.e2e.regional import regional_commands
from scripts.e2e.regional import run_boot019_admin_lifecycle as lifecycle
from scripts.e2e.regional import run_cap005_postgres_suite as cap005
from tests.regional._cov95_focused_mock_receipts import write_focused_receipt
from tests.regional.test_acceptance_alignment_cpu_observation import Inventory
from tools import pytest_result_identity


@pytest.mark.parametrize("invoke", [boot.run, capacity.command])
def test_owned_commands_use_the_public_supervised_boundary(
    invoke: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[Any, dict[str, Any]]] = []

    def run(command: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "observed", "")

    monkeypatch.setattr(regional_commands, "run_command", run)

    result = invoke(
        ["tool", "argument"], input_text="input", timeout=17, env={"X": "1"}
    )

    assert result.stdout == "observed", (
        "successful stdout must remain available to parsers"
    )
    assert len(calls) == 1, "one command must have one supervised invocation"
    assert calls[0][1]["timeout_seconds"] == 17, (
        "the caller's deadline must be preserved"
    )
    assert calls[0][1]["environment"] == {"X": "1"}, (
        "environment must reach the supervisor"
    )
    assert calls[0][1]["input_text"] == "input", "stdin must not be moved into argv"


@pytest.mark.parametrize(
    "invoke,error_type",
    [(boot.run, boot.BootAcceptanceError), (capacity.command, capacity.CapError)],
)
def test_owned_command_failures_do_not_expose_raw_diagnostics(
    invoke: Any, error_type: type[Exception], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        regional_commands,
        "run_command",
        lambda command, **_k: subprocess.CompletedProcess(
            command, 1, "", "password=private-value-that-must-not-be-published"
        ),
    )

    with pytest.raises(error_type) as failure:
        invoke(["tool"])

    assert "private-value-that-must-not-be-published" not in str(failure.value), (
        "the public failure must contain only the shared safe diagnostic"
    )


def test_boot_build_umask_runs_inside_the_supervised_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(
        regional_commands,
        "run_command",
        lambda command, **_k: commands.append(list(command))
        or subprocess.CompletedProcess(command, 0, "", ""),
    )

    boot.run(["python", "build.py"], umask=0o077)

    assert commands == [
        [
            "sh",
            "-c",
            'umask "$1"; shift; exec "$@"',
            "gpu-fault-umask",
            "077",
            "python",
            "build.py",
        ]
    ], "threaded builds must not use a preexec_fn outside supervision"


@pytest.fixture
def cap005_transport(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setattr(cap005, "validate_server", lambda _url: None)
    created: list[str] = []
    dropped: list[str] = []
    monkeypatch.setattr(
        cap005, "_create_database", lambda _url, name: created.append(name)
    )
    monkeypatch.setattr(
        cap005, "_drop_database", lambda _url, name: dropped.append(name)
    )
    monkeypatch.setattr(cap005, "_database_exists", lambda *_a: False)
    checkout = cap005.ROOT
    identity = pytest_result_identity.source_identity(checkout)
    # The temporary working directory models the runner's own checkout; the
    # receipt verifier itself is real.
    monkeypatch.setattr(cap005, "ROOT", tmp_path)

    def mock_source_identity(root: Path) -> str:
        assert root == tmp_path, "CAP005 must bind its supplied mock working directory"
        return identity

    monkeypatch.setattr(pytest_result_identity, "source_identity", mock_source_identity)
    state: dict[str, Any] = {
        "commands": [],
        "created": created,
        "dropped": dropped,
        "receipt": "complete",
        "shards": [],
    }

    def shards(workdir: Path, _report_dir: Path, workers: int) -> dict[str, Any]:
        # The PostgreSQL suite stage: owned PG16 instances through the shard
        # launcher (exercised in test_cap005_parallel); here only its ordering
        # relative to the supervised contract child matters.
        assert workdir == tmp_path, "shards run in the runner's checkout"
        state["shards"].append(workers)
        return {"source_identity": identity, "workers": workers, "executed_tests": 2}

    monkeypatch.setattr(cap005, "run_sharded_postgres", shards)

    def run(command: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        environment = kwargs["environment"]
        options = argparse.ArgumentParser(add_help=False)
        options.add_argument("--junitxml", "--junit-xml")
        arguments, _ = options.parse_known_args(
            [*command, *shlex.split(environment.get("PYTEST_ADDOPTS", ""))]
        )
        state["commands"].append(
            {
                "command": list(command),
                "environment": environment,
                "junit": arguments.junitxml,
            }
        )
        if command[1:3] == ["-m", "pytest"]:
            write_focused_receipt(
                command,
                environment=environment,
                cwd=checkout,
                nodeids=[
                    "tests/store/test_store_contracts.py::test_store_classes_cover_application_protocol",
                    "tests/store/test_store_contracts.py::test_processor_queue_contract[postgres]",
                ],
                defect=state["receipt"],
            )
        if arguments.junitxml:
            Path(arguments.junitxml).write_text(
                '<testsuite tests="2" failures="0" errors="0" skipped="0"/>',
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(regional_commands, "run_command", run)
    return state


def test_cap005_pytest_children_are_supervised(
    tmp_path: Path, cap005_transport: dict[str, Any]
) -> None:
    result = cap005.run_suite("postgresql://localhost:55432/postgres", tmp_path)

    assert result["status"] == "PASS", (
        "a complete receipt and both real child output contracts are required",
        result,
    )
    commands = cap005_transport["commands"]
    assert cap005_transport["shards"] == [1] and len(commands) == 1, (
        "the owned-shard stage and then the Store contract child must both run"
    )
    assert commands[0]["command"][commands[0]["command"].index("-n") + 1] == "0", (
        "this runner must not use shared PG xdist"
    )
    assert result["exit_codes"]["contract"] == 0, (
        "the real receipt verifier must accept complete evidence through the public boundary"
    )
    assert cap005_transport["created"] == cap005_transport["dropped"], (
        "receipt failure must preserve cleanup of the generated fake database"
    )


@pytest.mark.parametrize("receipt", ["missing", "failed"])
def test_cap005_refuses_missing_or_failed_child_receipts(
    tmp_path: Path, cap005_transport: dict[str, Any], receipt: str
) -> None:
    cap005_transport["receipt"] = receipt
    result = cap005.run_suite("postgresql://localhost:55432/postgres", tmp_path)
    assert result["status"] == "FAIL" and result["exit_codes"]["contract"] == 1, (
        "the real receipt verifier must reject missing or failed evidence despite exit zero",
        result,
    )
    assert len(cap005_transport["commands"]) == 1, (
        "receipt refusal must not retry the supervised contract child"
    )
    assert cap005_transport["created"] == cap005_transport["dropped"], (
        "receipt failure must preserve cleanup of the generated fake database"
    )


@pytest.mark.parametrize("handoff", ["junit", "isolated-postgres"])
def test_cap005_contract_child_receives_explicit_report_and_isolated_database(
    tmp_path: Path, cap005_transport: dict[str, Any], handoff: str
) -> None:
    result = cap005.run_suite("postgresql://localhost:55432/postgres", tmp_path)
    assert result["exit_codes"]["contract"] == 0, (
        "a complete receipt must reach the caller before testing its output handoff",
        result,
    )
    (contract,) = cap005_transport["commands"]
    if handoff == "junit":
        assert contract["junit"] is not None, (
            "the direct pytest child needs an explicit JUnit output argument; "
            "the sanitizer deliberately discards inherited PYTEST_ADDOPTS"
        )
        assert Path(contract["junit"]).name == "contract.xml", (
            "the child must receive the exact report path later consumed by CAP005"
        )
        assert "PYTEST_ADDOPTS" not in contract["environment"], (
            "the direct contract report must not depend on stripped inherited options"
        )
        assert "--durations=20" in contract["command"], (
            "the explicit direct-child arguments must preserve duration reporting"
        )
        assert contract["command"][contract["command"].index("-n") + 1] == "0", (
            "the direct Store contract must remain serial independently of defaults"
        )
    else:
        assert contract["environment"].get("GPU_FAULT_TEST_POSTGRES_URL") == (
            cap005.database_url(
                "postgresql://localhost:55432/postgres", cap005_transport["created"][0]
            )
        ), "the authorized private database must reach the Store contract's PG variants"


def test_cap005_clears_inherited_selection_and_authority_for_both_child_types(
    tmp_path: Path, cap005_transport: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    inherited = {
        "GPU_FAULT_STORE_URL": "postgresql://production.invalid/business",
        "GPU_FAULT_STORE_URL_FILE": "/private/production-dsn",
        "GPU_FAULT_TEST_POSTGRES_URL": "postgresql://foreign.invalid/other",
        "GPU_FAULT_EXECUTION_TOKEN": "unit-not-a-real-token",
        "AWS_ACCESS_KEY_ID": "unit-not-a-real-key",
        "AWS_PROFILE": "production",
        "PGHOST": "production.invalid",
        "PGSERVICE": "production",
        "PGPASSFILE": "/private/production-pgpass",
        "KUBECONFIG": "/private/production-kubeconfig",
        "PYTEST_ADDOPTS": "-k wrong-selection",
        "PYTEST_GPU_FAULT_CASE_REPORT": "/private/foreign-report.json",
        "PYTEST_GPU_FAULT_PARTITION_COUNT": "2",
        "PYTEST_GPU_FAULT_PARTITION_INDEX": "1",
        "PYTEST_XDIST_WORKER": "gw1",
        "PYTEST_XDIST_WORKER_COUNT": "2",
        "PYTEST_CURRENT_TEST": "foreign-session",
        "MAKEFLAGS": "--eval=unapproved",
    }
    for name, value in inherited.items():
        monkeypatch.setenv(name, value)

    result = cap005.run_suite("postgresql://localhost:55432/postgres", tmp_path)

    assert result["status"] == "PASS", result
    assert len(cap005_transport["commands"]) == 1, (
        "the supervised contract child must be inspected"
    )
    for child in cap005_transport["commands"]:
        environment = child["environment"]
        assert environment["GPU_FAULT_STORE_URL"] == "", (
            "both child types must explicitly disable the production Store URL"
        )
        assert environment["GPU_FAULT_TEST_POSTGRES_URL"] == cap005.database_url(
            "postgresql://localhost:55432/postgres", result["database"]
        ), "only the generated database may reach either suite"
        assert environment["KUBECONFIG"] == "/dev/null", (
            "database test children must have no Kubernetes access"
        )
        allowed = {
            "GPU_FAULT_STORE_URL",
            "GPU_FAULT_TEST_POSTGRES_URL",
            "KUBECONFIG",
            "PYTEST_ADDOPTS",
            "PYTEST_GPU_FAULT_CASE_REPORT",
        }
        assert not (set(inherited) - allowed) & set(environment), (
            "inherited credentials, workers, shards and make options must be absent"
        )
        assert (
            environment.get("PYTEST_GPU_FAULT_CASE_REPORT")
            != inherited["PYTEST_GPU_FAULT_CASE_REPORT"]
        ), "the previous report must not be reused by either child"
    (contract,) = cap005_transport["commands"]
    assert "PYTEST_ADDOPTS" not in contract["environment"], (
        "the direct child must carry its reporting options in argv"
    )


@pytest.fixture
def boot019_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> SimpleNamespace:
    namespace = "test-namespace"
    site = SimpleNamespace(
        release_config={
            "cpu_kubeconfig": str(tmp_path / "synthetic-cpu"),
            "gpu_kubeconfig": str(tmp_path / "synthetic-gpu"),
            "namespace": namespace,
            "clusters": [{"cluster_id": "cluster-a", "context": "gpu-context"}],
        },
        environment={},
    )
    monkeypatch.setattr(lifecycle, "load_site", lambda *_a, **_k: site)
    monkeypatch.setattr(
        lifecycle,
        "fetch_installation_resource_registry",
        lambda _site: SimpleNamespace(
            resources=[
                SimpleNamespace(
                    resource_key="cluster/cluster-a/eks",
                    status=SimpleNamespace(value="ACTIVE"),
                )
            ]
        ),
    )
    inventory = Inventory()
    deployments = {item["metadata"]["name"]: item for item in inventory.deployments}
    for deployment in deployments.values():
        deployment.update(apiVersion="apps/v1", kind="Deployment")
        deployment["metadata"]["namespace"] = namespace
    nodes = [
        {
            "kind": "Node",
            "metadata": {
                "name": "gpu-node",
                "uid": "gpu-node-uid",
                "labels": {"sagemaker.amazonaws.com/instance-group-name": "group-a"},
            },
        }
    ]
    manifest = domains.failure_domain_configmap(
        failure_domain_map("cluster-a", nodes), namespace=namespace
    )
    manifest["metadata"]["uid"] = "map-uid"
    digest = failure_domain_map_sha256(manifest)
    worker = deployments[domains.CONTROL_WORKER_DEPLOYMENT]
    worker["spec"]["template"]["metadata"]["annotations"] = {
        FAILURE_DOMAIN_MAP_ANNOTATION: digest
    }
    cpu = [
        "kubectl",
        "--kubeconfig",
        site.release_config["cpu_kubeconfig"],
        "-n",
        namespace,
    ]
    gpu = [
        "kubectl",
        "--kubeconfig",
        site.release_config["gpu_kubeconfig"],
        "--context",
        "gpu-context",
    ]
    reads: list[tuple[list[str], int, dict[str, Any] | None]] = [
        (
            [*cpu, "get", "secret", "gpu-fault-regional-clusters", "-o", "json"],
            300,
            {
                "data": {
                    "clusters.json": base64.b64encode(
                        b'[{"cluster_id":"cluster-a"}]'
                    ).decode()
                }
            },
        ),
        (
            [
                *cpu,
                "get",
                "configmap",
                "gpu-fault-regional-release-state",
                "-o",
                "json",
            ],
            300,
            {"data": {"state.json": '{"cluster_ids":["cluster-a"]}'}},
        ),
        (
            [*cpu, "get", "deployment", "gpu-fault-api-ha", "-o", "json"],
            300,
            deployments["gpu-fault-api-ha"],
        ),
        ([*cpu, "get", "deployment", "-o", "json"], 120, None),
        *[
            (
                [*cpu, "get", "pod,replicaset", "-l", f"app={name}", "-o", "json"],
                120,
                None,
            )
            for name in CPU_RUNTIME_DEPLOYMENTS
        ],
        ([*gpu, "get", "nodes", "-o", "json"], 120, {"items": nodes}),
        *[
            (
                [
                    *cpu,
                    "get",
                    kind,
                    name,
                    "--ignore-not-found",
                    "-o",
                    "json",
                    "--request-timeout=15s",
                ],
                20,
                value,
            )
            for kind, name, value in (
                ("configmap", domains.FAILURE_DOMAIN_CONFIGMAP, manifest),
                ("deployment", domains.CONTROL_WORKER_DEPLOYMENT, worker),
                ("configmap", domains.FAILURE_DOMAIN_CONFIGMAP, manifest),
            )
        ],
    ]
    commands: list[tuple[list[str], int]] = []

    def run(command: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert len(commands) < len(reads), "snapshot issued an unexpected extra read"
        expected, timeout, value = reads[len(commands)]
        assert list(command) == expected, (
            "snapshot must preserve the exact read order, resource and CPU/GPU scope"
        )
        assert kwargs["timeout_seconds"] == timeout, (
            "each snapshot read must retain its supervised deadline"
        )
        commands.append((list(command), kwargs["timeout_seconds"]))
        if value is None:
            return inventory.run(command, **kwargs)
        return subprocess.CompletedProcess(command, 0, json.dumps(value), "")

    for module in (regional_commands, membership, domains):
        monkeypatch.setattr(module, "run_command", run)
    backend = lifecycle.LiveAdminLifecycleBackend(
        site_path=tmp_path / "unused-site",
        gpu_cluster_arn="synthetic-arn",
        cluster_id=None,
        allowed_namespaces=("training",),
        join_state_dir=tmp_path / "join",
        run_dir=tmp_path,
    )
    return SimpleNamespace(
        backend=backend,
        inventory=inventory,
        manifest=manifest,
        commands=commands,
        expected_reads=[(command, timeout) for command, timeout, _value in reads],
        publication={
            "map_sha256": digest,
            "worker_uid": worker["metadata"]["uid"],
            "configmap_uid": manifest["metadata"]["uid"],
        },
    )


def test_boot019_snapshot_reads_are_supervised(
    boot019_snapshot: SimpleNamespace,
) -> None:
    result = boot019_snapshot.backend.snapshot()

    assert result["cpu_control_plane_ready"] is True, "snapshot must retain readiness"
    assert result["registry_secret_cluster_ids"] == ["cluster-a"], (
        "registry IDs must be parsed"
    )
    assert boot019_snapshot.commands == boot019_snapshot.expected_reads, (
        "all eleven snapshot reads must reach their supervised transport in order"
    )
    observed = result["membership_cpu"]
    assert observed["publication"] == boot019_snapshot.publication, (
        "snapshot must bind the fresh Node map to the observed worker and ConfigMap UIDs"
    )
    assert set(observed["deployments"]) == set(CPU_RUNTIME_DEPLOYMENTS), (
        "snapshot must retain every required CPU role"
    )
    for name, deployment in observed["deployments"].items():
        assert deployment["uid"] == name + "-uid", (
            "CPU role observations must retain the original Deployment identity"
        )
        assert [pod["uid"] for pod in deployment["pods"]] == [name + "-pod-uid"], (
            "snapshot must retain the complete owned Pod census for each CPU role"
        )


@pytest.mark.parametrize(
    ("kind", "message", "read_count"),
    [
        ("deployment", "CPU role inventory query failed", 4),
        ("pod,replicaset", "CPU Pod ownership query failed", 5),
    ],
)
def test_boot019_snapshot_refuses_failed_membership_reads(
    boot019_snapshot: SimpleNamespace, kind: str, message: str, read_count: int
) -> None:
    boot019_snapshot.inventory.failed = kind
    with pytest.raises(ValueError, match=message):
        boot019_snapshot.backend.snapshot()
    assert boot019_snapshot.commands == boot019_snapshot.expected_reads[:read_count], (
        "failed membership reads must stop the snapshot before subsequent observations"
    )


@pytest.mark.parametrize("invalid", ["pod-owner", "map-publication"])
def test_boot019_snapshot_requires_real_ownership_and_publication(
    boot019_snapshot: SimpleNamespace, invalid: str
) -> None:
    if invalid == "pod-owner":
        _replicaset, pod = boot019_snapshot.inventory.children[
            CPU_RUNTIME_DEPLOYMENTS[0]
        ]
        pod["metadata"]["ownerReferences"][0]["uid"] = "foreign-replicaset"
        error, message, read_count = ValueError, "CPU Pod is not owned", 5
    else:
        boot019_snapshot.manifest["data"][FAILURE_DOMAIN_FILE] = json.dumps(
            {"cluster-a": {"gpu-node": "different-group"}}
        )
        error, message, read_count = BootstrapError, "map changed after publication", 9
    with pytest.raises(error, match=message):
        boot019_snapshot.backend.snapshot()
    assert boot019_snapshot.commands == boot019_snapshot.expected_reads[:read_count], (
        "invalid ownership or publication must stop the real snapshot validation"
    )


@pytest.mark.parametrize(
    ("plane", "python"),
    [
        ("cpu", "/opt/gpu-fault/control-plane/bin/python"),
        ("gpu", "/opt/gpu-fault/executor/bin/python"),
    ],
)
def test_boot_pod_json_selects_the_installed_component(plane: str, python: str) -> None:
    commands: list[tuple[str, ...]] = []
    fixture = boot.SiteFixture.__new__(boot.SiteFixture)
    fixture.regional = SimpleNamespace(  # type: ignore[assignment]
        kubectl=lambda _plane, *args, **_kw: commands.append(args) or '{"ok": true}'
    )

    assert fixture.pod_json(plane, "test-pod", "import gpu_fault") == {"ok": True}, (
        "the component probe must retain its parsed response"
    )
    assert commands[0][commands[0].index("--") + 1] == python, (
        "Pod probes must not use the deliberately empty system Python"
    )


def test_capacity_fallback_executes_under_the_cpu_component() -> None:
    harness = capacity.CapHarnessBase.__new__(capacity.CapHarnessBase)
    harness.run_id = "captest"
    commands: list[tuple[str, ...]] = []
    harness.kubectl_json = lambda *_a: {  # type: ignore[method-assign]
        "items": [
            {
                "metadata": {"name": "worker", "uid": "worker-uid", "labels": {}},
                "spec": {"containers": [{"name": "worker"}]},
                "status": {
                    "phase": "Running",
                    "conditions": [{"type": "Ready", "status": "True"}],
                    "containerStatuses": [{"name": "worker", "ready": True}],
                },
            }
        ]
    }
    harness.kubectl = lambda *args, **_kw: commands.append(
        args
    ) or subprocess.CompletedProcess(  # type: ignore[method-assign]
        args, 0, "", ""
    )

    harness.drop_database_fallback("gpu_fault_captest_cap001")

    assert (
        commands[0][commands[0].index("--") + 1]
        == "/opt/gpu-fault/control-plane/bin/python"
    ), "CPU cleanup must use its installed psycopg/component environment"
