from __future__ import annotations

import base64
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault_release import regional_release_orchestration as ORCHESTRATION_MODULE
from gpu_fault_release import regional_release_registry as REGISTRY_MODULE
from gpu_fault_release import regional_release_state as STATE_MODULE
from gpu_fault_release import rollout as ROLLOUT_MODULE

ROOT = Path(__file__).resolve().parents[2]


def cluster_target(
    tmp_path: Path, *, cidrs: tuple[str, ...], cluster_id: str = "gpu-a"
):
    token = tmp_path / f"{cluster_id}.token"
    token.write_text("t" * 32)
    return REGISTRY_MODULE.ClusterTarget(
        cluster_id=cluster_id,
        context=f"{cluster_id}-context",
        executor_irsa_role_arn="arn:aws:iam::1:role/a",
        region="us-east-1",
        hyperpod_cluster_name=f"hp-{cluster_id}",
        eks_cluster_arn=(f"arn:aws:eks:us-east-1:123456789012:cluster/{cluster_id}"),
        token_file=str(token),
        allowed_namespaces=("training",),
        agent_endpoint_allowed_cidrs=cidrs,
    )


class RegistrySecret:
    """An in-memory stand-in for the registry Secret and the kubectl round trip.

    ``write_registry`` renders with ``create secret --dry-run=client -o yaml``
    and persists with ``apply -f -``, then reads the Secret back and refuses to
    continue if it did not persist. Modelling both halves keeps that read-back
    check live, so these tests exercise the staging invariant instead of
    matching the text of the function that implements it.
    """

    dry_run = False

    def __init__(self, data: dict[str, list[dict[str, str]]] | None = None) -> None:
        self.data: dict[str, str] | None = (
            None
            if data is None
            else {
                key: base64.b64encode(json.dumps(value).encode()).decode()
                for key, value in data.items()
            }
        )
        self.applies = 0

    def payload(self, key: str) -> list[dict[str, str]] | None:
        encoded = (self.data or {}).get(key)
        return None if encoded is None else json.loads(base64.b64decode(encoded))

    def probe(self, _arguments) -> bool:
        return self.data is not None

    def run(self, arguments, *, input_text=None, capture=False, sensitive=False):
        if "apply" in arguments:
            self.data = json.loads(input_text)["data"]
            self.applies += 1
            return ""
        rendered = {
            key: base64.b64encode(Path(path).read_bytes()).decode()
            for key, path in (
                item.removeprefix("--from-file=").split("=", 1)
                for item in arguments
                if item.startswith("--from-file=")
            )
        }
        return json.dumps({"data": rendered})


def registry_release(secret: RegistrySecret, *targets):
    return SimpleNamespace(
        runner=secret,
        config=SimpleNamespace(namespace="gpu-fault-system", clusters=targets),
        _cpu=lambda *arguments: ["kubectl", *arguments],
        _get_json=lambda _arguments: {"data": dict(secret.data or {})},
    )


def test_registry_entry_requires_and_normalizes_agent_endpoint_cidrs(
    tmp_path: Path,
) -> None:
    entry = REGISTRY_MODULE.registry_entry(
        cluster_target(tmp_path, cidrs=("10.0.1.15/16",))
    )

    assert entry["agent_endpoint_allowed_cidrs"] == ["10.0.0.0/16"]
    assert entry["token"] == "t" * 32

    with pytest.raises(
        REGISTRY_MODULE.ReleaseError, match="requires Agent endpoint CIDRs"
    ):
        REGISTRY_MODULE.registry_entry(cluster_target(tmp_path, cidrs=()))


def test_registry_stage_keeps_the_original_secret_until_commit(tmp_path: Path) -> None:
    original = REGISTRY_MODULE.registry_entry(
        cluster_target(tmp_path, cidrs=("10.0.0.0/24",), cluster_id="gpu-a")
    )
    target = cluster_target(tmp_path, cidrs=("10.0.1.0/24",), cluster_id="gpu-b")
    secret = RegistrySecret({REGISTRY_MODULE.REGISTRY_CURRENT_KEY: [original]})
    release = registry_release(secret, target)

    assert REGISTRY_MODULE.stage_registry(release) is True
    assert [item["cluster_id"] for item in secret.payload("clusters.json")] == ["gpu-b"]
    assert secret.payload("previous-clusters.json") == [original], (
        "staging must keep the pre-release registry as the backup"
    )

    REGISTRY_MODULE.commit_registry_update(release)

    assert [item["cluster_id"] for item in secret.payload("clusters.json")] == ["gpu-b"]
    assert secret.payload("previous-clusters.json") is None, (
        "commit must drop the backup so the next release can stage again"
    )


def test_restaging_does_not_overwrite_the_backup_with_the_staged_registry(
    tmp_path: Path,
) -> None:
    """A retried release must still be able to roll back to the original."""

    original = REGISTRY_MODULE.registry_entry(
        cluster_target(tmp_path, cidrs=("10.0.0.0/24",), cluster_id="gpu-a")
    )
    staged = REGISTRY_MODULE.registry_entry(
        cluster_target(tmp_path, cidrs=("10.0.1.0/24",), cluster_id="gpu-b")
    )
    target = cluster_target(tmp_path, cidrs=("10.0.2.0/24",), cluster_id="gpu-c")
    secret = RegistrySecret(
        {
            REGISTRY_MODULE.REGISTRY_CURRENT_KEY: [staged],
            REGISTRY_MODULE.REGISTRY_BACKUP_KEY: [original],
        }
    )
    release = registry_release(secret, target)

    assert REGISTRY_MODULE.stage_registry(release) is True
    assert secret.payload("previous-clusters.json") == [original]

    assert REGISTRY_MODULE.restore_registry_backup(release) is True
    assert [item["cluster_id"] for item in secret.payload("clusters.json")] == ["gpu-a"]


def test_registry_stage_is_a_no_op_when_the_registry_already_matches(
    tmp_path: Path,
) -> None:
    target = cluster_target(tmp_path, cidrs=("10.0.0.0/24",), cluster_id="gpu-a")
    secret = RegistrySecret(
        {REGISTRY_MODULE.REGISTRY_CURRENT_KEY: [REGISTRY_MODULE.registry_entry(target)]}
    )
    release = registry_release(secret, target)

    assert REGISTRY_MODULE.stage_registry(release) is False
    assert REGISTRY_MODULE.restore_registry_backup(release) is False
    assert secret.applies == 0, "an unchanged registry must not be rewritten"


def test_registry_update_retries_resource_version_conflict_without_lost_update(
    tmp_path: Path,
) -> None:
    target = cluster_target(tmp_path, cidrs=("10.0.0.0/24",), cluster_id="gpu-a")
    concurrent = REGISTRY_MODULE.registry_entry(
        cluster_target(tmp_path, cidrs=("10.0.1.0/24",), cluster_id="gpu-b")
    )
    secret = {
        "metadata": {"resourceVersion": "1"},
        "data": {"clusters.json": base64.b64encode(b"[]").decode()},
    }

    class Runner:
        dry_run = False

        def __init__(self) -> None:
            self.calls = 0

        def run(self, arguments, *, sensitive=False, **_kwargs):
            assert sensitive is True
            self.calls += 1
            patch = json.loads(arguments[arguments.index("-p") + 1])
            if self.calls == 1:
                secret["metadata"]["resourceVersion"] = "2"
                secret["data"]["clusters.json"] = base64.b64encode(
                    json.dumps([concurrent]).encode()
                ).decode()
                raise REGISTRY_MODULE.ReleaseError("resourceVersion conflict")
            assert patch[0]["value"] == "2"
            secret["data"]["clusters.json"] = patch[1]["value"]
            secret["metadata"]["resourceVersion"] = "3"
            return ""

    runner = Runner()

    class Release:
        config = SimpleNamespace(namespace="gpu-fault-system")

        @staticmethod
        def _cpu(*arguments):
            return ["kubectl", *arguments]

        @staticmethod
        def _get_json(_arguments):
            return copy.deepcopy(secret)

    release = Release()
    release.runner = runner

    REGISTRY_MODULE.update_registry(release, target, remove=False)

    registrations = json.loads(base64.b64decode(secret["data"]["clusters.json"]))
    assert [item["cluster_id"] for item in registrations] == ["gpu-a", "gpu-b"]
    assert runner.calls == 2


class IdleProbe:
    """Answers the two kubectl calls the remote-command idle check makes."""

    def __init__(self, *, pod: str = "cpu-ingress-0", exec_results=()) -> None:
        self.pod = pod
        self.exec_results = list(exec_results)
        self.execs = 0

    def run(self, arguments, *, capture=False, **_kwargs) -> str:
        if "exec" in arguments:
            self.execs += 1
            result = self.exec_results.pop(0)
            if isinstance(result, Exception):
                raise result
            return result
        return self.pod

    def probe(self, _arguments) -> bool:
        """The memoised Pod is still there, so a failed exec really failed.

        A replaced Pod is the other case, and it buys an extra attempt on a
        freshly resolved name; these tests are about the retry budget for an exec
        that reached the Pod and died, so the Pod has to still exist.
        """

        return True


def idle_release(probe: IdleProbe):
    return SimpleNamespace(
        runner=probe,
        config=SimpleNamespace(namespace="gpu-fault-system"),
        _cpu=lambda *arguments: ["kubectl", *arguments],
    )


def remote_commands_are_idle(probe: IdleProbe) -> bool:
    return ROLLOUT_MODULE.RegionalRelease._remote_commands_are_idle(idle_release(probe))


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ROLLOUT_MODULE.time, "sleep", lambda _seconds: None)


def test_remote_command_idle_check_reads_the_queue_and_reports_backlog() -> None:
    assert remote_commands_are_idle(IdleProbe(exec_results=["\n"])) is True
    busy = IdleProbe(exec_results=["{'PENDING': 2}"])

    assert remote_commands_are_idle(busy) is False
    assert busy.execs == 1, "a decisive answer must not be re-probed"


def test_remote_command_idle_check_treats_a_missing_ingress_pod_as_idle() -> None:
    probe = IdleProbe(pod="")

    assert remote_commands_are_idle(probe) is True
    assert probe.execs == 0, "there is no Pod to exec into"


def test_remote_command_idle_check_retries_a_transient_exec_failure() -> None:
    probe = IdleProbe(
        exec_results=[ROLLOUT_MODULE.ReleaseError("exec closed"), "{'LEASED': 1}"]
    )

    assert remote_commands_are_idle(probe) is False
    assert probe.execs == 2


def test_remote_command_idle_check_has_bounded_retries() -> None:
    """Fail closed after a bounded number of attempts, never in a loop."""

    probe = IdleProbe(
        exec_results=[ROLLOUT_MODULE.ReleaseError("exec closed") for _ in range(3)]
    )

    with pytest.raises(ROLLOUT_MODULE.ReleaseError, match="exec closed"):
        remote_commands_are_idle(probe)

    assert probe.execs == 3


def upgrade_release(*, staged: bool):
    """A release whose only recorded effect is how it applied the CPU role."""

    applies: list[dict] = []
    commits: list[None] = []
    release = SimpleNamespace(
        applies=applies,
        commits=commits,
        state={"phase": "preflight"},
        config=SimpleNamespace(clusters=(), upgrade_max_parallel_clusters=1),
        _upload_release=lambda _diff: None,
        _validate_release_quick=lambda _plan: None,
        _stage_registry=lambda: staged,
        _commit_registry_update=lambda: commits.append(None),
        _capture_active_agent_node_sets=lambda: {},
        _wait_candidate_cpu_agent_heartbeats=lambda _expected, **_kwargs: None,
        _apply_cpu=lambda **kwargs: applies.append(kwargs),
        _save_state=lambda phase, **updates: release.state.update(
            {"phase": phase, **updates}
        ),
    )
    return release


def run_registry_and_cpu_stage(release) -> None:
    diff = ORCHESTRATION_MODULE.ReleaseDiff(
        kind=ORCHESTRATION_MODULE.ReleaseChangeKind.FULL,
        changed=frozenset({"regional_cluster_registry"}),
    )
    ORCHESTRATION_MODULE.run_upgrade_phases(
        release,
        diff=diff,
        plan=ORCHESTRATION_MODULE.ReleaseExecutionPlan(
            nodes=(
                ORCHESTRATION_MODULE.ReleaseComponent.REGISTRY,
                ORCHESTRATION_MODULE.ReleaseComponent.CPU_STAGE,
                ORCHESTRATION_MODULE.ReleaseComponent.VERIFY,
            )
        ),
        previous={},
        completed_phases=set(),
        completed_clusters=set(),
        registry_staged=False,
    )


def test_registry_stage_forces_cpu_secret_reload() -> None:
    """A rewritten registry Secret is only picked up by restarted CPU Pods."""

    release = upgrade_release(staged=True)

    run_registry_and_cpu_stage(release)

    assert [item["force_restart"] for item in release.applies] == [True]
    assert release.applies[0]["finalize"] is False
    assert len(release.commits) == 1, (
        "a staged registry must be committed once the release verified"
    )


def test_unchanged_registry_does_not_force_a_cpu_restart() -> None:
    release = upgrade_release(staged=False)

    run_registry_and_cpu_stage(release)

    assert [item["force_restart"] for item in release.applies] == [False]
    assert release.commits == [], "there is no backup to commit"


def test_cpu_role_config_snapshot_rejects_sensitive_keys() -> None:
    class SnapshotRelease:
        config = SimpleNamespace(namespace="gpu-fault-system")

        @staticmethod
        def _cpu(*args):
            return ["kubectl", *args]

        @staticmethod
        def _get_json(arguments):
            if "deployment" in arguments:
                return {
                    "spec": {
                        "template": {
                            "spec": {
                                "containers": [
                                    {
                                        "envFrom": [
                                            {
                                                "configMapRef": {
                                                    "name": (
                                                        "gpu-fault-api-ha-config-core"
                                                    )
                                                }
                                            }
                                        ]
                                    }
                                ]
                            }
                        }
                    }
                }
            return {}

        @staticmethod
        def _config_map_data(_name):
            return {"GPU_FAULT_DATABASE_PASSWORD": "unsafe"}

    with pytest.raises(STATE_MODULE.ReleaseError, match="sensitive-looking keys"):
        STATE_MODULE.cpu_role_config_maps(SnapshotRelease())


def test_previous_state_captures_the_live_administrator_config() -> None:
    class SnapshotRelease:
        config = SimpleNamespace(namespace="gpu-fault-system")

        @staticmethod
        def _cpu(*args):
            return ["kubectl", *args]

        @staticmethod
        def _get_json(arguments):
            name = arguments[arguments.index("deployment") + 1]
            return {
                "spec": {
                    "replicas": (3 if name == "gpu-fault-telemetry-spool-worker" else 6)
                }
            }

    snapshots = {
        "gpu-fault-api-ha-config-telemetry": {"GPU_FAULT_TELEMETRY_SPOOL": "true"},
        "gpu-fault-control-worker-config-core": {
            "GPU_FAULT_REMEDIATION_MAX_ACTIVE_REGION": "128",
            "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_CLUSTER": "4",
            "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_NODE": "1",
            "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_FAILURE_DOMAIN": "1",
            "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_RESOURCE_CLASS": "4",
        },
    }

    config = STATE_MODULE.captured_admin_config(SnapshotRelease(), snapshots)

    assert config.capacity.control_worker_replicas == 6
    assert config.capacity.telemetry_spool.enabled is True
    assert config.capacity.telemetry_spool.replicas == 3
    assert config.capacity.remediation.max_active_region == 128


def test_previous_state_captures_node_counts_only_from_the_live_environment() -> None:
    # A rollback restores what this returns, so a node count that was never in
    # the live ConfigMaps must not be invented from the release defaults (the
    # Aurora 0.5/8 fabrication had exactly that shape).
    class SnapshotRelease:
        config = SimpleNamespace(namespace="gpu-fault-system")

        @staticmethod
        def _cpu(*args):
            return ["kubectl", *args]

        @staticmethod
        def _get_json(arguments):
            name = arguments[arguments.index("deployment") + 1]
            return {
                "spec": {
                    "replicas": (0 if name == "gpu-fault-telemetry-spool-worker" else 6)
                }
            }

    declared = STATE_MODULE.captured_admin_config(
        SnapshotRelease(),
        {
            "gpu-fault-control-worker-config-core": {
                "GPU_FAULT_CAPACITY_LARGEST_CLUSTER_NODE_COUNT": "1000",
                "GPU_FAULT_CAPACITY_MANAGED_NODE_COUNT": "4000",
            },
            "gpu-fault-api-ha-config-processor": {
                "GPU_FAULT_PROCESSOR_MAX_CLUSTER_QUEUE_DEPTH": "4096"
            },
        },
    )
    assert declared.capacity.largest_cluster_node_count == 1000
    assert declared.capacity.managed_node_count == 4000
    assert declared.fault_reserved_cluster_depth() == 1000

    legacy = STATE_MODULE.captured_admin_config(
        SnapshotRelease(),
        {
            "gpu-fault-api-ha-config-processor": {
                "GPU_FAULT_PROCESSOR_MAX_CLUSTER_QUEUE_DEPTH": "1024"
            }
        },
    )
    assert legacy.processor.max_cluster_queue_depth == 1024
    assert legacy.capacity.largest_cluster_node_count == 256
    assert legacy.capacity.managed_node_count == 256
