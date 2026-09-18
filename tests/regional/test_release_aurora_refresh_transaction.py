from __future__ import annotations

import base64
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gpu_fault.admin import bootstrap_services
from gpu_fault_release import regional_release_aurora_refresh as REFRESH
from gpu_fault_release import regional_release_orchestration as ORCHESTRATION
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_diff import (
    ReleaseComponent,
    build_execution_plan,
    diff_from_changed,
)
from gpu_fault_release.regional_release_progress import build_rollback_compensation_plan
from gpu_fault_release.regional_release_rollback_context import (
    rollback_identity_context,
)

NAMESPACE = "gpu-fault-system"
OLD = "registry/runtime@sha256:" + "1" * 64
CANDIDATE = "registry/runtime@sha256:" + "2" * 64
MASTER_ARN = "arn:aws:secretsmanager:us-east-1:123456789012:secret:rds!test"


def objects(image: str = OLD) -> list[dict]:
    return list(
        yaml.safe_load_all(
            REFRESH.render_aurora_refresh(
                namespace=NAMESPACE,
                runtime_image=image,
                wheel_config_map="previous-wheel",
                master_secret_arn=MASTER_ARN,
            )
        )
    )


class ObjectRunner:
    dry_run = False

    def __init__(self, documents: list[dict]) -> None:
        self.live = {
            REFRESH.resource_argument(item["apiVersion"], item["kind"]): copy.deepcopy(
                item
            )
            for item in documents
        }
        self.events: list[str] = []
        self.fail_read = False
        self.diff_code: int | None = None

    def run(self, args, **kwargs):
        if "get" in args:
            self.events.append("read")
            if self.fail_read:
                raise ReleaseError("Forbidden")
            if "jsonpath={.data.master-secret-arn}" in args:
                assert "secret" in args
                return base64.b64encode(MASTER_ARN.encode()).decode()
            resource = args[args.index("get") + 1]
            if resource == "cronjob":
                resource = "cronjob.batch"
            item = copy.deepcopy(self.live.get(resource))
            if item is not None:
                item["metadata"].update(uid="previous-uid", resourceVersion="2")
                item["status"] = {}
            return json.dumps(item) if item is not None else ""
        if "apply" in args:
            document = kwargs.get("input_text")
            if document is None:
                document = Path(args[args.index("-f") + 1]).read_text()
            documents = list(yaml.safe_load_all(document))
            if len(documents) == 1 and documents[0].get("kind") == "List":
                documents = documents[0]["items"]
            assert all(item["kind"] != "Secret" for item in documents), (
                "refresher restore must never apply a database Secret snapshot"
            )
            if "--dry-run=server" in args:
                self.events.append("dry-run")
            else:
                self.events.append("apply")
                for item in documents:
                    self.live[
                        REFRESH.resource_argument(item["apiVersion"], item["kind"])
                    ] = item
            return "configured"
        if "delete" in args:
            self.events.append("delete")
            self.live.pop(args[args.index("delete") + 1], None)
            return ""
        raise AssertionError(f"unexpected command: {args}")

    def probe_output(self, args, **kwargs):
        if "diff" not in args:
            return 0, self.run(args), ""
        self.events.append("diff")
        if self.diff_code is not None:
            return self.diff_code, "", ""
        documents = list(
            yaml.safe_load_all(Path(args[args.index("-f") + 1]).read_text())
        )
        if len(documents) == 1 and documents[0].get("kind") == "List":
            documents = documents[0]["items"]
        matches = all(
            self.live.get(REFRESH.resource_argument(item["apiVersion"], item["kind"]))
            == item
            for item in documents
        )
        return int(not matches), "", ""


def release(runner: ObjectRunner, **hooks: object) -> SimpleNamespace:
    defaults = dict(
        runner=runner,
        config=SimpleNamespace(
            namespace=NAMESPACE,
            aws_region="us-east-1",
            cpu_eks_arn="arn:aws:eks:us-east-1:123456789012:cluster/cpu",
            auto_rollback=True,
            delivery_component_digests={"aurora_refresh": "manifest"},
            clusters=(),
        ),
        runtime_image=CANDIDATE,
        wheel_cm="candidate-wheel",
        _cpu=lambda *args: ["kubectl", "--context", "cpu", *args],
        _get_json=lambda _args: {
            "kind": "Deployment",
            "metadata": {"name": "gpu-fault-api-ha", "namespace": NAMESPACE},
        },
        enforce_manifest_plan_pin=lambda: runner.events.append("pin"),
        _refresh_aurora_credentials=lambda **_kwargs: runner.events.append("refresh"),
    )
    return SimpleNamespace(**(defaults | hooks))


def test_renderer_substitutes_namespace_env_rbac_image_wheel_and_secret_reference() -> (
    None
):
    docs = list(
        yaml.safe_load_all(
            REFRESH.render_aurora_refresh(
                namespace="isolated",
                runtime_image=CANDIDATE,
                wheel_config_map="candidate-wheel",
                master_secret_arn=MASTER_ARN,
            )
        )
    )
    assert all(item["metadata"]["namespace"] == "isolated" for item in docs), (
        "refresher objects escaped the selected namespace"
    )
    binding = next(item for item in docs if item["kind"] == "RoleBinding")
    assert binding["subjects"][0]["namespace"] == "isolated"
    pod = docs[-1]["spec"]["jobTemplate"]["spec"]["template"]["spec"]
    container = pod["containers"][0]
    assert container["image"] == CANDIDATE
    env = {item["name"]: item["value"] for item in container["env"]}
    assert env["GPU_FAULT_NAMESPACE"] == "isolated"
    assert env["GPU_FAULT_AURORA_MASTER_SECRET_ARN"] == MASTER_ARN
    assert pod["volumes"][0]["configMap"]["name"] == "candidate-wheel"


def test_full_snapshot_restores_arguments_env_rbac_and_volumes_not_only_image() -> None:
    original = objects()
    original[-1]["spec"]["jobTemplate"]["spec"]["template"]["spec"]["containers"][0][
        "args"
    ] = ["--old-argument"]
    runner = ObjectRunner(original)
    instance = release(runner)
    previous = REFRESH.capture_aurora_refresh_snapshot(instance)
    assert "status" not in previous["objects"][-1]
    assert "uid" not in previous["objects"][-1]["metadata"]
    REFRESH.apply_aurora_refresh(instance)
    assert runner.live["cronjob.batch"] != original[-1]
    REFRESH.restore_aurora_refresh_snapshot(instance, previous)
    assert list(runner.live.values()) == original
    assert (
        runner.events.index("pin")
        < runner.events.index("dry-run")
        < runner.events.index("apply")
    )
    assert runner.events.index("apply") < runner.events.index("refresh")


def test_confirmed_absence_is_restored_without_deleting_the_database_secret() -> None:
    runner = ObjectRunner([])
    instance = release(runner)
    previous = REFRESH.capture_aurora_refresh_snapshot(instance)
    assert previous["absence_verified"] is True
    REFRESH.apply_aurora_refresh(instance)
    REFRESH.restore_aurora_refresh_snapshot(instance, previous)
    assert runner.live == {}
    assert runner.events.count("delete") == 4


def test_absence_without_the_cpu_namespace_anchor_is_rejected() -> None:
    runner = ObjectRunner([])
    instance = release(
        runner, _get_json=lambda _args: {"metadata": {"namespace": "other"}}
    )
    with pytest.raises(ReleaseError, match="namespace anchor"):
        REFRESH.capture_aurora_refresh_snapshot(instance)
    assert set(runner.events) == {"read"}


@pytest.mark.parametrize(
    "damage", ["namespace", "missing", "duplicate", "secret", "unverified"]
)
def test_invalid_snapshot_is_rejected_before_any_restore(damage: str) -> None:
    runner = ObjectRunner(objects())
    instance = release(runner)
    previous = REFRESH.capture_aurora_refresh_snapshot(instance)
    if damage == "namespace":
        previous["namespace"] = "other"
    elif damage == "missing":
        previous["objects"].pop()
    elif damage == "duplicate":
        previous["objects"].append(copy.deepcopy(previous["objects"][0]))
    elif damage == "secret":
        previous["objects"][0]["kind"] = "Secret"
    else:
        previous["objects"].pop()
        previous["absent"] = [
            {"resource": "cronjob.batch", "name": REFRESH.CRONJOB_NAME}
        ]
    runner.events.clear()
    with pytest.raises(ReleaseError, match="snapshot"):
        REFRESH.restore_aurora_refresh_snapshot(instance, previous)
    assert runner.events == []


@pytest.mark.parametrize("code", [2, 126, 255])
def test_drift_read_errors_stop_deploy_without_a_mutation(code: int) -> None:
    runner = ObjectRunner(objects())
    runner.diff_code = code
    with pytest.raises(ReleaseError, match="state is unknown"):
        REFRESH.aurora_refresh_drift(release(runner))
    assert runner.events == ["read", "diff"]


def test_manifest_only_refresher_changes_do_not_roll_cpu_gpu_or_run_schema() -> None:
    plan = build_execution_plan(diff_from_changed({"aurora_refresh_manifests"}))
    assert plan.nodes == (ReleaseComponent.AURORA_REFRESH, ReleaseComponent.VERIFY)
    compensation = build_rollback_compensation_plan(
        {
            "execution_plan": plan.as_dict(),
            "component_progress": {
                "schema_version": 1,
                "global": {"aurora-refresh": {"status": "STARTED"}},
                "clusters": {},
            },
        },
        (),
    )
    assert compensation.global_has(ReleaseComponent.AURORA_REFRESH), (
        "a started refresher mutation was omitted from rollback"
    )
    assert not compensation.restores_cpu, "refresher-only rollback selected CPU rollout"
    assert not compensation.restores_data_plane, (
        "refresher-only rollback selected GPU rollout"
    )


def test_managed_bootstrap_only_reconciles_iam_not_candidate_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = ObjectRunner(objects())
    before = copy.deepcopy(runner.live)
    monkeypatch.setattr(bootstrap_services, "_pod_identity_trust", lambda _cpu: {})
    monkeypatch.setattr(
        bootstrap_services,
        "_ensure_role",
        lambda *_args, **_kwargs: {"role_arn": "role"},
    )
    monkeypatch.setattr(
        bootstrap_services,
        "_ensure_pod_identity_association",
        lambda *_args, **_kwargs: {
            "association_id": "association",
            "ownership": "CREATED",
            "cluster_name": "cpu",
            "namespace": NAMESPACE,
            "service_account": REFRESH.CRONJOB_NAME,
        },
    )
    result = bootstrap_services.install_aurora_refresh(
        runner,
        repository_root=tmp_path,
        cpu=SimpleNamespace(account_id="123456789012"),
        cpu_kubeconfig=tmp_path / "cpu",
        namespace=NAMESPACE,
        site_id="test",
        release_manifest=tmp_path / "candidate-not-read.json",
        runtime_image=CANDIDATE,
        aurora={"master_secret_arn": MASTER_ARN},
        runtime_managed_by_release=True,
    )
    assert result["runtime_managed_by_release"] is True
    assert runner.live == before
    assert runner.events == []


class ReachedWorkloadRestore(RuntimeError):
    pass


def test_database_secret_cannot_be_restored_as_a_cpu_rollback_backup() -> None:
    runner = ObjectRunner(objects())
    compensation = build_rollback_compensation_plan(
        {
            "execution_plan": {"nodes": ["cpu-finalize", "verify"]},
            "component_progress": {
                "schema_version": 1,
                "global": {"cpu-finalize": {"status": "STARTED"}},
                "clusters": {},
            },
        },
        (),
    )
    with pytest.raises(
        ReleaseError, match="database credentials cannot be rolled back"
    ):
        rollback_identity_context(
            release(runner),
            {
                "secret_backups": {
                    "cpu": {"source": "gpu-fault-aurora", "backup": "old-database"}
                }
            },
            compensation,
        )
    assert runner.events == []


def test_missing_full_snapshot_refuses_rollback_before_running_the_candidate() -> None:
    runner = ObjectRunner(objects(CANDIDATE))
    instance = release(runner)
    instance.state = {
        "execution_plan": {"nodes": ["aurora-refresh", "verify"]},
        "component_progress": {
            "schema_version": 1,
            "global": {"aurora-refresh": {"status": "STARTED"}},
            "clusters": {},
        },
    }
    with pytest.raises(ReleaseError, match="snapshot"):
        ORCHESTRATION.rollback_release(instance, state={"runtime_image": OLD})
    assert runner.events == [], (
        "rollback used the candidate despite an unprovable restore"
    )


def test_rollback_restores_good_refresher_before_refresh_and_workload_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = ObjectRunner(objects())
    instance = release(runner)
    previous = {
        "runtime_image": OLD,
        "aurora_refresh": REFRESH.capture_aurora_refresh_snapshot(instance),
    }
    REFRESH.apply_aurora_refresh(instance)
    runner.events.clear()
    instance.state = {
        "execution_plan": {"nodes": ["aurora-refresh", "verify"]},
        "component_progress": {
            "schema_version": 1,
            "global": {"aurora-refresh": {"status": "FAILED"}},
            "clusters": {},
        },
    }

    def refresh() -> None:
        image = runner.live["cronjob.batch"]["spec"]["jobTemplate"]["spec"]["template"][
            "spec"
        ]["containers"][0]["image"]
        assert image == OLD, "rollback tried to run the broken candidate refresher"
        runner.events.append("refresh")

    instance = release(
        runner,
        state=instance.state,
        _save_state=lambda phase, **_kwargs: runner.events.append(phase),
        _refresh_aurora_credentials=refresh,
        _require_no_inflight_installs=lambda **_kwargs: runner.events.append(
            "workload-gate"
        ),
    )
    monkeypatch.setattr(
        ORCHESTRATION,
        "_rollback_target_arguments",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ReachedWorkloadRestore()),
    )
    with pytest.raises(ReachedWorkloadRestore):
        ORCHESTRATION.rollback_release(instance, state=previous)
    assert runner.events.index(
        "rollback-aurora-refresh-restoring"
    ) < runner.events.index("apply")
    assert runner.events.index(
        "rollback-aurora-refresh-restored"
    ) < runner.events.index("refresh")
    assert runner.events.index("refresh") < runner.events.index("workload-gate")
