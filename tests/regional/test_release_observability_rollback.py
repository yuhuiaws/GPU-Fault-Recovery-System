import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
ADOT = lazy_script_module(
    ROOT / "deploy/control-plane/regional/regional_observability_rollback.py"
)
STATE = lazy_script_module(
    ROOT / "deploy/control-plane/regional/regional_release_state.py"
)
ORCHESTRATION = lazy_script_module(
    ROOT / "deploy/control-plane/regional/regional_release_orchestration.py"
)
DIFF = lazy_script_module(
    ROOT / "deploy/control-plane/regional/regional_release_diff.py"
)
MODULE = lazy_script_module(
    ROOT / "deploy/control-plane/regional/rollout_regional_release.py"
)


def _declared() -> tuple[dict[str, str], ...]:
    # Read the manifest the installer actually applies: the point of the capture
    # is that it covers whatever that file declares today.
    return ADOT.declared_adot_objects(ADOT.ADOT_MANIFEST.read_text(encoding="utf-8"))


def _live_object(kind: str, name: str) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": kind,
        "metadata": {
            "name": name,
            "namespace": "gpu-fault-system",
            "uid": "1a2b",
            "resourceVersion": "4711",
            "creationTimestamp": "2026-09-01T00:00:00Z",
            "generation": 3,
            "managedFields": [{"manager": "kubectl-client-side-apply"}],
            "annotations": {
                "kubectl.kubernetes.io/last-applied-configuration": "{}",
                "deployment.kubernetes.io/revision": "7",
                "gpu-fault.io/previous-release": "kept",
            },
            "labels": {"app": name},
        },
        "spec": {"marker": "previous"},
        "status": {"observedGeneration": 3},
    }


def _capture_release(
    *, absent: frozenset[tuple[str, str]] = frozenset()
) -> tuple[SimpleNamespace, list[list[str]]]:
    live = {
        (item["resource"], item["name"]): _live_object(item["kind"], item["name"])
        for item in _declared()
        if (item["resource"], item["name"]) not in absent
    }
    reads: list[list[str]] = []

    def run(arguments: list[str], **_kwargs: Any) -> str:
        reads.append(list(arguments))
        index = arguments.index("get")
        key = (arguments[index + 1], arguments[index + 2])
        document = live.get(key)
        return json.dumps(document) if document is not None else ""

    release = SimpleNamespace(
        config=SimpleNamespace(namespace="gpu-fault-system"),
        runner=SimpleNamespace(run=run),
        _cpu=lambda *args: ["kubectl", "--kubeconfig", "/secure/cpu", *args],
    )
    return release, reads


def test_capture_reads_every_declared_object_and_drops_server_owned_fields() -> None:
    release, reads = _capture_release()

    snapshot = ADOT.capture_adot_objects(release)

    assert [(item["resource"], item["name"]) for item in _declared()] == [
        (arguments[arguments.index("get") + 1], arguments[arguments.index("get") + 2])
        for arguments in reads
    ]
    for arguments in reads:
        assert arguments[:3] == ["kubectl", "--kubeconfig", "/secure/cpu"]
        assert arguments[3:5] == ["-n", "gpu-fault-system"]
        # An object the candidate adds is not an unreadable object: the capture
        # has to tell those apart without failing the upgrade.
        assert "--ignore-not-found" in arguments
    assert snapshot["namespace"] == "gpu-fault-system"
    assert snapshot["absent"] == []
    assert len(snapshot["objects"]) == len(_declared())
    for item in snapshot["objects"]:
        metadata = item["metadata"]
        assert "status" not in item
        for field in ("uid", "resourceVersion", "creationTimestamp", "generation"):
            assert field not in metadata, field
        assert "managedFields" not in metadata
        assert metadata["annotations"] == {"gpu-fault.io/previous-release": "kept"}
        assert metadata["labels"] == {"app": item["metadata"]["name"]}
        assert item["spec"] == {"marker": "previous"}


def test_capture_records_candidate_added_objects_for_deletion() -> None:
    added = ("poddisruptionbudget.policy", "gpu-fault-adot")
    release, _reads = _capture_release(absent=frozenset({added}))

    snapshot = ADOT.capture_adot_objects(release)

    assert snapshot["absent"] == [{"resource": added[0], "name": added[1]}]
    assert all(item["kind"] != "PodDisruptionBudget" for item in snapshot["objects"]), (
        snapshot["objects"]
    )


def test_capture_fails_closed_when_the_live_collector_is_not_there() -> None:
    release, _reads = _capture_release(
        absent=frozenset({("deployment.apps", "gpu-fault-adot")})
    )

    with pytest.raises(MODULE.ReleaseError, match="ADOT collector Deployment"):
        ADOT.capture_adot_objects(release)


def _restore_release() -> tuple[SimpleNamespace, list[list[str]], list[Any]]:
    calls: list[list[str]] = []
    applied: list[Any] = []

    def run(arguments: list[str], **_kwargs: Any) -> str:
        calls.append(list(arguments))
        if "apply" in arguments:
            # The file lives in a temporary directory that is gone by the time
            # the assertions run, so read it while the call is happening.
            applied.append(
                json.loads(
                    Path(arguments[arguments.index("-f") + 1]).read_text(
                        encoding="utf-8"
                    )
                )
            )
        return ""

    release = SimpleNamespace(
        config=SimpleNamespace(namespace="gpu-fault-system"),
        runner=SimpleNamespace(run=run),
        _cpu=lambda *args: ["kubectl", *args],
    )
    return release, calls, applied


def _snapshot() -> dict[str, Any]:
    return {
        "namespace": "gpu-fault-system",
        "objects": [
            {
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "metadata": {"name": "gpu-fault-adot"},
                "data": {"collector.yaml": "previous"},
            },
            {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {"name": "gpu-fault-adot"},
                "spec": {"replicas": 1},
            },
        ],
        "absent": [
            {"resource": "poddisruptionbudget.policy", "name": "gpu-fault-adot"}
        ],
    }


def test_restore_applies_the_previous_objects_removes_the_new_ones_and_restarts() -> (
    None
):
    release, calls, applied = _restore_release()

    ADOT.restore_adot_objects(release, {"adot": _snapshot()})

    verbs = [
        next(word for word in arguments if word in {"apply", "delete", "rollout"})
        for arguments in calls
    ]
    assert verbs == ["apply", "delete", "rollout", "rollout"]
    assert applied == [
        {"apiVersion": "v1", "kind": "List", "items": _snapshot()["objects"]}
    ]
    delete = calls[1]
    assert delete[1:] == [
        "-n",
        "gpu-fault-system",
        "delete",
        "poddisruptionbudget.policy",
        "gpu-fault-adot",
        "--ignore-not-found",
    ]
    # Restoring the mounted ConfigMap only takes effect once the Pod is
    # replaced, and the rollback has to find out whether the restored collector
    # actually comes back up.
    assert calls[2][-2:] == ["restart", "deployment/gpu-fault-adot"]
    assert calls[3][-3:] == [
        "status",
        "deployment/gpu-fault-adot",
        f"--timeout={ADOT.ADOT_ROLLOUT_TIMEOUT}",
    ]


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "snapshot",
    [
        {"rule_namespace": "rules"},
        {"adot": {"namespace": "gpu-fault-system", "objects": [], "absent": []}},
        {"adot": {"objects": [{"kind": "Deployment"}], "absent": []}},
        {"adot": "gpu-fault-adot"},
    ],
)
def test_restore_refuses_a_snapshot_it_cannot_put_back(snapshot: object) -> None:
    release, calls, _applied = _restore_release()

    with pytest.raises(MODULE.ReleaseError, match="ADOT collector"):
        ADOT.restore_adot_objects(release, snapshot)

    assert calls == [], "a snapshot that cannot be restored still mutated the cluster"


def test_observability_snapshot_carries_and_restores_the_collector() -> None:
    live = {
        (item["resource"], item["name"]): _live_object(item["kind"], item["name"])
        for item in _declared()
    }
    describe = {
        "describe-rule-groups-namespace": {"ruleGroupsNamespace": {"data": "cnVsZXM="}},
        "describe-alert-manager-definition": {
            "alertManagerDefinition": {"data": "YWxlcnRz"}
        },
    }
    calls: list[list[str]] = []

    def run(arguments: list[str], **_kwargs: Any) -> str:
        calls.append(list(arguments))
        for verb, payload in describe.items():
            if verb in arguments:
                return json.dumps(payload)
        if "get" in arguments:
            index = arguments.index("get")
            document = live.get((arguments[index + 1], arguments[index + 2]))
            return json.dumps(document) if document is not None else ""
        return ""

    release = SimpleNamespace(
        config=SimpleNamespace(
            namespace="gpu-fault-system",
            aws_region="us-west-2",
            health=SimpleNamespace(
                amp_workspace_id="ws-test", amp_rule_namespace="gpu-fault-rules"
            ),
        ),
        runner=SimpleNamespace(run=run),
        _cpu=lambda *args: ["kubectl", *args],
    )

    snapshot = ADOT.capture_observability_snapshot(release)
    assert snapshot["rules_data_base64"] == "cnVsZXM="
    assert [item["kind"] for item in snapshot["adot"]["objects"]] == [
        item["kind"] for item in _declared()
    ]

    calls.clear()
    ADOT.restore_observability_snapshot(release, snapshot)
    aws_verbs = [arguments[2] for arguments in calls if arguments[0] == "aws"]
    kubectl_verbs = [
        word
        for arguments in calls
        if arguments[0] == "kubectl"
        for word in arguments
        if word in {"apply", "delete", "rollout"}
    ]
    # The AMP blobs and the collector are one compensation: a rollback that put
    # the rules back but left the collector on the candidate would be the state
    # the non-transactional refusal used to exist for.
    assert aws_verbs == ["put-rule-groups-namespace", "put-alert-manager-definition"]
    assert kubectl_verbs == ["apply", "rollout", "rollout"]


def test_a_snapshot_without_collector_objects_refuses_before_any_mutation() -> None:
    # A resumed transaction keeps the snapshot it was started with, so a release
    # begun before the collector was compensable must not be told it now has
    # automatic rollback: the restore would fail after other components were
    # already put back.
    legacy = {
        "observability": {
            "rule_namespace": "rules",
            "rules_data_base64": "cnVsZXM=",
            "alertmanager_data_base64": "YWxlcnRz",
        }
    }
    release = SimpleNamespace(
        config=SimpleNamespace(
            auto_rollback=True, schema_rollback_compatible=True, clusters=()
        ),
        state={},
        _ensure_contexts=lambda: None,
        _require_cpu_secrets=lambda: None,
        _remote_commands_are_idle=lambda: True,
        _capture_previous=lambda **_kwargs: dict(legacy),
        _backup_release_secrets=lambda: {},
    )
    changed = frozenset({"observability_adot"})

    with pytest.raises(MODULE.ReleaseError, match="predates ADOT rollback capture"):
        ORCHESTRATION.upgrade_release(
            release,
            diff=DIFF.ReleaseDiff(kind=DIFF.ReleaseChangeKind.FULL, changed=changed),
        )


def test_adot_change_keeps_automatic_rollback_while_clusters_still_refuse() -> None:
    class ValidationPassed(RuntimeError):
        pass

    def release(changed: frozenset[str]) -> SimpleNamespace:
        return SimpleNamespace(
            config=SimpleNamespace(
                auto_rollback=True, schema_rollback_compatible=False, clusters=()
            ),
            state={"release_diff": {"changed": sorted(changed)}},
            _ensure_contexts=lambda: None,
            _require_cpu_secrets=lambda: None,
            _remote_commands_are_idle=lambda: True,
            _capture_previous=lambda **_kwargs: (_ for _ in ()).throw(
                ValidationPassed()
            ),
        )

    adot = frozenset({"observability_adot", "adot_image"})
    with pytest.raises(ValidationPassed):
        ORCHESTRATION.upgrade_release(
            release(adot),
            diff=DIFF.ReleaseDiff(kind=DIFF.ReleaseChangeKind.FULL, changed=adot),
        )
    with pytest.raises(
        MODULE.ReleaseError, match="previous release pins are incomplete"
    ):
        ORCHESTRATION.rollback_release(release(adot), state={"metadata": {}})

    registry = frozenset({"clusters"})
    with pytest.raises(
        MODULE.ReleaseError, match="not yet transactional for: clusters"
    ):
        ORCHESTRATION.upgrade_release(
            release(registry),
            diff=DIFF.ReleaseDiff(kind=DIFF.ReleaseChangeKind.FULL, changed=registry),
        )
    with pytest.raises(MODULE.ReleaseError, match="not transactional for: clusters"):
        ORCHESTRATION.rollback_release(release(registry), state={"metadata": {}})
