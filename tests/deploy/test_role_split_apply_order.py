"""The role-split apply rolls one Deployment at a time, ingress first, behind a
schedulability preflight that refuses before anything changes.

The functions are lifted out of the script and run in a bash harness against
the checked-in generated manifests: the sequence and its stop-on-failure, the
preflight's wiring (live reads, evidence, the planned order, refuse/warn), the
surge steering around the ingress apply (old Pods annotated with their
deletion cost before the template is applied, re-steered from the rollout
wait, the stall explained when the wait expires) and the stability of the
final Deployment document across a resume.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "deploy" / "control-plane" / "tools"
SCRIPT = TOOLS / "apply-control-plane-role-split.sh"
GENERATED = ROOT / "deploy" / "control-plane" / "regional" / "generated"
SCHEDULABILITY = lazy_script_module(TOOLS / "control_plane_schedulability.py")
FUNCTIONS = (
    "role_selected",
    "role_forced",
    "role_restart_token",
    "apply_spool_role",
    "apply_worker_role",
    "apply_ingress_role",
    "apply_role",
    "apply_roles_in_order",
    "render_manifest",
    "render_manifest_for_apply",
    "role_deployment_sha",
    "final_deployment_document",
    "schedulability_preflight",
    "wait_for_rollout",
    "steer_surge_roll",
    "diagnose_rollout_stall",
)
FAKE_KUBECTL = r"""#!/usr/bin/env bash
# Records every invocation; serves the preflight's and the steering's live
# reads from files; records Pod annotations in the harness log so their place
# in the apply sequence is visible.
printf '%s\n' "$*" >>"${FAKE_KUBECTL_LOG}"
case "$*" in
  *"get nodes -o json"*) cat "${FAKE_NODES}" ;;
  *"get pods -A -o json"*) cat "${FAKE_PODS}" ;;
  *"get pods -l app="*) cat "${FAKE_DEPLOYMENT_PODS}" ;;
  *"get replicasets -l app="*) cat "${FAKE_REPLICASETS}" ;;
  *"annotate pod "*)
    [[ "$*" =~ annotate\ pod\ ([^ ]+)\ controller\.kubernetes\.io/pod-deletion-cost=([0-9]+) ]]
    printf 'annotate %s %s\n' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" >>"${HARNESS_LOG}"
    ;;
  *"rollout status "*)
    sleep 0.2
    exit "${FAKE_ROLLOUT_STATUS}"
    ;;
  *)
    echo "unexpected kubectl call: $*" >&2
    exit 9
    ;;
esac
"""
STUBS = """
apply_manifest() { printf 'apply %s\\n' "$1" >>"${HARNESS_LOG}"; }
wait_for_rollout() { printf 'wait %s\\n' "$1" >>"${HARNESS_LOG}"; }
steer_surge_roll() { printf 'steer %s %s\\n' "$1" "$2" >>"${HARNESS_LOG}"; }
"""
# Steering stubbed out, the rest real: the steering's own kubectl traffic.
APPLY_STUBS = """
apply_manifest() { printf 'apply %s\\n' "$1" >>"${HARNESS_LOG}"; }
wait_for_rollout() { printf 'wait %s\\n' "$1" >>"${HARNESS_LOG}"; }
"""
GI = 1024**3
OLD_HASH, NEW_HASH = "7c9d5b8f4", "55767f6fbd"


def _extract_function(name: str) -> str:
    text = SCRIPT.read_text(encoding="utf-8")
    start = text.index(f"\n{name}() {{\n") + 1
    end = text.index("\n}\n", start) + 3
    return text[start:end]


def _node(name: str, cpu: str) -> dict[str, Any]:
    return {
        "metadata": {"name": name, "labels": {"kubernetes.io/hostname": name}},
        "spec": {},
        "status": {
            "allocatable": {"cpu": cpu, "memory": "30Gi"},
            "conditions": [{"type": "Ready", "status": "True"}],
        },
    }


def _pod(
    name: str,
    node: str | None,
    cpu: str,
    app: str = "filler",
    *,
    template_hash: str = "",
    conditions: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    labels = {"app": app}
    if template_hash:
        labels["pod-template-hash"] = template_hash
    spec: dict[str, Any] = {
        "containers": [
            {"name": "api", "resources": {"requests": {"cpu": cpu, "memory": "4Gi"}}}
        ]
    }
    if node is not None:
        spec["nodeName"] = node
    status: dict[str, Any] = {"phase": "Running" if node else "Pending"}
    if conditions is not None:
        status["conditions"] = conditions
    return {
        "metadata": {"name": name, "labels": labels},
        "spec": spec,
        "status": status,
    }


def _one_slot_cluster() -> tuple[dict[str, Any], dict[str, Any]]:
    """Three nodes, one old api-ha Pod each; only node-0 has a free 4-core
    slot (6.02 cores free; node-1 3.67, node-2 3.62)."""

    pods = [
        _pod(
            f"gpu-fault-api-ha-{OLD_HASH}-old{index}",
            f"node-{index}",
            "4",
            "gpu-fault-api-ha",
            template_hash=OLD_HASH,
        )
        for index in range(3)
    ]
    for index, used in enumerate(("5870m", "8220m", "8270m")):
        pods.append(_pod(f"filler-{index}", f"node-{index}", used))
    live = {
        "items": [
            {
                "metadata": {"name": "gpu-fault-api-ha"},
                "spec": {
                    "replicas": 3,
                    "template": {
                        "metadata": {
                            "annotations": {"gpu-fault.io/release-wheel-sha256": "old"}
                        },
                        "spec": {"containers": [{"name": "api", "image": "old"}]},
                    },
                },
            }
        ]
    }
    return {"items": pods}, live


def _replicasets() -> dict[str, Any]:
    template = yaml.safe_load(
        (GENERATED / "gpu-fault-api-ha-ingress.yaml").read_text(encoding="utf-8")
    )["spec"]["template"]
    return {
        "items": [
            {
                "metadata": {
                    "name": f"gpu-fault-api-ha-{template_hash}",
                    "labels": {
                        "app": "gpu-fault-api-ha",
                        "pod-template-hash": template_hash,
                    },
                    "annotations": {"deployment.kubernetes.io/revision": revision},
                },
                "spec": {"template": template},
            }
            for template_hash, revision in ((OLD_HASH, "7"), (NEW_HASH, "8"))
        ]
    }


def _run(
    tmp_path: Path,
    body: str,
    *,
    role_targets: str = "ingress,worker,spool",
    live: dict[str, Any] | None = None,
    restart_token: str = "",
    mode: str = "refuse",
    evidence_dir: Path | None = None,
    node_cores: str = "15890m",
    extra: str = "",
    pods: dict[str, Any] | None = None,
    deployment_pods: dict[str, Any] | None = None,
    replicasets: dict[str, Any] | None = None,
    rollout_status: int = 0,
) -> tuple[subprocess.CompletedProcess[str], list[str], list[str]]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    kubectl = bin_dir / "kubectl"
    kubectl.write_text(FAKE_KUBECTL, encoding="utf-8")
    kubectl.chmod(kubectl.stat().st_mode | stat.S_IXUSR)
    contract = tmp_path / "contract"
    contract.mkdir(exist_ok=True)
    live_path = tmp_path / "live.json"
    live_path.write_text(json.dumps(live if live is not None else {"items": []}))
    nodes_path = tmp_path / "nodes.json"
    nodes_path.write_text(
        json.dumps(
            {"items": [_node(f"node-{index}", node_cores) for index in range(3)]}
        )
    )
    pods_path = tmp_path / "pods.json"
    pods_path.write_text(json.dumps(pods if pods is not None else {"items": []}))
    deployment_pods_path = tmp_path / "deployment-pods.json"
    deployment_pods_path.write_text(
        json.dumps(deployment_pods if deployment_pods is not None else {"items": []})
    )
    replicasets_path = tmp_path / "replicasets.json"
    replicasets_path.write_text(
        json.dumps(replicasets if replicasets is not None else {"items": []})
    )
    harness_log = tmp_path / "harness.log"
    kubectl_log = tmp_path / "kubectl.log"
    preamble = "\n".join(
        [
            "set -euo pipefail",
            "kubectl_args=()",
            "NAMESPACE=gpu-fault-system",
            f"SCRIPT_DIR={TOOLS}",
            f"REPO_DIR={ROOT}",
            f"GENERATED={GENERATED}",
            f"CONTRACT_DIR={contract}",
            "WHEEL_CONFIGMAP=gpu-fault-control-plane-wheel-aaaaaaaaaaaa",
            "AWS_REGION=us-west-2",
            "RUNTIME_PROFILE_VERSION=hyperpod-v1",
            f"WHEEL_SHA256={'a' * 64}",
            "ALLOW_EMAIL=true",
            "ACKNOWLEDGE_NO_ALERT_CHANNEL=false",
            'SES_CONFIGURATION_SET=""',
            'DEFAULT_RUNTIME_IMAGE="public.ecr.aws/docker/library/python:3.12-slim"',
            f'RUNTIME_IMAGE="registry.example/runtime@sha256:{"b" * 64}"',
            "LEGACY_COMPONENT_PINS=false",
            "RELEASE_ID=aaaaaaaaaaaa",
            f"NOTIFICATION_CONFIG_SHA256={'c' * 64}",
            f"ADMIN_CONFIG_SHA256={'e' * 64}",
            f"ADMIN_CONFIG_INGRESS_SHA256={'f' * 64}",
            f"ADMIN_CONFIG_WORKER_SHA256={'1' * 64}",
            f"ADMIN_CONFIG_SPOOL_SHA256={'2' * 64}",
            f"PIN_METADATA_SHA256={'3' * 64}",
            f"FAILURE_DOMAIN_MAP_SHA256={'4' * 64}",
            f'ROLE_RESTART_TOKEN="{restart_token}"',
            "PER_ROLE_RESTART=false",
            'FORCED_ROLES=""',
            f"LIVE_DEPLOYMENTS={live_path}",
            f'ROLE_TARGETS="{role_targets}"',
            f"SCHEDULABILITY_MODE={mode}",
            f'SCHEDULABILITY_EVIDENCE_DIR="{evidence_dir or ""}"',
            f"SCHEDULABILITY_EVIDENCE={contract}/schedulability.json",
            'SCHEDULABILITY_ORDER=""',
            "SCHEDULABILITY_RUNS=0",
            f"STEERING_DIR={contract}/steering",
            "FAST_ROLLOUT_TIMEOUT_SECONDS=1",
            *(_extract_function(name) for name in FUNCTIONS),
            extra,
            body,
        ]
    )
    env = {
        **os.environ,
        # The venv's python3 (PyYAML) ahead of the fake kubectl's directory.
        "PATH": f"{Path(sys.executable).parent}:{bin_dir}:{os.environ['PATH']}",
        "HARNESS_LOG": str(harness_log),
        "FAKE_KUBECTL_LOG": str(kubectl_log),
        "FAKE_NODES": str(nodes_path),
        "FAKE_PODS": str(pods_path),
        "FAKE_DEPLOYMENT_PODS": str(deployment_pods_path),
        "FAKE_REPLICASETS": str(replicasets_path),
        "FAKE_ROLLOUT_STATUS": str(rollout_status),
    }
    result = subprocess.run(
        ["bash", "-c", preamble], check=False, text=True, capture_output=True, env=env
    )
    events = harness_log.read_text().splitlines() if harness_log.exists() else []
    calls = kubectl_log.read_text().splitlines() if kubectl_log.exists() else []
    return result, events, calls


def test_the_script_dispatches_every_role_through_one_sequential_function() -> None:
    script = SCRIPT.read_text(encoding="utf-8")
    for role in ("ingress", "worker", "spool"):
        assert f"{role}) apply_{role}_role ;;" in script
    assert 'python3 "${SCRIPT_DIR}/control_plane_schedulability.py"' in script
    disable = script.index('if [[ "${CURRENT_SPOOL_ADMISSION}" == "true"')
    assert 'schedulability_preflight "ingress,worker,spool"' in script[disable:]


def test_roles_roll_one_at_a_time_in_the_planned_order(tmp_path: Path) -> None:
    result, events, _ = _run(
        tmp_path, 'apply_roles_in_order "ingress,worker,spool"', extra=STUBS
    )

    assert result.returncode == 0, result.stderr
    assert events == [
        "apply gpu-fault-api-ha-pdb",
        "steer gpu-fault-api-ha plan",
        "apply gpu-fault-api-ha-ingress",
        "wait gpu-fault-api-ha",
        "apply gpu-fault-control-worker-pdb",
        "steer gpu-fault-control-worker plan",
        "apply gpu-fault-control-worker",
        "wait gpu-fault-control-worker",
        "apply gpu-fault-telemetry-spool-worker-pdb",
        "steer gpu-fault-telemetry-spool-worker plan",
        "apply gpu-fault-telemetry-spool-worker",
        "wait gpu-fault-telemetry-spool-worker",
    ], (
        "each Deployment is steered before its template is applied and waits "
        "for its own rollout before the next one is applied"
    )


def test_a_planned_order_that_frees_room_first_is_followed_as_given(
    tmp_path: Path,
) -> None:
    result, events, _ = _run(
        tmp_path, 'apply_roles_in_order "spool,ingress,worker"', extra=STUBS
    )

    assert result.returncode == 0, result.stderr
    assert [event for event in events if event.startswith("wait ")] == [
        "wait gpu-fault-telemetry-spool-worker",
        "wait gpu-fault-api-ha",
        "wait gpu-fault-control-worker",
    ]


def test_unselected_roles_are_skipped_in_the_sequence(tmp_path: Path) -> None:
    result, events, _ = _run(
        tmp_path,
        'apply_roles_in_order "ingress,worker,spool"',
        role_targets="worker",
        extra=STUBS,
    )

    assert result.returncode == 0, result.stderr
    assert events == [
        "apply gpu-fault-control-worker-pdb",
        "steer gpu-fault-control-worker plan",
        "apply gpu-fault-control-worker",
        "wait gpu-fault-control-worker",
    ]


def test_a_failed_rollout_wait_stops_the_sequence_before_the_next_role(
    tmp_path: Path,
) -> None:
    failing = STUBS + (
        "wait_for_rollout() {\n"
        '  printf \'wait %s\\n\' "$1" >>"${HARNESS_LOG}"\n'
        '  [[ "$1" != gpu-fault-api-ha ]]\n'
        "}\n"
    )
    result, events, _ = _run(
        tmp_path, 'apply_roles_in_order "ingress,worker,spool"', extra=failing
    )

    assert result.returncode != 0
    assert events == [
        "apply gpu-fault-api-ha-pdb",
        "steer gpu-fault-api-ha plan",
        "apply gpu-fault-api-ha-ingress",
        "wait gpu-fault-api-ha",
    ], "the consumer tier is not touched once the ingress roll failed"


def test_an_unknown_role_in_the_order_is_refused(tmp_path: Path) -> None:
    result, events, _ = _run(
        tmp_path, 'apply_roles_in_order "ingress,bogus"', extra=STUBS
    )

    assert result.returncode == 2
    assert "unknown control-plane role: bogus" in result.stderr
    assert events[-1] == "wait gpu-fault-api-ha"


def test_preflight_reads_the_live_cluster_plans_the_order_and_keeps_evidence(
    tmp_path: Path,
) -> None:
    evidence_dir = tmp_path / "state" / "control-plane-apply"
    result, _, calls = _run(
        tmp_path,
        'schedulability_preflight ""\nprintf \'order=%s\\n\' "${SCHEDULABILITY_ORDER}"',
        evidence_dir=evidence_dir,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "order=ingress,worker,spool"
    assert "get nodes -o json" in calls and "get pods -A -o json" in calls
    assert all("apply" not in call and "patch" not in call for call in calls), (
        "the preflight changes nothing in the cluster"
    )
    kept = sorted(evidence_dir.glob("schedulability-aaaaaaaaaaaa-*.json"))
    assert len(kept) == 1, kept
    assert stat.S_IMODE(kept[0].stat().st_mode) == 0o600
    evidence = json.loads(kept[0].read_text(encoding="utf-8"))
    assert evidence["verdict"] == "fits" and evidence["mode"] == "refuse"
    assert evidence["order"] == ["ingress", "worker", "spool"]
    assert [step["deployment"] for step in evidence["steps"]] == [
        "gpu-fault-api-ha",
        "gpu-fault-control-worker",
        "gpu-fault-telemetry-spool-worker",
    ]
    ingress = evidence["steps"][0]
    assert ingress["pod_request"] == {
        "cpu_millicores": 4000,
        "memory_bytes": 4 * 1024**3,
    }, "the generated api-ha manifest's requests reach the model"
    assert ingress["added_replicas"] == 3 and ingress["template_rolls"] is False
    assert "control-plane schedulability: {" in result.stderr


def test_preflight_refuses_nodes_too_small_for_the_consumer_tier(
    tmp_path: Path,
) -> None:
    result, _, _ = _run(
        tmp_path, 'schedulability_preflight ""\necho "reached"', node_cores="8"
    )

    assert result.returncode == 2, result.stderr
    assert "reached" not in result.stdout, "set -e ends the script at the refusal"
    assert (
        "step 2 (gpu-fault-control-worker roll) cannot schedule 6 added replica(s)"
        in result.stderr
    )
    assert "more free cores on one node would fit it" in result.stderr
    assert (
        "node-0 4.0 cores" in result.stderr
        and "8.00 cores allocatable" in result.stderr
    )


def test_preflight_warn_mode_logs_the_shortfall_and_proceeds(tmp_path: Path) -> None:
    result, _, _ = _run(
        tmp_path,
        'schedulability_preflight ""\nprintf \'order=%s\\n\' "${SCHEDULABILITY_ORDER}"',
        node_cores="8",
        mode="warn",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "order=ingress,worker,spool"
    assert "proceeding because the caller asked to warn only" in result.stderr


def test_preflight_models_the_spool_disable_order_as_fixed(tmp_path: Path) -> None:
    result, _, _ = _run(
        tmp_path,
        'schedulability_preflight "ingress,worker,spool"\n'
        "printf 'order=%s\\n' \"${SCHEDULABILITY_ORDER}\"",
        role_targets="ingress,spool",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "order=ingress,spool"


def _document(tmp_path: Path, **kwargs: Any) -> dict[str, Any]:
    result, _, _ = _run(
        tmp_path,
        "final_deployment_document gpu-fault-api-ha-ingress "
        '"$(role_deployment_sha gpu-fault-api-ha-ingress)"',
        **kwargs,
    )
    assert result.returncode == 0, result.stderr
    document: dict[str, Any] = json.loads(result.stdout)
    return document


def test_final_document_is_stable_across_a_resume_and_rolls_only_on_a_restart_token(
    tmp_path: Path,
) -> None:
    first = _document(tmp_path)
    annotations = first["spec"]["template"]["metadata"]["annotations"]
    assert annotations["gpu-fault.io/role-config-sha256"] == "f" * 64
    assert annotations["gpu-fault.io/release-wheel-sha256"] == "a" * 64
    assert annotations["gpu-fault.io/runtime-profile-version"] == "hyperpod-v1"
    assert annotations["gpu-fault.io/runtime-image"].startswith(
        "registry.example/runtime@"
    ), annotations
    assert "kubectl.kubernetes.io/restartedAt" not in annotations
    assert (
        first["metadata"]["annotations"]["gpu-fault.io/admin-config-sha256"] == "e" * 64
    )

    # The live Deployment carries an earlier forced restart; a resume with the
    # same inputs inherits it and renders the live template again: no roll.
    live_item = json.loads(json.dumps(first))
    live_item["spec"]["template"]["metadata"]["annotations"][
        "kubectl.kubernetes.io/restartedAt"
    ] = "2000-01-01T00:00:00.000000000Z"
    resumed = _document(tmp_path, live={"items": [live_item]})
    assert resumed["spec"]["template"] == live_item["spec"]["template"]
    assert SCHEDULABILITY.template_rolls(resumed, live_item) is False

    forced = _document(
        tmp_path,
        live={"items": [live_item]},
        restart_token="2000-01-02T00:00:00.000000000Z",
    )
    assert (
        forced["spec"]["template"]["metadata"]["annotations"][
            "kubectl.kubernetes.io/restartedAt"
        ]
        == "2000-01-02T00:00:00.000000000Z"
    )
    assert SCHEDULABILITY.template_rolls(forced, live_item) is True


def _pending_new_pod() -> dict[str, Any]:
    return _pod(
        f"gpu-fault-api-ha-{NEW_HASH}-8t4p5",
        None,
        "4",
        "gpu-fault-api-ha",
        template_hash=NEW_HASH,
        conditions=[
            {
                "type": "PodScheduled",
                "status": "False",
                "reason": "Unschedulable",
                "message": "0/3 nodes are available: 1 node(s) didn't match pod "
                "topology spread constraints, 2 Insufficient cpu.",
            }
        ],
    )


def test_the_steering_annotates_the_old_pods_before_the_ingress_template_is_applied(
    tmp_path: Path,
) -> None:
    """The live deadlock's shape: one node with the free slot. The plan runs
    after the preflight, writes the deletion costs onto the three old Pods --
    the one sharing the predicted surge node last -- and only then is the new
    template applied."""

    pods, live = _one_slot_cluster()
    result, events, calls = _run(
        tmp_path,
        'schedulability_preflight ""\napply_ingress_role',
        role_targets="ingress",
        live=live,
        pods=pods,
        extra=APPLY_STUBS,
    )

    assert result.returncode == 0, result.stderr
    assert events == [
        "apply gpu-fault-api-ha-pdb",
        f"annotate gpu-fault-api-ha-{OLD_HASH}-old2 100",
        f"annotate gpu-fault-api-ha-{OLD_HASH}-old1 200",
        f"annotate gpu-fault-api-ha-{OLD_HASH}-old0 300",
        "apply gpu-fault-api-ha-ingress",
        "wait gpu-fault-api-ha",
    ], "the old Pods carry their deletion cost before the template changes"
    assert all("--overwrite" in call for call in calls if "annotate" in call), calls
    assert (
        "surge steering: gpu-fault-api-ha: surge Pod expected on node-0"
        in result.stderr
    )
    assert "old0 on node-0 (cost 300, co-located with a new Pod, last)" in result.stderr
    state = json.loads(
        (tmp_path / "contract/steering/gpu-fault-api-ha.json").read_text()
    )
    assert state["active"] is True and state["progress_guaranteed"] is True
    evidence = json.loads((tmp_path / "contract/schedulability.json").read_text())
    steering = evidence["steps"][0]["steering"]
    assert steering["without_steering"]["progress_guaranteed"] is False
    assert steering["without_steering"]["stalling_order"] == [
        f"gpu-fault-api-ha-{OLD_HASH}-old0"
    ], "unsteered, the controller deletes the Pod co-located with the surge Pod first"


def test_a_tier_rolling_with_max_unavailable_one_is_not_steered(tmp_path: Path) -> None:
    # Warn mode: the one-slot cluster cannot take six fresh workers, and the
    # preflight's verdict is not what is under test here -- the plan's is.
    pods, live = _one_slot_cluster()
    result, events, calls = _run(
        tmp_path,
        'schedulability_preflight ""\napply_worker_role',
        role_targets="worker",
        live=live,
        pods=pods,
        mode="warn",
        extra=APPLY_STUBS,
    )

    assert result.returncode == 0, result.stderr
    assert events == [
        "apply gpu-fault-control-worker-pdb",
        "apply gpu-fault-control-worker",
        "wait gpu-fault-control-worker",
    ]
    assert not any("annotate" in call for call in calls), calls
    assert "surge steering: gpu-fault-control-worker: nothing to steer" in result.stderr


def test_the_rollout_wait_resteers_from_the_actual_surge_node_and_explains_a_stall(
    tmp_path: Path,
) -> None:
    """The surge Pod landed on node-1, not the predicted node-0: the reconcile
    inside the rollout wait rewrites the costs so node-1's old Pod goes last.
    The wait still expires (the fake rollout never finishes) and the stall is
    explained -- the pending Pod's scheduler verdict, every node's room -- before
    the wait fails as before."""

    pods, live = _one_slot_cluster()
    surge = _pod(
        f"gpu-fault-api-ha-{NEW_HASH}-sf7g8",
        "node-1",
        "4",
        "gpu-fault-api-ha",
        template_hash=NEW_HASH,
    )
    deployment_pods = {
        "items": [
            *(
                pod
                for pod in pods["items"]
                if pod["metadata"]["labels"]["app"] == "gpu-fault-api-ha"
            ),
            surge,
            _pending_new_pod(),
        ]
    }
    result, events, calls = _run(
        tmp_path,
        "\n".join(
            [
                "startup_failure_reason() { return 1; }",
                'schedulability_preflight ""',
                "steer_surge_roll gpu-fault-api-ha plan",
                "wait_for_rollout gpu-fault-api-ha",
                "echo unreachable",
            ]
        ),
        role_targets="ingress",
        live=live,
        pods=pods,
        deployment_pods=deployment_pods,
        replicasets=_replicasets(),
        rollout_status=1,
    )

    assert result.returncode != 0, "an expired rollout wait still fails the apply"
    assert "unreachable" not in result.stdout
    assert events[:3] == [
        f"annotate gpu-fault-api-ha-{OLD_HASH}-old2 100",
        f"annotate gpu-fault-api-ha-{OLD_HASH}-old1 200",
        f"annotate gpu-fault-api-ha-{OLD_HASH}-old0 300",
    ], "the plan steers for the predicted surge node"
    assert events[3:5] == [
        f"annotate gpu-fault-api-ha-{OLD_HASH}-old0 200",
        f"annotate gpu-fault-api-ha-{OLD_HASH}-old1 300",
    ], (
        "the reconcile moves node-1's old Pod to the end once the surge Pod is seen there"
    )
    assert len(events) == 5, (
        f"a settled order is not rewritten on later polls: {events}"
    )
    assert (
        "surge steering: gpu-fault-api-ha: new Pod(s) on node-1 instead of the planned "
        "node-0; deletion order now" in result.stderr
    )
    assert (
        f"rollout stalled: gpu-fault-api-ha: new Pod gpu-fault-api-ha-{NEW_HASH}-8t4p5 is "
        "Pending (Unschedulable): 0/3 nodes are available: 1 node(s) didn't match pod "
        "topology spread constraints, 2 Insufficient cpu." in result.stderr
    )
    assert (
        "rollout stalled: gpu-fault-api-ha: node node-0: 6.0 cores / " in result.stderr
    )
    assert (
        "0 new Pod(s) []; 1 old Pod(s) [gpu-fault-api-ha-7c9d5b8f4-old0 (deletion cost None)]"
        in result.stderr
    ), "the fake API does not persist annotations; the diagnosis reports what it reads"
    assert (
        "a new Pod needs 4 cores / 4Gi on a node that hosts no new Pod of this template "
        "(hostname spread maxSkew 1); free such a slot on a second node or add a CPU node"
        in result.stderr
    )
    assert [call for call in calls if "rollout status" in call], calls
