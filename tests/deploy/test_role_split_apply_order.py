"""The role-split apply rolls one Deployment at a time, ingress first, behind a
schedulability preflight that refuses before anything changes.

The functions are lifted out of the script and run in a bash harness against
the checked-in generated manifests: the sequence and its stop-on-failure, the
preflight's wiring (live reads, evidence, the planned order, refuse/warn) and
the stability of the final Deployment document across a resume.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "deploy" / "control-plane" / "tools"
SCRIPT = TOOLS / "apply-control-plane-role-split.sh"
GENERATED = ROOT / "deploy" / "control-plane" / "regional" / "generated"
SCHEDULABILITY = lazy_script_module(TOOLS / "control_plane_schedulability.py")
FUNCTIONS = (
    "role_selected",
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
)
FAKE_KUBECTL = r"""#!/usr/bin/env bash
# Records every invocation; serves the preflight's two live reads from files.
printf '%s\n' "$*" >>"${FAKE_KUBECTL_LOG}"
case "$*" in
  *"get nodes -o json"*) cat "${FAKE_NODES}" ;;
  *"get pods -A -o json"*) cat "${FAKE_PODS}" ;;
  *)
    echo "unexpected kubectl call: $*" >&2
    exit 9
    ;;
esac
"""
STUBS = """
apply_manifest() { printf 'apply %s\\n' "$1" >>"${HARNESS_LOG}"; }
wait_for_rollout() { printf 'wait %s\\n' "$1" >>"${HARNESS_LOG}"; }
"""


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
    pods_path.write_text(json.dumps({"items": []}))
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
            f"LIVE_DEPLOYMENTS={live_path}",
            f'ROLE_TARGETS="{role_targets}"',
            f"SCHEDULABILITY_MODE={mode}",
            f'SCHEDULABILITY_EVIDENCE_DIR="{evidence_dir or ""}"',
            f"SCHEDULABILITY_EVIDENCE={contract}/schedulability.json",
            'SCHEDULABILITY_ORDER=""',
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
        "apply gpu-fault-api-ha-ingress",
        "wait gpu-fault-api-ha",
        "apply gpu-fault-control-worker-pdb",
        "apply gpu-fault-control-worker",
        "wait gpu-fault-control-worker",
        "apply gpu-fault-telemetry-spool-worker-pdb",
        "apply gpu-fault-telemetry-spool-worker",
        "wait gpu-fault-telemetry-spool-worker",
    ], "each Deployment waits for its own rollout before the next one is applied"


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
