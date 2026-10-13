from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "deploy/control-plane/tools/apply-control-plane-role-split.sh"
DEPLOYMENTS = (
    "gpu-fault-api-ha",
    "gpu-fault-control-worker",
    "gpu-fault-telemetry-spool-worker",
)
PIN_ANNOTATION = "gpu-fault.io/pin-config-sha256"
RESTARTED_AT = "2026-01-01T00:00:00Z"
FAKE_API = """#!/usr/bin/env python3
import fcntl
import hashlib
import json
import os
import sys
import yaml

args = sys.argv[1:]
with open(os.environ["CALLS"], "a") as handle:
    fcntl.flock(handle, fcntl.LOCK_EX)
    handle.write(json.dumps(args) + "\\n")
state = json.loads(os.environ["LIVE_STATE"])
# The release-metadata ConfigMap the script "created"; the Pods below serve it.
pins_file = os.path.join(os.environ["HOME"], "release-metadata.json")
if "get" in args:
    kind = args[args.index("get") + 1]
    if kind == "deployments":
        print(json.dumps(state["deployments"]))
    elif kind == "deployment":
        print("{}")
    elif kind == "configmap":
        if "gpu-fault-release-metadata" in args:
            print(json.dumps(state["pins"]))
        else:
            print("false")
    elif kind == "nodes":
        # One roomy Ready node: the schedulability preflight's reads.
        print(json.dumps({"items": [{
            "metadata": {"name": "node-a", "labels": {"kubernetes.io/hostname": "node-a"}},
            "spec": {},
            "status": {"allocatable": {"cpu": "64", "memory": "256Gi"},
                       "conditions": [{"type": "Ready", "status": "True"}]},
        }]}))
    elif kind == "pods" and "-A" in args:
        print(json.dumps({"items": []}))
    elif kind == "pods":
        deployment = next(arg for arg in args if arg.startswith("app="))[4:]
        print(json.dumps({"items": [{
            "metadata": {"name": deployment + "-0", "labels": {"app": deployment}},
            "spec": {"containers": [{"name": "api",
                                     "ports": [{"name": "http", "containerPort": 8080}]}]},
            "status": {"phase": "Running"},
        }]}))
    elif kind != "pod":
        raise SystemExit("unexpected get")
elif "exec" in args:
    # A Pod polls the ConfigMap itself and reports the digest it serves --
    # unless POD_PINS says its tier runs a wheel older than fleet pins.
    pod = args[args.index("exec") + 1]
    pins = json.loads(os.environ.get("POD_PINS") or "{}")
    if pins.get(pod.rsplit("-", 1)[0]) == "unsupported":
        print(json.dumps({"status": "ok"}))
        raise SystemExit(0)
    with open(pins_file) as handle:
        data = json.load(handle)
    digest = hashlib.sha256(json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")).hexdigest()
    print(json.dumps({"status": "ok", "fleet_pins": {"content_sha256": digest}}))
elif "create" in args:
    name = args[args.index("configmap") + 1]
    data = dict(arg[len("--from-literal="):].split("=", 1)
                for arg in args if arg.startswith("--from-literal="))
    with open(pins_file, "w") as handle:
        json.dump(data, handle)
    print(json.dumps({"apiVersion":"v1", "kind":"ConfigMap",
                     "metadata":{"name":name}, "data":data}))
elif "apply" in args:
    documents = [item for item in yaml.safe_load_all(sys.stdin) if item]
    with open(os.environ["APPLIES"], "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        for document in documents:
            handle.write(json.dumps(document) + "\\n")
elif "patch" in args:
    patch = json.loads(args[args.index("-p") + 1])
    if "spec" in patch:
        raise SystemExit("a second Pod template mutation was attempted")
elif "rollout" in args and "status" in args:
    pass
else:
    raise SystemExit("unexpected mutation: " + " ".join(args))
"""


# Stands in for wait_control_plane_pins.py when a test needs an outcome the fake
# API cannot produce quickly (a tier still serving the old digest at the
# deadline): pops the next {exit, report} from PIN_WAIT_OUTCOMES and writes the
# report where --report asks for it.
PIN_WAIT_STUB = """import json, os, sys
args = sys.argv[1:]
path = os.environ["PIN_WAIT_OUTCOMES"]
with open(path) as handle:
    outcomes = json.load(handle)
outcome = outcomes.pop(0)
with open(path, "w") as handle:
    json.dump(outcomes, handle)
if "--report" in args and outcome.get("report") is not None:
    with open(args[args.index("--report") + 1], "w") as handle:
        json.dump(outcome["report"], handle)
sys.exit(outcome["exit"])
"""
ROLE_DEPLOYMENTS = {
    "ingress": "gpu-fault-api-ha",
    "worker": "gpu-fault-control-worker",
    "spool": "gpu-fault-telemetry-spool-worker",
}


def pin_content_sha256(data: dict[str, str]) -> str:
    """The shared fleet-pin digest: what every control-plane Pod reports."""

    return hashlib.sha256(
        json.dumps(
            data, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


# The shared contract's vector: gpu_fault.fleet_pins.pin_content_sha256 and the
# script's PIN_METADATA_SHA256 must both produce it. ``jq -cS | sha256sum``
# would hash jq's trailing newline as well and yield c2f96399... instead.
PIN_VECTOR = {"b": "x,y", "a": "1", "z": ""}
PIN_VECTOR_SHA256 = "585414de19bbac53560be605120197fd16f19f6ccc14d77b275817cf39280b65"


def test_script_pin_digest_matches_the_shared_vector_without_jqs_newline() -> None:
    line = next(
        line
        for line in SCRIPT.read_text(encoding="utf-8").splitlines()
        if line.startswith("PIN_METADATA_SHA256=")
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            "\n".join(
                [
                    "set -euo pipefail",
                    'RELEASE_METADATA_JSON="$1"',
                    line,
                    'printf "%s" "${PIN_METADATA_SHA256}"',
                ]
            ),
            "digest",
            json.dumps({"data": PIN_VECTOR}),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout == PIN_VECTOR_SHA256
    assert pin_content_sha256(PIN_VECTOR) == PIN_VECTOR_SHA256


def run_apply(
    tmp_path: Path,
    *,
    targets: str,
    force: bool,
    namespace: str = "gpu-fault-system",
    environment: dict[str, str] | None = None,
    live_annotations: dict[str, dict[str, str]] | None = None,
    pod_pins: dict[str, str] | None = None,
    pin_wait_outcomes: list[dict[str, Any]] | None = None,
) -> subprocess.CompletedProcess[str]:
    binary = tmp_path / "kubectl"
    binary.write_text(FAKE_API, encoding="utf-8")
    binary.chmod(0o755)
    python = tmp_path / "python3"
    python.write_text(
        "#!/bin/bash\nset -eu\n"
        'printf \'%s\\n\' "$*" >>"$PYTHON_CALLS"\n'
        'case "$*" in\n'
        "  *gpu_fault.config_cli*|*verify_control_plane_role_split.py*) exit 0 ;;\n"
        "  *wait_control_plane_pins.py*)\n"
        '    if [ -n "${PIN_WAIT_STUB:-}" ]; then exec "$REAL_PYTHON" "$PIN_WAIT_STUB" "$@"; fi\n'
        '    exec "$REAL_PYTHON" "$@" ;;\n'
        '  *) exec "$REAL_PYTHON" "$@" ;;\n'
        "esac\n",
        encoding="utf-8",
    )
    python.chmod(0o755)
    stub: dict[str, str] = {}
    if pin_wait_outcomes is not None:
        (tmp_path / "pin-wait-stub.py").write_text(PIN_WAIT_STUB, encoding="utf-8")
        (tmp_path / "pin-wait-outcomes.json").write_text(
            json.dumps(pin_wait_outcomes), encoding="utf-8"
        )
        stub = {
            "PIN_WAIT_STUB": str(tmp_path / "pin-wait-stub.py"),
            "PIN_WAIT_OUTCOMES": str(tmp_path / "pin-wait-outcomes.json"),
        }
    state = {
        "pins": {
            "data": {
                "required-agent-artifact-sha256": "a" * 64,
                "required-agent-config-digest": "b" * 64,
                "required-agent-protocol-version": "3",
                "required-regional-executor-protocol-version": "1",
            }
        },
        "deployments": {
            "items": [
                {
                    "metadata": {"name": name},
                    "spec": {
                        "template": {
                            "metadata": {
                                "annotations": {
                                    "kubectl.kubernetes.io/restartedAt": RESTARTED_AT,
                                    **(live_annotations or {}).get(name, {}),
                                }
                            },
                            "spec": {"containers": [{"name": "api", "env": []}]},
                        }
                    },
                }
                for name in DEPLOYMENTS
            ]
        },
    }
    return subprocess.run(
        ["bash", str(SCRIPT)],
        env={
            "PATH": str(tmp_path)
            + os.pathsep
            + os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(tmp_path),
            "PYTHONPATH": str(ROOT / "src"),
            "REAL_PYTHON": sys.executable,
            "LIVE_STATE": json.dumps(state),
            "POD_PINS": json.dumps(pod_pins or {}),
            "CALLS": str(tmp_path / "calls.jsonl"),
            "APPLIES": str(tmp_path / "applies.jsonl"),
            "PYTHON_CALLS": str(tmp_path / "python-calls.txt"),
            "KUBECONFIG": str(tmp_path / "cpu"),
            "GPU_FAULT_AWS_REGION": "us-east-1",
            "GPU_FAULT_NAMESPACE": namespace,
            "GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION": "profile-test",
            "GPU_FAULT_NOTIFICATION_CONFIG_SHA256": "c" * 64,
            "GPU_FAULT_WHEEL_SHA256": "a" * 64,
            "GPU_FAULT_REQUIRED_AGENT_CONFIG_DIGEST": "b" * 64,
            "GPU_FAULT_REQUIRED_AGENT_PROTOCOL_VERSION": "3",
            "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_PROTOCOL_VERSION": "1",
            "GPU_FAULT_CONTROL_PLANE_ROLE_TARGETS": targets,
            "GPU_FAULT_FORCE_ROLE_RESTART": str(force).lower(),
            "GPU_FAULT_RUNTIME_IMAGE": "registry.invalid/runtime@sha256:" + "d" * 64,
            "GPU_FAULT_FAILURE_DOMAIN_MAP_SHA256": "e" * 64,
            "GPU_FAULT_SYNC_INSTALLED_RESOURCE_REGISTRY": "false",
            **stub,
            **(environment or {}),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def outputs(tmp_path: Path) -> tuple[list[dict], list[list[str]], list[list[str]]]:
    """(applied documents, kubectl calls, python3 calls) of the last run."""

    def lines(name: str) -> list[str]:
        path = tmp_path / name
        return path.read_text(encoding="utf-8").splitlines() if path.exists() else []

    return (
        [json.loads(line) for line in lines("applies.jsonl")],
        [json.loads(line) for line in lines("calls.jsonl")],
        [line.split() for line in lines("python-calls.txt")],
    )


def apply_roles(
    tmp_path: Path,
    *,
    targets: str,
    force: bool,
    namespace: str = "gpu-fault-system",
    environment: dict[str, str] | None = None,
    **harness: Any,
) -> tuple[list[dict], list[list[str]], list[list[str]]]:
    completed = run_apply(
        tmp_path,
        targets=targets,
        force=force,
        namespace=namespace,
        environment=environment,
        **harness,
    )
    assert completed.returncode == 0, completed.stderr
    return outputs(tmp_path)


def applied_pin_digest(documents: list[dict]) -> str:
    """Digest of the release-metadata ConfigMap the script applied."""

    metadata = [
        item
        for item in documents
        if item["kind"] == "ConfigMap"
        and item["metadata"]["name"] == "gpu-fault-release-metadata"
    ]
    assert len(metadata) == 1, "the release-metadata ConfigMap was not applied once"
    return pin_content_sha256(metadata[0]["data"])


def pin_wait_call(python_calls: list[list[str]]) -> list[str]:
    waits = [
        call
        for call in python_calls
        if any(argument.endswith("/wait_control_plane_pins.py") for argument in call)
    ]
    assert len(waits) == 1, (
        "the pin convergence wait must run exactly once per apply: "
        f"{[' '.join(call) for call in python_calls]}"
    )
    return waits[0]


@pytest.mark.parametrize("force", [False, True])
def test_each_cpu_role_receives_one_complete_template_update(
    tmp_path: Path, force: bool
) -> None:
    documents, calls, python_calls = apply_roles(
        tmp_path, targets="spool,worker,ingress", force=force
    )
    deployments = [item for item in documents if item["kind"] == "Deployment"]
    expected_pins = applied_pin_digest(documents)

    assert sorted(item["metadata"]["name"] for item in deployments) == sorted(
        DEPLOYMENTS
    )
    for deployment in deployments:
        annotations = deployment["spec"]["template"]["metadata"]["annotations"]
        assert annotations["gpu-fault.io/control-plane-wheel-sha256"] == "a" * 64
        assert annotations["gpu-fault.io/runtime-image"].endswith("d" * 64), (
            "final template lacks the candidate image digest"
        )
        # A pin change must not roll the tier: the digest is recorded on the
        # Deployment and proven on /healthz, never mixed into the Pod template.
        assert PIN_ANNOTATION not in annotations, (
            "the pin digest on the Pod template rolls every replica per release"
        )
        assert deployment["metadata"]["annotations"][PIN_ANNOTATION] == expected_pins
        assert ("gpu-fault.io/failure-domain-map-sha256" in annotations) is (
            deployment["metadata"]["name"] == "gpu-fault-control-worker"
        )
        assert (
            annotations["kubectl.kubernetes.io/restartedAt"] != RESTARTED_AT
        ) is force
    assert not any(
        "restart" in call or "patch" in call or "set" in call for call in calls
    ), "normal apply issued another mutation after the final template"

    wait = pin_wait_call(python_calls)
    assert wait[wait.index("--expected-sha256") + 1] == expected_pins
    assert wait[wait.index("--namespace") + 1] == "gpu-fault-system"
    assert wait[wait.index("--kubeconfig") + 1] == str(tmp_path / "cpu")
    assert sorted(wait[wait.index("--deployments") + 1].split(",")) == sorted(
        DEPLOYMENTS
    )
    verify = next(
        index
        for index, call in enumerate(python_calls)
        if any(arg.endswith("/verify_control_plane_role_split.py") for arg in call)
    )
    assert python_calls.index(wait) < verify, (
        "the pin convergence wait must run before the role-split verifier"
    )
    probes = [call for call in calls if "exec" in call]
    assert sorted(call[call.index("exec") + 1] for call in probes) == sorted(
        f"{name}-0" for name in DEPLOYMENTS
    ), "every selected role's Pod must be asked for its /healthz pins"
    assert all("/healthz" in " ".join(call) for call in probes), probes
    last_apply = max(index for index, call in enumerate(calls) if "apply" in call)
    assert last_apply < min(calls.index(call) for call in probes), (
        "the convergence wait ran before the final templates were applied"
    )


def test_role_scoped_apply_only_changes_the_selected_template(tmp_path: Path) -> None:
    documents, calls, python_calls = apply_roles(
        tmp_path, targets="worker", force=False
    )
    deployments = [item for item in documents if item["kind"] == "Deployment"]

    assert [item["metadata"]["name"] for item in deployments] == [
        "gpu-fault-control-worker"
    ]
    annotations = deployments[0]["spec"]["template"]["metadata"]["annotations"]
    assert annotations["gpu-fault.io/failure-domain-map-sha256"] == "e" * 64
    patches = [call for call in calls if "patch" in call]
    assert len(patches) == 2
    assert all(
        "spec" not in json.loads(call[call.index("-p") + 1]) for call in patches
    ), "unselected CPU roles received a Pod template patch"
    wait = pin_wait_call(python_calls)
    assert wait[wait.index("--deployments") + 1] == "gpu-fault-control-worker", (
        "the convergence wait must cover only the selected roles"
    )
    assert [call[call.index("exec") + 1] for call in calls if "exec" in call] == [
        "gpu-fault-control-worker-0"
    ]


def test_verbatim_rollback_skips_the_pin_convergence_wait(tmp_path: Path) -> None:
    """A rollback restores the previous release's container env and forces a
    restart of every tier, so the restored Pods start from the ConfigMap's env
    snapshot; their wheel may predate ``fleet_pins`` on ``/healthz``. The
    previous release's own contract applies, as it does for the verifier."""

    snapshot = tmp_path / "container-env.json"
    snapshot.write_text("{}", encoding="utf-8")
    documents, calls, python_calls = apply_roles(
        tmp_path,
        targets="spool,worker,ingress",
        force=True,
        environment={"GPU_FAULT_ROLE_SPLIT_CONTAINER_ENV_FILE": str(snapshot)},
    )

    assert len([item for item in documents if item["kind"] == "Deployment"]) == 3
    assert not any(
        any(argument.endswith("/wait_control_plane_pins.py") for argument in call)
        for call in python_calls
    ), "a verbatim rollback must not wait for /healthz pins the old wheel lacks"
    assert not any("exec" in call for call in calls), (
        "a verbatim rollback probed a Pod for its pins"
    )


def test_final_cpu_templates_and_commands_keep_the_requested_namespace(
    tmp_path: Path,
) -> None:
    documents, calls, python_calls = apply_roles(
        tmp_path, targets="spool,worker,ingress", force=False, namespace="isolated-cpu"
    )
    deployments = [item for item in documents if item["kind"] == "Deployment"]
    assert len(deployments) == 3
    assert {item["metadata"]["namespace"] for item in deployments} == {"isolated-cpu"}
    for call in calls:
        if "-n" in call:
            assert call[call.index("-n") + 1] == "isolated-cpu"
    wait = pin_wait_call(python_calls)
    assert wait[wait.index("--namespace") + 1] == "isolated-cpu"


# --- Per-tier restart decision (a rollback names the tiers it must restart) ---


def per_role(
    tmp_path: Path, roles: str, *, verbatim: bool, **extra: str
) -> dict[str, str]:
    """The rollback's environment: the forced tiers, the report path and, for
    the verbatim CPU restore, the previous release's container env."""

    environment = {
        "GPU_FAULT_FORCE_ROLE_RESTART_ROLES": roles,
        "GPU_FAULT_ROLE_RESTART_REPORT": str(tmp_path / "restart-report.json"),
        **extra,
    }
    if verbatim:
        snapshot = tmp_path / "container-env.json"
        snapshot.write_text("{}", encoding="utf-8")
        environment["GPU_FAULT_ROLE_SPLIT_CONTAINER_ENV_FILE"] = str(snapshot)
    return environment


def restart_report(tmp_path: Path) -> dict[str, Any]:
    report: dict[str, Any] = json.loads(
        (tmp_path / "restart-report.json").read_text(encoding="utf-8")
    )
    return report


def restarted_roles(documents: list[dict]) -> list[str]:
    """Roles whose applied template carries a fresh restart token, in apply order."""

    roles = {deployment: role for role, deployment in ROLE_DEPLOYMENTS.items()}
    return [
        roles[item["metadata"]["name"]]
        for item in documents
        if item["kind"] == "Deployment"
        and item["spec"]["template"]["metadata"]["annotations"][
            "kubectl.kubernetes.io/restartedAt"
        ]
        != RESTARTED_AT
    ]


def pin_waits(python_calls: list[list[str]]) -> list[list[str]]:
    return [
        call
        for call in python_calls
        if any(argument.endswith("/wait_control_plane_pins.py") for argument in call)
    ]


def waited(call: list[str]) -> list[str]:
    return sorted(call[call.index("--deployments") + 1].split(","))


def verdicts(**deployments: str) -> dict[str, Any]:
    """A wait report: role -> verdict, keyed by Deployment like the tool's."""

    return {
        "converged": all(verdict == "converged" for verdict in deployments.values()),
        "deployments": {
            ROLE_DEPLOYMENTS[role]: {"verdict": verdict, "pods": 1}
            for role, verdict in deployments.items()
        },
    }


def test_rollback_forces_only_the_named_tier_and_proves_the_others_on_healthz(
    tmp_path: Path,
) -> None:
    completed = run_apply(
        tmp_path,
        targets="spool,worker,ingress",
        force=True,
        environment=per_role(tmp_path, "worker", verbatim=False),
    )
    assert completed.returncode == 0, completed.stderr
    documents, calls, python_calls = outputs(tmp_path)

    assert len([item for item in documents if item["kind"] == "Deployment"]) == 3
    assert restarted_roles(documents) == ["worker"], (
        "only the tier whose start-up inputs changed may receive a restart token"
    )
    wait = pin_wait_call(python_calls)
    assert waited(wait) == sorted(DEPLOYMENTS), (
        "the controller stage proves the restored pins on every selected tier"
    )
    assert "--unsupported-fails-fast" in wait, wait
    assert wait[wait.index("--expected-sha256") + 1] == applied_pin_digest(documents)
    assert restart_report(tmp_path) == {
        "forced": ["worker"],
        "fallback": [],
        "pin_wait": {name: "converged" for name in DEPLOYMENTS},
        "restart_wait": {},
    }
    assert "control-plane restart: forced=worker" in completed.stderr


def test_verbatim_restore_without_forced_tiers_proves_every_tier_instead_of_rolling(
    tmp_path: Path,
) -> None:
    documents, calls, python_calls = apply_roles(
        tmp_path,
        targets="spool,worker,ingress",
        force=False,
        environment=per_role(tmp_path, "", verbatim=True),
    )

    assert restarted_roles(documents) == [], "an untouched tier was rolled"
    wait = pin_wait_call(python_calls)
    assert waited(wait) == sorted(DEPLOYMENTS)
    probes = [call[call.index("exec") + 1] for call in calls if "exec" in call]
    assert sorted(probes) == sorted(f"{name}-0" for name in DEPLOYMENTS)
    report = restart_report(tmp_path)
    assert (report["forced"], report["fallback"]) == ([], [])
    assert report["pin_wait"] == {name: "converged" for name in DEPLOYMENTS}


def test_verbatim_restore_does_not_wait_on_the_tier_it_restarts(tmp_path: Path) -> None:
    documents, _, python_calls = apply_roles(
        tmp_path,
        targets="spool,worker,ingress",
        force=True,
        environment=per_role(tmp_path, "ingress", verbatim=True),
    )

    assert restarted_roles(documents) == ["ingress"]
    assert waited(pin_wait_call(python_calls)) == sorted(
        [ROLE_DEPLOYMENTS["spool"], ROLE_DEPLOYMENTS["worker"]]
    ), "a restarted verbatim tier starts from the pin ConfigMap; only the rest wait"
    report = restart_report(tmp_path)
    assert report["forced"] == ["ingress"]
    assert sorted(report["pin_wait"]) == sorted(
        [ROLE_DEPLOYMENTS["spool"], ROLE_DEPLOYMENTS["worker"]]
    )


@pytest.mark.parametrize("verbatim", [True, False])
def test_a_tier_without_fleet_pins_on_healthz_falls_back_to_the_forced_restart(
    tmp_path: Path, verbatim: bool
) -> None:
    documents, calls, python_calls = apply_roles(
        tmp_path,
        targets="spool,worker,ingress",
        force=False,
        environment=per_role(tmp_path, "", verbatim=verbatim),
        pod_pins={ROLE_DEPLOYMENTS["worker"]: "unsupported"},
    )

    workers = [
        item
        for item in documents
        if item["kind"] == "Deployment"
        and item["metadata"]["name"] == ROLE_DEPLOYMENTS["worker"]
    ]
    assert len(workers) == 2, "the unproven tier is applied again with a restart token"
    assert restarted_roles(documents) == ["worker"]
    applies = [index for index, call in enumerate(calls) if "apply" in call]
    probes = [index for index, call in enumerate(calls) if "exec" in call]
    assert max(applies) > min(probes), "the fallback restart must follow the proof"
    waits = pin_waits(python_calls)
    if verbatim:
        assert len(waits) == 1, "a restarted verbatim tier starts from the ConfigMap"
    else:
        assert len(waits) == 2, waits
        assert waited(waits[1]) == [ROLE_DEPLOYMENTS["worker"]]
    report = restart_report(tmp_path)
    assert report["forced"] == []
    assert report["fallback"] == ["worker"]
    assert report["pin_wait"][ROLE_DEPLOYMENTS["worker"]] == "unsupported"
    assert report["restart_wait"] == (
        {} if verbatim else {ROLE_DEPLOYMENTS["worker"]: "unsupported"}
    )


def test_an_unconverged_tier_is_restarted_and_proven_again(tmp_path: Path) -> None:
    documents, _, python_calls = apply_roles(
        tmp_path,
        targets="spool,worker,ingress",
        force=False,
        environment=per_role(tmp_path, "", verbatim=False),
        pin_wait_outcomes=[
            {
                "exit": 1,
                "report": verdicts(
                    spool="converged", worker="unconverged", ingress="converged"
                ),
            },
            {"exit": 0, "report": verdicts(worker="converged")},
        ],
    )

    assert restarted_roles(documents) == ["worker"]
    waits = pin_waits(python_calls)
    assert len(waits) == 2, waits
    assert waited(waits[1]) == [ROLE_DEPLOYMENTS["worker"]]
    report = restart_report(tmp_path)
    assert report["fallback"] == ["worker"]
    assert report["restart_wait"] == {ROLE_DEPLOYMENTS["worker"]: "converged"}


def test_a_tier_unproven_after_its_fallback_restart_fails_the_apply(
    tmp_path: Path,
) -> None:
    completed = run_apply(
        tmp_path,
        targets="spool,worker,ingress",
        force=False,
        environment=per_role(tmp_path, "", verbatim=False),
        pin_wait_outcomes=[
            {"exit": 1, "report": verdicts(spool="converged", worker="unconverged")},
            {"exit": 1, "report": verdicts(worker="unconverged")},
        ],
    )

    assert completed.returncode == 1, completed.stderr
    assert "gpu-fault-control-worker did not converge" in completed.stderr
    assert "after its fallback restart" in completed.stderr
    _, _, python_calls = outputs(tmp_path)
    assert not any(
        any(
            argument.endswith("/verify_control_plane_role_split.py")
            for argument in call
        )
        for call in python_calls
    ), "a failed proof must stop the apply before the verifier"


def test_a_forced_tier_that_does_not_converge_fails_without_a_second_restart(
    tmp_path: Path,
) -> None:
    completed = run_apply(
        tmp_path,
        targets="spool,worker,ingress",
        force=True,
        environment=per_role(tmp_path, "worker", verbatim=False),
        pin_wait_outcomes=[
            {
                "exit": 1,
                "report": verdicts(
                    spool="converged", worker="unconverged", ingress="converged"
                ),
            }
        ],
    )

    assert completed.returncode == 1, completed.stderr
    assert "forced restart" in completed.stderr
    documents, _, python_calls = outputs(tmp_path)
    assert restarted_roles(documents) == ["worker"], "a forced tier is not re-rolled"
    assert len(pin_waits(python_calls)) == 1


def test_a_pin_wait_usage_error_fails_the_apply(tmp_path: Path) -> None:
    completed = run_apply(
        tmp_path,
        targets="spool,worker,ingress",
        force=False,
        environment=per_role(tmp_path, "", verbatim=False),
        pin_wait_outcomes=[{"exit": 2, "report": None}],
    )

    assert completed.returncode == 2, completed.stderr
    documents, _, _ = outputs(tmp_path)
    assert restarted_roles(documents) == [], "a usage error is not a proof failure"


def test_a_wait_without_a_report_restarts_every_waited_tier(tmp_path: Path) -> None:
    documents, _, python_calls = apply_roles(
        tmp_path,
        targets="spool,worker,ingress",
        force=False,
        environment=per_role(tmp_path, "", verbatim=False),
        pin_wait_outcomes=[
            {"exit": 1, "report": None},
            {
                "exit": 0,
                "report": verdicts(
                    spool="converged", worker="converged", ingress="converged"
                ),
            },
        ],
    )

    assert sorted(restarted_roles(documents)) == sorted(ROLE_DEPLOYMENTS)
    waits = pin_waits(python_calls)
    assert len(waits) == 2, waits
    assert waited(waits[1]) == sorted(DEPLOYMENTS)
    report = restart_report(tmp_path)
    assert report["fallback"] == ["spool", "worker", "ingress"]
    assert report["pin_wait"] == {}


@pytest.mark.parametrize(
    ("targets", "force", "environment"),
    [
        (
            "spool,worker,ingress",
            False,
            {"GPU_FAULT_FORCE_ROLE_RESTART_ROLES": "worker"},
        ),
        ("spool,worker,ingress", True, {"GPU_FAULT_FORCE_ROLE_RESTART_ROLES": ""}),
        ("ingress", True, {"GPU_FAULT_FORCE_ROLE_RESTART_ROLES": "worker"}),
        ("spool,worker,ingress", True, {"GPU_FAULT_FORCE_ROLE_RESTART_ROLES": "bogus"}),
        (
            "spool,worker,ingress",
            False,
            {"GPU_FAULT_ROLE_RESTART_REPORT": "/tmp/r.json"},
        ),
    ],
    ids=[
        "roles-without-force",
        "force-without-roles",
        "role-not-targeted",
        "unknown-role",
        "report-without-roles",
    ],
)
def test_inconsistent_restart_requests_are_refused_before_anything_changes(
    tmp_path: Path, targets: str, force: bool, environment: dict[str, str]
) -> None:
    completed = run_apply(
        tmp_path, targets=targets, force=force, environment=environment
    )

    assert completed.returncode == 2, completed.stderr
    documents, calls, _ = outputs(tmp_path)
    assert (documents, calls) == ([], []), "a refused request reached the cluster"


@pytest.mark.parametrize(
    ("requested", "live", "expected"),
    [("", "f" * 64, "f" * 64), ("", None, None), ("e" * 64, "f" * 64, "e" * 64)],
    ids=["empty-inherits-live", "empty-without-live", "rendered-wins"],
)
def test_an_unrendered_failure_domain_map_leaves_the_live_stamp_alone(
    tmp_path: Path, requested: str, live: str | None, expected: str | None
) -> None:
    """A rollback does not re-render the failure-domain map; dropping the live
    stamp from the worker template would roll the worker for nothing."""

    annotation = "gpu-fault.io/failure-domain-map-sha256"
    documents, _, _ = apply_roles(
        tmp_path,
        targets="worker",
        force=False,
        environment={"GPU_FAULT_FAILURE_DOMAIN_MAP_SHA256": requested},
        live_annotations=(
            {ROLE_DEPLOYMENTS["worker"]: {annotation: live}} if live else None
        ),
    )

    worker = next(item for item in documents if item["kind"] == "Deployment")
    assert worker["spec"]["template"]["metadata"]["annotations"].get(annotation) == (
        expected
    )
