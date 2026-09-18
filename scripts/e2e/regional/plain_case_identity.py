"""Read-only identity and canonical evidence for seeded CMD/NET cases."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeGuard
from uuid import uuid4

from gpu_fault.admin.diagnostics import diagnostic_text

from scripts.e2e.regional.acceptance_runner_common import write_json_atomic
from scripts.e2e.regional.acceptance_scope import current_acceptance_scope
from scripts.e2e.regional.regional_case_contract import predecessor_path
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from scripts.e2e.regional.regional_live_fixture import (
    RegionalLiveFixture,
    RegionalLiveSettings,
    predecessor_evidence,
    runtime_identity_errors,
    settings_from_arguments,
)

EXECUTOR_CLUSTER_IDENTITY = """
import json
import os
print(json.dumps({"cluster_id": os.environ.get("GPU_FAULT_CLUSTER_ID", "")}))
"""


def add_plain_identity_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cpu-kubeconfig", default="")
    parser.add_argument("--gpu-kubeconfig", default="")
    parser.add_argument("--gpu-context", default="")
    parser.add_argument(
        "--namespace",
        default=os.getenv(
            "GPU_FAULT_NAMESPACE",
            os.getenv("GPU_FAULT_PERF_DATAPLANE_NAMESPACE", "gpu-fault-system"),
        ),
    )
    parser.add_argument("--cluster-id", default="")
    parser.add_argument("--region", default=os.getenv("GPU_FAULT_PERF_AWS_REGION", ""))


def configure_plain_case(arguments: argparse.Namespace) -> RegionalLiveSettings:
    settings = settings_from_arguments(arguments)
    if settings.cluster_id.startswith("perf-cap-"):
        raise RegionalFixtureError("plain case identity requires a physical cluster")
    return settings


def plain_execution_targets(settings: RegionalLiveSettings) -> dict[str, str]:
    """Refuse stale import-time perf targets before issuing any probe."""

    expected = {
        "CONTROL_KUBECONFIG": str(settings.cpu_kubeconfig),
        "CONTROL_NAMESPACE": settings.namespace,
        "NAMESPACE": settings.namespace,
        "IDENTITY_NAMESPACE": settings.namespace,
        "DATAPLANE_CONTEXT": settings.gpu_context,
        "AWS_REGION": settings.region,
    }
    fields = {
        "scripts.e2e.regional.seeded_command_fixture": (
            "AWS_REGION",
            "CONTROL_NAMESPACE",
            "DATAPLANE_CONTEXT",
            "NAMESPACE",
        ),
        "regional_capacity_registry": tuple(expected),
        "regional_capacity_suite": (
            "AWS_REGION",
            "CONTROL_NAMESPACE",
            "DATAPLANE_CONTEXT",
            "NAMESPACE",
        ),
        "regional_action_capacity_suite": ("NAMESPACE",),
    }
    targets = {}
    for name, keys in fields.items():
        module = importlib.import_module(name)
        for key in keys:
            value = getattr(module, key)
            if key == "CONTROL_KUBECONFIG":
                value = str(Path(value).expanduser().resolve())
            if value != expected[key]:
                raise RegionalFixtureError(
                    f"plain case execution target differs at {name}.{key}"
                )
            targets[f"{name}.{key}"] = value
    return targets


def plain_environment(
    settings: RegionalLiveSettings, environment: dict[str, str]
) -> dict[str, str]:
    expected = {
        "GPU_FAULT_CONTROL_KUBECONFIG": str(settings.cpu_kubeconfig),
        "KUBECONFIG": str(settings.gpu_kubeconfig),
        "GPU_FAULT_DATAPLANE_CONTEXT": settings.gpu_context,
        "GPU_FAULT_PERF_AWS_REGION": settings.region,
        "GPU_FAULT_PERF_CONTROL_NAMESPACE": settings.namespace,
        "GPU_FAULT_PERF_DATAPLANE_NAMESPACE": settings.namespace,
    }
    if set(environment) != set(expected):
        raise RegionalFixtureError("plain case environment identity is incomplete")
    for key, value in environment.items():
        current = (
            str(Path(value).expanduser().resolve()) if "KUBECONFIG" in key else value
        )
        if current != expected[key]:
            raise RegionalFixtureError(f"plain case environment differs at {key}")
    return environment


def plain_identity_valid(identity: Any) -> TypeGuard[dict[str, Any]]:
    return (
        isinstance(identity, dict)
        and all(
            isinstance(identity.get(key), str) and identity[key].strip()
            for key in ("release_id", "cluster_id")
        )
        and not identity["cluster_id"].startswith("perf-cap-")
    )


def plain_case_preflight(
    settings: RegionalLiveSettings,
    case_dir: Path,
    *,
    read_environment: Callable[[], dict[str, str]],
) -> dict[str, Any]:
    errors: list[str] = []
    result: dict[str, Any] = {"errors": errors}
    try:
        result["environment"] = plain_environment(settings, read_environment())
        result["targets"] = plain_execution_targets(settings)
        fixture = RegionalLiveFixture(settings)
        identity = fixture.evidence_identity()
        if (
            not plain_identity_valid(identity)
            or identity["cluster_id"] != settings.cluster_id
        ):
            raise RegionalFixtureError(
                "plain case release/physical cluster is unavailable"
            )
        result["identity"] = identity
        executor = fixture.executor_python(EXECUTOR_CLUSTER_IDENTITY)
        if executor.get("cluster_id") != identity["cluster_id"]:
            raise RegionalFixtureError("plain case physical executor cluster differs")
        runtime = fixture.runtime_identity()
        result["runtime_identity"] = runtime
        errors.extend(runtime_identity_errors(runtime))
        if runtime.get("release_state", {}).get("release_id") != identity["release_id"]:
            errors.append("plain case release changed during identity preflight")
        previous, path = predecessor_path(case_dir.parents[1], case_dir.name, None)
        if previous is None or path is None:
            raise RegionalFixtureError("plain case formal predecessor is unavailable")
        proof = predecessor_evidence(
            path,
            previous,
            release_id=identity["release_id"],
            cluster_id=identity["cluster_id"],
        )
        result["predecessor"] = proof
        if proof.get("valid") is not True:
            errors.append("formal predecessor is not PASS for this release and cluster")
    except Exception as exc:
        errors.append(
            f"{type(exc).__name__}: {diagnostic_text(str(exc), sensitive=True)}"
        )
    write_json_atomic(case_dir / "plain-preflight.json", result)
    return result


class PlainCaseEvidence:
    """Keep body evidence on failure without letting an older PASS survive."""

    def __init__(self, case_dir: Path, case_id: str, attempt: int) -> None:
        self.path = case_dir / f"{case_id}.json"
        self.binding: dict[str, Any] = {"case_id": case_id, "attempt": attempt}
        if self.path.exists():
            previous = self.read()
            write_json_atomic(
                case_dir / "plain-history" / f"{uuid4().hex}.json",
                {"previous_result": previous},
            )
        write_json_atomic(
            self.path,
            {
                **self.binding,
                "status": "RUNNING",
                "verdict": "NOT_RUN",
                "formal_sequence_satisfied": False,
            },
        )

    def read(self) -> dict[str, Any]:
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise RegionalFixtureError("plain case result is not an object")
        return value

    def bind(
        self, preflight: dict[str, Any], details: dict[str, Any], *, passed: bool
    ) -> None:
        identity = preflight.get("identity")
        proof = preflight.get("predecessor")
        if passed and (
            not plain_identity_valid(identity)
            or not isinstance(proof, dict)
            or proof.get("valid") is not True
        ):
            raise RegionalFixtureError(
                "plain preflight lacks release/cluster/predecessor proof"
            )
        if plain_identity_valid(identity):
            self.binding.update(
                release_id=identity["release_id"],
                cluster_id=identity["cluster_id"],
            )
        self.binding.update(
            identity_preflight=preflight,
            synthetic_cluster_id=details.get("synthetic_cluster_id"),
        )
        write_json_atomic(self.path, {**self.read(), **self.binding})

    def planned(self, passed: bool) -> None:
        write_json_atomic(
            self.path,
            {
                **self.read(),
                "status": "COMPLETED" if passed else "FAILED",
                "verdict": "NOT_RUN",
                "preflight_passed": passed,
            },
        )

    def finish(self, code: int) -> int:
        result = self.read()
        scope = current_acceptance_scope()
        for key, expected in scope.plan_fields().items():
            if key in result and result[key] != expected:
                raise RegionalFixtureError(f"plain case result scope differs at {key}")
        for key in (
            "case_id",
            "attempt",
            "release_id",
            "cluster_id",
            "synthetic_cluster_id",
        ):
            if key in result and result[key] != self.binding.get(key):
                raise RegionalFixtureError(
                    f"plain case result identity differs at {key}"
                )
        if type(code) is not int:
            raise RegionalFixtureError("plain case returned an invalid exit code")
        if result.get("verdict") not in {"PASS", "FAIL", "PARTIAL", "NOT_RUN"}:
            raise RegionalFixtureError("plain case result has no valid verdict")
        if result.get("status") == "RUNNING":
            raise RegionalFixtureError("plain case did not write a completed result")
        formal_refused = (
            "formal_sequence_satisfied" in result
            and result["formal_sequence_satisfied"] is not True
        )
        passed = (
            code == 0
            and result["verdict"] == "PASS"
            and result.get("status", "COMPLETED") == "COMPLETED"
            and (scope.selective or not formal_refused)
        )
        if not passed and result["verdict"] == "PASS":
            result["verdict"] = "FAIL"
        result.update(
            **self.binding,
            status="COMPLETED" if passed else "FAILED",
            **scope.result_fields(),
        )
        result["formal_sequence_satisfied"] = passed and not scope.selective
        write_json_atomic(self.path, result)
        return code if code else (0 if passed else 1)

    def fail(self, exc: BaseException) -> None:
        try:
            result = self.read()
        except (OSError, ValueError, RegionalFixtureError):
            result = {}
        result.update(self.binding)
        if result.get("verdict") not in {"FAIL", "PARTIAL"}:
            result["verdict"] = "FAIL"
        result.update(
            status="FAILED",
            formal_sequence_satisfied=False,
            guard_error=f"{type(exc).__name__}: {diagnostic_text(str(exc), sensitive=True)}",
        )
        write_json_atomic(self.path, result)
