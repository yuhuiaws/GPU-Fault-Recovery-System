from __future__ import annotations

import time
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import pytest

from tests._script_loader import lazy_script_module

MODULE = lazy_script_module(
    "regional_release_runtime_identity_test",
    Path("deploy/control-plane/regional/regional_release_runtime_identity.py"),
)


class Runner:
    def __init__(self, *, mismatch: bool = False) -> None:
        self.mismatch = mismatch

    def run(self, arguments, **_kwargs):
        if f"PATH={MODULE.CONTROL_PLANE_PATH}" in arguments:
            return "c" * 64
        if f"PATH={MODULE.EXECUTOR_PATH}" in arguments:
            return "f" * 64 if self.mismatch else "e" * 64
        raise AssertionError(f"unexpected command: {arguments}")


class ConcurrentRunner(Runner):
    def __init__(self) -> None:
        super().__init__()
        self._lock = Lock()
        self.active = 0
        self.max_active = 0

    def run(self, arguments, **kwargs):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.02)
            return super().run(arguments, **kwargs)
        finally:
            with self._lock:
                self.active -= 1


class Release:
    def __init__(
        self, *, mismatch: bool = False, replicas: int = 1, runner=None
    ) -> None:
        self.config = SimpleNamespace(
            namespace="gpu-fault-system",
            component_digests={"control_plane": "c" * 64, "executor": "e" * 64},
            clusters=(SimpleNamespace(cluster_id="gpu-a", context="gpu-a-context"),),
        )
        self.runner = runner or Runner(mismatch=mismatch)
        self.replicas = replicas

    @staticmethod
    def _cpu(*arguments):
        return ["kubectl", "--kubeconfig", "cpu", *arguments]

    @staticmethod
    def _gpu(target, *arguments):
        return ["kubectl", "--context", target.context, *arguments]

    def _get_json(self, arguments):
        if "deployment" in arguments:
            return {"spec": {"replicas": self.replicas}}
        selector = arguments[arguments.index("-l") + 1]
        deployment = selector.removeprefix("app=")
        return {
            "items": [
                {"metadata": {"name": f"{deployment}-pod-{index}"}}
                for index in reversed(range(self.replicas))
            ]
        }


def test_runtime_component_identity_accepts_matching_pods() -> None:
    result = MODULE.validate_runtime_component_identity(Release())

    assert result["control_plane"]["expected"] == "c" * 64
    assert result["executor"]["expected"] == "e" * 64


def test_runtime_component_identity_rejects_wrong_executor_package() -> None:
    with pytest.raises(MODULE.ReleaseError, match="module digest"):
        MODULE.validate_runtime_component_identity(Release(mismatch=True))


def test_runtime_component_identity_checks_pods_concurrently_with_stable_output() -> (
    None
):
    runner = ConcurrentRunner()
    result = MODULE.validate_runtime_component_identity(
        Release(replicas=10, runner=runner)
    )

    assert runner.max_active > 1
    assert runner.max_active == MODULE.MAX_RUNTIME_IDENTITY_WORKERS
    cpu_pods = result["control_plane"]["deployments"]["gpu-fault-api-ha"]
    assert list(cpu_pods) == sorted(cpu_pods)


def test_runtime_component_identity_aggregates_every_failed_pod() -> None:
    release = Release(mismatch=True, replicas=3)

    with pytest.raises(MODULE.ReleaseError) as error:
        MODULE.validate_runtime_component_identity(release)

    message = str(error.value)
    for index in range(3):
        assert f"gpu-fault-cluster-executor-pod-{index}" in message
