from __future__ import annotations

import time
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import pytest

from tests._script_loader import lazy_script_module

MODULE = lazy_script_module(
    Path("deploy/control-plane/regional/regional_release_runtime_identity.py")
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


class ProbeRelease:
    """A release whose ingress Pod can be replaced between two exec attempts.

    ``pod`` is what a fresh resolution returns; ``live_pods`` is what the API
    server still has. They differ exactly while the ingress Deployment rolls,
    which is the window this probe has to survive.
    """

    def __init__(self, *, pod: str, live_pods: list[str], failures: int) -> None:
        self.config = SimpleNamespace(namespace="gpu-fault-system")
        self.pod = pod
        self.live_pods = live_pods
        self.failures = failures
        self.execs: list[str] = []
        self.runner = SimpleNamespace(run=self._run, probe=self._probe, dry_run=False)

    @staticmethod
    def _cpu(*arguments):
        return ["kubectl", "--kubeconfig", "cpu", *arguments]

    def _run(self, arguments, **_kwargs):
        if "get" in arguments and "pod" in arguments and "-l" in arguments:
            return self.pod
        if "exec" in arguments:
            # ``kubectl exec`` may carry flags such as ``-i`` between the verb
            # and the Pod, so the Pod is the last word before ``--``.
            pod = arguments[arguments.index("--") - 1]
            self.execs.append(pod)
            if self.failures > 0:
                self.failures -= 1
                raise MODULE.ReleaseError("command failed (1): kubectl")
            return "probe-output"
        raise AssertionError(f"unexpected command: {arguments}")

    def _probe(self, arguments, **_kwargs):
        return arguments[-1] in self.live_pods

    def prime(self) -> None:
        """Memoise the current Pod the way the run's first probe would.

        The cache is primed through the public resolver so the test starts from
        the state a real barrier leaves behind, whatever attribute the
        memoisation happens to use.
        """
        assert (
            MODULE.cpu_ingress_pod(self, failure="release safety checks") == self.pod
        ), "priming the cache did not resolve the pod under test"


def test_replaced_ingress_pod_buys_one_more_probe_even_without_retries() -> None:
    # CPU finalize rolls the ingress Deployment, so the barrier that runs right
    # after it can exec into a name that no longer exists. The exec never
    # started, so the ``retries=0`` rationale -- the in-Pod barrier already
    # spent its window -- does not apply, and refusing to re-resolve would fail
    # the release for a routine Pod replacement.
    release = ProbeRelease(pod="stale-pod", live_pods=["stale-pod"], failures=1)
    release.prime()
    release.pod = "fresh-pod"
    release.live_pods = ["fresh-pod"]

    assert (
        MODULE.exec_cpu_ingress_probe(
            release, script="print(1)", failure="release safety checks", retries=0
        )
        == "probe-output"
    )
    assert release.execs == ["stale-pod", "fresh-pod"]


def test_a_probe_that_ran_against_a_live_pod_is_not_retried_without_retries() -> None:
    # The replacement allowance is not a general retry: a probe whose Pod is
    # still there failed for its own reasons, and running a barrier a second
    # time would pay its window twice for nothing.
    release = ProbeRelease(pod="live-pod", live_pods=["live-pod"], failures=2)
    release.prime()

    with pytest.raises(MODULE.ReleaseError, match="command failed"):
        MODULE.exec_cpu_ingress_probe(
            release, script="print(1)", failure="release safety checks", retries=0
        )
    assert release.execs == ["live-pod"]


def test_the_replacement_allowance_is_granted_once() -> None:
    # A Pod that keeps vanishing must not loop forever: one extra attempt, then
    # the release reports the failure.
    # Every resolution answers, and every answer is already gone by the time the
    # exec lands.
    release = ProbeRelease(pod="ghost-pod", live_pods=[], failures=5)
    release.prime()

    with pytest.raises(MODULE.ReleaseError, match="command failed"):
        MODULE.exec_cpu_ingress_probe(
            release, script="print(1)", failure="release safety checks", retries=1
        )
    assert len(release.execs) == 3


def test_runtime_component_identity_aggregates_every_failed_pod() -> None:
    release = Release(mismatch=True, replicas=3)

    with pytest.raises(MODULE.ReleaseError) as error:
        MODULE.validate_runtime_component_identity(release)

    message = str(error.value)
    for index in range(3):
        assert f"gpu-fault-cluster-executor-pod-{index}" in message
