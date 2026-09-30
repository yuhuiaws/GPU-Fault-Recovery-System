"""effective_cpu_runtime_profile_version: the profile a Running api-ha Pod loaded."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from gpu_fault_release import regional_release_state as STATE

NAMESPACE = "gpu-fault-system"


def _release(pods: list[dict[str, Any]], answer: str) -> tuple[SimpleNamespace, list]:
    calls: list[list[str]] = []

    def get_json(arguments: list[str]) -> dict[str, Any]:
        assert arguments[arguments.index("get") + 1] == "pods", arguments
        assert "app=gpu-fault-api-ha" in arguments, "only the ingress role is asked"
        return {"items": pods}

    def run(arguments: list[str], **kwargs: Any) -> str:
        calls.append(arguments)
        assert kwargs.get("capture") is True, "the value is read, not streamed"
        return answer

    return (
        SimpleNamespace(
            config=SimpleNamespace(namespace=NAMESPACE),
            _cpu=lambda *args: ["cpu", *args],
            _get_json=get_json,
            runner=SimpleNamespace(run=run),
        ),
        calls,
    )


def _pod(name: str, phase: str) -> dict[str, Any]:
    return {"metadata": {"name": name}, "status": {"phase": phase}}


def test_the_first_running_replica_answers_from_its_own_env() -> None:
    release, calls = _release(
        [_pod("api-ha-pending", "Pending"), _pod("api-ha-0", "Running")],
        "hyperpod-v1-boot020\n",
    )

    value = STATE.effective_cpu_runtime_profile_version(release)

    assert value == "hyperpod-v1-boot020", value
    assert len(calls) == 1 and "exec" in calls[0] and "api-ha-0" in calls[0], calls
    assert STATE.CPU_PROFILE_ENV in " ".join(calls[0]), "the Pod's env, not a ConfigMap"


def test_no_running_replica_is_unknown_not_a_value() -> None:
    release, calls = _release([_pod("api-ha-pending", "Pending")], "")

    assert STATE.effective_cpu_runtime_profile_version(release) is None, "unknown"
    assert calls == [], "nothing to exec into"


def test_an_empty_env_reads_as_unknown() -> None:
    release, _calls = _release([_pod("api-ha-0", "Running")], "   ")

    assert STATE.effective_cpu_runtime_profile_version(release) is None, "blank env"
