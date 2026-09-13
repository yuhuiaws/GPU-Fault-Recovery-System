"""Rolling the CPU ingress Deployment invalidates the memoised ingress Pod name.

Live 2026-09-13: the preflight probes memoised an api-ha Pod, the role-split
apply rolled the Deployment (and failed), and the automatic rollback's first
mutating fleet exec went to the vanished Pod -- ``NotFound`` -- leaving the site
in ``rollback-failed``. The memo is dropped after every role-split apply
(upgrade and rollback restore) and at the rollback entry.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gpu_fault_release import regional_release_orchestration as ORCHESTRATION_MODULE
from gpu_fault_release import (
    regional_release_rollback_context as ROLLBACK_CONTEXT_MODULE,
)
from gpu_fault_release import rollout as ROLLOUT_MODULE
from gpu_fault_release.regional_release_runtime_identity import (
    CPU_INGRESS_POD_ATTRIBUTE,
)

STALE_POD = "gpu-fault-api-ha-6dfb57fc9-5q5dw"


class ScriptRunner:
    dry_run = False

    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def run(self, arguments, **_kwargs) -> str:
        self.commands.append(list(arguments))
        return ""


def release_with_memo() -> SimpleNamespace:
    release = SimpleNamespace(runner=ScriptRunner(), state={})
    setattr(release, CPU_INGRESS_POD_ATTRIBUTE, STALE_POD)
    return release


def test_upgrade_cpu_role_apply_forgets_the_memoised_ingress_pod(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ROLLOUT_MODULE, "apply_failure_domain_map", lambda _release: "")
    release = release_with_memo()

    ROLLOUT_MODULE.render_and_apply_cpu_roles(release, {})

    scripts = [command[1] for command in release.runner.commands]
    assert any(
        script.endswith("apply-control-plane-role-split.sh") for script in scripts
    ), scripts
    assert getattr(release, CPU_INGRESS_POD_ATTRIBUTE) == "", (
        "the apply rolled api-ha; the next exec must re-resolve the Pod"
    )


def test_rollback_cpu_restore_forgets_the_memoised_ingress_pod() -> None:
    release = release_with_memo()

    ROLLBACK_CONTEXT_MODULE.apply_rollback_cpu_environment(release, {})

    scripts = [command[1] for command in release.runner.commands]
    assert any(
        script.endswith("apply-control-plane-role-split.sh") for script in scripts
    ), scripts
    assert getattr(release, CPU_INGRESS_POD_ATTRIBUTE) == "", (
        "the restore rolled api-ha; the next exec must re-resolve the Pod"
    )


def test_rollback_entry_forgets_the_memoised_ingress_pod_before_any_exec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whatever the failed upgrade memoised is suspect once a rollback starts."""

    release = release_with_memo()
    seen: dict[str, str] = {}

    def fake_context(self, _state):
        seen["memo_at_context"] = getattr(self, CPU_INGRESS_POD_ATTRIBUTE)
        return {}, {}

    monkeypatch.setattr(ORCHESTRATION_MODULE, "_rollback_context", fake_context)

    ORCHESTRATION_MODULE.rollback_release(release, state=None, automatic=True)

    assert seen["memo_at_context"] == "", (
        "the memo is dropped before the rollback plans"
    )
