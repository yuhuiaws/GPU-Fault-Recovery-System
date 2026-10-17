"""A refused node preflight names every node with its whole reasons and keeps
each node's DCGM state for ``status --full``.

Live 2026-10-17 (a host-mode release while ``nvidia-dcgm`` was inactive on both
plain EKS nodes): the refusal printed the first node only, cut mid-reason --
``ERROR: host preflight ip-10-0-42-179...: {"node_id":...,"reasons":["host
prerequisites unavailable: n`` -- because a sensitive command keeps three
``ERROR:`` lines of 200 characters; the second node was not named, the failed
Jobs were gone, and ``dcgm_hostengine_nodes`` was absent from the status. The
preflight batch now writes a per-node report to a private file the engine
names in the reconciler environment, and the engine refuses from it.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin.command_log import FAILURE_EXIT_CODE_ATTRIBUTE
from gpu_fault_release import regional_node_batch as batch
from gpu_fault_release import regional_release_node_preflight as nodes
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_dcgm import NODE_DCGM_STATE_KEY
from gpu_fault_release.regional_release_observation import ApiUnavailable
from tests.regional._cov95_release_support import ResourceRelease
from tests.regional.test_cov95_release_wait_guards import candidate

NODE_A = "ip-10-0-42-179.us-west-2.compute.internal"
NODE_B = "ip-10-0-37-86.us-west-2.compute.internal"
MISSING = (
    "host prerequisites unavailable: nvidia-dcgm.service (inactive), "
    "dcgmi discovery -l (exit 226)"
)
REMEDY = (
    "the managed node group declares gpu-fault.io/dcgm-hostengine=host: install "
    "the pinned datacenter-gpu-manager-4 packages and systemctl enable --now "
    "nvidia-dcgm in the launch template user data (see 管理员日常运维 / "
    "administrator-operations §5.0.1), or remove the label and deploy to run "
    "the exporter's embedded hostengine"
)
#: What the live refusal looked like: the generic sensitive-command message.
CUT = (
    "command failed (1): deploy-node-installer-reconciler.sh: <sensitive output "
    "redacted>: ERROR: Job <redacted> failed: BackoffLimitExceeded; ERROR: host "
    f'preflight {NODE_A}: {{"node_id":"{NODE_A}","status":"FAILED","reasons":'
    '["host prerequisites unavailable: n'
)


def _eks_release() -> tuple[ResourceRelease, Any]:
    release = ResourceRelease()
    target = replace(
        release.config.clusters[0],
        cluster_id="gpu-eks-cluster",
        provider="eks-managed",
        hyperpod_cluster_name="",
        node_selector="gpu-fault.io/control-plane=example-site",
        node_selector_source="label-convention",
        dcgm_hostengine="host",
        dcgm_hostengine_source="node-group-label",
    )
    release.config.clusters = (target,)
    return release, target


def _report(refused: dict[str, Any], *, status: str = "FAILED") -> dict[str, Any]:
    inactive = {"mode": "host", "state": "installed-inactive", "conflict": False}
    return {
        "status": status,
        "node_count": 2,
        "dcgm_hostengine": {NODE_A: dict(inactive), NODE_B: dict(inactive)},
        "refused": refused,
    }


def _preflight(
    monkeypatch: pytest.MonkeyPatch,
    release: ResourceRelease,
    target: Any,
    report: dict[str, Any] | None,
    failure: ReleaseError,
) -> pytest.ExceptionInfo[ReleaseError]:
    seen: list[str] = []

    def failing_script(_arguments: list[str], options: dict[str, Any]) -> str:
        path = Path(options["env"][nodes.NODE_PREFLIGHT_REPORT_ENV])
        seen.append(str(path))
        if report is not None:
            path.write_text(json.dumps(report), encoding="utf-8")
        raise failure

    release.runner.handler = failing_script
    monkeypatch.setattr(
        nodes, "build_reconciler_environment", lambda *_args, **_kwargs: {}
    )
    with pytest.raises(ReleaseError) as raised:
        nodes.run_node_installer_preflight(
            release, target, replace(candidate(), node_names=(NODE_A, NODE_B))
        )
    (path,) = seen
    assert not Path(path).exists(), "the private report does not outlive the call"
    return raised


def test_a_refused_node_preflight_names_every_node_with_its_whole_reasons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release, target = _eks_release()
    failure = ReleaseError(CUT)
    setattr(failure, FAILURE_EXIT_CODE_ATTRIBUTE, 1)
    refused = {
        NODE_A: {"reasons": [MISSING], "remedies": [REMEDY]},
        NODE_B: {"reasons": [MISSING], "remedies": [REMEDY]},
    }

    raised = _preflight(monkeypatch, release, target, _report(refused), failure)

    assert str(raised.value).splitlines() == [
        "gpu-eks-cluster node preflight refused 2 of 2 node(s) before any node "
        "changed:",
        f"  {NODE_B}: {MISSING}",
        f"  {NODE_A}: {MISSING}",
        f"  remedy: {REMEDY}",
    ], "every node on its own line, each reason whole, the remedy once"
    assert raised.value.__cause__ is failure, "the command's own failure is kept"
    assert getattr(raised.value, FAILURE_EXIT_CODE_ATTRIBUTE) == 1
    record = release.state[NODE_DCGM_STATE_KEY]["gpu-eks-cluster"]
    assert record["mode"] == "host" and record["observed_at"], record
    assert record["nodes"] == {
        NODE_A: {"state": "installed-inactive", "conflict": False},
        NODE_B: {"state": "installed-inactive", "conflict": False},
    }, "status --full shows every node's state after the refusal too"


def test_a_node_line_keeps_whole_reasons_within_its_bound() -> None:
    reasons = [f"reason {index}: " + "x" * 150 for index in range(8)]

    line = nodes.node_refusal_line(NODE_A, {"reasons": reasons})

    assert len(line) <= nodes.NODE_REFUSAL_LINE_CHARACTERS
    kept = line.removeprefix(f"  {NODE_A}: ").rsplit(" (+", 1)[0].split("; ")
    assert kept == reasons[: len(kept)], "only whole reasons, in order"
    assert line.endswith(f" (+{len(reasons) - len(kept)} more)"), (
        "the reasons left out are counted"
    )
    giant = "y" * (2 * nodes.NODE_REFUSAL_LINE_CHARACTERS)
    assert nodes.node_refusal_line(NODE_A, {"reasons": [giant, "z"]}) == (
        f"  {NODE_A}: {giant} (+1 more)"
    ), "a reason is never cut, even one longer than the bound"


def test_without_a_report_or_on_an_api_blip_the_original_failure_stands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release, target = _eks_release()
    original = ReleaseError(CUT)
    raised = _preflight(monkeypatch, release, target, None, original)
    assert raised.value is original, "an older batch writes no report"
    assert NODE_DCGM_STATE_KEY not in release.state

    release, target = _eks_release()
    blip = ApiUnavailable(CUT, reason="credentials")
    refused = {NODE_A: {"reasons": ["Unauthorized"], "remedies": []}}
    raised = _preflight(monkeypatch, release, target, _report(refused), blip)
    assert raised.value is blip, "an unobservable preflight stays retryable"
    assert set(release.state[NODE_DCGM_STATE_KEY]["gpu-eks-cluster"]["nodes"]) == {
        NODE_A,
        NODE_B,
    }, "what the preflight did observe is still recorded"


def test_the_engine_and_the_batch_name_the_same_report_variable() -> None:
    assert nodes.NODE_PREFLIGHT_REPORT_ENV == batch.NODE_PREFLIGHT_REPORT_ENV, (
        "the engine names the file the batch writes"
    )
