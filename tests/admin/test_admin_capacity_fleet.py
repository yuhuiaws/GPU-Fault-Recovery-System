"""The declared capacity topology against the registered GPU fleet (b229).

``status`` puts the declared ``largestClusterNodeCount`` / ``managedNodeCount``
next to each configured cluster's registered GPU nodes (Agents ACTIVE or
DRAINING in the ``control_api`` check) and warns where the fleet outgrew them;
``config`` refuses to lower a declared count below the registered fleet unless
``--accept-below-registered-fleet``, reading the fleet only for a lowering.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gpu_fault.admin import capacity_fleet, operator_identity
from gpu_fault.admin import cli as admin_cli
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.capacity_evidence import aurora_min_acu_floor
from gpu_fault.admin.capacity_fleet import (
    ACCEPT_FLAG,
    LARGEST,
    MANAGED,
    STATUS_FIELD,
    capacity_findings,
    capacity_topology_field,
    counts_below_fleet,
    lowered_node_counts,
    refuse_lowered_node_counts,
)
from gpu_fault.admin.config import (
    AdminConfig,
    AdminConfigError,
    CapacityConfig,
    admin_config_pending_path,
    load_desired_admin_config,
)
from gpu_fault.admin.config_file import initialize_desired_admin_config
from gpu_fault.admin.resource_registry_outcome import status_view
from tests.admin._registry_sync_support import record_registry_sync
from tests.admin.test_admin_site import mock_live_release, site_file
from tests.admin.test_admin_site_config import Stubs

APPROVER = "arn:aws:sts::123456789012:assumed-role/Admin/alice"


@pytest.fixture(autouse=True)
def fixed_approver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        operator_identity, "caller_identity_arn", lambda **_kwargs: APPROVER
    )


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def status_report(
    clusters: dict[str, list[str]],
    *,
    largest: int = 512,
    managed: int = 512,
    details: bool = True,
) -> dict[str, object]:
    """A status document: the configured topology and each cluster's Agents."""

    agents = {
        cluster_id: {
            "agents": [
                {"node_id": f"{cluster_id}-{index}", "lifecycle_state": state}
                for index, state in enumerate(states)
            ]
        }
        for cluster_id, states in clusters.items()
    }
    return {
        "mode": "status",
        "healthy": True,
        "health_scope": "quick",
        "configured_release": {
            "admin_config": {
                "capacity": {
                    "largest_cluster_node_count": largest,
                    "managed_node_count": managed,
                }
            }
        },
        "live_release": {
            "release_id": "release-a",
            "phase": "complete",
            "transaction_committed": True,
        },
        "next_deploy": {"kind": "NOOP", "changed": []},
        "health": {
            "summary": {"PASS": 2, "WARN": 0, "FAIL": 0, "SKIP": 0},
            "checks": [
                {"name": "cpu_workloads", "status": "PASS", "details": {}},
                {
                    "name": "control_api",
                    "status": "PASS" if details else "FAIL",
                    "details": {"clusters": agents} if details else None,
                },
            ],
        },
    }


def test_status_puts_the_declared_counts_next_to_the_registered_nodes() -> None:
    report = status_report(
        {
            "hp-a": ["ACTIVE", "ACTIVE", "DRAINING", "REVOKED"],
            "eks-b": ["ACTIVE", "REVOKED"],
        }
    )

    assert capacity_findings(report, state_dir="/secure/site") == [
        "capacity: registered GPU nodes 4 of the declared managedNodeCount 512; "
        "per cluster against the declared largestClusterNodeCount 512: hp-a 3, "
        "eks-b 1"
    ], "a retired (REVOKED) Agent is not a registered GPU node"
    assert capacity_topology_field(report) == {
        STATUS_FIELD: {
            "declared": {"largest_cluster_node_count": 512, "managed_node_count": 512},
            "registered": {"clusters": {"hp-a": 3, "eks-b": 1}, "managed": 4},
            "clusters_above_declared": [],
            "managed_above_declared": False,
        }
    }


def test_status_warns_for_a_cluster_above_the_declared_largest_cluster() -> None:
    report = status_report(
        {"hp-a": ["ACTIVE"] * 5, "eks-b": ["ACTIVE"] * 3}, largest=4, managed=16
    )

    findings = capacity_findings(report, state_dir="/secure/site")

    assert findings[1:] == [
        "WARN capacity: cluster hp-a has 5 registered GPU nodes, above the declared "
        "largestClusterNodeCount 4; raise spec.capacity.largestClusterNodeCount to "
        "at least 5 (spec.processor.maxClusterQueueDepth at least 20) and run "
        "gpu-fault-admin config --state-dir /secure/site"
    ], findings
    assert capacity_topology_field(report)[STATUS_FIELD]["clusters_above_declared"] == [
        "hp-a"
    ], "only the cluster above the declared count is named"


def test_status_warns_when_the_managed_fleet_outgrew_the_declaration() -> None:
    report = status_report(
        {"hp-a": ["ACTIVE"] * 300, "eks-b": ["ACTIVE"] * 300}, largest=512, managed=512
    )

    findings = capacity_findings(report)

    assert findings[1:] == [
        "WARN capacity: 600 registered GPU nodes, above the declared "
        "managedNodeCount 512; raise spec.capacity.managedNodeCount to at least 600 "
        f"with spec.aurora.minAcu at least {aurora_min_acu_floor(600):g} "
        "(ceil(managedNodeCount / 100, 0.5 ACU)) and run gpu-fault-admin config "
        "--state-dir <state-dir>"
    ], findings
    assert aurora_min_acu_floor(600) == 6.0, "the coupled Aurora floor for 600 nodes"


def test_equal_counts_are_not_exceeded() -> None:
    findings = capacity_findings(
        status_report({"hp-a": ["ACTIVE"] * 4}, largest=4, managed=4)
    )

    assert [line for line in findings if line.startswith("WARN")] == [], findings


def test_a_large_fleet_names_five_clusters_and_points_at_the_document() -> None:
    clusters = {f"c-{index:02d}": ["ACTIVE"] * (10 + index) for index in range(8)}
    report = status_report(clusters, largest=5, managed=1000)

    findings = capacity_findings(report)

    assert findings[0].endswith(
        "c-07 17, c-06 16, c-05 15, c-04 14, c-03 13, and 3 more in capacity_topology"
    ), findings[0]
    warnings = [line for line in findings if line.startswith("WARN")]
    assert len(warnings) == 6, warnings
    assert warnings[-1] == (
        "WARN capacity: 3 more clusters above the declared largestClusterNodeCount "
        "in capacity_topology"
    ), warnings[-1]


@pytest.mark.parametrize(
    "report",
    [
        None,
        {"healthy": True},
        status_report({"hp-a": ["ACTIVE"]}, details=False),
        {
            **status_report({"hp-a": ["ACTIVE"]}),
            "configured_release": {"release_id": "release-a"},
        },
    ],
    ids=[
        "no-document",
        "no-checks",
        "control-api-failed",
        "no-configured-admin-config",
    ],
)
def test_status_says_nothing_without_both_halves(report) -> None:
    assert capacity_findings(report) == [], "no capacity line without both halves"
    assert capacity_topology_field(report) == {}, "no field without both halves"


def test_the_status_verb_prints_the_findings_and_adds_the_document_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    site_file(tmp_path)
    record_registry_sync(tmp_path, release_id="release-a")
    report = status_report({"hp-a": ["ACTIVE"] * 3}, largest=2, managed=512)
    document = json.dumps(report, indent=2, sort_keys=True) + "\n"

    def fake_run(arguments, **_kwargs):
        return subprocess.CompletedProcess(arguments, 0, stdout=document)

    monkeypatch.setattr(admin_cli, "run_driver", fake_run)
    monkeypatch.setattr(admin_cli, "_live_release_state", lambda _site: {})
    arguments = argparse.Namespace(
        command="status",
        file=None,
        state_dir=tmp_path,
        repo_root=None,
        show_effective_config=False,
        full=False,
    )

    assert admin_cli.run(arguments) == 0, "an advisory finding never fails status"

    captured = capsys.readouterr()
    findings = [
        line.removeprefix("gpu-fault-admin status: finding: ")
        for line in captured.err.splitlines()
        if line.startswith("gpu-fault-admin status: finding: ")
    ]
    assert findings == capacity_findings(report, state_dir=tmp_path), findings
    assert f"--state-dir {tmp_path}" in findings[-1], findings
    expected = {
        **report,
        "installation_registry_sync": status_view(tmp_path, release_id="release-a"),
        **capacity_topology_field(report),
    }
    assert json.loads(captured.out) == expected, "engine fields must stay unchanged"


# ---------------------------------------------------------------------------
# config: the pure rules
# ---------------------------------------------------------------------------


def capacity(largest: int, managed: int) -> CapacityConfig:
    return CapacityConfig(
        largest_cluster_node_count=largest, managed_node_count=managed
    )


@pytest.mark.parametrize(
    ("before", "after", "lowered"),
    [
        ((512, 512), (512, 512), ()),
        ((512, 512), (1024, 2048), ()),
        ((512, 2048), (256, 2048), (LARGEST,)),
        ((512, 2048), (512, 1024), (MANAGED,)),
        ((512, 2048), (256, 1024), (LARGEST, MANAGED)),
    ],
)
def test_only_a_lower_declared_count_is_a_lowering(before, after, lowered) -> None:
    assert lowered_node_counts(capacity(*before), capacity(*after)) == lowered


def test_a_lowered_count_is_judged_against_the_largest_cluster_and_the_total() -> None:
    fleet = {"eks-b": 120, "hp-a": 400}

    assert counts_below_fleet(fleet, capacity(256, 300), (LARGEST, MANAGED)) == [
        "spec.capacity.largestClusterNodeCount 256 is below the 400 registered GPU "
        "nodes of cluster hp-a",
        "spec.capacity.managedNodeCount 300 is below the 520 registered GPU nodes of "
        "the managed fleet",
    ]
    assert counts_below_fleet(fleet, capacity(400, 520), (LARGEST, MANAGED)) == [], (
        "a lowered count at the registered nodes is not below them"
    )
    assert counts_below_fleet(fleet, capacity(256, 300), ()) == [], (
        "a count that was not lowered is not judged"
    )


def _configs(before: tuple[int, int], after: tuple[int, int]):
    return (
        AdminConfig(capacity=capacity(*before)),
        AdminConfig(capacity=capacity(*after)),
    )


def test_nothing_is_read_unless_a_count_is_lowered() -> None:
    def forbidden(_site):
        raise AssertionError("the fleet was read for a config that lowers nothing")

    current, desired = _configs((512, 512), (1024, 2048))

    refuse_lowered_node_counts(None, current, desired, accepted=False, read=forbidden)


def test_a_lowering_below_the_fleet_is_refused_and_names_the_override() -> None:
    current, desired = _configs((512, 2048), (256, 2048))

    with pytest.raises(AdminConfigError) as error:
        refuse_lowered_node_counts(
            None, current, desired, accepted=False, read=lambda _site: {"hp-a": 400}
        )

    message = str(error.value)
    assert message.startswith(
        "refusing to lower the declared topology below the registered GPU fleet: "
        "spec.capacity.largestClusterNodeCount 256 is below the 400 registered GPU "
        "nodes of cluster hp-a."
    ), message
    assert f"rerun with {ACCEPT_FLAG}" in message, message


def test_the_override_turns_the_refusal_into_a_warning() -> None:
    warnings: list[str] = []
    current, desired = _configs((512, 2048), (256, 300))

    refuse_lowered_node_counts(
        None,
        current,
        desired,
        accepted=True,
        read=lambda _site: {"hp-a": 400},
        warn=warnings.append,
    )

    assert warnings == [
        "spec.capacity.largestClusterNodeCount 256 is below the 400 registered GPU "
        "nodes of cluster hp-a; spec.capacity.managedNodeCount 300 is below the 400 "
        "registered GPU nodes of the managed fleet"
    ], warnings


@pytest.mark.parametrize(
    "failure",
    [
        BootstrapError("workflow reconcile found no Running CPU ingress Pod"),
        subprocess.TimeoutExpired(["kubectl"], 120),
        OSError("kubectl: not found"),
    ],
    ids=["no-ingress-pod", "timeout", "no-kubectl"],
)
def test_a_lowering_that_cannot_be_judged_is_refused(failure: Exception) -> None:
    def unreadable(_site):
        raise failure

    current, desired = _configs((512, 2048), (256, 2048))

    with pytest.raises(AdminConfigError, match="cannot read the registered GPU nodes"):
        refuse_lowered_node_counts(
            None, current, desired, accepted=False, read=unreadable
        )
    warnings: list[str] = []
    refuse_lowered_node_counts(
        None, current, desired, accepted=True, read=unreadable, warn=warnings.append
    )
    assert len(warnings) == 1, warnings
    assert warnings[0].startswith(
        "cannot read the registered GPU nodes to judge the lowered "
        "spec.capacity.largestClusterNodeCount: "
    ), warnings


# ---------------------------------------------------------------------------
# config: the in-Pod read
# ---------------------------------------------------------------------------


class PodStore:
    """What the in-Pod script reads: every Agent and every registration."""

    def list_agents(self):
        return [
            SimpleNamespace(cluster_id=cluster_id, lifecycle_state=state)
            for cluster_id, state in (
                ("hp-a", "ACTIVE"),
                ("hp-a", "DRAINING"),
                ("hp-a", "REVOKED"),
                ("eks-b", "ACTIVE"),
                ("perf-cap-000", "ACTIVE"),
            )
        ]

    def list_regional_clusters(self):
        return [
            SimpleNamespace(cluster_id="hp-a", synthetic=False),
            SimpleNamespace(cluster_id="eks-b", synthetic=False),
            SimpleNamespace(cluster_id="perf-cap-000", synthetic=True),
        ]


def test_the_pod_script_counts_what_the_control_plane_gauges_count(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    from gpu_fault.app import ApplicationContext

    context = SimpleNamespace(store=PodStore())
    monkeypatch.setattr(
        ApplicationContext, "from_environment", classmethod(lambda cls: context)
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"mode": "registered-fleet"}'))

    exec(compile(capacity_fleet.FLEET_SCRIPT, "<registered-fleet>", "exec"), {})

    result = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert result == {"clusters": {"eks-b": 1, "hp-a": 2}}, (
        "Agents ACTIVE or DRAINING, synthetic drill clusters left out"
    )


def test_the_fleet_read_goes_through_the_administrator_exec_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, dict, str]] = []

    def control_plane(site, payload, *, script):
        calls.append((site, payload, script))
        return {"clusters": {"hp-a": 4}}

    monkeypatch.setattr(capacity_fleet, "run_control_plane_script", control_plane)

    assert capacity_fleet.read_registered_fleet("site") == {"hp-a": 4}
    assert calls == [
        ("site", {"mode": "registered-fleet"}, capacity_fleet.FLEET_SCRIPT)
    ], calls


@pytest.mark.parametrize(
    "answer",
    [{}, {"clusters": [4]}, {"clusters": {"hp-a": -1}}, {"clusters": {"hp-a": True}}],
    ids=["no-clusters", "a-list", "negative", "a-bool"],
)
def test_an_invalid_fleet_answer_is_a_failed_read(
    monkeypatch: pytest.MonkeyPatch, answer: dict
) -> None:
    monkeypatch.setattr(
        capacity_fleet,
        "run_control_plane_script",
        lambda _site, _payload, *, script: answer,
    )

    with pytest.raises(BootstrapError, match="invalid answer"):
        capacity_fleet.read_registered_fleet("site")


# ---------------------------------------------------------------------------
# config: the verb
# ---------------------------------------------------------------------------


def _config_file(path: Path, spec: dict[str, object]) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "apiVersion": "gpu-fault.aws/v1alpha1",
                "kind": "AdminConfig",
                "spec": spec,
            }
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)
    return path


def _apply(tmp_path: Path, spec: dict[str, object], *extra: str) -> int:
    config = _config_file(tmp_path / "admin-input.yaml", spec)
    return admin_cli.run(
        admin_cli.parser().parse_args(
            ["config", "--state-dir", str(tmp_path), "--file", str(config), *extra]
        )
    )


@pytest.fixture
def live_site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Stubs:
    """A committed live release of the default (512 / 512) topology."""

    site_file(tmp_path)
    mock_live_release(monkeypatch)
    initialize_desired_admin_config(tmp_path)
    return Stubs(monkeypatch)


def _fleet(monkeypatch: pytest.MonkeyPatch, clusters: dict[str, int]) -> list[str]:
    reads: list[str] = []

    def control_plane(_site, _payload, *, script):
        reads.append(script)
        return {"clusters": clusters}

    monkeypatch.setattr(capacity_fleet, "run_control_plane_script", control_plane)
    return reads


LOWERED = {"capacity": {"largestClusterNodeCount": 256}}


def test_config_refuses_lowering_below_the_registered_fleet_before_any_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, live_site: Stubs
) -> None:
    reads = _fleet(monkeypatch, {"hp-a": 400, "eks-b": 2})

    with pytest.raises(AdminConfigError, match="refusing to lower") as error:
        _apply(tmp_path, LOWERED)

    assert "largestClusterNodeCount 256 is below the 400 registered GPU nodes" in str(
        error.value
    ), error.value
    assert reads == [capacity_fleet.FLEET_SCRIPT], "one fleet read, nothing else"
    assert live_site.events == [], "nothing may roll"
    assert not admin_config_pending_path(tmp_path).exists(), "no apply was recorded"
    assert load_desired_admin_config(tmp_path).capacity.largest_cluster_node_count == (
        512
    ), "desired.json must keep the declared count"


def test_config_applies_a_lowering_below_the_fleet_with_the_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, live_site: Stubs, capsys
) -> None:
    _fleet(monkeypatch, {"hp-a": 400})

    assert _apply(tmp_path, LOWERED, ACCEPT_FLAG) == 0, "the override applies"

    captured = capsys.readouterr()
    assert json.loads(captured.out)["status"] == "APPLIED", captured.out
    assert (
        f"gpu-fault-admin config: {ACCEPT_FLAG}: spec.capacity.largestClusterNodeCount "
        "256 is below the 400 registered GPU nodes of cluster hp-a; applying anyway"
    ) in captured.err, captured.err
    assert live_site.events == ["verify", "release"], live_site.events
    assert load_desired_admin_config(tmp_path).capacity.largest_cluster_node_count == (
        256
    ), "the accepted target is the new declaration"


def test_config_applies_a_lowering_that_still_holds_the_fleet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, live_site: Stubs, capsys
) -> None:
    _fleet(monkeypatch, {"hp-a": 200})

    assert _apply(tmp_path, LOWERED) == 0, "256 still holds a 200-node cluster"
    assert ACCEPT_FLAG not in capsys.readouterr().err, "no warning without a finding"


def test_config_raising_the_counts_reads_no_fleet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, live_site: Stubs
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("a raise must not read the fleet")

    monkeypatch.setattr(capacity_fleet, "run_control_plane_script", forbidden)

    assert (
        _apply(
            tmp_path,
            {
                "capacity": {"largestClusterNodeCount": 1024, "managedNodeCount": 2048},
                "aurora": {"minAcu": 20.5, "maxAcu": 64},
            },
        )
        == 0
    ), "raising the topology with its Aurora floor applies"


def test_config_dry_run_stays_local_for_a_lowering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    site_file(tmp_path)
    initialize_desired_admin_config(tmp_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dry-run read the fleet")

    monkeypatch.setattr(capacity_fleet, "run_control_plane_script", forbidden)

    assert _apply(tmp_path, LOWERED, "--dry-run") == 0, "dry-run plans locally"
    assert json.loads(capsys.readouterr().out)["status"] == "DRY_RUN"


def test_config_help_names_the_override() -> None:
    parsed = admin_cli.parser().parse_args(["config", "--state-dir", "s", ACCEPT_FLAG])

    assert parsed.accept_below_registered_fleet is True, "the flag parses"
    assert (
        admin_cli.parser().parse_args(["config", "--state-dir", "s"])
    ).accept_below_registered_fleet is False, "off by default"
