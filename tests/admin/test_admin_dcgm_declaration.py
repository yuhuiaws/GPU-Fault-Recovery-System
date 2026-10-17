"""A DCGM host engine label flipped since the last deploy is the next deploy.

Live 2026-10-17: the operator labelled the plain EKS GPU node group
``gpu-fault.io/dcgm-hostengine=host`` and ran ``deploy`` over the unchanged
source; ``status`` said ``next deploy: NOOP`` (the engine classifies the
recorded ``site.yaml``) and the deploy re-signed its success record without
reading a node group. The declaration is now read with discovery's own code
by ``status`` (quick and ``--full``) and ``node-scope``; the deploy half is in
``tests/test_staging_deploy_dcgm_declaration.py``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pytest

from gpu_fault.admin import bootstrap as admin_bootstrap
from gpu_fault.admin import cli as admin_cli
from gpu_fault.admin import dcgm_declaration
from gpu_fault.admin import node_scope_command as node_scope
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.site import load_site
from gpu_fault.cluster_scope import DCGM_HOSTENGINE_LABEL
from tests.admin._dcgm_declaration_world import (
    EKS_ARN,
    EKS_NAME,
    LABEL_SELECTOR,
    SITE_ID,
    NodeGroupWorld,
    eks_entry,
    eks_site,
    unreadable_world,
)
from tests.admin._registry_sync_support import record_registry_sync
from tests.admin.test_admin_site import site_file

HEADER = "gpu-fault-admin status:"


def _engine_report(*, full: bool) -> dict[str, object]:
    """What the release engine prints: it classifies the recorded entries."""

    checks = [
        {"name": "cpu_workloads", "status": "PASS"},
        {"name": "control_api", "status": "PASS"},
    ]
    if full:
        checks.append({"name": "regional_contexts", "status": "PASS"})
    return {
        "mode": "status",
        "healthy": True,
        "health_scope": "full" if full else "quick",
        "live_release": {
            "release_id": "release-a",
            "phase": "complete",
            "transaction_committed": True,
        },
        "configured_release": {"release_id": "release-a"},
        "next_deploy": {"kind": "NOOP", "changed": []},
        "health": {
            "summary": {"PASS": len(checks), "WARN": 0, "FAIL": 0, "SKIP": 0},
            "checks": checks,
        },
    }


def _status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    world: NodeGroupWorld,
    *,
    full: bool = False,
) -> tuple[int, dict[str, object], list[str], list[list[str]]]:
    """``gpu-fault-admin status`` over the site in ``tmp_path``: ``(exit code,
    JSON document, header lines, engine commands)``."""

    record_registry_sync(tmp_path, release_id="release-a")
    document = json.dumps(_engine_report(full=full), indent=2, sort_keys=True) + "\n"
    engine: list[list[str]] = []

    def fake_run(arguments, **_kwargs):
        engine.append([str(item) for item in arguments])
        return subprocess.CompletedProcess(arguments, 0, stdout=document)

    monkeypatch.setattr(admin_cli, "run_driver", fake_run)
    monkeypatch.setattr(admin_cli, "_live_release_state", lambda _site: {})
    monkeypatch.setattr(dcgm_declaration, "CommandRunner", lambda: world)
    monkeypatch.setenv(admin_cli.ADMIN_LOG_ENVIRONMENT, "/var/log/status.log")
    code = admin_cli.run(
        argparse.Namespace(
            command="status",
            file=None,
            state_dir=tmp_path,
            repo_root=None,
            show_effective_config=False,
            full=full,
        )
    )
    captured = capsys.readouterr()
    header = [line for line in captured.err.splitlines() if HEADER in line]
    return code, json.loads(captured.out), header, engine


# --- status -----------------------------------------------------------------------


@pytest.mark.parametrize("full", [False, True], ids=["quick", "full"])
def test_status_reports_a_flipped_label_as_the_next_deploy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    full: bool,
) -> None:
    eks_site(tmp_path, recorded="embedded")
    world = NodeGroupWorld({"gpu-a": "host", "gpu-b": "host"})

    code, report, header, engine = _status(
        tmp_path, monkeypatch, capsys, world, full=full
    )

    assert code == 0, "the engine's exit status passes through"
    assert ("--full" in engine[0]) is full
    reason = f"dcgm_hostengine: {EKS_NAME} declared host, recorded embedded"
    assert f"{HEADER} next deploy: DATA_PLANE_COMPATIBLE ({reason})" in header, header
    assert report["next_deploy"] == {
        "kind": "DATA_PLANE_COMPATIBLE",
        "changed": ["dcgm_hostengines"],
        "pending": [reason],
    }, "the release diff input the deploy's rewritten entry moves"
    declaration = report["dcgm_hostengine_declarations"][EKS_NAME]
    assert declaration == {
        "recorded": "embedded",
        "recorded_source": "default",
        "declared": "host",
        "declared_source": "node-group-label",
        "node_groups": ["gpu-a", "gpu-b"],
        "state": "pending deploy",
        "refusal": None,
        "error": None,
    }
    # The checks are the engine's: a declared change is the next deploy's.
    expected = _engine_report(full=full)
    assert report["healthy"] is True and report["health"] == expected["health"]
    assert f"{HEADER} failing checks: none" in header
    assert world.node_group_calls() == [
        ("eks", "list-nodegroups", ""),
        ("eks", "describe-nodegroup", "cpu-node-group"),
        ("eks", "describe-nodegroup", "gpu-a"),
        ("eks", "describe-nodegroup", "gpu-b"),
    ], "discovery's node group read, nothing more"
    assert world.calls == world.node_group_calls()


def test_status_reports_a_mixed_declaration_as_the_deploys_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    eks_site(tmp_path, recorded="embedded")
    world = NodeGroupWorld({"gpu-a": "host", "gpu-b": None})

    code, report, header, _engine = _status(tmp_path, monkeypatch, capsys, world)

    assert code == 0
    (line,) = [item for item in header if "next deploy:" in item]
    assert line.startswith(
        f"{HEADER} next deploy: refused (dcgm_hostengine: {EKS_NAME}: managed node "
        f"groups of EKS cluster {EKS_NAME} disagree on {DCGM_HOSTENGINE_LABEL}: "
        f"gpu-a {DCGM_HOSTENGINE_LABEL}=host; gpu-b {DCGM_HOSTENGINE_LABEL}=absent."
    ), line
    assert "--nodegroup-name gpu-b" in line, "the refusal keeps discovery's remedy"
    next_deploy = report["next_deploy"]
    assert next_deploy["kind"] == "NOOP" and next_deploy["changed"] == [], (
        "the release the deploy would roll is unchanged; the deploy refuses first"
    )
    assert len(next_deploy["refused"]) == 1, next_deploy
    declaration = report["dcgm_hostengine_declarations"][EKS_NAME]
    assert declaration["state"] == "blocked" and declaration["declared"] is None


def test_status_of_an_unchanged_declaration_and_of_a_hyperpod_site_stays_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    eks_site(tmp_path / "eks", recorded=None)
    unlabelled = NodeGroupWorld({"gpu-a": None})

    _code, report, header, _engine = _status(
        tmp_path / "eks", monkeypatch, capsys, unlabelled
    )

    assert f"{HEADER} next deploy: NOOP" in header
    assert report["next_deploy"] == {"kind": "NOOP", "changed": []}, (
        "embedded -> embedded leaves the engine's verdict byte for byte"
    )
    declaration = report["dcgm_hostengine_declarations"][EKS_NAME]
    assert (declaration["recorded"], declaration["declared"]) == (
        "embedded",
        "embedded",
    ), "an entry recorded before the field reads as embedded"
    assert declaration["state"] == "converged"
    assert unlabelled.node_group_calls() == [
        ("eks", "list-nodegroups", ""),
        ("eks", "describe-nodegroup", "cpu-node-group"),
        ("eks", "describe-nodegroup", "gpu-a"),
    ]

    site_file(tmp_path / "hyperpod")
    untouched = NodeGroupWorld({"gpu-a": "host"})
    _code, report, header, _engine = _status(
        tmp_path / "hyperpod", monkeypatch, capsys, untouched
    )
    assert "dcgm_hostengine_declarations" not in report, (
        "a HyperPod-only site's document is the engine's"
    )
    assert report["next_deploy"] == {"kind": "NOOP", "changed": []}
    assert untouched.calls == [], "HyperPod declares nothing: no provider read"


def test_status_reports_an_unreadable_declaration_as_a_finding_not_a_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    eks_site(tmp_path, recorded="host")

    code, report, header, _engine = _status(
        tmp_path, monkeypatch, capsys, unreadable_world()
    )

    assert code == 0 and report["healthy"] is True
    assert f"{HEADER} next deploy: NOOP" in header
    assert (
        f"{HEADER} finding: DCGM host engine declaration of {EKS_NAME} could not "
        "be read (command failed (254): aws eks list-nodegroups); the next deploy "
        "reads it again"
    ) in header
    declaration = report["dcgm_hostengine_declarations"][EKS_NAME]
    assert declaration["state"] == "unreadable" and declaration["recorded"] == "host"


# --- node-scope -------------------------------------------------------------------


class NodeScopeWorld(NodeGroupWorld):
    """The node groups plus the kubectl reads ``node-scope`` makes (no Secret,
    every GPU node of the cluster labelled for the site)."""

    def run(self, arguments, **_options):
        if "secret" in arguments:
            return ""
        return json.dumps(
            {
                "items": [
                    {
                        "metadata": {
                            "name": "ip-10-0-38-180",
                            "labels": {
                                "eks.amazonaws.com/nodegroup": "gpu-a",
                                "gpu-fault.io/control-plane": SITE_ID,
                            },
                        }
                    }
                ]
            }
        )


def _node_scope(tmp_path: Path, capsys, world: NodeScopeWorld) -> tuple[dict, str]:
    arguments = admin_cli.parser().parse_args(
        ["node-scope", "--state-dir", str(tmp_path), "--gpu-cluster-arn", EKS_ARN]
    )
    assert (
        node_scope.run_node_scope_command(
            arguments, site=load_site(eks_site(tmp_path)), runner=world
        )
        == 0
    )
    captured = capsys.readouterr()
    (cluster,) = json.loads(captured.out)["clusters"]
    return cluster["dcgm_hostengine"], captured.err


def test_node_scope_names_the_next_deploy_for_a_flipped_label(
    tmp_path: Path, capsys
) -> None:
    report, summary = _node_scope(tmp_path, capsys, NodeScopeWorld({"gpu-a": "host"}))

    assert report == {
        "recorded": "embedded",
        "source": "default",
        "declared": "host",
        "declared_source": "node-group-label",
        "refusal": None,
        "state": "pending deploy",
        "next_deploy": (
            "DATA_PLANE_COMPATIBLE: records dcgmHostEngine host, re-renders the "
            "DCGM exporter and re-installs the Agents"
        ),
    }
    assert (
        "DCGM host engine embedded, declared host (pending deploy: the next deploy "
        "is DATA_PLANE_COMPATIBLE, not NOOP)"
    ) in summary, summary


def test_node_scope_reports_a_selector_covering_no_group_as_the_refusal(
    tmp_path: Path, capsys
) -> None:
    """Discovery refuses a recorded selector that covers no node group; the
    declaration is blocked by that same refusal, not read as embedded."""

    world = NodeScopeWorld({"gpu-a": "host"}, unowned=("gpu-a",))
    report, summary = _node_scope(tmp_path, capsys, world)

    assert report["state"] == "blocked" and report["next_deploy"] == (
        "refused: see refusal"
    )
    assert "covers none of its managed node groups" in str(report["refusal"])
    assert "(blocked: the next deploy refuses, see refusal)" in summary


# --- one reading: discovery's ---------------------------------------------------------


@pytest.mark.parametrize(
    ("groups", "unowned"),
    [
        ({"gpu-a": "host", "gpu-b": "host"}, ()),
        ({"gpu-a": None, "gpu-b": None}, ()),
        ({"gpu-a": "host", "gpu-b": None}, ()),
        ({"gpu-a": "host", "gpu-b": "Host"}, ()),
        ({"gpu-a": "host", "gpu-b": None}, ("gpu-b",)),
        ({"gpu-a": "host"}, ("gpu-a",)),
    ],
    ids=["host", "embedded", "mixed", "unknown", "unowned-absent", "covers-none"],
)
def test_the_declaration_is_what_the_deploys_discovery_records(
    groups: dict[str, str | None], unowned: tuple[str, ...]
) -> None:
    entry = eks_entry("embedded")
    reader = NodeGroupWorld(groups, unowned=unowned)
    (declaration,) = dcgm_declaration.read_dcgm_declarations([entry], reader)
    discovery = NodeGroupWorld(groups, unowned=unowned)
    try:
        identity = admin_bootstrap.discover_cluster(
            discovery,
            cluster_arn=EKS_ARN,
            role="gpu",
            context=EKS_NAME,
            site_id=SITE_ID,
            existing_clusters=(entry,),
        )
    except BootstrapError as exc:
        assert declaration.state == "blocked", declaration
        assert declaration.refusal == str(exc), "the refusal is discovery's own"
    else:
        assert declaration.state in {"converged", "pending deploy"}, declaration
        assert (declaration.declared, declaration.declared_source) == (
            identity.dcgm_hostengine,
            identity.dcgm_hostengine_source,
        ), "the declared mode is the one the deploy writes into site.yaml"
    assert reader.calls == discovery.node_group_calls(), (
        "the same node group reads, in the same order"
    )
    assert entry["nodeSelector"] == LABEL_SELECTOR
