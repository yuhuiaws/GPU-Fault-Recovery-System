"""A deploy of an unchanged source re-reads the DCGM host engine declaration.

Live 2026-10-17: the GPU node group of a plain EKS cluster was labelled
``gpu-fault.io/dcgm-hostengine=host`` and ``deploy`` ran over the source the
site already ran. Classified UNCHANGED (same fingerprint, same live evidence),
it re-signed its success record in 90 s without one node group read; the mode
flip waited for the next source change. Every NOOP shortcut now compares the
declaration with ``site.yaml`` first: a changed one takes the application
release, whose bootstrap rewrites the entry and whose release diff then sees
``dcgm_hostengines``; a mixed one refuses before anything.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

from gpu_fault.admin import bootstrap as admin_bootstrap
from gpu_fault.admin import dcgm_declaration
from gpu_fault.admin.bootstrap_discovery import cluster_provider_entries
from gpu_fault.admin.bootstrap_site import finalize_bootstrap_site
from gpu_fault.admin.command_log import ADMIN_LOG_ENVIRONMENT
from gpu_fault_release.regional_release_dcgm import (
    DCGM_HOSTENGINE_DRIFT_INPUT,
    DCGM_HOSTENGINES_INPUT,
)
from scripts import staging_deploy
from tests._staging_deploy_support import (
    DEPLOY_EVENT,
    EVIDENCE_EVENT,
    RECORD_EVIDENCE_EVENT,
    deploy_arguments,
    install_admin_stub,
    live_evidence,
    signing_material,
    source_identities,
    status_report,
    stub_deploy_orchestration,
)
from tests.admin._dcgm_declaration_world import (
    EKS_ARN,
    EKS_NAME,
    SITE_ID,
    NodeGroupWorld,
    eks_entry,
    unreadable_world,
)

#: What the pre-deploy reading and the shortcut cost without a flip.
UNCHANGED_EVENTS = ["lock-enter", EVIDENCE_EVENT, "success", "prune", "lock-exit"]
SITE_REWRITTEN = "site-rewritten"


class _BootstrapState:
    path = Path("bootstrap-state.json")

    def record(self, *_arguments: object) -> None:
        return None

    def phase(self, *_arguments: object) -> None:
        return None


def _site(state: Path, recorded: str | None, *, hyperpod_only: bool = False) -> Path:
    clusters: list[dict[str, Any]] = [
        {"clusterId": "hp-gpu-a", "hyperpodClusterName": "hp-gpu-a"}
    ]
    if not hyperpod_only:
        clusters.append(eks_entry(recorded))
    path = state / "site.yaml"
    path.write_text(
        yaml.safe_dump(
            {"kind": "RegionalSite", "spec": {"clusters": clusters}}, sort_keys=False
        ),
        encoding="utf-8",
    )
    return path


def _entry(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    return next(
        item for item in document["spec"]["clusters"] if item["clusterId"] == EKS_NAME
    )


def _deploy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    world: NodeGroupWorld,
    recorded: str | None = "embedded",
    old_mode: str = "UNCHANGED",
    report: dict[str, Any] | None = None,
    hyperpod_only: bool = False,
    events: list[str] | None = None,
    release_fails: bool = False,
) -> tuple[dict[str, object], list[str], Path]:
    """One staging deploy over a site the last deploy recorded ``recorded`` for.

    ``old_mode`` is what the source identities alone classify; ``report`` the
    quick status the deploy reads (its ``dcgm_hostengine_declarations`` decide
    whether the deploy reads the node groups itself). The application
    release's inner hop is the bootstrap's own site regeneration: discovery
    over the recorded entries, then ``finalize_bootstrap_site``; with
    ``release_fails`` the release after it fails.
    """

    monkeypatch.delenv(ADMIN_LOG_ENVIRONMENT, raising=False)
    repository = tmp_path / "repo"
    repository.mkdir(parents=True)
    state = tmp_path / "state"
    state.mkdir()
    site = _site(state, recorded, hyperpod_only=hyperpod_only)
    install_admin_stub(state)
    signing = signing_material(state)
    source = staging_deploy.SourceCheckout(
        repository_root=repository,
        git_commit="a" * 40,
        fingerprint="a" * 64,
        snapshot=False,
        isolated=True,
    )
    previous: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASSED",
        "identities": source_identities(),
        "source": {"fingerprint": source.fingerprint, "git_commit": source.git_commit},
        "live": live_evidence(),
    }
    if old_mode == "DEPLOY_HOST_ONLY":
        previous["identities"]["deploy_host"]["sha256"] = "different"
    elif old_mode == "QUALITY_ONLY":
        previous["source"]["fingerprint"] = "different"
    elif old_mode == "APPLICATION_RELEASE":
        previous["identities"]["application"]["sha256"] = "different"
    events = events if events is not None else []
    stub_deploy_orchestration(
        monkeypatch,
        state=state,
        source=source,
        signing=signing,
        previous=previous,
        events=events,
    )
    reading = report if report is not None else status_report()
    monkeypatch.setattr(staging_deploy, "read_live_status", lambda **_k: reading)
    monkeypatch.setattr(staging_deploy, "run_source_impact_gate", lambda **_k: {})
    monkeypatch.setattr(dcgm_declaration, "CommandRunner", lambda: world)
    inner = NodeGroupWorld(world.groups, unowned=world.unowned)

    def application_release(**kwargs: object) -> None:
        events.append(f"deploy(lock_fd={kwargs.get('lock_fd')})")
        existing = yaml.safe_load(site.read_text(encoding="utf-8"))
        identity = admin_bootstrap.discover_cluster(
            inner,
            cluster_arn=EKS_ARN,
            role="gpu",
            context=EKS_NAME,
            site_id=SITE_ID,
            existing_clusters=existing["spec"]["clusters"],
        )
        generated = deepcopy(existing)
        for item in generated["spec"]["clusters"]:
            if item["clusterId"] == EKS_NAME:
                item.update(cluster_provider_entries(identity))
        finalize_bootstrap_site(
            site,
            generated,
            existing,
            [identity],
            _BootstrapState(),
            lambda path, document: path.write_text(
                yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
            ),
        )
        events.append(SITE_REWRITTEN)
        if release_fails:
            raise staging_deploy.StagingDeployError("release-deploy exited 4")

    monkeypatch.setattr(staging_deploy, "run_admin_deploy", application_release)
    result = staging_deploy.deploy(deploy_arguments(repository, state))
    return result, events, site


def _reading(declaration: dict[str, Any]) -> dict[str, Any]:
    """A quick status from a deploy host that reads the declarations."""

    return {**status_report(), dcgm_declaration.STATUS_FIELD: {EKS_NAME: declaration}}


def _declaration(state: str, **overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "recorded": "embedded",
        "recorded_source": "default",
        "declared": "embedded",
        "declared_source": "default",
        "node_groups": ["gpu-a"],
        "state": state,
        "refusal": None,
        "error": None,
    }
    value.update(overrides)
    return value


@pytest.mark.parametrize("old_mode", ["UNCHANGED", "DEPLOY_HOST_ONLY", "QUALITY_ONLY"])
@pytest.mark.parametrize("read_by", ["deploy", "status"])
def test_a_flipped_label_on_an_unchanged_source_releases_and_rewrites_site_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, old_mode: str, read_by: str
) -> None:
    world = NodeGroupWorld({"gpu-a": "host"})
    report = (
        _reading(
            _declaration(
                "pending deploy", declared="host", declared_source="node-group-label"
            )
        )
        if read_by == "status"
        else None
    )

    result, events, site = _deploy(
        tmp_path, monkeypatch, world=world, old_mode=old_mode, report=report
    )

    assert result["deploy_mode"] == "APPLICATION_RELEASE", (
        f"{old_mode} skipped the release the flipped label needs"
    )
    entry = _entry(site)
    assert (entry["dcgmHostEngine"], entry["dcgmHostEngineSource"]) == (
        "host",
        "node-group-label",
    ), "the release path's bootstrap records the declared mode"
    assert entry["nodeSelector"] == eks_entry(None)["nodeSelector"], (
        "membership is kept as recorded"
    )
    assert events.index(DEPLOY_EVENT) < events.index(SITE_REWRITTEN)
    assert events.index(SITE_REWRITTEN) < events.index(RECORD_EVIDENCE_EVENT)
    assert events.index(RECORD_EVIDENCE_EVENT) < events.index("success"), (
        "the success record is signed only after the release that records it"
    )
    expected_reads = (
        []
        if read_by == "status"
        else [
            ("eks", "list-nodegroups", ""),
            ("eks", "describe-nodegroup", "cpu-node-group"),
            ("eks", "describe-nodegroup", "gpu-a"),
        ]
    )
    assert world.calls == expected_reads, (
        "the quick status' reading is reused; read here only without it"
    )


def test_a_failed_release_after_the_rewrite_signs_no_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rewritten entry whose release failed is never a success: the engine
    compares it with the live release state, so the next deploy releases it."""

    events: list[str] = []
    with pytest.raises(staging_deploy.StagingDeployError, match="exited 4"):
        _deploy(
            tmp_path,
            monkeypatch,
            world=NodeGroupWorld({"gpu-a": "host"}),
            events=events,
            release_fails=True,
        )

    assert SITE_REWRITTEN in events and "success" not in events, events
    assert _entry(tmp_path / "state/site.yaml")["dcgmHostEngine"] == "host"


def test_unchanged_declarations_keep_their_path_and_cost_one_read_at_most(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unlabelled = NodeGroupWorld({"gpu-a": None})

    result, events, site = _deploy(tmp_path / "old-host", monkeypatch, world=unlabelled)

    assert result["deploy_mode"] == "UNCHANGED"
    assert events == UNCHANGED_EVENTS, "the shortcut is exactly what it was"
    assert unlabelled.calls == [
        ("eks", "list-nodegroups", ""),
        ("eks", "describe-nodegroup", "cpu-node-group"),
        ("eks", "describe-nodegroup", "gpu-a"),
    ], "a status without declarations costs discovery's node group read, nothing more"
    assert _entry(site) == eks_entry("embedded"), "site.yaml is not touched"

    reused = NodeGroupWorld({"gpu-a": None})
    result, events, _site_file = _deploy(
        tmp_path / "new-host",
        monkeypatch,
        world=reused,
        report=_reading(_declaration("converged")),
    )
    assert result["deploy_mode"] == "UNCHANGED" and events == UNCHANGED_EVENTS
    assert reused.calls == [], "the quick status already read the declaration"

    # HyperPod is always host and an application release's bootstrap reads the
    # declaration itself: neither pays a read here.
    hyperpod = NodeGroupWorld({"gpu-a": "host"})
    result, events, _site_file = _deploy(
        tmp_path / "hyperpod", monkeypatch, world=hyperpod, hyperpod_only=True
    )
    assert result["deploy_mode"] == "UNCHANGED" and events == UNCHANGED_EVENTS
    assert hyperpod.calls == [], "HyperPod is always host: nothing to read"

    release = NodeGroupWorld({"gpu-a": "host"})
    result, _events, site = _deploy(
        tmp_path / "release", monkeypatch, world=release, old_mode="APPLICATION_RELEASE"
    )
    assert result["deploy_mode"] == "APPLICATION_RELEASE"
    assert release.calls == [], "the release's bootstrap reads the declaration itself"
    assert _entry(site)["dcgmHostEngine"] == "host", (
        "and records it, as every release did"
    )


@pytest.mark.parametrize("read_by", ["deploy", "status"])
@pytest.mark.parametrize("case", ["mixed", "unreadable"])
def test_a_mixed_or_unreadable_declaration_refuses_before_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str, read_by: str
) -> None:
    world = (
        NodeGroupWorld({"gpu-a": "host", "gpu-b": None})
        if case == "mixed"
        else unreadable_world()
    )
    report = None
    if read_by == "status":
        if case == "unreadable":
            # A status that could not read it is not reused: the deploy reads.
            report = _reading(
                _declaration("unreadable", declared=None, declared_source=None)
                | {"error": "throttled", "node_groups": []}
            )
        else:
            (declaration,) = dcgm_declaration.read_dcgm_declarations(
                [eks_entry("embedded")], NodeGroupWorld(world.groups)
            )
            report = _reading(declaration.as_dict())

    events: list[str] = []
    with pytest.raises(staging_deploy.StagingDeployError) as refusal:
        _deploy(tmp_path, monkeypatch, world=world, report=report, events=events)

    message = str(refusal.value)
    assert message.startswith(
        "the DCGM host engine declaration stops this deploy before any change: "
        f"dcgm_hostengine: {EKS_NAME}"
    ), message
    if case == "mixed":
        assert "disagree on gpu-fault.io/dcgm-hostengine" in message
    else:
        assert "declaration unreadable: command failed (254)" in message
    assert _entry(tmp_path / "state/site.yaml") == eks_entry("embedded"), (
        "nothing was recorded"
    )
    assert events == ["lock-enter", EVIDENCE_EVENT, "lock-exit"], (
        "refused right after the pre-deploy reading: no deploy host, gate or apply"
    )
    if read_by == "status" and case == "mixed":
        assert world.calls == [], "the refusal the quick status read is reused"


@pytest.mark.parametrize("old_mode", ["UNCHANGED", "DEPLOY_HOST_ONLY", "QUALITY_ONLY"])
@pytest.mark.parametrize(
    "pending", [DCGM_HOSTENGINE_DRIFT_INPUT, DCGM_HOSTENGINES_INPUT]
)
def test_a_dcgm_change_the_status_reports_pending_takes_the_release_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, old_mode: str, pending: str
) -> None:
    """After a refused flip the record can say host while the exporter runs
    embedded (``dcgm_hostengine_drift``), or say a mode no release rolled
    (``dcgm_hostengines``): the label agrees with the record, so only the
    status' next deploy shows it, and no shortcut may record it as success."""

    host = {
        "recorded": "host",
        "recorded_source": "node-group-label",
        "declared": "host",
        "declared_source": "node-group-label",
    }
    report = _reading(_declaration("converged", **host))
    report["next_deploy"] = {"kind": "DATA_PLANE_COMPATIBLE", "changed": [pending]}

    result, events, _site_file = _deploy(
        tmp_path,
        monkeypatch,
        world=NodeGroupWorld({"gpu-a": "host"}),
        recorded="host",
        old_mode=old_mode,
        report=report,
    )

    assert result["deploy_mode"] == "APPLICATION_RELEASE", old_mode
    assert DEPLOY_EVENT in events and events.index(DEPLOY_EVENT) < events.index(
        "success"
    )
    assert {DCGM_HOSTENGINES_INPUT, DCGM_HOSTENGINE_DRIFT_INPUT} <= (
        staging_deploy.RELEASE_ONLY_CHANGES
    ), "the shortcut's list names the engine's own inputs"
