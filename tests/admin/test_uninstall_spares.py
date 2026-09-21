"""An uninstall releases the warm spares the site declared before node cleanup."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin import uninstall_spares as module
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.site import load_site
from gpu_fault.admin.uninstall_types import UninstallRequest
from gpu_fault.admin.warm_spare import (
    SPARE_LABEL,
    SPARE_RESERVATION_ANNOTATION,
    AgentState,
    record_path,
)
from tests.admin.test_admin_site import site_file
from tests.admin.test_admin_warm_spare import NODE, FakeCoreApi, _raw_node, _record

CLUSTER = "gpu-a"  # the cluster id the site fixture manages


def _declared(
    tmp_path: Path, **node_overrides: Any
) -> tuple[UninstallRequest, Path, FakeCoreApi]:
    site = load_site(site_file(tmp_path))
    api = FakeCoreApi(
        _raw_node(
            NODE, labels={SPARE_LABEL: "true"}, unschedulable=True, **node_overrides
        )
    )
    record = record_path(tmp_path, NODE)
    record.parent.mkdir(parents=True)
    survey = {"node": {"uid": f"uid-{NODE}"}}
    record.write_text(
        json.dumps(_record(cluster_id=CLUSTER, pre_declaration_survey=survey)),
        encoding="utf-8",
    )
    request = UninstallRequest(
        site=site,
        cpu_disposition="keep",
        confirmation="UNINSTALL_GPU_FAULT",
        reset_database=True,
    )
    return request, record, api


def _agent(_cluster: str, _node: str) -> AgentState:
    return AgentState(lifecycle_state=None, error="control plane already stopped")


def test_declared_spares_return_to_their_baseline_and_are_journaled(
    tmp_path: Path,
) -> None:
    request, record, api = _declared(tmp_path)
    state_path = tmp_path / "uninstall" / "state.json"
    state_path.parent.mkdir()
    state: dict[str, Any] = {"phase": "REGISTRY_EXPORTED"}

    released = module.release_declared_spares(
        request,
        state_path,
        state,
        reference="uninstall/UNINSTALL_GPU_FAULT",
        api_factory=lambda _site, _target: api,
        agent_lookup=_agent,
        actor="uninstall-test",
    )

    assert released == [NODE], "the declared spare was released"
    node = api.nodes[NODE]
    assert SPARE_LABEL not in node["metadata"]["labels"], "the spare label is gone"
    assert node["spec"]["unschedulable"] is False, "the recorded baseline is back"
    assert json.loads(record.read_text(encoding="utf-8"))["released_at"], (
        "the declaration record is closed"
    )
    journal = json.loads(state_path.read_text(encoding="utf-8"))["released_warm_spares"]
    assert [entry["node"] for entry in journal] == [NODE], "journaled in state.json"
    assert journal[0]["cluster_id"] == CLUSTER, "with its cluster"

    again = module.release_declared_spares(
        request,
        state_path,
        state,
        reference="uninstall/UNINSTALL_GPU_FAULT",
        api_factory=lambda _site, _target: pytest.fail(
            "a released record is not re-released"
        ),
        agent_lookup=_agent,
        actor="uninstall-test",
    )
    assert again == [], "nothing left to release"
    assert (
        len(json.loads(state_path.read_text(encoding="utf-8"))["released_warm_spares"])
        == 1
    )


def test_a_reserved_spare_fails_the_uninstall_closed(tmp_path: Path) -> None:
    request, record, api = _declared(
        tmp_path, annotations={SPARE_RESERVATION_ANNOTATION: "INC-7"}
    )
    state_path = tmp_path / "uninstall" / "state.json"
    state_path.parent.mkdir()

    with pytest.raises(BootstrapError, match="reserved by an incident"):
        module.release_declared_spares(
            request,
            state_path,
            {"phase": "REGISTRY_EXPORTED"},
            reference="uninstall/UNINSTALL_GPU_FAULT",
            api_factory=lambda _site, _target: api,
            agent_lookup=_agent,
            actor="uninstall-test",
        )
    assert api.patches == [], "a refused release touches nothing"
    assert not json.loads(record.read_text(encoding="utf-8")).get("released_at"), (
        "a refused release leaves the record open"
    )


def test_a_site_without_declarations_releases_nothing(tmp_path: Path) -> None:
    site = load_site(site_file(tmp_path))
    request = UninstallRequest(
        site=site, cpu_disposition="keep", confirmation="UNINSTALL_GPU_FAULT"
    )
    assert module.unreleased_spare_records(tmp_path) == [], "no records, no work"
    assert (
        module.release_declared_spares(
            request,
            tmp_path / "state.json",
            {"phase": "STARTED"},
            reference="uninstall/UNINSTALL_GPU_FAULT",
            api_factory=lambda _site, _target: pytest.fail("no api needed"),
            agent_lookup=_agent,
        )
        == []
    )
    assert not (tmp_path / "state.json").exists(), "nothing was journaled"
