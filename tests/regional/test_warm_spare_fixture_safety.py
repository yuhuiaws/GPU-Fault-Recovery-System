from __future__ import annotations

import json
import sys
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, cast

import pytest

from scripts.e2e.regional import run_destr003_warm_spare_failover as failover
from scripts.e2e.regional import warm_spare_fixture as warm
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError


class NodeApi:
    def __init__(self) -> None:
        self.node: dict[str, Any] = {
            "metadata": {
                "name": "spare-a",
                "uid": "node-uid-a",
                "resourceVersion": "1",
                "labels": {warm.SPARE_LABEL: "true"},
                "annotations": {"gpu-fault.io/installer-id": "keep"},
            },
            "spec": {"unschedulable": True},
            "status": {"conditions": [{"type": "Ready", "status": "True"}]},
        }
        self.patches: list[dict[str, Any]] = []
        self.lose_ack = False
        self.conflict = False

    def kubectl(self, _plane: str, *args: str, **_kwargs: Any) -> str:
        if args[:2] == ("get", "node"):
            return json.dumps(self.node)
        assert args[:3] == ("patch", "node", "spare-a"), args
        body = json.loads(args[args.index("-p") + 1])
        self.patches.append(deepcopy(body))
        meta = self.node["metadata"]
        if self.conflict:
            meta["resourceVersion"] = str(int(meta["resourceVersion"]) + 1)
            self.conflict = False
        requested = body["metadata"]
        for key in ("uid", "resourceVersion"):
            if key in requested and requested[key] != meta[key]:
                raise RegionalFixtureError("409: node precondition conflict")
        for section in ("labels", "annotations"):
            for key, value in requested.get(section, {}).items():
                if value is None:
                    meta[section].pop(key, None)
                else:
                    meta[section][key] = value
        self.node["spec"].update(body.get("spec", {}))
        meta["resourceVersion"] = str(int(meta["resourceVersion"]) + 1)
        if self.lose_ack:
            self.lose_ack = False
            raise TimeoutError("applied but response lost")
        return json.dumps(self.node)


def fixture(api: NodeApi, **kwargs: Any) -> warm.NodeMutationFixture:
    return warm.NodeMutationFixture(
        warm.WarmSpareLiveFixture(cast(Any, api), "cluster-a"),
        "spare-a",
        annotation_keys=(warm.SPARE_RESERVATION_ANNOTATION,),
        **kwargs,
    )


def reservation(value: str | None) -> warm.NodePatch:
    return warm.NodePatch(
        labels={}, annotations={warm.SPARE_RESERVATION_ANNOTATION: value}
    )


def test_node_patch_is_bound_to_uid_and_current_resource_version() -> None:
    api = NodeApi()
    mutation = fixture(api)

    mutation.apply(reservation("acceptance-owned"))

    assert api.patches[0]["metadata"]["uid"] == "node-uid-a"
    assert api.patches[0]["metadata"]["resourceVersion"] == "1"


def test_restore_recovers_an_applied_patch_whose_ack_was_lost() -> None:
    api = NodeApi()
    mutation = fixture(api)
    api.lose_ack = True
    with pytest.raises(TimeoutError, match="response lost"):
        mutation.apply(reservation("acceptance-owned"))

    restored = mutation.restore()

    assert restored["annotations"][warm.SPARE_RESERVATION_ANNOTATION] is None
    assert api.node["metadata"]["annotations"]["gpu-fault.io/installer-id"] == "keep"
    assert api.patches[-1]["metadata"]["resourceVersion"] == "2"


@pytest.mark.parametrize("before_apply", [True, False])
def test_node_recreation_never_authorizes_mutation_or_restore(
    before_apply: bool,
) -> None:
    api = NodeApi()
    mutation = fixture(api)
    if not before_apply:
        mutation.apply(reservation("acceptance-owned"))
    api.node["metadata"]["uid"] = "replacement-uid"
    count = len(api.patches)

    with pytest.raises(RegionalFixtureError, match="UID"):
        if before_apply:
            mutation.apply(reservation("acceptance-owned"))
        else:
            mutation.restore()
    assert len(api.patches) == count


def test_restore_preserves_a_foreign_reservation_instead_of_overwriting_it() -> None:
    api = NodeApi()
    mutation = fixture(api)
    mutation.apply(reservation("acceptance-owned"))
    api.node["metadata"]["annotations"][warm.SPARE_RESERVATION_ANNOTATION] = "foreign"
    count = len(api.patches)

    with pytest.raises(RegionalFixtureError, match="changed"):
        mutation.restore()
    assert len(api.patches) == count
    assert (
        api.node["metadata"]["annotations"][warm.SPARE_RESERVATION_ANNOTATION]
        == "foreign"
    )


def test_spare_label_restore_cannot_release_a_new_foreign_reservation() -> None:
    api = NodeApi()
    mutation = warm.NodeMutationFixture(
        warm.WarmSpareLiveFixture(cast(Any, api), ""),
        "spare-a",
        label_keys=(warm.SPARE_LABEL,),
    )
    mutation.apply(warm.NodePatch(labels={warm.SPARE_LABEL: None}, annotations={}))
    api.node["metadata"]["annotations"][warm.SPARE_RESERVATION_ANNOTATION] = "foreign"
    count = len(api.patches)

    with pytest.raises(RegionalFixtureError, match="ownership or reservation"):
        mutation.restore()
    assert len(api.patches) == count
    assert warm.SPARE_LABEL not in api.node["metadata"]["labels"]


def test_conflicting_patch_does_not_retry_or_overwrite_a_new_revision() -> None:
    api = NodeApi()
    mutation = fixture(api)
    api.conflict = True

    with pytest.raises(RegionalFixtureError, match="409"):
        mutation.apply(reservation("acceptance-owned"))

    assert len(api.patches) == 1
    assert warm.SPARE_RESERVATION_ANNOTATION not in api.node["metadata"]["annotations"]


def test_restore_without_an_attempted_write_is_a_read_only_noop() -> None:
    api = NodeApi()
    mutation = fixture(api)

    mutation.restore()

    assert api.patches == []


def test_untracked_metadata_cannot_be_mutated() -> None:
    api = NodeApi()
    mutation = fixture(api)

    with pytest.raises(RegionalFixtureError, match="tracked"):
        mutation.apply(warm.NodePatch(labels={warm.SPARE_LABEL: None}, annotations={}))
    assert api.patches == []


def test_tracked_plugin_annotations_are_observed_and_restored() -> None:
    api = NodeApi()
    key = "gpu-fault.io/efa-plugin-restart-incident"
    mutation = warm.NodeMutationFixture(
        warm.WarmSpareLiveFixture(cast(Any, api), ""), "spare-a", annotation_keys=(key,)
    )
    mutation.apply(warm.NodePatch(labels={}, annotations={key: "acceptance-owned"}))

    mutation.restore()

    assert key not in api.node["metadata"]["annotations"]


def test_store_probe_never_serializes_the_command_lease_credential(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from gpu_fault.app import ApplicationContext

    class Record:
        request_id = "workflow-a"
        workflow_request_id = "workflow-a"
        incident_id = "incident-a"

        def model_dump(self, **kwargs: Any) -> dict[str, Any]:
            values = {"command_id": "command-a", "lease_token": "example-local-lease"}
            return {
                key: value
                for key, value in values.items()
                if key not in kwargs.get("exclude", set())
            }

    record = Record()
    store = SimpleNamespace(
        get_incident_by_event=lambda _event: record,
        get_workflow=lambda _request: record,
        list_remote_commands=lambda: [record],
        list_notifications=lambda: [],
        list_markers=lambda: [],
        list_attempt_observations=lambda _cluster: [],
        list_agents=lambda _cluster: [],
    )
    monkeypatch.setattr(
        ApplicationContext, "from_environment", lambda: SimpleNamespace(store=store)
    )
    monkeypatch.setattr(sys, "argv", ["probe", "cluster-a", "event-a", "", ""])

    exec(warm.STORE_PROBE, {"__name__": "isolated_test_probe"})

    output = capsys.readouterr().out
    report = json.loads(output)
    assert "lease_token" not in report["commands"][0], report["commands"][0].keys()
    assert "example-local-lease" not in json.dumps(report["commands"])


def test_ack_loss_during_restore_can_be_rechecked_without_another_write() -> None:
    api = NodeApi()
    mutation = fixture(api)
    mutation.apply(reservation("acceptance-owned"))
    api.lose_ack = True
    with pytest.raises(TimeoutError, match="response lost"):
        mutation.restore()
    count = len(api.patches)

    mutation.restore()

    assert len(api.patches) == count


def test_a_proven_reclaimed_spare_can_return_to_its_declared_baseline() -> None:
    api = NodeApi()
    mutation = warm.NodeMutationFixture(
        warm.WarmSpareLiveFixture(cast(Any, api), ""),
        "spare-a",
        annotation_keys=(
            warm.SPARE_RESERVATION_ANNOTATION,
            warm.SPARE_RESERVED_AT_ANNOTATION,
            warm.SPARE_POOL_STATE_ANNOTATION,
        ),
        allow_reclaimed_reservation=True,
    )
    mutation.apply(
        warm.NodePatch(
            labels={},
            annotations={
                warm.SPARE_RESERVATION_ANNOTATION: "acceptance-owned",
                warm.SPARE_RESERVED_AT_ANNOTATION: "2026-09-01T00:00:00Z",
                warm.SPARE_POOL_STATE_ANNOTATION: "ALLOCATED",
            },
        )
    )
    values = api.node["metadata"]["annotations"]
    values.pop(warm.SPARE_RESERVATION_ANNOTATION)
    values.pop(warm.SPARE_RESERVED_AT_ANNOTATION)
    values[warm.SPARE_POOL_STATE_ANNOTATION] = "AVAILABLE"

    result = mutation.restore()

    assert result["annotations"][warm.SPARE_POOL_STATE_ANNOTATION] is None
    assert result["unschedulable"] is True


def test_cleanup_waits_for_commands_even_after_the_workflow_is_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]
    calls: list[tuple[str, ...]] = []
    incident = {
        "incident_id": "incident-a",
        "cluster_id": "cluster-a",
        "workflow_request_id": "workflow-a",
    }
    states = [
        {
            "incident": incident,
            "workflow_status": "FAILED",
            "active_workflows": [],
            "open_commands": [{"command_id": "command-a", "status": "LEASED"}],
        },
        {
            "incident": incident,
            "workflow_status": "FAILED",
            "active_workflows": [],
            "open_commands": [],
        },
    ]

    def read(_script: str, *args: str) -> dict[str, Any]:
        calls.append(args)
        return states.pop(0)

    monkeypatch.setattr(
        warm,
        "time",
        SimpleNamespace(
            monotonic=lambda: clock[0],
            sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
        ),
    )
    regional = SimpleNamespace(
        settings=SimpleNamespace(cluster_id="cluster-a"), cpu_python=read
    )

    result = warm.WarmSpareLiveFixture(cast(Any, regional), "").wait_incident_idle(
        "incident-a", timeout_seconds=30, quiet_seconds=0
    )

    assert result == incident
    assert calls == [("cluster-a", "incident-a")] * 2


def test_missing_cleanup_observation_is_not_idle() -> None:
    regional = SimpleNamespace(
        settings=SimpleNamespace(cluster_id="cluster-a"), cpu_python=lambda *_args: {}
    )
    with pytest.raises(RegionalFixtureError, match="incomplete"):
        warm.WarmSpareLiveFixture(cast(Any, regional), "").wait_incident_idle(
            "incident-a", timeout_seconds=30, quiet_seconds=0
        )


@pytest.mark.parametrize("incident_id", ["incident-a", ""])
def test_failover_cleanup_does_not_release_resources_after_unproved_quiescence(
    incident_id: str,
) -> None:
    calls: list[str] = []

    def busy(_incident: str) -> None:
        raise RegionalFixtureError("command is still leased")

    def delete() -> None:
        calls.append("delete")

    def release(*_args: Any) -> None:
        calls.append("release")

    def prewarm() -> dict[str, bool]:
        calls.append("prewarm")
        return {}

    result = failover.cleanup_case(
        warm=cast(
            Any, SimpleNamespace(wait_incident_idle=busy, release_spares=release)
        ),
        regional=cast(Any, object()),
        workload=cast(Any, SimpleNamespace(delete=delete)),
        prewarm=cast(Any, SimpleNamespace(cleanup=prewarm)),
        settings=cast(Any, object()),
        incident_id=incident_id,
        profile_version="profile-a",
        trigger_attempted=True,
    )

    assert result["workload_and_spare_cleanup_deferred"] is True
    assert result["errors"]
    assert calls == ["prewarm"]
