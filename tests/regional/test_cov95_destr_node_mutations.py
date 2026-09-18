from __future__ import annotations

from typing import Any, cast

import pytest

from scripts.e2e.regional import warm_spare_fixture as warm
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError
from tests.regional.test_warm_spare_fixture_safety import NodeApi, fixture, reservation


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("uid", "", "UID or resourceVersion is unknown"),
        ("resourceVersion", "", "UID or resourceVersion is unknown"),
        ("name", "foreign", "names another target"),
    ],
)
def test_node_mutation_cannot_capture_an_unbound_baseline(
    field: str, value: str, expected: str
) -> None:
    api = NodeApi()
    api.node["metadata"][field] = value
    with pytest.raises(RegionalFixtureError, match=expected):
        fixture(api)
    assert api.patches == [], api.patches


@pytest.mark.parametrize("field", ["label", "cordon"])
def test_tracked_node_field_drift_refuses_before_transport(field: str) -> None:
    api = NodeApi()
    mutation = warm.NodeMutationFixture(
        warm.WarmSpareLiveFixture(cast(Any, api), "unit"),
        "spare-a",
        label_keys=(warm.SPARE_LABEL,),
        track_unschedulable=True,
    )
    if field == "label":
        api.node["metadata"]["labels"][warm.SPARE_LABEL] = "foreign"
        patch = warm.NodePatch(labels={warm.SPARE_LABEL: None}, annotations={})
        expected = "tracked node field changed"
    else:
        api.node["spec"]["unschedulable"] = False
        patch = warm.NodePatch(labels={}, annotations={}, unschedulable=False)
        expected = "cordon state changed"
    with pytest.raises(RegionalFixtureError, match=expected):
        mutation.apply(patch)
    assert api.patches == [], api.patches


@pytest.mark.parametrize("field", ["taint", "cordon"])
def test_untracked_scheduling_changes_do_not_authorize_reservation_write(
    field: str,
) -> None:
    api = NodeApi()
    mutation = fixture(api)
    if field == "taint":
        api.node["spec"]["taints"] = [{"key": "foreign", "effect": "NoSchedule"}]
        expected = "node taints changed"
    else:
        api.node["spec"]["unschedulable"] = False
        expected = "node scheduling changed"
    with pytest.raises(RegionalFixtureError, match=expected):
        mutation.apply(reservation("owned"))
    assert api.patches == [], api.patches
    assert (
        warm.SPARE_RESERVATION_ANNOTATION not in api.node["metadata"]["annotations"]
    ), api.node


def test_cordon_write_must_be_observed_after_the_revision_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = NodeApi()
    mutation = fixture(api, track_unschedulable=True)
    original = api.kubectl

    def transport(plane: str, *args: str, **kwargs: Any) -> str:
        value = original(plane, *args, **kwargs)
        if args[0] == "patch":
            api.node["spec"]["unschedulable"] = True
        return value

    monkeypatch.setattr(api, "kubectl", transport)
    with pytest.raises(RegionalFixtureError, match="cordon patch was not observed"):
        mutation.apply(warm.NodePatch(labels={}, annotations={}, unschedulable=False))
    assert len(api.patches) == 1, api.patches
    assert api.patches[0]["metadata"] == {
        "uid": "node-uid-a",
        "resourceVersion": "1",
    }, api.patches


@pytest.mark.parametrize("already_restored", [False, True])
def test_cordon_restore_writes_only_when_the_owned_change_still_exists(
    already_restored: bool,
) -> None:
    api = NodeApi()
    mutation = fixture(api, track_unschedulable=True)
    mutation.apply(warm.NodePatch(labels={}, annotations={}, unschedulable=False))
    if already_restored:
        api.node["spec"]["unschedulable"] = True
    restored = mutation.restore()
    assert restored["unschedulable"] is True, restored
    assert len(api.patches) == (1 if already_restored else 2), api.patches
