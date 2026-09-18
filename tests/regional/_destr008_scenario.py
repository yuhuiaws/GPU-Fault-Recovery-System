"""A valid cancellation identity for isolated non-mutating fixture tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from scripts.e2e.regional.run_destr008_warm_spare_shortage import ScenarioFixture
from tests.regional.test_destr008_cancellation_probe import plan


def bind_fixture(fixture: ScenarioFixture) -> None:
    source = plan()
    bound = plan(
        run_id=fixture.run_id,
        cluster_id=fixture.settings.regional.cluster_id,
        fault_node=fixture.settings.fault_node,
        spare_node=fixture.settings.spare_node,
        fence={**source.fence.model_dump(), "node": fixture.settings.spare_node},
    )
    fixture.bind_safety(bound, datetime.now(timezone.utc) + timedelta(hours=1))
