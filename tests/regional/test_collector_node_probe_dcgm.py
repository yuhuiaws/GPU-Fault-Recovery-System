"""The collector node probe reports whether the DCGM load generator exists."""

from __future__ import annotations

import pytest

from scripts.e2e.regional.probes import collector_node_probe as probe


def test_the_snapshot_field_is_the_binary_or_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(probe, "proftester_binary", lambda: "/usr/bin/dcgmproftester13")
    assert probe.dcgm_load_generator() == "/usr/bin/dcgmproftester13"

    def missing() -> str:
        raise probe.ProbeError("no dcgmproftester binary is installed on this node")

    monkeypatch.setattr(probe, "proftester_binary", missing)
    assert probe.dcgm_load_generator() is None, (
        "a node without DCGM reports None instead of failing the snapshot"
    )
