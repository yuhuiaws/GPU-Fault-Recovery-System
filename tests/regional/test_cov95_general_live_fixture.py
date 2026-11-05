"""GF-GENERAL-LIVE shared fixture: run ids, preflight lanes, probe handles, backing dirs."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_applogs as applogs
from scripts.e2e.regional import general_live_fixture as fx
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tests.regional._general_live_fakes import FakeRegional, make_settings, store_script

NODE = "ip-10-0-43-5.us-west-2.compute.internal"
#: A managed-job variant on both lanes whose own preflight needs --site-file.
SPEC = next(item for item in applogs.VARIANTS if item.variant == "live_pid")
STORE = {"agent": {"lifecycle_state": "ACTIVE"}, "collector_statuses": []}


def test_a_variant_name_that_yields_no_safe_run_id_is_refused(tmp_path: Path) -> None:
    assert fx.run_identity("GF-GENERAL-LIVE-017", "live_pid", 1, tmp_path).startswith(
        "gl017-live-pid-a1-"
    ), "a reviewed variant name derives its run id"

    with pytest.raises(RegionalFixtureError, match="cannot derive a safe run id"):
        fx.run_identity("GF-GENERAL-LIVE-017", "Live PID!", 1, tmp_path)


def preflight(
    tmp_path: Path, *, labels: dict[str, str], run_output: str = "null", **settings: Any
) -> dict[str, Any]:
    site = tmp_path / "site.yaml"
    site.write_text("site: test\n", encoding="utf-8")
    values = dataclasses.replace(
        make_settings(SPEC, node=NODE), site_file=site, **settings
    )
    regional = FakeRegional(
        node_metadata={"name": NODE, "labels": labels},
        cpu_python=store_script({"preflight": STORE}),
        run_output=lambda command: run_output,
    )
    return fx.read_only_preflight(values, SPEC, regional)


def test_a_node_outside_the_variants_lanes_is_named(tmp_path: Path) -> None:
    result = preflight(tmp_path, labels={})

    assert f"{NODE} is a unknown node; {SPEC.key} runs on eks/hp" in result["errors"]
    assert "a managed test job needs --site-file" not in result["errors"]


def test_hyperpod_needs_its_cluster_name_and_node_recovery_off(tmp_path: Path) -> None:
    hp = {fx.HYPERPOD_LABEL: "hyperpod"}

    unnamed = preflight(tmp_path, labels=hp, hyperpod_cluster="")
    automatic = preflight(tmp_path, labels=hp, run_output='"Automatic"')
    off = preflight(tmp_path, labels=hp, run_output='"None"')

    assert "--hyperpod-cluster is required on HyperPod" in unnamed["errors"]
    assert unnamed["node_recovery"] is None
    assert (
        "HyperPod NodeRecovery is Automatic; it would race the product"
        in automatic["errors"]
    )
    assert off["node_recovery"] == "None"
    assert not [error for error in off["errors"] if "NodeRecovery" in error], off
    assert off["provider"] == "hp"


def test_a_pair_node_equal_to_the_target_is_refused(tmp_path: Path) -> None:
    result = preflight(
        tmp_path, labels={fx.EKS_NODEGROUP_LABEL: "gpu-ng"}, peer_node=NODE
    )

    assert "the pair node equals the target node" in result["errors"]


def test_discarding_an_undo_that_is_not_on_the_stack_changes_nothing() -> None:
    stack = fx.CleanupStack()
    stack.push("export-a", lambda: None)

    stack.discard("never-pushed")

    assert stack.pending() == ["export-a"]


class Fixture:
    def __init__(self, residuals: dict[str, bool], *, fail: bool = False) -> None:
        self.answer = residuals
        self.fail = fail
        self.reads = 0

    def residuals(self) -> dict[str, bool]:
        self.reads += 1
        return dict(self.answer)

    def cleanup(self) -> dict[str, bool]:
        self.reads += 1
        if self.fail:
            raise RuntimeError("API server unavailable")
        return dict(self.answer)


def handle(tmp_path: Path, fixture: Fixture) -> fx.ProbeHandle:
    script = tmp_path / "probe.py"
    script.write_text("print('{}')\n", encoding="utf-8")
    value = fx.ProbeHandle(
        FakeRegional(),
        script=script,
        node=NODE,
        image="public.ecr.aws/x@sha256:" + "0" * 64,
        key="GF-GENERAL-LIVE-017",
        run_id="gl017-live-pid-a1-deadbeef",
        case_dir=tmp_path,
        deadline_seconds=600,
    )
    value.fixture = fixture  # type: ignore[assignment]
    return value


def test_a_handle_with_residue_is_not_proven_clean(tmp_path: Path) -> None:
    fixture = Fixture({"pod": True, "configmap": False})
    probe = handle(tmp_path, fixture)

    assert probe.cleanup() == {"pod": True, "configmap": False}
    assert probe.cleanup() == {"pod": True, "configmap": False}
    assert fixture.reads == 2, "residue is audited again, never answered from a proof"


def test_a_replaced_handle_whose_cleanup_fails_is_kept_for_the_teardown(
    tmp_path: Path,
) -> None:
    registry = fx.ProbeRegistry()
    first = handle(tmp_path, Fixture({"pod": True}, fail=True))
    first.created = True
    second = handle(tmp_path, Fixture({}))
    registry["node@a"] = first

    registry["node@a"] = second

    assert registry["node@a"] is second
    assert registry.replaced == {"node@a#1": first}


def backing(root: str, available: Any, used: Any, fstype: str = "xfs") -> dict:
    return {
        "root": root,
        "statvfs": {"available_bytes": available, "used_percent": used},
        "mount": {"fstype": fstype},
    }


def test_backing_root_picks_the_roomiest_eligible_directory() -> None:
    gib = 1024**3
    survey = {
        "backing": [
            backing("/unreadable", None, 10.0),
            backing("/full", 500 * gib, 71.0),
            backing("/small", 4 * gib, 10.0),
            backing("/big", 100 * gib, 20.0),
            backing("/smaller", 50 * gib, 20.0),
            backing("/ram", 200 * gib, 1.0, fstype="tmpfs"),
        ]
    }

    assert fx.backing_root(survey) == "/big"
