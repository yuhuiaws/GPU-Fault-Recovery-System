"""GF-REGIONAL-BOOT-038 configuration: the site lookup refuses every input it
cannot bind before anything live is read, and the script entry runs ``main``
under the abort wrapper."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.e2e.regional import boot038_membership as membership
from scripts.e2e.regional import live_driver_guard, regional_live_fixture
from scripts.e2e.regional import run_boot038_cross_cluster_membership as runner
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError
from scripts.e2e.regional.site_profile import SITE_PROFILE_ENV
from tests.regional._cov95_boot038_harness import (
    CPU_ARN,
    OTHER_ID,
    OTHER_NODE,
    TARGET_ARN,
    TARGET_ID,
    settings,
    write_site,
)


@pytest.fixture
def profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An installed site profile naming the host-probe image."""

    path = tmp_path / "profile.yaml"
    path.write_text(
        yaml.safe_dump({"arguments": {"host-probe-image": "example.test/probe@x"}}),
        encoding="utf-8",
    )
    path.chmod(0o600)
    monkeypatch.setenv(SITE_PROFILE_ENV, str(path))
    return path


def arguments(tmp_path: Path, *extra: str, **flags: str) -> argparse.Namespace:
    values = {
        "state-dir": str(tmp_path / "state"),
        "target-cluster-arn": TARGET_ARN,
        "other-cluster-id": OTHER_ID,
        "other-node": OTHER_NODE,
        **flags,
    }
    argv = ["--run-dir", str(tmp_path / "run")]
    for name, value in values.items():
        argv += [f"--{name}", value]
    return runner.parser().parse_args([*argv, *extra])


def test_a_json_site_document_binds_the_target_and_the_plan_b_window(
    tmp_path: Path, profile: Path
) -> None:
    write_site(tmp_path / "state")
    path = tmp_path / "state" / "site.yaml"
    path.write_text(json.dumps(yaml.safe_load(path.read_text())), encoding="utf-8")

    value = runner.configure(
        arguments(tmp_path, "--dry-run", **{"window": "none", "other-node": ""})
    )

    assert (value.target_cluster_id, value.other_node) == (TARGET_ID, ""), value
    assert value.window == membership.WINDOW_NONE and value.dry_run is True, value
    assert value.cpu_cluster_arn == CPU_ARN, value
    assert value.host_probe_image == "example.test/probe@x", value
    assert value.predecessor_id == "GF-REGIONAL-BOOT-029", value
    assert len(value.run_token) == 8, value.run_token
    environment = value.environment()
    assert environment["GPU_FAULT_STATE_DIR"] == str(tmp_path / "state"), environment
    assert environment["GPU_EKS_CONTEXT"] == "ctx-other", environment


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("kind: [unclosed\n", "cannot read"),
        ("- not\n- a site\n", "is not a RegionalSite"),
        ("kind: RegionalProfile\nspec: {}\n", "is not a RegionalSite"),
    ],
    ids=["invalid-yaml", "a-list", "another-kind"],
)
def test_a_site_document_that_is_not_a_regional_site_is_refused(
    content: str, message: str, tmp_path: Path, profile: Path
) -> None:
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "site.yaml").write_text(content, encoding="utf-8")

    with pytest.raises(RegionalFixtureError, match=message):
        runner.configure(arguments(tmp_path))


def test_an_unreadable_site_document_is_refused(tmp_path: Path, profile: Path) -> None:
    (tmp_path / "state" / "site.yaml").mkdir(parents=True)

    with pytest.raises(RegionalFixtureError, match="cannot read .*site.yaml"):
        runner.configure(arguments(tmp_path))


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        ({"state-dir": " "}, "--state-dir .* is required"),
        ({"other-cluster-id": ""}, "--other-cluster-id is required"),
        ({"other-node": ""}, "--other-node is required for --window kubelet"),
        ({"target-cluster-arn": f"{TARGET_ARN}-absent"}, "no member cluster with EKS"),
        ({"other-cluster-id": "absent-cluster"}, "names no member cluster absent"),
    ],
    ids=["state-dir", "other-cluster", "other-node", "target-arn", "other-id"],
)
def test_an_input_the_site_cannot_bind_is_refused(
    flags: dict[str, str], message: str, tmp_path: Path, profile: Path
) -> None:
    write_site(tmp_path / "state")

    with pytest.raises(RegionalFixtureError, match=message):
        runner.configure(arguments(tmp_path, **flags))


def test_a_window_outside_the_two_plans_is_refused(
    tmp_path: Path, profile: Path
) -> None:
    write_site(tmp_path / "state")
    value = arguments(tmp_path)
    value.window = "kubelet-and-agent"

    with pytest.raises(RegionalFixtureError, match="--window must be one of"):
        runner.configure(value)


@pytest.mark.parametrize(
    "change",
    [{"awsRegion": ""}, {"clusters": "other-context"}],
    ids=["region", "other-context"],
)
def test_a_site_without_its_region_or_the_other_context_is_refused(
    change: dict[str, Any], tmp_path: Path, profile: Path
) -> None:
    if change.get("clusters"):
        write_site(tmp_path / "state")
        path = tmp_path / "state" / "site.yaml"
        document = yaml.safe_load(path.read_text())
        document["spec"]["clusters"][0].pop("context")
        path.write_text(yaml.safe_dump(document), encoding="utf-8")
    else:
        write_site(tmp_path / "state", **change)

    with pytest.raises(RegionalFixtureError, match="missing the GPU/CPU kubeconfig"):
        runner.configure(arguments(tmp_path))


def test_a_profile_without_a_host_probe_image_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(SITE_PROFILE_ENV, raising=False)
    write_site(tmp_path / "state")

    with pytest.raises(RegionalFixtureError, match="names no host-probe-image"):
        runner.configure(arguments(tmp_path))


def test_the_plan_b_mutation_names_the_external_departure(tmp_path: Path) -> None:
    value = settings(tmp_path, window=membership.WINDOW_NONE)
    preflight: dict[str, Any] = {
        "identity": {"release_id": "release-test", "cluster_id": OTHER_ID},
        "product_generation": "b250",
    }

    details = runner.plan_details(value, preflight)

    assert "No window is opened" in details["mutation"], details["mutation"]
    assert details["window"] == membership.WINDOW_NONE, details
    assert details["preflight_identity"] == {
        "release_id": "release-test",
        "cluster_id": OTHER_ID,
        "product_generation": "b250",
    }, details["preflight_identity"]
    assert preflight["preflight_identity"] == details["preflight_identity"], (
        "the plan pins the identity the execute re-checks"
    )


def test_the_script_entry_runs_main_under_the_abort_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        sys, "path", [path for path in sys.path if path != str(runner.ROOT)]
    )

    def standard(case: Any) -> int:
        calls.append(case.case_id)
        return 0

    def wrapper(entry: Any) -> int:
        calls.append("wrapper")
        return 5 + entry()

    monkeypatch.setattr(live_driver_guard, "run_standard_case", standard)
    monkeypatch.setattr(regional_live_fixture, "run_case_main", wrapper)

    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(runner.__file__, run_name="__main__")

    assert stopped.value.code == 5, stopped.value
    assert calls == ["wrapper", runner.CASE_ID], calls
    assert sys.path[0] == str(runner.ROOT), "the script puts its repository first"
