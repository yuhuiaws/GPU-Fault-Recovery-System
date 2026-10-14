"""The BOOT isolated-site cleanup and the uninstall's registry sync gate.

BOOT-016 (after a failed bootstrap) and BOOT-018 remove their isolated site with
``gpu-fault-admin uninstall``, which refuses a new attempt while the site's
last installation registry sync is failed, missing -- a first deploy that
stopped before its release step recorded none -- or unreadable. The cleanup
judges that record first (``registry_sync_preflight``): it records the refusal
with the product's remedy and the exact uninstall to run afterwards instead of
running a command that can only exit non-zero, never passes
``--accept-unsynchronized-registry``, and leaves an uninstall that already
started to the verb.
"""

from __future__ import annotations

import shlex
import subprocess
from argparse import Namespace
from pathlib import Path

import pytest

from gpu_fault.admin.resource_registry_outcome import ACCEPT_FLAG, record_path
from scripts.e2e.regional import boot_acceptance_lifecycle as lifecycle
from tests.admin._registry_sync_support import record_registry_sync
from tests.regional._cov95_boot_site import BootSite

FAILURE = "fixture registry sync failed"


def isolated_site(tmp_path: Path, record: str) -> Path:
    state = tmp_path / "state"
    state.mkdir()
    (state / "site.yaml").write_text("site: true\n", encoding="utf-8")
    if record == "synced":
        record_registry_sync(state)
    elif record == "failed":
        record_registry_sync(state, error=FAILURE)
    elif record == "unreadable":
        record_path(state).parent.mkdir(parents=True)
        record_path(state).write_text("{", encoding="utf-8")
    return state


class Uninstall:
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def __call__(self, state_dir: Path) -> subprocess.CompletedProcess[str]:
        self.calls.append(state_dir)
        return subprocess.CompletedProcess(["uninstall"], 0, "uninstalled", "")


@pytest.mark.parametrize(
    ("record", "status", "reason"),
    [
        ("missing", "MISSING", "no installation registry sync outcome is recorded"),
        ("failed", "UNAVAILABLE", FAILURE),
        ("unreadable", "UNREADABLE", "is not valid JSON"),
    ],
    ids=["missing", "failed", "unreadable"],
)
def test_an_unsynchronized_site_is_reported_with_its_remedy_not_uninstalled(
    tmp_path: Path, record: str, status: str, reason: str
) -> None:
    state = isolated_site(tmp_path, record)
    uninstall = Uninstall()

    cleanup = lifecycle.cleanup_isolated_site(
        state, tmp_path, log_name="cleanup.log", retain=False, uninstall=uninstall
    )

    assert uninstall.calls == [], "an uninstall that can only refuse is not run"
    assert not (tmp_path / "cleanup.log").exists(), "no uninstall log is fabricated"
    assert cleanup["disposition"] == "RECONCILIATION_REQUIRED", cleanup
    assert cleanup["uninstall_ran"] is False and cleanup["cleanup_verified"] is False
    text = cleanup["reason"]
    assert text.startswith(
        f"uninstall would refuse to start (installation registry sync {status}): "
    ), text
    assert reason in text, f"the record names the recorded outcome: {text}"
    assert f"gpu-fault-admin deploy --state-dir {state}" in text, (
        f"the record names the product's remedy: {text}"
    )
    assert f"never passes {ACCEPT_FLAG}" in text, "the cleanup never overrides the gate"
    assert text.endswith(
        "gpu-fault-admin " + shlex.join(lifecycle.uninstall_arguments(state))
    ), f"the record names the uninstall to run after the remedy: {text}"


def test_a_synchronized_site_is_uninstalled_as_before(tmp_path: Path) -> None:
    state = isolated_site(tmp_path, "synced")
    uninstall = Uninstall()

    cleanup = lifecycle.cleanup_isolated_site(
        state, tmp_path, log_name="cleanup.log", retain=False, uninstall=uninstall
    )

    assert uninstall.calls == [state]
    assert cleanup["disposition"] == "UNINSTALLED" and "reason" not in cleanup


def test_an_uninstall_that_already_started_resumes_without_the_record(
    tmp_path: Path,
) -> None:
    state = isolated_site(tmp_path, "missing")
    journal = state / "uninstall" / "state.json"
    journal.parent.mkdir()
    journal.write_text("{}", encoding="utf-8")
    uninstall = Uninstall()

    cleanup = lifecycle.cleanup_isolated_site(
        state, tmp_path, log_name="cleanup.log", retain=False, uninstall=uninstall
    )

    assert uninstall.calls == [state], "a started attempt keeps its own verdict"
    assert cleanup["disposition"] == "UNINSTALLED"


def arguments(root: Path) -> Namespace:
    return Namespace(
        bootstrap_state_dir=root / "state",
        cpu_cluster_arn="arn:aws:eks:us-east-1:000000000000:cluster/cpu",
        gpu_cluster_arn=["arn:aws:eks:us-east-1:000000000000:cluster/gpu"],
        admin_email="unit@example.invalid",
        retain_bootstrap_site=False,
    )


def test_boot016_names_the_registry_refusal_after_a_deploy_that_recorded_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = BootSite(tmp_path, monkeypatch)
    model.failure = "deploy"
    model.registry_sync = False

    outcome = lifecycle.run_boot016(arguments(tmp_path), tmp_path / "case")

    assert outcome["verdict"] == "FAIL"
    assert [call[0][0] for call in model.admin_calls] == ["deploy"], (
        "no uninstall runs over a site whose registry sync is not recorded"
    )
    cleanup = outcome["cleanup"]
    assert cleanup["disposition"] == "RECONCILIATION_REQUIRED", cleanup
    assert "installation registry sync MISSING" in cleanup["reason"], cleanup


def test_boot018_fails_while_its_site_cannot_be_uninstalled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = isolated_site(tmp_path, "failed")
    uninstall = Uninstall()
    monkeypatch.setattr(lifecycle, "uninstall_site", uninstall)
    monkeypatch.setattr(
        lifecycle,
        "boot018_body",
        lambda state_dir, case_dir: {"verdict": "PASS", "checks": {"x": True}},
    )

    outcome = lifecycle.run_boot018(arguments(tmp_path), tmp_path / "case")

    assert uninstall.calls == []
    assert outcome["checks"]["bootstrap_site_cleanup"] is False
    assert outcome["verdict"] == "FAIL", "a site left behind is not a PASS"
    assert FAILURE in outcome["cleanup"]["reason"]
    assert state.joinpath("site.yaml").is_file(), "the site is left as it was"
