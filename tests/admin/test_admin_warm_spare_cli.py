"""``gpu-fault-admin config spare`` rides the ``config`` verb.

No new top-level verb: the CLI was slimmed to ten on purpose. The sub-action
has its own ``--help``, needs ``--reference`` and the exact ``--confirm`` token
to mutate, and the plain ``config --state-dir ...`` invocation still parses.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin import cli, warm_spare
from gpu_fault.admin.command_log import ADMIN_LOG_ENVIRONMENT
from gpu_fault.admin.site import SiteConfigError
from gpu_fault.admin.warm_spare import (
    DECLARE_CONFIRMATION,
    RELEASE_CONFIRMATION,
    SPARE_LABEL,
    AgentState,
    WarmSpareError,
)
from tests.admin.test_admin_warm_spare import NODE, FakeCoreApi, _raw_node

ACTIVE = AgentState(lifecycle_state="ACTIVE")


def _state_dir(tmp_path: Path) -> Path:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "site.yaml").write_text("placeholder\n", encoding="utf-8")
    return state_dir


def _site() -> Any:
    return SimpleNamespace(
        release_config={
            "clusters": [{"cluster_id": "cluster-a", "context": "ctx-a"}],
            "gpu_kubeconfig": "/state/gpu.kubeconfig",
        },
        environment={},
    )


def test_config_spare_has_its_own_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit, match="0"):
        cli.parser().parse_args(["config", "spare", "--help"])

    help_text = capsys.readouterr().out
    for flag in (
        "--state-dir",
        "--node",
        "--fault-node",
        "--cluster-id",
        "--reference",
        "--declare",
        "--release",
        "--confirm",
        DECLARE_CONFIRMATION,
        RELEASE_CONFIRMATION,
    ):
        assert flag in help_text, flag


def test_the_plain_config_invocation_still_parses(tmp_path: Path) -> None:
    arguments = cli.parser().parse_args(
        ["config", "--state-dir", str(tmp_path), "--reference", "CHG-1", "--dry-run"]
    )

    assert arguments.command == "config"
    assert arguments.config_action is None
    assert arguments.state_dir == tmp_path
    assert arguments.reference == "CHG-1"
    assert arguments.dry_run is True


def test_config_without_a_state_dir_is_refused_by_name(tmp_path: Path) -> None:
    # ``--state-dir`` moved off the parser's required list so the sub-action can
    # carry its own; the refusal now comes from the site lookup instead.
    with pytest.raises(SiteConfigError, match="config requires --state-dir"):
        cli.run(cli.parser().parse_args(["config"]))


def test_the_plain_config_path_still_requires_a_reference(tmp_path: Path) -> None:
    # ``--reference`` left the parser's required list so a read-only
    # ``config spare`` needs none; the audited apply refuses without it by name.
    from gpu_fault.admin.config import AdminConfigError, prepare_admin_config_apply

    parsed = cli.parser().parse_args(["config", "--state-dir", str(tmp_path)])
    assert parsed.reference is None

    with pytest.raises(AdminConfigError, match="requires --reference"):
        prepare_admin_config_apply(
            tmp_path,
            expected_plan_sha256="a" * 64,
            reference=parsed.reference or "",
            current_release_identity={},
        )


def test_config_spare_parses_the_sub_action_flags(tmp_path: Path) -> None:
    arguments = cli.parser().parse_args(
        [
            "config",
            "spare",
            "--state-dir",
            str(tmp_path),
            "--node",
            NODE,
            "--fault-node",
            "node-fault",
            "--cluster-id",
            "cluster-a",
            "--reference",
            "CHG-1",
            "--declare",
            "--confirm",
            DECLARE_CONFIRMATION,
        ]
    )

    assert arguments.command == "config"
    assert arguments.config_action == "spare"
    request = warm_spare.spare_request(arguments)
    assert request.state_dir == tmp_path
    assert request.node == NODE
    assert request.fault_node == "node-fault"
    assert request.cluster_id == "cluster-a"
    assert request.reference == "CHG-1"
    assert request.mode == "declare"
    assert request.confirmation == DECLARE_CONFIRMATION


def test_declare_and_release_are_mutually_exclusive_and_node_is_required(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit):
        cli.parser().parse_args(
            ["config", "spare", "--state-dir", str(tmp_path), "--node", NODE]
            + ["--declare", "--release"]
        )
    with pytest.raises(SystemExit):
        cli.parser().parse_args(["config", "spare", "--state-dir", str(tmp_path)])


def test_run_dispatches_config_spare_to_the_warm_spare_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = _state_dir(tmp_path)
    calls: list[dict[str, Any]] = []
    site = SimpleNamespace(name="site")
    monkeypatch.setattr(cli, "load_site", lambda *_args, **_kwargs: site)

    def fake_command(arguments, *, site, **kwargs):
        calls.append({"arguments": arguments, "site": site, **kwargs})
        return 0

    monkeypatch.setattr(warm_spare, "run_config_spare_command", fake_command)

    exit_code = cli.run(
        cli.parser().parse_args(
            ["config", "spare", "--state-dir", str(state_dir), "--node", NODE]
        )
    )

    assert exit_code == 0
    (call,) = calls
    assert call["site"] is site
    assert call["arguments"].node == NODE


def _spare_arguments(state_dir: Path, *extra: str) -> Any:
    return cli.parser().parse_args(
        ["config", "spare", "--state-dir", str(state_dir), "--node", NODE, *extra]
    )


def test_declare_requires_the_exact_confirmation_token(tmp_path: Path) -> None:
    api = FakeCoreApi(_raw_node(NODE))

    with pytest.raises(WarmSpareError, match=f"--confirm {DECLARE_CONFIRMATION}"):
        warm_spare.run_config_spare_command(
            _spare_arguments(tmp_path, "--declare", "--reference", "CHG-1"),
            site=_site(),
            api=api,
            agent_lookup=lambda *_: ACTIVE,
        )
    with pytest.raises(WarmSpareError, match=f"--confirm {DECLARE_CONFIRMATION}"):
        warm_spare.run_config_spare_command(
            _spare_arguments(
                tmp_path,
                "--declare",
                "--reference",
                "CHG-1",
                "--confirm",
                RELEASE_CONFIRMATION,
            ),
            site=_site(),
            api=api,
            agent_lookup=lambda *_: ACTIVE,
        )
    assert api.patches == []


def test_declare_requires_a_reference(tmp_path: Path) -> None:
    api = FakeCoreApi(_raw_node(NODE))

    with pytest.raises(WarmSpareError, match="requires --reference"):
        warm_spare.run_config_spare_command(
            _spare_arguments(tmp_path, "--declare", "--confirm", DECLARE_CONFIRMATION),
            site=_site(),
            api=api,
            agent_lookup=lambda *_: ACTIVE,
        )
    assert api.patches == []


def test_release_requires_its_own_token_and_reference(tmp_path: Path) -> None:
    api = FakeCoreApi(_raw_node(NODE, labels={SPARE_LABEL: "true"}, unschedulable=True))

    with pytest.raises(WarmSpareError, match=f"--confirm {RELEASE_CONFIRMATION}"):
        warm_spare.run_config_spare_command(
            _spare_arguments(
                tmp_path,
                "--release",
                "--reference",
                "CHG-1",
                "--confirm",
                DECLARE_CONFIRMATION,
            ),
            site=_site(),
            api=api,
            agent_lookup=lambda *_: ACTIVE,
        )
    assert api.patches == []


def test_the_read_only_check_exits_one_on_refusals_and_zero_when_ready(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    not_ready = FakeCoreApi(_raw_node(NODE, ready="False"))

    assert (
        warm_spare.run_config_spare_command(
            _spare_arguments(tmp_path),
            site=_site(),
            api=not_ready,
            agent_lookup=lambda *_: ACTIVE,
        )
        == 1
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed["refusals"] == ["node is not Ready"]
    assert printed["mode"] == "check"

    assert (
        warm_spare.run_config_spare_command(
            _spare_arguments(tmp_path),
            site=_site(),
            api=FakeCoreApi(_raw_node(NODE)),
            agent_lookup=lambda *_: ACTIVE,
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["ready"] is True


def test_declare_through_the_command_takes_the_site_lock_and_writes_the_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    api = FakeCoreApi(_raw_node(NODE))

    exit_code = warm_spare.run_config_spare_command(
        _spare_arguments(
            tmp_path,
            "--declare",
            "--reference",
            "CHG-1",
            "--confirm",
            DECLARE_CONFIRMATION,
        ),
        site=_site(),
        api=api,
        agent_lookup=lambda *_: ACTIVE,
    )

    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["declaration"]["reference"] == "CHG-1"
    assert (tmp_path / "warm-spares" / f"{NODE}.json").is_file(), (
        "the declaration record must live under <state-dir>/warm-spares/"
    )
    assert api.nodes[NODE]["spec"]["unschedulable"] is True


def test_a_mutating_spare_action_is_teed_into_the_admin_command_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``config spare`` is a ``config`` invocation, so ``main`` opens the same
    ``logs/mutating/config-*.log`` every other mutating verb writes."""

    state_dir = _state_dir(tmp_path)
    monkeypatch.delenv(ADMIN_LOG_ENVIRONMENT, raising=False)
    monkeypatch.setattr(cli, "load_site", lambda *_args, **_kwargs: _site())
    monkeypatch.setattr(cli, "enforce_deploy_host_state_dir", lambda _arguments: None)
    monkeypatch.setattr(
        warm_spare,
        "run_config_spare_command",
        lambda arguments, *, site: print(f"declared {arguments.node}") or 0,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gpu-fault-admin",
            "config",
            "spare",
            "--state-dir",
            str(state_dir),
            "--node",
            NODE,
            "--declare",
            "--reference",
            "CHG-1",
            "--confirm",
            DECLARE_CONFIRMATION,
        ],
    )

    assert cli.main() == 0

    logs = sorted((state_dir / "logs" / "mutating").glob("config-*.log"))
    assert len(logs) == 1, logs
    assert "logging to" in logs[0].read_text(encoding="utf-8")
