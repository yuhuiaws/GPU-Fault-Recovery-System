"""``gpu-fault-admin aurora-failback``: the supported way back to a writer
beside the control plane.

Live 2026-10-13 an operator had to run ``aws rds failover-db-cluster
--target-db-instance-identifier <same-AZ instance>`` by hand after HA-010. The
verb plans read-only, executes only with the exact token and an approved
reference under the site operation lock, refuses while work is in flight by the
site's own checks, journals every step, runs bootstrap's writer-zone step and
waits for the cluster and the control plane. It resumes an interrupted run and
never moves a writer that is already placed.
"""

from __future__ import annotations

import json
import re
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin import aurora_failback as module
from gpu_fault.admin import cli
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.deploy_host_binding import is_readonly_command
from gpu_fault.admin.operation_lock import site_operation_lock
from gpu_fault_release.regional_release_config import ReleaseError
from tests.admin._aurora_placement_support import CLUSTER_ID, REGION, World, site

COMMITTED = {
    "release_id": "release-a",
    "phase": "complete",
    "transaction_committed": True,
}


class _Site:
    """The live collaborators of an execute, recorded in call order."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, world: World) -> None:
        self.world = world
        self.events: list[str] = []
        self.state: dict[str, Any] = dict(COMMITTED)
        self.idle = True
        self.safety_error: str | None = None
        self.failover_error: str | None = None
        self.ready_error: str | None = None
        self.store_errors = 0
        self.failovers: list[dict[str, Any]] = []
        release = SimpleNamespace(_remote_commands_are_idle=self.remote_idle)
        self.release = release

        def live_state(_site: Any) -> dict[str, Any]:
            self.events.append("release-state")
            return self.state

        def safety(target: Any) -> dict[str, Any]:
            assert target is release, "the checks read through the site's engine"
            self.events.append("workflow-safety")
            if self.safety_error:
                raise ReleaseError(self.safety_error)
            return {"blocker_count": 0, "resolved_blocked_count": 2, "extra": "x"}

        def affinity(runner: Any, **kwargs: Any) -> dict[str, Any]:
            assert runner is world, "the failover runs through the verb's runner"
            self.events.append("failover")
            self.failovers.append(kwargs)
            if self.failover_error:
                raise BootstrapError(self.failover_error)
            (target,) = kwargs["instance_ids"]
            world.writer = target
            return {"action": "failover", "target": target}

        def ready(target: Any, **kwargs: Any) -> dict[str, Any]:
            assert target is release, "the control plane is read through the engine"
            self.events.append("control-plane-ready")
            if self.ready_error:
                raise ReleaseError(self.ready_error)
            return {"waited_seconds": 4.0, "registry_generation": 7}

        def stats(target: Any) -> dict[str, Any]:
            self.events.append("store-read")
            if self.store_errors:
                self.store_errors -= 1
                raise ReleaseError("unit store reconnecting")
            return {"by_status": {"SUCCEEDED": 3}, "open_by_cluster": {}}

        monkeypatch.setattr(module, "live_release_state", live_state)
        monkeypatch.setattr(module, "workflow_safety_snapshot", safety)
        monkeypatch.setattr(module, "ensure_writer_zone_affinity", affinity)
        monkeypatch.setattr(module, "wait_control_plane_ready", ready)
        monkeypatch.setattr(module, "remote_command_stats", stats)

    def remote_idle(self) -> bool:
        self.events.append("remote-commands")
        return self.idle

    def execute(self, path: Path, **kwargs: Any) -> dict[str, Any]:
        return module.run_aurora_failback(
            site(path),  # type: ignore[arg-type]
            reference="CHG-9",
            runner=self.world,  # type: ignore[arg-type]
            release_factory=lambda _site: self.release,
            identity=lambda: "unit-operator",
            sleep=lambda _seconds: None,
            **kwargs,
        )


def _journals(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(item.read_text())
        for item in sorted((path / module.STATE_ROOT).glob("*.json"))
    ]


def _argv(path: Path, *extra: str) -> list[str]:
    return ["aurora-failback", "--state-dir", str(path), *extra]


def _site_file(arguments: Any, *, command: str) -> Path:
    """``cli._managed_site_file`` for a test: the state directory's site.yaml."""

    assert command == "aurora-failback", command
    return Path(arguments.state_dir) / "site.yaml"


def _load(monkeypatch: pytest.MonkeyPatch, path: Path) -> Any:
    """The managed site on disk, as ``load_site`` would hand it to the verb."""

    loaded = site(path)
    monkeypatch.setattr(module, "load_site", lambda _file, repository_root=None: loaded)
    return loaded


def test_execute_fails_back_journals_each_step_and_waits_for_the_control_plane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = _Site(monkeypatch, World())
    live.store_errors = 1

    result = live.execute(tmp_path)

    assert live.events == [
        "release-state",
        "remote-commands",
        "workflow-safety",
        "failover",
        "control-plane-ready",
        "store-read",
        "store-read",
    ], "checks before the failover, waits after it"
    assert live.failovers == [
        {
            "aws_region": REGION,
            "cluster_id": CLUSTER_ID,
            "instance_ids": ["aurora-a-writer"],
            "control_plane_zones": ["us-west-2a"],
            "sleep": live.failovers[0]["sleep"],
        }
    ]
    assert result["status"] == module.STATUS_COMPLETED
    assert result["target"]["instance_id"] == "aurora-a-writer"
    assert result["target"]["promotion_tier"] == 0
    assert result["before"]["writer"] == "aurora-a-reader"
    assert result["after"]["writer"] == "aurora-a-writer"
    assert result["after"]["placed"] is True
    assert result["guards"] == {
        "release_id": "release-a",
        "release_phase": "complete",
        "remote_commands_idle": True,
        "workflow_safety": {
            "blocker_count": 0,
            "resolved_blocked_count": 2,
            "abandoned_generation_count": None,
            "compile_blocked_count": None,
        },
    }
    assert result["control_plane"]["registry_generation"] == 7
    assert result["control_plane"]["open_remote_commands"] == {}
    assert list(result["steps"]) == [
        module.STEP_PLANNED,
        module.STEP_CLUSTER_AVAILABLE,
        module.STEP_CONTROL_PLANE_HEALTHY,
    ]
    (journal,) = _journals(tmp_path)
    assert journal == {key: value for key, value in result.items() if key != "journal"}
    assert journal["operator"] == "unit-operator" and journal["reference"] == "CHG-9"
    assert Path(result["journal"]).parent == tmp_path / module.STATE_ROOT


def test_the_planned_step_is_on_disk_before_the_failover_is_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = _Site(monkeypatch, World())
    seen: list[dict[str, Any]] = []
    original = module.ensure_writer_zone_affinity

    def observe(runner: Any, **kwargs: Any) -> dict[str, Any]:
        seen.extend(_journals(tmp_path))
        return original(runner, **kwargs)

    monkeypatch.setattr(module, "ensure_writer_zone_affinity", observe)

    live.execute(tmp_path)

    (planned,) = seen
    assert planned["status"] == module.STATUS_IN_PROGRESS
    assert list(planned["steps"]) == [module.STEP_PLANNED]
    assert planned["target"]["instance_id"] == "aurora-a-writer"


def test_a_writer_already_placed_is_a_noop_with_no_checks_and_no_failover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = _Site(monkeypatch, World(writer="aurora-a-writer"))

    result = live.execute(tmp_path)

    assert result["status"] == module.STATUS_NOOP
    assert live.events == [], "nothing to fail back: no checks, no request, no wait"
    (journal,) = _journals(tmp_path)
    assert journal["status"] == module.STATUS_NOOP
    assert journal["before"]["placed"] is True


@pytest.mark.parametrize(
    ("defect", "message", "events"),
    [
        ("transaction", "a release transaction is open (phase cpu-staged)", 1),
        ("remote", "remote commands are PENDING/LEASED/WAITING", 2),
        ("safety", "active destructive workflows block release", 3),
        ("state", "cannot read the live release state", 1),
    ],
)
def test_work_in_flight_refuses_before_anything_is_journaled_or_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
    message: str,
    events: int,
) -> None:
    live = _Site(monkeypatch, World())
    if defect == "transaction":
        live.state = {
            **COMMITTED,
            "phase": "cpu-staged",
            "transaction_committed": False,
        }
    elif defect == "remote":
        live.idle = False
    elif defect == "safety":
        live.safety_error = "active destructive workflows block release: {}"
    else:

        def unreadable(_site: Any) -> dict[str, Any]:
            live.events.append("release-state")
            raise module.SiteConfigError("cannot read live ConfigMap state")

        monkeypatch.setattr(module, "live_release_state", unreadable)

    with pytest.raises(
        BootstrapError, match=f"aurora-failback refused: .*{re.escape(message)}"
    ):
        live.execute(tmp_path)

    assert len(live.events) == events and "failover" not in live.events
    assert _journals(tmp_path) == [], "a refusal writes nothing"


@pytest.mark.parametrize("defect", ["settling", "no-member", "no-writer"])
def test_a_cluster_with_nowhere_to_go_is_refused_before_the_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    world = World()
    if defect == "settling":
        world.cluster_status = "failing-over"
        message = "is failing-over; a failover may still be settling"
    elif defect == "no-member":
        world.statuses["aurora-a-writer"] = "rebooting"
        message = "no available Aurora member is in us-west-2a to fail back to"
    else:
        world.writer = "nobody"
        message = "reports 0 writers"
    live = _Site(monkeypatch, world)

    with pytest.raises(BootstrapError, match=message):
        live.execute(tmp_path)

    assert live.events == [] and _journals(tmp_path) == []


def test_a_failover_that_does_not_settle_is_journaled_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = _Site(monkeypatch, World())
    live.failover_error = "Aurora writer failover to aurora-a-writer did not settle"

    with pytest.raises(BootstrapError, match="did not settle; journal "):
        live.execute(tmp_path)

    (journal,) = _journals(tmp_path)
    assert journal["status"] == module.STATUS_FAILED
    assert "did not settle" in journal["error"]
    assert list(journal["steps"]) == [module.STEP_PLANNED]


def test_a_rerun_resumes_the_waits_once_the_writer_moved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = _Site(monkeypatch, World())
    live.ready_error = "control plane did not become ready within 300 seconds"
    with pytest.raises(BootstrapError, match="rerun to resume"):
        live.execute(tmp_path)
    (failed,) = _journals(tmp_path)
    assert failed["status"] == module.STATUS_FAILED
    assert module.STEP_CLUSTER_AVAILABLE in failed["steps"]

    live.ready_error = None
    live.events.clear()
    result = live.execute(tmp_path)

    assert live.events == ["control-plane-ready", "store-read"], (
        "the resumed run only waits: no checks, no second failover"
    )
    assert result["run_id"] == failed["run_id"] and len(result["resumed_at"]) == 1
    (journal,) = _journals(tmp_path)
    assert journal["status"] == module.STATUS_COMPLETED
    assert module.STEP_CONTROL_PLANE_HEALTHY in journal["steps"]


def test_a_rerun_supersedes_an_unfinished_run_whose_writer_did_not_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = World()
    live = _Site(monkeypatch, world)
    live.failover_error = "InvalidDBClusterStateFault"
    with pytest.raises(BootstrapError):
        live.execute(tmp_path)

    live.failover_error = None
    result = live.execute(tmp_path)

    assert result["status"] == module.STATUS_COMPLETED
    first, second = _journals(tmp_path)
    assert first["status"] == module.STATUS_SUPERSEDED
    assert first["superseded_with"]["writer"] == "aurora-a-reader"
    assert second["run_id"] == result["run_id"] != first["run_id"]
    assert live.events.count("failover") == 2


def test_the_store_must_answer_within_the_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = iter(range(0, 10_000, 100))
    release = SimpleNamespace()
    monkeypatch.setattr(
        module, "wait_control_plane_ready", lambda *a, **k: {"waited_seconds": 0}
    )

    def unanswered(_release: Any) -> dict[str, Any]:
        raise ReleaseError("unit store down")

    monkeypatch.setattr(module, "remote_command_stats", unanswered)

    with pytest.raises(ReleaseError, match="did not answer within 300 s"):
        module.wait_control_plane_healthy(
            release, sleep=lambda _seconds: None, monotonic=lambda: float(next(clock))
        )


def test_plan_is_read_only_and_names_the_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    world = World()
    live = _Site(monkeypatch, world)
    monkeypatch.setattr(module, "CommandRunner", lambda: world)
    _load(monkeypatch, tmp_path)
    monkeypatch.setattr(
        module,
        "administrator_operation_lock",
        lambda _path: pytest.fail("a plan must not take the site operation lock"),
    )
    arguments = cli.parser().parse_args(_argv(tmp_path, "--reference", "CHG-9"))

    assert module.run_command(arguments, _site_file) == 0

    plan = json.loads(capsys.readouterr().out)
    assert plan["mode"] == "plan" and plan["action"] == "failover"
    assert plan["target"]["instance_id"] == "aurora-a-writer"
    assert plan["placement"]["writer_zone"] == "us-west-2b"
    assert plan["execute"].endswith("--reference CHG-9 --confirm AURORA_FAILBACK"), (
        plan["execute"]
    )
    assert live.events == [] and _journals(tmp_path) == []
    assert [call[2] for call in world.calls if call[0] == "aws"] == [
        "describe-db-clusters",
        "describe-db-instances",
    ], "a plan only describes"
    assert is_readonly_command(arguments) is True


@pytest.mark.parametrize(
    ("writer", "status", "action", "reason"),
    [
        ("aurora-a-writer", "available", "none", "already runs in the preferred zone"),
        ("aurora-a-reader", "failing-over", "wait", "Aurora cluster is failing-over"),
    ],
)
def test_plan_reports_why_there_is_nothing_to_do(
    tmp_path: Path, writer: str, status: str, action: str, reason: str
) -> None:
    world = World(writer=writer, cluster_status=status)

    plan = module.plan_failback(
        site(tmp_path),  # type: ignore[arg-type]
        reference="CHG-9",
        runner=world,  # type: ignore[arg-type]
    )

    assert plan["action"] == action and reason in plan["reason"], plan
    assert plan["target"] is None


def test_execute_needs_the_exact_token_and_a_valid_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = _Site(monkeypatch, World())
    _load(monkeypatch, tmp_path)
    wrong = cli.parser().parse_args(
        _argv(tmp_path, "--reference", "CHG-9", "--confirm", "YES")
    )
    with pytest.raises(BootstrapError, match="requires --confirm AURORA_FAILBACK"):
        module.run_command(wrong, _site_file)
    bad = cli.parser().parse_args(
        _argv(tmp_path, "--reference", "bad ref!", "--confirm", module.CONFIRMATION)
    )
    with pytest.raises(BootstrapError, match="reference is invalid"):
        module.run_command(bad, _site_file)
    with pytest.raises(SystemExit):
        cli.parser().parse_args(_argv(tmp_path, "--confirm", module.CONFIRMATION))
    assert live.events == [], "refusals precede every read"
    assert is_readonly_command(wrong) is False


def test_execute_takes_the_lock_reloads_the_site_and_lends_its_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    events: list[str] = []
    loaded = site(tmp_path)

    @contextmanager
    def lock(path: Path):
        assert path == tmp_path
        events.append("lock")
        try:
            yield 7
        finally:
            events.append("unlock")

    @contextmanager
    def environment(target: Any):
        assert target is loaded
        events.append("environment")
        yield

    def reload(target: Any) -> Any:
        assert target is loaded and events == ["lock"]
        events.append("reload")
        return target

    def run(target: Any, *, reference: str) -> dict[str, Any]:
        events.append(f"run:{reference}")
        return {"status": module.STATUS_COMPLETED}

    monkeypatch.setattr(module, "load_site", lambda path, repository_root=None: loaded)
    monkeypatch.setattr(module, "administrator_operation_lock", lock)
    monkeypatch.setattr(module, "reload_site_for_mutation", reload)
    monkeypatch.setattr(module, "site_process_environment", environment)
    monkeypatch.setattr(module, "run_aurora_failback", run)
    arguments = cli.parser().parse_args(
        _argv(tmp_path, "--reference", "CHG-9", "--confirm", module.CONFIRMATION)
    )

    assert cli.run(arguments) == 0

    assert events == ["lock", "reload", "environment", "run:CHG-9", "unlock"]
    assert json.loads(capsys.readouterr().out)["status"] == module.STATUS_COMPLETED


def test_execute_refuses_while_another_mutation_holds_the_site_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = _Site(monkeypatch, World())
    _load(monkeypatch, tmp_path)
    held = threading.Event()
    release = threading.Event()

    def hold() -> None:
        with site_operation_lock(tmp_path, wait=False):
            held.set()
            release.wait(timeout=30)

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    assert held.wait(timeout=10), "the holder must take the site lock first"
    arguments = cli.parser().parse_args(
        _argv(tmp_path, "--reference", "CHG-9", "--confirm", module.CONFIRMATION)
    )
    try:
        with pytest.raises(BootstrapError, match="another administrator mutation"):
            module.run_command(arguments, _site_file)
    finally:
        release.set()
        holder.join(timeout=10)
    assert live.events == [], "a held lock refuses before any read"


def test_the_cli_offers_the_verb_beside_the_other_site_verbs(capsys) -> None:
    arguments = cli.parser().parse_args(_argv(Path("/srv/site"), "--reference", "C-1"))
    assert (arguments.command, arguments.reference, arguments.confirm) == (
        "aurora-failback",
        "C-1",
        None,
    )
    with pytest.raises(SystemExit):
        cli.parser().parse_args(["--help"])
    assert "aurora-failback" in capsys.readouterr().out
