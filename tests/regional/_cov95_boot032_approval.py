from __future__ import annotations

from datetime import datetime, timedelta, timezone

from scripts.e2e.regional import boot032_lifecycle as lifecycle
from scripts.e2e.regional import live_driver_guard as guard
from scripts.e2e.regional import run_boot032_full_uninstall as entry
from scripts.e2e.regional.boot032_contract import CASE_ID, CONFIRMATION


def approve(world):
    arguments = entry.parser().parse_args(
        [
            "--run-dir",
            str(world.root),
            "--fixture-id",
            world.settings.fixture_id,
            "--protected-site",
            str(world.protected.source),
            "--protected-cluster-id",
            world.settings.protected_cluster_id,
            "--plan",
        ]
    )
    settings = entry.configure(arguments)
    world.settings = settings
    preflight = lifecycle.read_only_preflight(settings, settings.case_dir)
    assert preflight["errors"] == [], f"fake site must pass preflight: {preflight}"
    plan = guard.build_plan(
        run_dir=world.root,
        case_id=CASE_ID,
        attempt=1,
        confirmation=CONFIRMATION,
        details=lifecycle.plan_details(settings, preflight),
        arguments=arguments,
        preflight_passed=True,
        environment=settings.environment(),
    )
    arguments.plan = False
    arguments.execute = True
    arguments.confirm = CONFIRMATION
    deadline = datetime.now(timezone.utc) + timedelta(minutes=20)
    arguments.maintenance_window_end = deadline.isoformat()
    return settings, plan, deadline


def execute(world, deadline):
    return lifecycle.execute_case(world.settings, world.root, 1, deadline)


def restart(world):
    world.process = {**world.process, "pid": world.process["pid"] + 1}
