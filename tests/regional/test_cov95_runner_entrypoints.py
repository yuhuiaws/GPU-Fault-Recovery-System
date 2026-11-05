"""Runner scripts started as programs: repository imports and the usage text.

Operators start these files directly (``python scripts/e2e/regional/<runner>.py``,
``python tools/<tool>.py``), from any working directory. Each one must make the
repository importable itself and route ``__main__`` to its parser.
"""

from __future__ import annotations

import importlib
import runpy
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from scripts.e2e.regional.site_profile import SITE_PROFILE_ENV

ROOT = Path(__file__).resolve().parents[2]
REGIONAL = "scripts/e2e/regional"


@pytest.fixture
def program(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """No site profile; ``sys.path`` and bare-named tool modules restored after."""

    monkeypatch.delenv(SITE_PROFILE_ENV, raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    before = set(sys.modules)
    yield
    tools = str(ROOT / "tools")
    for name in set(sys.modules) - before:
        if str(getattr(sys.modules[name], "__file__", "") or "").startswith(tools):
            if "." not in name:
                del sys.modules[name]


@pytest.mark.parametrize(
    ("relative", "removed"),
    [
        (f"{REGIONAL}/run_boot020_release_rolling.py", [ROOT]),
        (f"{REGIONAL}/run_destr010_fabric_manager_restart.py", [ROOT]),
        (f"{REGIONAL}/run_destr014_branch_exhaustion.py", [ROOT]),
        (f"{REGIONAL}/run_destr019_agent_restart_ledger.py", [ROOT]),
        (f"{REGIONAL}/collect002_power_cleanup.py", [ROOT]),
        (f"{REGIONAL}/capacity_acceptance_base.py", [ROOT, ROOT / "src"]),
    ],
)
def test_a_runner_started_outside_the_repository_puts_it_first_on_the_path(
    program: None, relative: str, removed: list[Path]
) -> None:
    # The runner's imports resolve from this checkout (sys.modules), never from
    # an installed copy of the packages while the path lacks the checkout.
    importlib.import_module(relative.removesuffix(".py").replace("/", "."))
    dropped = {str(path) for path in removed}
    sys.path[:] = [entry for entry in sys.path if entry not in dropped]

    namespace = runpy.run_path(str(ROOT / relative), run_name="cov95_entry_probe")

    assert sys.path[: len(removed)] == [str(path) for path in reversed(removed)]
    assert namespace["ROOT"] == ROOT


@pytest.mark.parametrize(
    ("relative", "flags"),
    [
        (
            f"{REGIONAL}/run_boot020_release_rolling.py",
            ["--admin-state-dir", "--execute", "--confirm"],
        ),
        (f"{REGIONAL}/run_destr010_fabric_manager_restart.py", ["--execute"]),
        (f"{REGIONAL}/run_destr014_branch_exhaustion.py", ["--cluster-id"]),
        (f"{REGIONAL}/run_destr019_agent_restart_ledger.py", ["--gpu-context"]),
        (f"{REGIONAL}/collect002_power_cleanup.py", ["--power-run-id"]),
        (f"{REGIONAL}/run_cap005_postgres_suite.py", ["--postgres-workers"]),
        (f"{REGIONAL}/probes/collector_node_probe.py", ["firmware-premise"]),
        ("tools/gpu_environment_replay.py", ["--plan", "--gpu-uuid"]),
        ("tools/run_regional_acceptance.py", ["--propose-dependencies"]),
    ],
)
def test_a_runner_program_routes_main_to_its_parser(
    program: None,
    relative: str,
    flags: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    if relative.startswith("tools/"):
        # ``python tools/x.py`` runs with tools/ first on the path: the script
        # imports its siblings by bare name when it is not a package member.
        sys.path.insert(0, str(ROOT / "tools"))
    monkeypatch.setattr(sys, "argv", [relative, "--help"])

    with pytest.raises(SystemExit) as stop:
        runpy.run_path(str(ROOT / relative), run_name="__main__")

    assert stop.value.code == 0
    usage = capsys.readouterr().out
    assert usage.startswith("usage: "), usage[:200]
    missing = [flag for flag in flags if flag not in usage]
    assert missing == [], usage
