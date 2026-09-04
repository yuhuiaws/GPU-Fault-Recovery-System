from __future__ import annotations

import json
from pathlib import Path

from tests._script_loader import lazy_script_module
from tests.regional._release_orchestrator_support import config_file

ROOT = Path(__file__).resolve().parents[2]
MODULE = lazy_script_module(
    ROOT / "deploy/control-plane/regional/rollout_regional_release.py"
)
GPU_BOOTSTRAP = lazy_script_module(
    ROOT / "deploy/control-plane/regional/regional_gpu_bootstrap.py"
)


def test_cancel_active_installer_jobs_rechecks_for_running_jobs(tmp_path: Path) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))

    class CancelRunner:
        dry_run = False

        def __init__(self) -> None:
            self.reads = 0
            self.calls = []

        def run(self, arguments, **_kwargs):
            self.calls.append(arguments)
            if "get" in arguments and "jobs" in arguments:
                self.reads += 1
                items = (
                    [
                        {
                            "metadata": {"name": "running-job"},
                            "status": {"conditions": []},
                        }
                    ]
                    if self.reads == 1
                    else []
                )
                return json.dumps({"items": items})
            return ""

    runner = CancelRunner()
    release = MODULE.RegionalRelease(config, runner)

    GPU_BOOTSTRAP.cancel_active_installer_jobs(release, config.clusters[0])

    assert runner.reads == 2
    assert any(
        "delete" in arguments and "running-job" in arguments
        for arguments in runner.calls
    ), "active Installer Job was not cancelled"
