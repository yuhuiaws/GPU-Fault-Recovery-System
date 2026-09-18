from __future__ import annotations

from pathlib import Path
from types import ModuleType

from scripts.e2e.regional import control_plane_env_window, executor_env_window
from tests.regional.test_deployment_window_safety import WindowAPI

WINDOWS = (control_plane_env_window, executor_env_window)


def window_fixture(module: ModuleType, tmp_path: Path):
    api = WindowAPI(module)
    settings = module.Settings(tmp_path / "window.json", 1)
    assignments = (
        {module.LIFETIME_VARIABLE: "180"}
        if module is control_plane_env_window
        else {module.ALLOWED_VARIABLES[0]: "10"}
    )
    return api, settings, assignments


def change_variable(api: WindowAPI, name: str, value: str) -> None:
    entries = api.deployment["spec"]["template"]["spec"]["containers"][0]["env"]
    for item in entries:
        if item["name"] == name:
            item["value"] = value
            return
    entries.append({"name": name, "value": value})
