"""GF-REGIONAL-BOOT-020 ``restore`` stage: give the site back to itself.

The chain ends on candidate D (FULL: +control workers, a suffixed Runtime
Profile, a transaction the operator never committed). Leaving the site there
made every later case that needs a committed site release (BOOT-023) wait for
a hand-run ``gpu-fault-admin deploy`` -- and that deploy once tripped over the
control-plane Pods still carrying candidate D's start-up environment while the
ConfigMaps already said otherwise. So the runner restores the site itself,
through the same release engine path the candidates used, and proves the end
state:

* the site's own config (the ``noop`` candidate) classifies as a real change
  against candidate D and is applied as a normal, committed release;
* afterwards it classifies ``NOOP`` again -- the site is back where the chain
  started;
* every Running CPU Pod's effective ``GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION``
  equals its ``-config-core`` ConfigMap: no start-up-env drift is left behind.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

STAGE = "restore"
# The site's own release config: it classified NOOP before the chain began.
RESTORE_SCENARIO = "noop"
PROFILE_ENV = "GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION"


def stage_restore(
    backend: Any,
    recorder: Any,
    chain: Any,
    *,
    require: Callable[..., None],
) -> None:
    diff = recorder.stage(
        f"{STAGE}_classification", lambda: backend.classify(RESTORE_SCENARIO)
    )
    require(
        diff["kind"] != "NOOP",
        "restore classification: the site already matches its own config, so "
        "the full stage left nothing to restore",
        diff,
    )
    before = chain.before(backend, recorder, STAGE, diff)
    applied = recorder.stage(
        f"{STAGE}_apply",
        lambda: backend.deploy(RESTORE_SCENARIO, diff=diff, commit=True),
    )
    require(applied["phase"] == "complete", "restore phase", applied)
    require(
        applied.get("transaction_committed") is True,
        "restore release is not a committed transaction",
        applied,
    )
    after = chain.after(backend, recorder, STAGE)
    require(after["live"] != before["live"], "restore changed nothing", (before, after))
    alignment = recorder.stage(
        f"{STAGE}_cpu_env_alignment",
        lambda: backend.cpu_env_alignment(RESTORE_SCENARIO),
    )
    require(
        alignment.get("aligned") is True,
        "CPU Pods still carry a start-up environment that differs from their "
        "ConfigMaps after the restore",
        alignment,
    )
    following = recorder.stage(
        f"{STAGE}_next_classification", lambda: backend.classify(RESTORE_SCENARIO)
    )
    require(following["kind"] == "NOOP", "restore next classification", following)


def live_cpu_env_alignment(
    release: Any, deployments: tuple[str, ...]
) -> dict[str, Any]:
    """Compare each CPU Deployment's ``-config-core`` profile with its Running
    Pods' effective environment (``printenv`` inside the container)."""

    namespace = release.config.namespace
    report: dict[str, Any] = {}
    aligned = True
    for deployment in deployments:
        expected = release._config_map_data(f"{deployment}-config-core").get(
            PROFILE_ENV
        )
        pods = release._get_json(
            release._cpu("-n", namespace, "get", "pod", "-l", f"app={deployment}")
        )
        observed: dict[str, str | None] = {}
        for item in pods.get("items", []) if isinstance(pods, dict) else []:
            if (item.get("status") or {}).get("phase") != "Running":
                continue
            name = str(item["metadata"]["name"])
            value = release.runner.run(
                release._cpu(
                    "-n", namespace, "exec", name, "--", "printenv", PROFILE_ENV
                ),
                capture=True,
            ).strip()
            observed[name] = value or None
            if value != expected:
                aligned = False
        report[deployment] = {"config_map": expected, "pods": observed}
    return {"aligned": aligned, "deployments": report}


def summary(alignment: dict[str, Any]) -> str:
    return json.dumps(alignment, sort_keys=True)[:600]
