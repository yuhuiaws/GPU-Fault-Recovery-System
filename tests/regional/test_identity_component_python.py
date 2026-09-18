from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import audit_auth_boundary as audit
from scripts.e2e.regional.identity_acceptance_common import IdentitySite
from scripts.e2e.regional.regional_live_fixture import component_python


@pytest.mark.parametrize("plane", ["cpu", "gpu"])
def test_identity_pod_exec_uses_the_installed_component_interpreter(
    plane: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    regional = SimpleNamespace(
        kubectl=lambda *args, **kwargs: calls.append(args) or "{}"
    )
    site = IdentitySite.__new__(IdentitySite)
    monkeypatch.setattr(site, "regional", lambda target: regional)
    assert site.pod_json(plane, None, "pod-test", "probe-test") == {}
    command = calls[0]
    assert command[command.index("--") + 1] == component_python(plane)


def test_auth_store_probe_uses_cpu_component_python(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = []

    def command(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(
            args,
            0,
            json.dumps({"clusters": {}, "commands": {}, "probe_id": args[-1]}),
            "",
        )

    monkeypatch.setattr(audit, "run_fixture_command", command)
    assert audit.store_snapshot(
        tmp_path / "synthetic.kubeconfig", "gpu-system", "api", ["a", "b"]
    ) == {"clusters": {}, "commands": {}, "probe_id": "auth-probe-node"}
    assert calls[0][calls[0].index("--") + 1] == component_python("cpu")
