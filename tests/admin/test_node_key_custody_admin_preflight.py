from __future__ import annotations

import subprocess
from contextlib import contextmanager

import pytest

from gpu_fault.admin import bootstrap, cli
from gpu_fault.admin.node_key_custody_admin import (
    CustodyPreparationRequired,
    CustodyReconciliationRequired,
)
from tests.admin._node_key_custody_admin_support import AdminWorld
from tests.admin.test_node_key_custody_admin import provision
from tests.regional._cov95_identity_support import offline_guard as offline_guard


@pytest.mark.parametrize("ready", [False, True])
def test_public_admin_preflight_never_prepares_or_signs_custody(
    tmp_path, monkeypatch, ready
):
    world = AdminWorld(tmp_path, monkeypatch)
    world.configure()
    world.access()
    world.release_ready = True
    if ready:
        with pytest.raises(CustodyPreparationRequired):
            provision(world)
        world.authorize()
        provision(world)
    site = world.site()
    site.source.write_text("owned fake site")
    site.audit_summary = {}
    monkeypatch.setattr(cli, "load_site", lambda *a, **k: site)
    monkeypatch.setattr(cli, "CommandRunner", lambda: world)
    monkeypatch.setattr(bootstrap, "discover_cluster", lambda *a, **k: world.gpu)

    @contextmanager
    def materialized(_site):
        yield world.root / "release.json"

    monkeypatch.setattr(cli, "materialized_release_config", materialized)
    monkeypatch.setattr(cli, "effective_environment", lambda _: {})
    calls = []

    def driver(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(cli, "run_driver", driver)
    arguments = cli.parser().parse_args(
        ["preflight", "--state-dir", str(world.state_dir)]
    )
    before = len(world.authorities.calls)
    helpers = world.helper_calls
    if ready:
        assert cli.run(arguments) == 0
        assert calls and calls[0][1] == "preflight"
    else:
        with pytest.raises(CustodyReconciliationRequired, match="public preflight"):
            cli.run(arguments)
        assert calls == []
        assert not (world.state_dir / "node-key-custody/preparations").exists(), (
            "public preflight must not create custody preparations"
        )
    assert world.helper_calls == helpers
    assert not any(
        call[:3] == ["aws", "kms", "sign"] for call in world.authorities.calls[before:]
    ), "public preflight must not invoke KMS signing"
