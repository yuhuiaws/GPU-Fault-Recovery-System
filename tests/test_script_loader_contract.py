"""Pin the identity guarantees tests rely on when loading deploy/ scripts.

An earlier loader let one file become several module objects. Patches applied to
one copy left the other live, and the live copy shelled out to a real
``kubectl`` from a unit test. These assertions are the tripwire for that
regression, so they check object identity rather than behaviour.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tests._script_loader import lazy_script_module, load_script_module

ROOT = Path(__file__).resolve().parents[1]
REGIONAL = ROOT / "deploy/control-plane/regional"
NODE_RUNTIME_ROLLOUT = REGIONAL / "regional_release_node_runtime_rollout.py"
FLEET_ROLLOUT = REGIONAL / "regional_release_fleet_rollout.py"


def test_repeated_loads_return_one_module_object() -> None:
    first = load_script_module(NODE_RUNTIME_ROLLOUT)
    second = load_script_module(NODE_RUNTIME_ROLLOUT)
    assert first is second, "each load re-executed the file instead of caching it"


def test_lazy_and_eager_loads_agree() -> None:
    eager = load_script_module(FLEET_ROLLOUT)
    lazy = lazy_script_module(FLEET_ROLLOUT)
    assert lazy.load() is eager, "the lazy proxy forked a second copy of the file"


def test_module_is_registered_under_the_name_siblings_import() -> None:
    module = load_script_module(FLEET_ROLLOUT)
    assert sys.modules["regional_release_fleet_rollout"] is module, (
        "the module is not reachable by the bare name its siblings import"
    )


def test_imported_function_belongs_to_the_loaded_sibling() -> None:
    """``MODULE.f`` and the ``f`` the script calls must be one object.

    ``regional_release_node_runtime_rollout`` does
    ``from regional_release_fleet_rollout import run_fleet_waves``. If loading
    the fleet module directly produced a second copy, patching that copy would
    silently not affect the function the rollout actually calls.
    """
    rollout = load_script_module(NODE_RUNTIME_ROLLOUT)
    fleet = load_script_module(FLEET_ROLLOUT)
    assert rollout.run_fleet_waves is fleet.run_fleet_waves, (
        "the rollout calls a different run_fleet_waves than tests can patch"
    )
    assert rollout.run_fleet_waves.__globals__ is fleet.__dict__, (
        "monkeypatch.setattr on the fleet module would not reach run_fleet_waves"
    )


def test_lazy_proxy_forwards_attribute_writes_to_the_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``monkeypatch.setattr(PROXY, ...)`` must reach the script's own global.

    Tests patch collaborators through the proxy. If a write stopped at the proxy
    the script would keep calling the real collaborator and the assertion would
    pass against an unpatched code path.
    """
    proxy = lazy_script_module(FLEET_ROLLOUT)
    module = load_script_module(FLEET_ROLLOUT)
    sentinel = object()

    monkeypatch.setattr(proxy, "next_deployment_wave", sentinel)
    assert module.next_deployment_wave is sentinel, (
        "the patch landed on the proxy instead of the loaded module"
    )
    assert module.run_fleet_waves.__globals__["next_deployment_wave"] is sentinel, (
        "the patch did not reach the globals run_fleet_waves resolves against"
    )

    monkeypatch.undo()
    assert module.next_deployment_wave is not sentinel, (
        "undo left the patched collaborator in place for later tests"
    )


def test_two_files_cannot_claim_one_module_name(tmp_path: Path) -> None:
    """A name collision must fail loudly rather than shadow the real script."""
    impostor = tmp_path / "regional_release_fleet_rollout.py"
    impostor.write_text("run_fleet_waves = None\n", encoding="utf-8")
    load_script_module(FLEET_ROLLOUT)

    with pytest.raises(RuntimeError, match="already bound to"):
        load_script_module(impostor)
