"""Pin the identity guarantees tests rely on when loading scripts/ by path.

An earlier loader let one file become several module objects. Patches applied to
one copy left the other live, and the live copy shelled out to a real
``kubectl`` from a unit test. These assertions are the tripwire for that
regression, so they check object identity rather than behaviour.

The release orchestrator used to be the loader's main customer; it is the
``gpu_fault_release`` package now and is imported directly. What remains are the
``scripts/*.py`` files that import each other by bare name when run as files
(``from release_identity import file_set_identity`` behind ``if __package__``),
so the sibling pair here is ``deploy_source_identity`` -> ``release_identity``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tests._script_loader import lazy_script_module, load_script_module

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SOURCE_IDENTITY = SCRIPTS / "deploy_source_identity.py"
RELEASE_IDENTITY = SCRIPTS / "release_identity.py"


def test_repeated_loads_return_one_module_object() -> None:
    first = load_script_module(SOURCE_IDENTITY)
    second = load_script_module(SOURCE_IDENTITY)
    assert first is second, "each load re-executed the file instead of caching it"


def test_lazy_and_eager_loads_agree() -> None:
    eager = load_script_module(RELEASE_IDENTITY)
    lazy = lazy_script_module(RELEASE_IDENTITY)
    assert lazy.load() is eager, "the lazy proxy forked a second copy of the file"


def test_module_is_registered_under_the_name_siblings_import() -> None:
    module = load_script_module(RELEASE_IDENTITY)
    assert sys.modules["release_identity"] is module, (
        "the module is not reachable by the bare name its siblings import"
    )


def test_imported_function_belongs_to_the_loaded_sibling() -> None:
    """``MODULE.f`` and the ``f`` the script calls must be one object.

    ``deploy_source_identity`` does ``from release_identity import
    file_set_identity`` when run as a file. If loading the identity module
    directly produced a second copy, patching that copy would silently not
    affect the function the script actually calls.
    """
    consumer = load_script_module(SOURCE_IDENTITY)
    identity = load_script_module(RELEASE_IDENTITY)
    assert consumer.file_set_identity is identity.file_set_identity, (
        "the script calls a different file_set_identity than tests can patch"
    )
    assert consumer.file_set_identity.__globals__ is identity.__dict__, (
        "monkeypatch.setattr on the identity module would not reach file_set_identity"
    )


def test_lazy_proxy_forwards_attribute_writes_to_the_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``monkeypatch.setattr(PROXY, ...)`` must reach the script's own global.

    Tests patch collaborators through the proxy. If a write stopped at the proxy
    the script would keep calling the real collaborator and the assertion would
    pass against an unpatched code path.
    """
    proxy = lazy_script_module(RELEASE_IDENTITY)
    module = load_script_module(RELEASE_IDENTITY)
    sentinel = object()

    monkeypatch.setattr(proxy, "canonical_sha256", sentinel)
    assert module.canonical_sha256 is sentinel, (
        "the patch landed on the proxy instead of the loaded module"
    )
    assert module.file_set_identity.__globals__["canonical_sha256"] is sentinel, (
        "the patch did not reach the globals file_set_identity resolves against"
    )

    monkeypatch.undo()
    assert module.canonical_sha256 is not sentinel, (
        "undo left the patched collaborator in place for later tests"
    )


def test_two_files_cannot_claim_one_module_name(tmp_path: Path) -> None:
    """A name collision must fail loudly rather than shadow the real script."""
    impostor = tmp_path / "release_identity.py"
    impostor.write_text("file_set_identity = None\n", encoding="utf-8")
    load_script_module(RELEASE_IDENTITY)

    with pytest.raises(RuntimeError, match="already bound to"):
        load_script_module(impostor)
