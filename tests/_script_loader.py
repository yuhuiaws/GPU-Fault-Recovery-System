"""Import path-only repository scripts as modules for tests.

A few ``scripts/*.py`` files are executable scripts rather than part of an
installed package. They import each other by bare module name when run as
files (``from release_identity import file_set_identity`` behind an
``if __package__`` guard), so tests cannot reach them through a package
import. The release orchestrator used to be loaded here too; it is the
``gpu_fault_release`` package now and tests import it directly.

Two invariants hold here, and both exist because the earlier alias-based loader
broke them:

* One physical file maps to exactly one module object for the whole session.
  ``load_script_module`` used to build a fresh ``LazyScriptModule`` per call, so
  its instance cache never hit and every call re-executed the file. Two callers
  got two module objects whose functions had separate ``__globals__``.
* The module is registered under the bare name its siblings import it by.
  Registering a file under a test-local alias created a *second* copy that
  ``from regional_release_fleet_rollout import run_fleet_waves`` inside the
  other scripts could not see, so ``MODULE.run_fleet_waves`` and the function
  the script actually called were different objects. A test that patched one
  copy left the other live, and the live one shelled out to a real ``kubectl``.

Callers therefore pass only a path. There is no argument that can ask for a
second copy of a file.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

_MODULES: dict[Path, ModuleType] = {}


def _module_name(path: Path) -> str:
    """Bare name the sibling scripts import this file by."""
    return path.stem.replace("-", "_")


def _adopt_existing(name: str, resolved: Path) -> ModuleType | None:
    """Return the already-imported module for ``resolved``, if there is one.

    A sibling script's own ``from regional_release_x import ...`` imports
    ``regional_release_x`` before any test asks for it directly. Adopting that
    module keeps a single copy; re-executing the file would fork it.
    """
    existing = sys.modules.get(name)
    if existing is None:
        return None
    existing_file = getattr(existing, "__file__", None)
    if existing_file is not None and Path(existing_file).resolve() == resolved:
        return existing
    raise RuntimeError(
        f"script module name {name!r} is already bound to {existing_file!r}; "
        f"it cannot also load {resolved}"
    )


def _load(path: Path) -> ModuleType:
    resolved = path.resolve()
    cached = _MODULES.get(resolved)
    if cached is not None:
        return cached

    name = _module_name(resolved)
    adopted = _adopt_existing(name, resolved)
    if adopted is not None:
        _MODULES[resolved] = adopted
        return adopted

    spec = importlib.util.spec_from_file_location(name, resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load test script module {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    previous = sys.dont_write_bytecode
    # The siblings are imported by bare name while this file executes, so the
    # directory has to be importable for the duration. Nothing imports a
    # sibling lazily, so the entry does not need to outlive the exec.
    script_directory = str(resolved.parent)
    sys.path.insert(0, script_directory)
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    finally:
        sys.dont_write_bytecode = previous
        sys.path.remove(script_directory)
    shadowed = sorted(_PROXY_ATTRIBUTES.intersection(vars(module)))
    if shadowed:
        raise RuntimeError(
            f"{resolved} defines {', '.join(shadowed)} at module level, which the "
            f"lazy proxy would shadow; rename the proxy attribute in "
            f"tests/_script_loader.py"
        )
    _MODULES[resolved] = module
    return module


#: Names the proxy itself answers, so they never reach the loaded module. A
#: script that defined one of these at module level would be shadowed, so
#: ``_load`` rejects the collision instead of letting it pass silently.
_PROXY_ATTRIBUTES = frozenset({"load", "name", "path"})


class LazyScriptModule:
    """Defer loading a path-only repository script until first attribute use.

    Attribute *writes* are forwarded to the loaded module as well, so
    ``monkeypatch.setattr(MODULE, "DIST", tmp_path)`` reaches the global the
    script itself reads. Without that forwarding a patch would land on the proxy
    and silently do nothing, which is what pushed earlier tests into patching
    ``MODULE.some_function.__globals__`` by hand.
    """

    def __init__(self, path: Path) -> None:
        object.__setattr__(self, "path", path)

    @property
    def name(self) -> str:
        return _module_name(self.path.resolve())

    def load(self) -> ModuleType:
        """Return the loaded module, for tests that need the module itself."""
        return _load(self.path)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.load(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _PROXY_ATTRIBUTES:
            raise AttributeError(f"{name!r} belongs to the loader, not to the script")
        setattr(self.load(), name, value)

    def __delattr__(self, name: str) -> None:
        if name in _PROXY_ATTRIBUTES:
            raise AttributeError(f"{name!r} belongs to the loader, not to the script")
        delattr(self.load(), name)


def lazy_script_module(path: Path) -> LazyScriptModule:
    """Proxy that loads ``path`` on first attribute access."""
    return LazyScriptModule(path)


def load_script_module(path: Path) -> ModuleType:
    """Load ``path`` now, returning the session's single module object for it."""
    return _load(path)
