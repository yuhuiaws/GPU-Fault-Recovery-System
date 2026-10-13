"""Providers are compared only through ``cluster_scope.cluster_provider``.

A HyperPod cluster's provider is written nowhere: ``site.yaml`` and the
rendered release config carry ``provider`` only when it differs from the
default (``admin/site.py``, so a site deployed before plain EKS support
renders byte for byte as before), the control-plane registry leaves the
default out of its digests and an absent ``GPU_FAULT_CLUSTER_PROVIDER`` means
``hyperpod``. ``cluster_provider()`` is the one parser: absent or empty means
``hyperpod``, an unknown spelling fails closed. Two records' raw values can
therefore disagree about one cluster -- absent, ``None`` and ``"hyperpod"``
compare unequal -- so the check that keeps ``status`` from reporting a
re-joined cluster as a stale ownership claim
(``ownership_release.deregistered_clusters``) compares normalized describe
targets.

This AST walk over ``src/`` flags, in ``==``, ``!=``, ``in`` and ``not in``:

* two raw provider reads compared with each other, whatever default either
  read applies;
* a raw read compared with the HyperPod default (``"hyperpod"``,
  ``ClusterProvider.HYPERPOD`` or its ``.value``) or with a normalized
  provider (``cluster_provider(...)``), unless the read itself maps an absent
  value to ``hyperpod`` (``.get(KEY, "hyperpod")``, ``x or "hyperpod"``).

A raw read is a ``provider`` attribute, a ``["provider"]`` subscript, a
``.get("provider", ...)`` or ``getattr(x, "provider", ...)``, the same reads of
``GPU_FAULT_CLUSTER_PROVIDER`` (``environ``, ``getenv``), also inside
``str()``, ``.strip()`` or ``.lower()``, and a local name every binding of
which in its function is such a read; a module's string constants and the
first-party names it imports are resolved. A value passed through
``cluster_provider()`` or ``ClusterProvider()`` is normalized. A comparison
with ``"eks-managed"`` passes: an absent value already falls on the HyperPod
side.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator, Mapping
from functools import cache
from pathlib import Path

import pytest

from gpu_fault.cluster_scope import PROVIDER_ENV, ClusterProvider

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
FIRST_PARTY = frozenset({"gpu_fault", "gpu_fault_release"})
PROVIDER_KEYS = frozenset({"provider", PROVIDER_ENV})
HYPERPOD = ClusterProvider.HYPERPOD.value
NORMALIZERS = frozenset({"cluster_provider", ClusterProvider.__name__})
TEXT_METHODS = frozenset({"strip", "lower", "casefold"})
FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
RAW = "raw"
DEFAULTED = "defaulted"
#: Comparisons that stay raw, keyed by file, enclosing scope and the
#: comparison's source text (``ast.unparse``), each with why it cannot
#: mis-compare. Their operand is normalized where it is built, so a rewrite
#: would change no behaviour -- but ``admin/site.py``, ``regional.py`` and
#: ``regional_registry.py`` ship in the control-plane wheel, where any edit
#: rolls the control plane.
ALLOWED: dict[tuple[str, str, str], str] = {
    (
        "src/gpu_fault/admin/bootstrap_site.py",
        "preserve_existing_site_contract",
        "previous.get('provider', 'hyperpod') != item.get('provider', 'hyperpod')",
    ): (
        "two site.yaml documents read with the same absent-means-hyperpod "
        "default; a hand-written null or empty provider is refused as a provider "
        "change (fail closed)"
    ),
    (
        "src/gpu_fault/admin/site.py",
        "load_site",
        "cluster.provider != ClusterProvider.HYPERPOD.value",
    ): (
        "GpuClusterSiteConfig.provider is cluster_provider(...).value, set by "
        "from_value, its only constructor"
    ),
    (
        "src/gpu_fault/regional.py",
        "_registry_digest_payload",
        "payload.get(_PROVIDER_DIGEST_KEY) == _DEFAULT_PROVIDER",
    ): (
        "a RegionalClusterRegistration dump; the provider field validator "
        "stores cluster_provider(value).value"
    ),
    (
        "src/gpu_fault/regional_registry.py",
        "regional_registry_config_sha256",
        "item.provider != 'hyperpod'",
    ): (
        "a RegionalClusterRegistration; the provider field validator stores "
        "cluster_provider(value).value"
    ),
}
_CONSTANTS: dict[str, dict[str, str]] = {}


def module_name(path: Path) -> str:
    relative = path.relative_to(SOURCE).with_suffix("")
    parts = relative.parts[:-1] if relative.name == "__init__" else relative.parts
    return ".".join(parts)


def source_files() -> list[Path]:
    return sorted(
        path for path in SOURCE.rglob("*.py") if "__pycache__" not in path.parts
    )


def module_constants(module: str) -> Mapping[str, str]:
    """``module``'s module-level string constants (a cycle sees an empty table)."""

    if module not in _CONSTANTS:
        _CONSTANTS[module] = {}
        base = SOURCE.joinpath(*module.split("."))
        for path in (base.with_suffix(".py"), base / "__init__.py"):
            if path.is_file():
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                _CONSTANTS[module] = string_constants(tree, module)
                break
    return _CONSTANTS[module]


def string_constants(tree: ast.Module, module: str = "") -> dict[str, str]:
    """Module-level ``NAME = "..."`` bindings and first-party imports of them."""

    table: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            source = node.module or ""
            if node.level or source.partition(".")[0] not in FIRST_PARTY:
                continue
            if source == module:
                continue
            imported = module_constants(source)
            table.update(
                (alias.asname or alias.name, imported[alias.name])
                for alias in node.names
                if alias.name in imported
            )
            continue
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        else:
            continue
        if (
            isinstance(target, ast.Name)
            and isinstance(value, ast.Constant)
            and isinstance(value.value, str)
        ):
            table[target.id] = value.value
    return table


def _callee(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


class ProviderReads:
    """Classifies the operands of one scope's comparisons."""

    def __init__(
        self, constants: Mapping[str, str], names: Mapping[str, str] | None = None
    ) -> None:
        self.constants = constants
        #: The scope's local names bound only to provider reads (``local_reads``).
        self.names = names or {}

    def text(self, node: ast.expr) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return self.constants.get(node.id)
        return None

    def is_hyperpod(self, node: ast.expr) -> bool:
        """``"hyperpod"``, ``ClusterProvider.HYPERPOD`` or its ``.value``."""

        if self.text(node) == HYPERPOD:
            return True
        if isinstance(node, ast.Attribute) and node.attr == "value":
            node = node.value
        return (
            isinstance(node, ast.Attribute)
            and node.attr == ClusterProvider.HYPERPOD.name
            and isinstance(node.value, ast.Name)
            and node.value.id == ClusterProvider.__name__
        )

    def is_normalized(self, node: ast.expr) -> bool:
        if isinstance(node, ast.Attribute) and node.attr == "value":
            node = node.value
        return isinstance(node, ast.Call) and _callee(node) in NORMALIZERS

    def read(self, node: ast.expr) -> str | None:
        """``RAW`` or ``DEFAULTED`` when ``node`` reads a provider, else ``None``."""

        if isinstance(node, ast.Name):
            return self.names.get(node.id)
        if isinstance(node, ast.Attribute):
            return RAW if node.attr == "provider" else None
        if isinstance(node, ast.Subscript):
            return RAW if self.text(node.slice) in PROVIDER_KEYS else None
        if isinstance(node, ast.BoolOp):
            if not isinstance(node.op, ast.Or) or len(node.values) != 2:
                return None
            kind = self.read(node.values[0])
            if kind is not None and self.is_hyperpod(node.values[1]):
                return DEFAULTED
            return kind
        if not isinstance(node, ast.Call):
            return None
        callee = _callee(node)
        if callee == "str" and len(node.args) == 1:
            return self.read(node.args[0])
        if callee in TEXT_METHODS and isinstance(node.func, ast.Attribute):
            return None if node.args else self.read(node.func.value)
        if callee in {"get", "getenv"} and node.args:
            key, default = node.args[0], node.args[1:2]
        elif callee == "getattr" and len(node.args) >= 2:
            key, default = node.args[1], node.args[2:3]
        else:
            return None
        if self.text(key) not in PROVIDER_KEYS:
            return None
        return DEFAULTED if default and self.is_hyperpod(default[0]) else RAW


def _members(node: ast.expr) -> list[ast.expr]:
    """What ``in`` tests against: a literal's items, a comprehension's element."""

    if isinstance(node, (ast.Set, ast.List, ast.Tuple)):
        return list(node.elts)
    if isinstance(node, (ast.SetComp, ast.ListComp, ast.GeneratorExp)):
        return [node.elt]
    return [node]


def comparison_problem(node: ast.Compare, reads: ProviderReads) -> str | None:
    """Why ``node`` compares providers raw, or ``None``."""

    left = node.left
    for operator, right in zip(node.ops, node.comparators, strict=True):
        if isinstance(operator, (ast.Eq, ast.NotEq)):
            pairs = [(left, right)]
        elif isinstance(operator, (ast.In, ast.NotIn)):
            pairs = [(left, item) for item in _members(right)]
        else:
            pairs = []
        for first, second in pairs:
            kinds = reads.read(first), reads.read(second)
            if kinds[0] is not None and kinds[1] is not None:
                return "two raw provider reads"
            for kind, other in ((kinds[0], second), (kinds[1], first)):
                if kind == RAW and (
                    reads.is_hyperpod(other) or reads.is_normalized(other)
                ):
                    return "a raw provider read against the HyperPod default"
        left = right
    return None


def _scope_nodes(scope: ast.AST) -> Iterator[ast.AST]:
    """The nodes of one function body, nested functions and classes excluded."""

    pending = list(ast.iter_child_nodes(scope))
    while pending:
        node = pending.pop()
        yield node
        if not isinstance(node, (*FUNCTIONS, ast.ClassDef, ast.Lambda)):
            pending.extend(ast.iter_child_nodes(node))


def local_reads(
    function: ast.FunctionDef | ast.AsyncFunctionDef, reads: ProviderReads
) -> dict[str, str]:
    """The local names every binding of which in ``function`` is a provider read.

    Only plain ``name = <read>`` bindings count; a name also bound another way
    (a parameter, a loop or ``with`` target, unpacking) is never tracked.
    """

    kinds: dict[str, list[str | None]] = {}
    plain: set[int] = set()
    for node in _scope_nodes(function):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        else:
            continue
        if isinstance(target, ast.Name):
            plain.add(id(target))
            kinds.setdefault(target.id, []).append(reads.read(value))
    arguments = function.args
    other = {
        item.arg
        for item in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
            *(item for item in (arguments.vararg, arguments.kwarg) if item),
        )
    }
    other.update(
        node.id
        for node in _scope_nodes(function)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Store)
        and id(node) not in plain
    )
    return {
        name: RAW if RAW in values else DEFAULTED
        for name, values in kinds.items()
        if name not in other and all(values)
    }


class _Comparisons(ast.NodeVisitor):
    """Every comparison, its enclosing ``Class.function`` scope and the local
    provider reads of the innermost function."""

    def __init__(self, reads: ProviderReads) -> None:
        self.reads = reads
        self.scope: list[tuple[str, Mapping[str, str]]] = []
        self.found: list[tuple[str, Mapping[str, str], ast.Compare]] = []

    def _enter(self, node: ast.AST, name: str, names: Mapping[str, str]) -> None:
        self.scope.append((name, names))
        self.generic_visit(node)
        self.scope.pop()

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._enter(node, node.name, local_reads(node, self.reads))

    visit_FunctionDef = _function
    visit_AsyncFunctionDef = _function

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._enter(node, node.name, {})

    def visit_Compare(self, node: ast.Compare) -> None:
        scope = ".".join(name for name, _names in self.scope) or "<module>"
        names = self.scope[-1][1] if self.scope else {}
        self.found.append((scope, names, node))
        self.generic_visit(node)


def raw_provider_comparisons(
    tree: ast.Module, module: str = ""
) -> Iterator[tuple[str, ast.Compare, str]]:
    """``(scope, comparison, problem)`` for each raw provider comparison."""

    constants = string_constants(tree, module)
    visitor = _Comparisons(ProviderReads(constants))
    visitor.visit(tree)
    for scope, names, node in visitor.found:
        problem = comparison_problem(node, ProviderReads(constants, names))
        if problem is not None:
            yield scope, node, problem


@cache
def tree_comparisons() -> dict[tuple[str, str, str], str]:
    """Every raw provider comparison under ``src/``: key -> where and why."""

    found: dict[tuple[str, str, str], str] = {}
    for path in source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative = path.relative_to(ROOT).as_posix()
        for scope, node, problem in raw_provider_comparisons(tree, module_name(path)):
            found[(relative, scope, ast.unparse(node))] = (
                f"line {node.lineno}: {problem}"
            )
    return found


def _flagged(source: str) -> list[str]:
    return [
        problem
        for _scope, _node, problem in raw_provider_comparisons(ast.parse(source))
    ]


#: Fixture snippets the guard must flag.
FLAGGED = {
    "two-gets": 'a.get("provider") == b.get("provider")',
    "subscript-vs-attribute": 'left["provider"] != right.provider',
    "two-defaulted-gets": (
        'a.get("provider", "hyperpod") != b.get("provider", "hyperpod")'
    ),
    "in-comprehension": "target.provider in {item.provider for item in clusters}",
    "in-tuple-of-reads": 'target["provider"] not in (a.provider, b.provider)',
    "get-vs-default": 'entry.get("provider") == "hyperpod"',
    "attribute-vs-enum-value": "item.provider != ClusterProvider.HYPERPOD.value",
    "attribute-vs-enum": "item.provider == ClusterProvider.HYPERPOD",
    "empty-default-vs-default": (
        'str(entry.get("provider") or "").strip() == "hyperpod"'
    ),
    "in-provider-names": 'entry["provider"] in ("hyperpod", "eks-managed")',
    "raw-vs-normalized": 'cluster_provider(a.get("provider")) == b.get("provider")',
    "getattr": 'getattr(target, "provider", None) == "hyperpod"',
    "environment": f'environ.get("{PROVIDER_ENV}", "") != "hyperpod"',
    "imported-environment-key": (
        "from gpu_fault.cluster_scope import PROVIDER_ENV\n"
        'os.getenv(PROVIDER_ENV) == "hyperpod"'
    ),
    "module-constants": (
        'KEY = "provider"\nDEFAULT = "hyperpod"\npayload.get(KEY) == DEFAULT'
    ),
    "local-names": (
        "def same(state, item):\n"
        '    recorded = state["target"].get("provider")\n'
        '    current = item.get("provider")\n'
        "    return recorded == current\n"
    ),
    "local-name-vs-default": (
        "def hyperpod(entry):\n"
        '    provider = entry.get("provider")\n'
        '    return provider != "hyperpod"\n'
    ),
}
#: Fixture snippets the guard must pass.
PASSING = {
    "both-normalized": (
        'cluster_provider(a.get("provider")) == cluster_provider(b.get("provider"))'
    ),
    "identity": "cluster_provider(a.provider) is cluster_provider(b.provider)",
    "normalized-value": 'cluster_provider(item.provider).value != "hyperpod"',
    "enum-constructor": 'ClusterProvider(a["provider"]) == ClusterProvider.HYPERPOD',
    "get-with-default": 'entry.get("provider", "hyperpod") != "hyperpod"',
    "or-default": 'str(entry.get("provider") or "hyperpod").strip() == "hyperpod"',
    "imported-environment-key-with-default": (
        "from gpu_fault.cluster_scope import PROVIDER_ENV\n"
        'environment.get(PROVIDER_ENV, "hyperpod") != "hyperpod"'
    ),
    "eks-managed": 'target.provider == "eks-managed"',
    "eks-managed-enum": 'entry.get("provider") != ClusterProvider.EKS_MANAGED.value',
    "enum-member-value": 'groups.provider.value == "eks-managed"',
    "other-attribute": "a.provider_name == b.provider_name",
    "other-key": 'item.get("region") == other.get("region")',
    "resource-provider": 'resource.provider != "aws"',
    "local-name-normalized-again": (
        "def hyperpod(entry):\n"
        '    provider = entry.get("provider")\n'
        "    provider = cluster_provider(provider)\n"
        "    return provider == ClusterProvider.HYPERPOD\n"
    ),
    "local-name-with-default": (
        "def eks(values):\n"
        '    provider = str(values.get("provider") or "hyperpod").strip()\n'
        '    return provider == "hyperpod"\n'
    ),
    "parameter": 'def hyperpod(provider):\n    return provider == "hyperpod"\n',
}


@pytest.mark.parametrize("source", list(FLAGGED.values()), ids=list(FLAGGED))
def test_the_guard_flags_a_raw_provider_comparison(source: str) -> None:
    assert len(_flagged(source)) == 1, source


@pytest.mark.parametrize("source", list(PASSING.values()), ids=list(PASSING))
def test_the_guard_passes_a_normalized_or_eks_managed_comparison(source: str) -> None:
    assert _flagged(source) == [], source


def test_providers_are_compared_only_through_cluster_provider() -> None:
    offenders = sorted(
        f"{path} {scope} {where}: {text}"
        for (path, scope, text), where in tree_comparisons().items()
        if (path, scope, text) not in ALLOWED
    )
    assert offenders == [], (
        "compare providers through gpu_fault.cluster_scope.cluster_provider(): "
        "absent, None and 'hyperpod' are the same provider\n" + "\n".join(offenders)
    )


def test_every_allowed_comparison_still_exists_and_says_why() -> None:
    stale = sorted(set(ALLOWED) - set(tree_comparisons()))
    assert stale == [], f"allowlist entries that no longer match: {stale}"
    assert all(len(reason) > 40 for reason in ALLOWED.values()), ALLOWED
    assert len(source_files()) > 500, "the scan must cover the whole src tree"
