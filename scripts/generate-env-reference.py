from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "gpu_fault"
OUTPUT = ROOT / "docs" / "环境变量参考.md"
ADMIN_OUTPUT = ROOT / "docs" / "管理员环境变量参考.md"
ADMIN_SPEC = ROOT / "scripts" / "admin-env-reference.yaml"
INVENTORY_OUTPUT = ROOT / "src" / "gpu_fault" / "data" / "env-inventory.json"
DEPLOY_ENV_SOURCES = (
    ROOT / "deploy" / "node" / "install-gpu-fault-collector.sh",
    ROOT / "deploy" / "node" / "verify-gpu-fault-collector.sh",
    *(ROOT / "deploy" / "systemd").glob("*"),
)
REGIONAL_GENERATED = ROOT / "deploy" / "control-plane" / "regional" / "generated"
REGIONAL_ROLE_FILES = {
    "ingress": REGIONAL_GENERATED / "gpu-fault-api-ha-ingress.yaml",
    "worker": REGIONAL_GENERATED / "gpu-fault-control-worker.yaml",
    "spool": REGIONAL_GENERATED / "gpu-fault-telemetry-spool-worker.yaml",
}
NAME_PATTERN = re.compile(r"\bGPU_FAULT_[A-Z0-9_]+\b")
# Value kinds are inferred from the coercion each read site applies, so the
# schema cannot drift from the code that consumes it. Only unambiguous coercions
# are recorded: an unrecognised read shape leaves the variable untyped, which
# means unvalidated. A wrong kind would refuse a legal production value, so the
# inference stays conservative in that direction.
BOOLEAN_TOKENS = frozenset({"0", "1", "true", "false", "yes", "no", "on", "off"})
TRUE_TOKENS = frozenset({"1", "true", "yes", "on"})
STRING_METHODS = frozenset({"strip", "lower", "upper", "casefold"})
NUMERIC_KINDS = {"int": "integer", "float": "number"}
KIND_LABELS = {
    "integer": "整数",
    "number": "数值",
    "boolean": "布尔",
}
# `os.environ`, and the mapping parameter names the settings/validation helpers
# use for it. Restricting the receivers keeps an ordinary dictionary lookup from
# being read as an environment read.
ENVIRONMENT_RECEIVERS = frozenset({"environ", "values", "environment", "env"})
GUARD_EXCEPTIONS = frozenset({"ValueError", "TypeError", "Exception"})
DYNAMIC_PREFIXES = ("GPU_FAULT_NOTIFICATION_TTL_SECONDS_",)
NON_RUNTIME_PREFIXES = ("GPU_FAULT_TEST_", "GPU_FAULT_PERF_")
DYNAMIC_VARIABLES = tuple(
    f"GPU_FAULT_PROCESSOR_{prefix}ADMISSION_{suffix}"
    for prefix in ("", "FAULT_", "EVIDENCE_")
    for suffix in (
        "BATCH_SIZE",
        "BATCH_GROUPS",
        "BATCH_DELAY_SECONDS",
        "PROJECTION_MARGIN",
    )
)


def concrete_name(name: str) -> bool:
    return (
        bool(NAME_PATTERN.fullmatch(name))
        and not name.endswith("_")
        and not name.startswith(NON_RUNTIME_PREFIXES)
    )


def module_name(path: Path) -> str:
    relative = path.relative_to(SOURCE).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return "gpu_fault." + ".".join(parts)


def deployment_source_name(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _fstring_literal(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("{", "{{")
        .replace("}", "}}")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def _fstring_body(node: ast.JoinedStr) -> str:
    parts: list[str] = []
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            parts.append(_fstring_literal(value.value))
            continue
        if not isinstance(value, ast.FormattedValue):
            parts.append(ast.unparse(value))
            continue
        formatted = "{" + ast.unparse(value.value)
        if value.conversion != -1:
            formatted += f"!{chr(value.conversion)}"
        if value.format_spec is not None:
            formatted += ":" + _fstring_body(value.format_spec)
        parts.append(formatted + "}")
    return "".join(parts)


def expression(node: ast.AST | None) -> str:
    if node is None:
        return ""
    if isinstance(node, ast.JoinedStr):
        return f'f"{_fstring_body(node)}"'
    try:
        return ast.unparse(node).replace("\n", " ")
    except Exception:
        return ""


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _is_environment_mapping(node: ast.AST) -> bool:
    if isinstance(node, ast.Attribute):
        return node.attr == "environ"
    return isinstance(node, ast.Name) and node.id in ENVIRONMENT_RECEIVERS


def _read_key(node: ast.AST) -> str | None:
    """The environment key an expression reads, if it reads one at all.

    A returned name that is not a concrete ``GPU_FAULT_*`` variable is the
    parameter of a helper such as ``settings._boolean``; the call sites supply the
    concrete names for those.
    """

    key: ast.AST | None = None
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.args
    ):
        if node.func.attr == "getenv" or (
            node.func.attr == "get" and _is_environment_mapping(node.func.value)
        ):
            key = node.args[0]
    elif isinstance(node, ast.Subscript) and _is_environment_mapping(node.value):
        key = node.slice
    if isinstance(key, ast.Constant) and isinstance(key.value, str):
        return key.value if concrete_name(key.value) else None
    if isinstance(key, ast.Name):
        return key.id
    return None


def _passthrough_base(node: ast.AST) -> ast.AST:
    while (
        isinstance(node, ast.Call)
        and not node.args
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in STRING_METHODS
    ):
        node = node.func.value
    return node


def _guarded_by_except(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    """Whether a coercion sits under a ``try`` that swallows a bad value.

    Such a site accepts unparsable text by design, so the variable must stay
    untyped: validating it would reject a value the code deliberately tolerates.
    """

    child = node
    parent = parents.get(child)
    while parent is not None:
        if isinstance(parent, ast.Try) and any(item is child for item in parent.body):
            for handler in parent.handlers:
                if handler.type is None:
                    return True
                caught = (
                    handler.type.elts
                    if isinstance(handler.type, ast.Tuple)
                    else [handler.type]
                )
                if any(
                    isinstance(item, ast.Name) and item.id in GUARD_EXCEPTIONS
                    for item in caught
                ):
                    return True
        child, parent = parent, parents.get(parent)
    return False


def _string_tokens(node: ast.AST) -> frozenset[str] | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return frozenset({node.value.strip().lower()})
    if isinstance(node, (ast.Set, ast.List, ast.Tuple)):
        tokens = set()
        for element in node.elts:
            if not (
                isinstance(element, ast.Constant) and isinstance(element.value, str)
            ):
                return None
            tokens.add(element.value.strip().lower())
        return frozenset(tokens)
    return None


def _skip_string_methods(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> ast.AST:
    """Climb past ``.strip().lower()`` so the real coercion becomes visible.

    The parent of a call's receiver is the ``Attribute`` node, not the ``Call``,
    so both levels have to be stepped over.
    """

    current = node
    while True:
        attribute = parents.get(current)
        if not (
            isinstance(attribute, ast.Attribute)
            and attribute.value is current
            and attribute.attr in STRING_METHODS
        ):
            return current
        call = parents.get(attribute)
        if not (
            isinstance(call, ast.Call) and call.func is attribute and not call.args
        ):
            return current
        current = call


def _observe_use(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> tuple[str, frozenset[str]] | None:
    """Classify one use of an environment value by the coercion applied to it."""

    current = _skip_string_methods(node, parents)
    parent = parents.get(current)
    if (
        isinstance(parent, ast.Call)
        and isinstance(parent.func, ast.Name)
        and parent.func.id in NUMERIC_KINDS
        and parent.args
        and parent.args[0] is current
        and not _guarded_by_except(parent, parents)
    ):
        return NUMERIC_KINDS[parent.func.id], frozenset()
    if (
        isinstance(parent, ast.Compare)
        and parent.left is current
        and len(parent.ops) == 1
        and isinstance(parent.ops[0], (ast.Eq, ast.In))
    ):
        tokens = _string_tokens(parent.comparators[0])
        if tokens and tokens <= BOOLEAN_TOKENS:
            return "boolean", frozenset(tokens & TRUE_TOKENS)
    return None


def _scopes(tree: ast.Module):
    yield tree
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _aliases(scope: ast.AST) -> dict[str, ast.AST]:
    """Names assigned exactly once, so ``raw = os.getenv(...)`` stays traceable."""

    assigned: dict[str, ast.AST | None] = {}
    for node in ast.walk(scope):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            assigned[target.id] = None if target.id in assigned else node.value
    return {name: value for name, value in assigned.items() if value is not None}


def _keyed_nodes(scope: ast.AST) -> dict[ast.AST, str]:
    """Every expression in the scope that evaluates to an environment value."""

    keyed: dict[ast.AST, str] = {}
    for node in ast.walk(scope):
        key = _read_key(node)
        if key is not None:
            keyed[node] = key
    aliases = _aliases(scope)
    resolved: dict[str, str] = {}
    for _ in range(len(aliases) + 1):
        progressed = False
        for name, value in aliases.items():
            if name in resolved:
                continue
            base = _passthrough_base(value)
            key = keyed.get(base)
            if key is None and isinstance(base, ast.Name):
                key = resolved.get(base.id)
            if key is not None:
                resolved[name] = key
                progressed = True
        if not progressed:
            break
    if resolved:
        for node in ast.walk(scope):
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id in resolved
            ):
                keyed[node] = resolved[node.id]
    return keyed


def _module_observations(
    tree: ast.Module,
) -> tuple[
    dict[str, set[tuple[str, frozenset[str]]]],
    dict[tuple[str, int], set[tuple[str, frozenset[str]]]],
]:
    parents = _parent_map(tree)
    by_name: dict[str, set[tuple[str, frozenset[str]]]] = defaultdict(set)
    by_parameter: dict[tuple[str, int], set[tuple[str, frozenset[str]]]] = defaultdict(
        set
    )
    for scope in _scopes(tree):
        parameters: dict[str, int] = {}
        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            positional = [*scope.args.posonlyargs, *scope.args.args]
            parameters = {
                argument.arg: index for index, argument in enumerate(positional)
            }
        for node, key in _keyed_nodes(scope).items():
            observation = _observe_use(node, parents)
            if observation is None:
                continue
            if concrete_name(key):
                by_name[key].add(observation)
            elif key in parameters:
                by_parameter[(scope.name, parameters[key])].add(observation)
    return by_name, by_parameter


def _named_arguments(tree: ast.Module) -> dict[str, list[dict[int, str]]]:
    """Positional ``GPU_FAULT_*`` constants passed to each plain function name."""

    calls: dict[str, list[dict[int, str]]] = defaultdict(list)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        positions = {
            index: argument.value
            for index, argument in enumerate(node.args)
            if isinstance(argument, ast.Constant)
            and isinstance(argument.value, str)
            and concrete_name(argument.value)
        }
        if positions:
            calls[node.func.id].append(positions)
    return calls


def _merge_observations(
    observations: set[tuple[str, frozenset[str]]],
) -> tuple[str, frozenset[str]] | None:
    """Collapse every read site's verdict, preferring "unvalidated" over a guess."""

    labels = {label for label, _ in observations}
    if labels == {"boolean"}:
        return "boolean", frozenset().union(*(tokens for _, tokens in observations))
    if labels and labels <= {"integer", "number"}:
        # A float read accepts what an int read accepts, so the looser kind wins.
        return ("number" if "number" in labels else "integer"), frozenset()
    return None


def _target_path(node: ast.AST) -> str | None:
    """``limit`` or ``self.limit``: the two shapes a coerced read is stored in."""

    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return f"{node.value.id}.{node.attr}"
    return None


def _coerced_read(node: ast.AST, keyed: dict[ast.AST, str]) -> str | None:
    """The key of an expression that is *nothing but* a coerced read of it.

    A bound is only comparable with the configured text if the guarded value is
    the value the operator typed. ``int(os.getenv(SECONDS)) * 1000`` guarded at
    ``< 5000`` says nothing about the seconds an operator may write, so anything
    with arithmetic in it is refused here. The one wrapper that is allowed
    besides the coercion is the ``... if argument is None else argument``
    fallback every service constructor uses, because the env branch of it is
    still the raw value.
    """

    if isinstance(node, ast.IfExp):
        branches = [
            _coerced_read(node.body, keyed),
            _coerced_read(node.orelse, keyed),
        ]
        found = [item for item in branches if item is not None]
        return found[0] if len(found) == 1 else None
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in NUMERIC_KINDS
        and len(node.args) == 1
    ):
        node = node.args[0]
    node = _passthrough_base(node)
    return keyed.get(node)


def _bound_targets(scope: ast.AST, keyed: dict[ast.AST, str]) -> dict[str, str]:
    """Where in this function each environment value came to rest, if anywhere.

    A name assigned twice is dropped: the guard cannot be attributed to the
    environment read when something else may be in the variable by then.
    """

    seen: dict[str, str | None] = {}
    for node in ast.walk(scope):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        path = _target_path(node.targets[0])
        if path is None:
            continue
        key = _coerced_read(node.value, keyed)
        seen[path] = None if path in seen else key
    return {
        path: key
        for path, key in seen.items()
        if key is not None and concrete_name(key)
    }


def _literal_number(node: ast.AST) -> int | float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return None if isinstance(node.value, bool) else node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _literal_number(node.operand)
        return None if inner is None else -inner
    return None


# What each comparison operator says about the value when the comparison is the
# one that must *hold*. The guards read the other way round -- they raise on the
# rejected region -- so a non-negated guard is inverted before it gets here.
_HOLDING_BOUNDS = {
    ast.GtE: ("minimum", False),
    ast.Gt: ("minimum", True),
    ast.LtE: ("maximum", False),
    ast.Lt: ("maximum", True),
}
_INVERTED_OPS = {
    ast.Lt: ast.GtE,
    ast.LtE: ast.Gt,
    ast.Gt: ast.LtE,
    ast.GtE: ast.Lt,
}
_MIRRORED_OPS = {
    ast.Lt: ast.Gt,
    ast.LtE: ast.GtE,
    ast.Gt: ast.Lt,
    ast.GtE: ast.LtE,
}


def _compare_bounds(
    compare: ast.Compare,
    *,
    holds: bool,
) -> list[tuple[str, str, int | float, bool]]:
    """Bounds on a target implied by one comparison, as (path, side, value, strict)."""

    operands = [compare.left, *compare.comparators]
    bounds: list[tuple[str, str, int | float, bool]] = []
    for index, operator in enumerate(compare.ops):
        left, right = operands[index], operands[index + 1]
        kind = type(operator)
        path = _target_path(left)
        literal = _literal_number(right)
        if path is None or literal is None:
            # Mirror ``0 > limit`` into ``limit < 0`` before giving up.
            path, literal = _target_path(right), _literal_number(left)
            kind = _MIRRORED_OPS.get(kind, kind)
        if path is None or literal is None:
            continue
        if not holds:
            kind = _INVERTED_OPS.get(kind, kind)
        side = _HOLDING_BOUNDS.get(kind)
        if side is None:
            continue
        bounds.append((path, side[0], literal, side[1]))
    return bounds


def _statement_bounds(
    statement: ast.stmt,
) -> list[tuple[str, str, int | float, bool]]:
    """Bounds a top-level ``if ...: raise`` in a function body places on a target.

    Only a statement of the function body itself is read. A guard nested inside
    another branch is conditional on something this analysis cannot see, and a
    bound that only applies when some feature is switched on must not become a
    start-up refusal for every deployment.
    """

    if not isinstance(statement, ast.If):
        return []
    if not any(isinstance(item, ast.Raise) for item in statement.body):
        return []
    test, holds = statement.test, False
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        # ``if not 1 <= limit <= 100: raise`` states the accepted range directly.
        test, holds = test.operand, True
    if not isinstance(test, ast.Compare):
        return []
    if not holds and len(test.ops) != 1:
        # A chain that raises when it holds excludes an interval rather than
        # bounding one, which is not a range.
        return []
    return _compare_bounds(test, holds=holds)


def _module_bounds(tree: ast.Module) -> dict[str, list[tuple[str, int | float, bool]]]:
    """Every range its own read site already enforces, per environment variable."""

    bounds: dict[str, list[tuple[str, int | float, bool]]] = defaultdict(list)
    for scope in _scopes(tree):
        if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Module scope is skipped: its alias map spans every function in the
            # file, so a parameter that happens to share a name with a module
            # constant would collect the wrong bound.
            continue
        targets = _bound_targets(scope, _keyed_nodes(scope))
        if not targets:
            continue
        for statement in scope.body:
            for path, side, value, strict in _statement_bounds(statement):
                key = targets.get(path)
                if key is not None:
                    bounds[key].append((side, value, strict))
    return bounds


def _merge_bounds(
    observed: list[tuple[str, int | float, bool]],
    kind: str,
) -> dict[str, int | float]:
    """Collapse every read site's range into the widest one all of them reject.

    Two sites may guard the same variable differently -- one role needs at least
    30 seconds where another accepts 1 -- and refusing the stricter value at
    start-up would break the role that can use it. So the widest range wins, and
    start-up only rejects what no read site can use.

    A strict bound is only recorded for an integer, where "greater than 0" is
    exactly "at least 1". For a real number there is no next value to name, so
    the bound is dropped rather than approximated in either direction.
    """

    merged: dict[str, int | float] = {}
    for side, value, strict in observed:
        if strict:
            if kind != "integer" or not float(value).is_integer():
                continue
            value = int(value) + (1 if side == "minimum" else -1)
        current = merged.get(side)
        if current is None:
            merged[side] = value
        else:
            merged[side] = (
                min(current, value) if side == "minimum" else max(current, value)
            )
    if (
        "minimum" in merged
        and "maximum" in merged
        and merged["minimum"] > merged["maximum"]
    ):
        return {}
    return merged


def extract() -> tuple[
    dict[str, set[str]],
    dict[str, set[str]],
    dict[str, dict],
    dict[str, dict],
]:
    modules: dict[str, set[str]] = defaultdict(set)
    defaults: dict[str, set[str]] = defaultdict(set)
    observations: dict[str, set[tuple[str, frozenset[str]]]] = defaultdict(set)
    parameters: dict[tuple[str, int], set[tuple[str, frozenset[str]]]] = defaultdict(
        set
    )
    calls: dict[str, list[dict[int, str]]] = defaultdict(list)
    ranges: dict[str, list[tuple[str, int | float, bool]]] = defaultdict(list)
    for path in sorted(SOURCE.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        module = module_name(path)
        for name, observed in _module_bounds(tree).items():
            ranges[name].extend(observed)
        by_name, by_parameter = _module_observations(tree)
        for name, items in by_name.items():
            observations[name].update(items)
        for parameter, items in by_parameter.items():
            parameters[parameter].update(items)
        for function, positions in _named_arguments(tree).items():
            calls[function].extend(positions)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for name in NAME_PATTERN.findall(node.value):
                    if concrete_name(name):
                        modules[name].add(module)
            if not isinstance(node, ast.Call) or not node.args:
                continue
            first = node.args[0]
            if not (
                isinstance(first, ast.Constant)
                and isinstance(first.value, str)
                and concrete_name(first.value)
            ):
                continue
            name = first.value
            modules[name].add(module)
            if len(node.args) > 1:
                defaults[name].add(expression(node.args[1]))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Subscript):
                continue
            value = node.value
            if not (
                isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Name)
                and value.value.id == "os"
                and value.attr == "environ"
            ):
                continue
            key = node.slice
            if (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and concrete_name(key.value)
            ):
                modules[key.value].add(module)
                defaults[key.value].add("<required>")
    for name in DYNAMIC_VARIABLES:
        modules[name].add("gpu_fault.app.admission_runtime")
        defaults[name].add("<dynamic>")
    for path in sorted(DEPLOY_ENV_SOURCES):
        if not path.is_file():
            continue
        source = deployment_source_name(path)
        for name in NAME_PATTERN.findall(path.read_text(encoding="utf-8")):
            if concrete_name(name):
                modules[name].add(source)
    for (function, index), items in parameters.items():
        for positions in calls.get(function, ()):
            name = positions.get(index)
            if name is not None:
                observations[name].update(items)
    kinds: dict[str, dict] = {}
    for name in sorted(observations):
        merged = _merge_observations(observations[name])
        if merged is None or name not in modules:
            continue
        kind, tokens = merged
        entry: dict = {"kind": kind}
        if kind == "boolean" and tokens:
            entry["true_tokens"] = sorted(tokens)
        kinds[name] = entry
    bounds: dict[str, dict] = {}
    for name in sorted(ranges):
        kind = (kinds.get(name) or {}).get("kind", "")
        if kind not in {"integer", "number"} or name not in modules:
            # A range is meaningless without the coercion that makes the text a
            # number, and a variable the inference left untyped is deliberately
            # unvalidated.
            continue
        merged = _merge_bounds(ranges[name], kind)
        if merged:
            bounds[name] = merged
    return modules, defaults, kinds, bounds


def domain(name: str) -> str:
    groups = (
        (
            "PostgreSQL",
            ("GPU_FAULT_POSTGRES_", "GPU_FAULT_STORE_"),
        ),
        ("Processor", ("GPU_FAULT_PROCESSOR_",)),
        (
            "Agent/Fleet",
            ("GPU_FAULT_AGENT_", "GPU_FAULT_NODE_"),
        ),
        (
            "Notification",
            ("GPU_FAULT_NOTIFICATION_", "GPU_FAULT_SES_"),
        ),
        (
            "Telemetry",
            (
                "GPU_FAULT_GPU_",
                "GPU_FAULT_HOST_",
                "GPU_FAULT_EFA_",
                "GPU_FAULT_DCGM_",
                "GPU_FAULT_TELEMETRY_",
            ),
        ),
        ("HyperPod", ("GPU_FAULT_HYPERPOD_",)),
        ("Workflow", ("GPU_FAULT_WORKFLOW_",)),
    )
    for label, prefixes in groups:
        if name.startswith(prefixes):
            return label
    return "Core/Other"


def render() -> str:
    modules, defaults, kinds, _ = extract()
    lines = [
        "# 环境变量参考",
        "",
        "<!-- Generated by scripts/generate-env-reference.py; do not edit. -->",
        "",
        f"当前代码自动识别到 **{len(modules)}** 个 `GPU_FAULT_*` 配置项。",
        "修改代码中的配置项后必须重新生成本表。",
        "本表同时是生产进程的未知变量 allowlist，扫描范围为 `src/gpu_fault/**/*.py`、",
        "节点 install/verify 脚本和 systemd unit；不包含测试、性能驱动器或 fleet",
        "transport 临时变量。",
        "管理员请优先阅读按职责整理的[管理员环境变量参考](管理员环境变量参考.md)。",
        "",
        f"其中 **{len(kinds)}** 项的取值类型可以从代码里的转换方式唯一推断出来，进程启动时",
        "会校验（整数/数值必须可解析，布尔必须是 `0/1/true/false/yes/no/on/off`）；",
        "`文本` 表示没有唯一可推断的类型，只校验变量名是否已知。",
        "",
        "| 环境变量 | 领域 | 类型 | 默认值/要求 | 使用模块 |",
        "|---|---|---|---|---|",
    ]
    for name in sorted(modules):
        default = ", ".join(sorted(defaults.get(name, {"<implicit>"})))
        used = "<br>".join(f"`{item}`" for item in sorted(modules[name]))
        kind = KIND_LABELS.get((kinds.get(name) or {}).get("kind", ""), "文本")
        lines.append(f"| `{name}` | {domain(name)} | {kind} | `{default}` | {used} |")
    return "\n".join(lines) + "\n"


def render_inventory() -> str:
    modules, _, kinds, bounds = extract()
    return (
        json.dumps(
            {
                "schema_version": 3,
                "variables": sorted(modules),
                "value_kinds": kinds,
                "value_bounds": bounds,
                "dynamic_prefixes": list(DYNAMIC_PREFIXES),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _regional_role_from_config(path: Path) -> str:
    prefix = path.name.split("-config-", 1)[0]
    return {
        "gpu-fault-api-ha": "ingress",
        "gpu-fault-control-worker": "worker",
        "gpu-fault-telemetry-spool-worker": "spool",
    }[prefix]


def _value_from_label(value_from: dict) -> str:
    for kind, label in (
        ("secretKeyRef", "Secret"),
        ("configMapKeyRef", "ConfigMap"),
    ):
        reference = value_from.get(kind)
        if isinstance(reference, dict):
            return f"{label} {reference.get('name', '?')}/{reference.get('key', '?')}"
    field = value_from.get("fieldRef")
    if isinstance(field, dict):
        return f"fieldRef {field.get('fieldPath', '?')}"
    return "valueFrom"


def regional_values() -> dict[str, dict[str, str]]:
    values: dict[str, dict[str, str]] = defaultdict(dict)
    for role, path in REGIONAL_ROLE_FILES.items():
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        containers = document["spec"]["template"]["spec"]["containers"]
        for item in containers[0].get("env", []):
            name = item.get("name")
            if not isinstance(name, str):
                continue
            if "value" in item:
                values[name][role] = str(item["value"])
            elif isinstance(item.get("valueFrom"), dict):
                values[name][role] = _value_from_label(item["valueFrom"])
    for path in sorted(REGIONAL_GENERATED.glob("*-config-*.yaml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        role = _regional_role_from_config(path)
        for name, value in (document.get("data") or {}).items():
            values[str(name)][role] = str(value)
    return values


def _markdown(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _code(value: object) -> str:
    escaped = _markdown(value).replace("`", "\\`")
    return f"`{escaped}`"


def _regional_value(
    name: str,
    values: dict[str, dict[str, str]],
    override: object | None,
) -> str:
    if override is not None:
        return _markdown(override)
    by_role = values.get(name) or {}
    if not by_role:
        return "—"
    roles = ("ingress", "worker", "spool")
    if set(by_role) == set(roles) and len(set(by_role.values())) == 1:
        return _code(next(iter(by_role.values()))) + "（三角色）"
    return "<br>".join(
        f"{role}={_code(by_role[role])}" for role in roles if role in by_role
    )


def load_admin_spec() -> dict:
    document = yaml.safe_load(ADMIN_SPEC.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("admin environment reference schema_version must be 1")
    categories = document.get("categories")
    if not isinstance(categories, list) or not categories:
        raise ValueError("admin environment reference categories are required")
    return document


def render_admin() -> str:
    modules, defaults, _, _ = extract()
    spec = load_admin_spec()
    selected: set[str] = set()
    for category in spec["categories"]:
        for item in category.get("variables") or []:
            name = item.get("name")
            if name in selected:
                raise ValueError(f"duplicate admin environment variable: {name}")
            if name not in modules:
                raise ValueError(f"unknown admin environment variable: {name}")
            selected.add(name)

    values = regional_values()
    lines = [
        "# 管理员环境变量参考",
        "",
        "<!-- Generated by scripts/generate-env-reference.py; do not edit. -->",
        "",
        f"本文件从[环境变量参考](环境变量参考.md)的 **{len(modules)}** 个运行时配置中，",
        f"精选 **{len(selected)}** 个管理员需要理解或审批的变量，并按职责分类。",
        "完整机器清单仍是权威全集；本文不列测试、性能驱动器和纯内部传递变量。",
        "",
        "## 阅读约定",
        "",
        "- **区域清单值**来自 `deploy/control-plane/regional/generated/`；",
        "  `—` 表示该变量不由控制面角色 ConfigMap/Secret 直接注入，通常来自",
        "  `site.yaml`、节点安装器、数据面 Secret 或管理员命令。",
        "- **应用默认/要求**来自当前 Python 实现；`<required>` 表示缺失即失败，",
        "  `<implicit>` 表示通过上层配置、动态名称或校验器间接决定。",
        "- 取值类型能从代码里唯一推断出来的变量（见[环境变量参考](环境变量参考.md)的",
        "  `类型` 列），进程启动时会校验；写错类型的值不会带着一个看起来健康的进程上线。",
        "  开关类变量只认 `0/1/true/false/yes/no/on/off`，且必须使用该开关自身识别的",
        "  启用词——否则启动即失败，而不是静默停留在关闭状态。",
        "- 区域生产环境不要用 `kubectl set env` 临时覆盖生成值；修改事实源后重新渲染、",
        "  执行 `make docs-check`，再按发布流程部署。",
        "- `GPU_FAULT_TEST_*`、`GPU_FAULT_PERF_*` 和 fleet transport 临时变量不是生产",
        "  服务配置，不在本文中。",
        "",
    ]
    for index, category in enumerate(spec["categories"], start=1):
        lines.extend(
            [
                f"## {index}. {_markdown(category['title'])}",
                "",
                _markdown(category["description"]),
                "",
                "| 环境变量 | 区域清单值 | 应用默认/要求 | 含义与管理建议 |",
                "|---|---|---|---|",
            ]
        )
        for item in category["variables"]:
            name = item["name"]
            default = item.get("default")
            if default is None:
                default = ", ".join(sorted(defaults.get(name, {"<implicit>"})))
            lines.append(
                "| "
                + " | ".join(
                    (
                        _code(name),
                        _regional_value(name, values, item.get("production")),
                        _code(default),
                        _markdown(item["description"]),
                    )
                )
                + " |"
            )
        lines.append("")
    lines.extend(
        [
            "## 动态变量族",
            "",
            "| 模式 | 含义 |",
            "|---|---|",
            "| `GPU_FAULT_NOTIFICATION_TTL_SECONDS_<CATEGORY>` | "
            "按通知 category 覆盖保质期；未配置时使用通用通知 TTL。 |",
            "| `GPU_FAULT_FLEET_<FIELD>` | fleet CLI 传给 transport 子进程的临时上下文，"
            "不是管理员持久配置。 |",
            "",
            "未出现在本文的变量默认视为开发者或组件内部配置；需要调整时先查",
            "[完整环境变量参考](环境变量参考.md)的使用模块，再按",
            "[开发者部署实现](开发者部署实现.md)修改生成事实源。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = render()
    expected_admin = render_admin()
    expected_inventory = render_inventory()
    if args.check:
        actual = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        actual_inventory = (
            INVENTORY_OUTPUT.read_text(encoding="utf-8")
            if INVENTORY_OUTPUT.exists()
            else ""
        )
        actual_admin = (
            ADMIN_OUTPUT.read_text(encoding="utf-8") if ADMIN_OUTPUT.exists() else ""
        )
        if (
            actual != expected
            or actual_admin != expected_admin
            or actual_inventory != expected_inventory
        ):
            print(
                "environment reference/admin reference/inventory is stale; run "
                "python3 scripts/generate-env-reference.py",
                file=sys.stderr,
            )
            return 1
        return 0
    OUTPUT.write_text(expected, encoding="utf-8")
    ADMIN_OUTPUT.write_text(expected_admin, encoding="utf-8")
    INVENTORY_OUTPUT.write_text(
        expected_inventory,
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
