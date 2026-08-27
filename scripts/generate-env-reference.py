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
DYNAMIC_PREFIXES = ("GPU_FAULT_NOTIFICATION_TTL_SECONDS_",)
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
    return bool(NAME_PATTERN.fullmatch(name)) and not name.endswith("_")


def module_name(path: Path) -> str:
    relative = path.relative_to(SOURCE).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return "gpu_fault." + ".".join(parts)


def deployment_source_name(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def expression(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node).replace("\n", " ")
    except Exception:
        return ""


def extract() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    modules: dict[str, set[str]] = defaultdict(set)
    defaults: dict[str, set[str]] = defaultdict(set)
    for path in sorted(SOURCE.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        module = module_name(path)
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
    return modules, defaults


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
    modules, defaults = extract()
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
        "| 环境变量 | 领域 | 默认值/要求 | 使用模块 |",
        "|---|---|---|---|",
    ]
    for name in sorted(modules):
        default = ", ".join(sorted(defaults.get(name, {"<implicit>"})))
        used = "<br>".join(f"`{item}`" for item in sorted(modules[name]))
        lines.append(f"| `{name}` | {domain(name)} | `{default}` | {used} |")
    return "\n".join(lines) + "\n"


def render_inventory() -> str:
    modules, _ = extract()
    return (
        json.dumps(
            {
                "schema_version": 1,
                "variables": sorted(modules),
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
    modules, defaults = extract()
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
