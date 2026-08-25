from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

from gpu_fault.app.routes import (
    admin,
    completion,
    fleet,
    incidents,
    processor,
    telemetry,
    workflows,
)

ROOT = Path(__file__).resolve().parents[2]
API = ROOT / "src/gpu_fault/api.py"
FACTORY = ROOT / "src/gpu_fault/app/factory.py"


def test_route_handlers_are_importable_module_functions() -> None:
    handlers = (
        admin.healthz,
        admin.version,
        fleet.register_agent,
        fleet.list_agents,
        workflows.dispatch_workflows,
        incidents.get_incident,
        completion.terminal,
        telemetry.latest_gpu_metrics,
        processor.processor_status,
    )

    assert all(handler.__module__.startswith("gpu_fault.app") for handler in handlers)
    assert all("<locals>" not in handler.__qualname__ for handler in handlers)


def test_create_app_route_closures_only_shrink() -> None:
    tree = ast.parse(FACTORY.read_text(encoding="utf-8"))
    create_app = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "create_app"
    )
    route_handlers = [
        node
        for node in ast.walk(create_app)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr in {"get", "post", "put", "patch", "delete"}
            for decorator in node.decorator_list
        )
    ]

    assert route_handlers == []
    assert create_app.end_lineno - create_app.lineno + 1 <= 300


def test_api_module_has_no_import_time_app_factory_call() -> None:
    tree = ast.parse(API.read_text(encoding="utf-8"))
    offenders = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "create_app"
        ):
            offenders.append(node.lineno)

    assert offenders == []


def test_router_modules_do_not_reuse_method_and_path() -> None:
    routes = [
        route
        for module in (
            admin,
            completion,
            fleet,
            incidents,
            processor,
            telemetry,
            workflows,
        )
        for route in module.router.routes
    ]
    identities = [(method, route.path) for route in routes for method in route.methods]

    assert all(count == 1 for count in Counter(identities).values())
