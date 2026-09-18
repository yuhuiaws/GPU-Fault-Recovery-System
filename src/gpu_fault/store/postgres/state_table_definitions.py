"""Validate the declared state-table functions and triggers without repairing them."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpu_fault.store.postgres.ddl_helpers import _normalize_trigger_definition

FUNCTION = re.compile(
    r"CREATE\s+OR\s+REPLACE\s+FUNCTION\s+(\w+)[\s\S]*?\bAS\s+\$\$([\s\S]*?)\$\$",
    re.I,
)
TRIGGER = re.compile(r"CREATE\s+TRIGGER\s+(\w+)", re.I)


@dataclass(frozen=True)
class StateFunctionDefinition:
    body: str
    language: str
    volatility: str
    config: tuple[str, ...]
    arguments: str
    result: str
    strict: bool
    parallel: str
    defaults: str


def _normalize_signature(value: str) -> str:
    value = re.sub(r"\btimestamptz\b", "timestamp with time zone", value.lower())
    value = re.sub(r"\s+", " ", value.strip())
    return re.sub(r"\s*([,()])\s*", r"\1", value)


def declared_state_definitions() -> tuple[
    dict[str, StateFunctionDefinition], dict[str, str]
]:
    functions: dict[str, StateFunctionDefinition] = {}
    triggers: dict[str, str] = {}
    root = Path(__file__).parent
    paths = sorted(
        {
            *root.glob("ddl_control_state*.py"),
            *root.glob("ddl_remote_command_state.py"),
            *root.glob("ddl_workflow_state.py"),
        }
    )
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            text = node.value.strip()
            if match := FUNCTION.match(text):
                header = text[: match.start(2)]
                language = re.search(r"\bLANGUAGE\s+(\w+)", header, re.I)
                signature = re.search(
                    r"\(([\s\S]*?)\)\s+RETURNS\s+([\s\S]*?)\s+LANGUAGE\b",
                    header,
                    re.I,
                )
                if language is None or signature is None:
                    raise RuntimeError("state function declaration is incomplete")
                arguments = re.sub(
                    r"\s+DEFAULT\s+(NULL|FALSE|TRUE)\b",
                    "",
                    signature.group(1),
                    flags=re.I,
                )
                if re.search(r"\bDEFAULT\b", arguments, re.I):
                    raise RuntimeError("unsupported state function argument default")
                volatility = (
                    "i"
                    if re.search(r"\bIMMUTABLE\b", header, re.I)
                    else "s"
                    if re.search(r"\bSTABLE\b", header, re.I)
                    else "v"
                )
                config = (
                    ("timezone=utc",)
                    if re.search(r"SET\s+timezone\s*=\s*'UTC'", header, re.I)
                    else ()
                )
                functions[match.group(1)] = StateFunctionDefinition(
                    body=match.group(2),
                    language=language.group(1).lower(),
                    volatility=volatility,
                    config=config,
                    arguments=_normalize_signature(arguments),
                    result=_normalize_signature(signature.group(2)),
                    strict=bool(
                        re.search(
                            r"\bSTRICT\b|\bRETURNS\s+NULL\s+ON\s+NULL\s+INPUT\b",
                            header,
                            re.I,
                        )
                    ),
                    parallel=(
                        "s"
                        if re.search(r"\bPARALLEL\s+SAFE\b", header, re.I)
                        else "r"
                        if re.search(r"\bPARALLEL\s+RESTRICTED\b", header, re.I)
                        else "u"
                    ),
                    defaults=",".join(
                        value.lower()
                        for value in re.findall(
                            r"\bDEFAULT\s+(NULL|FALSE|TRUE)\b",
                            signature.group(1),
                            flags=re.I,
                        )
                    ),
                )
            elif trigger := TRIGGER.match(text):
                triggers[trigger.group(1)] = _normalize_trigger_definition(text)
    return functions, triggers


def validate_state_definitions(cursor: Any) -> None:
    functions, triggers = declared_state_definitions()
    cursor.execute(
        "SELECT p.proname, p.prosrc, l.lanname, p.provolatile, p.prosecdef, p.proconfig, "
        "pg_get_function_identity_arguments(p.oid), pg_get_function_result(p.oid), "
        "p.proisstrict, p.proparallel, pg_get_expr(p.proargdefaults, 0) "
        "FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang "
        "JOIN pg_namespace n ON n.oid=p.pronamespace "
        "WHERE n.nspname=current_schema() AND p.proname=ANY(%s)",
        (sorted(functions),),
    )
    observed: dict[str, StateFunctionDefinition] = {}
    for (
        name,
        body,
        language,
        volatility,
        definer,
        config,
        arguments,
        result,
        strict,
        parallel,
        defaults,
    ) in cursor.fetchall():
        if definer or name in observed:
            raise RuntimeError("state function ownership or overload differs")
        observed[name] = StateFunctionDefinition(
            body=body,
            language=language,
            volatility=volatility,
            config=tuple(sorted(str(item).lower() for item in (config or ()))),
            arguments=_normalize_signature(arguments),
            result=_normalize_signature(result),
            strict=strict,
            parallel=parallel,
            defaults=_normalize_signature(
                re.sub(r"\bNULL::\w+\b", "NULL", defaults or "", flags=re.I)
            ),
        )
    if observed != functions:
        raise RuntimeError(
            "control-state function definitions differ; run --ensure-schema"
        )
    cursor.execute(
        "SELECT t.tgname, pg_get_triggerdef(t.oid) FROM pg_trigger t "
        "JOIN pg_class c ON c.oid=t.tgrelid "
        "JOIN pg_namespace n ON n.oid=c.relnamespace "
        "WHERE n.nspname=current_schema() AND t.tgname=ANY(%s) AND NOT t.tgisinternal",
        (sorted(triggers),),
    )
    actual = {
        name: _normalize_trigger_definition(definition)
        for name, definition in cursor.fetchall()
    }
    if actual != triggers:
        raise RuntimeError(
            "control-state trigger definitions differ; run --ensure-schema"
        )
