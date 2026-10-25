"""Reviewed pass-through command wrapper of the GF-GENERAL-LIVE host source replay.

The host collector (``gpu-fault-host-collector.service``) reads several raw
sources by shelling out: ``smartctl`` (SMART scan and health), ``ipmitool``
(BMC sensors), ``ethtool -S`` (EFA netdev counters) and ``nvidia-smi`` (GPU
utilization, inventory, compute apps, driver version). A live-replay variant
replaces *one* of those readings without replacing a host binary: the host
probe (``scripts/e2e/regional/probes/general_live_source_probe.py``) installs
this file and a reviewed plan under ``/run/gfrs-live/<run>/`` with sha256
checks, writes a shim named after the tool into that run's ``bin/`` and puts
``bin/`` first on the PATH of exactly one allow-listed collector unit through a
``/run`` drop-in that cleanup and a deadman remove. The Node Agent is never in
the allow-list, and ``py-spy`` -- which only the Node Agent runs, by absolute
path -- is refused by name.

Every invocation is forwarded to the real binary (the first one on PATH outside
``/run/gfrs-live``) unchanged unless the argv is one of the catalogued queries
of the tool *and* the reviewed plan has a rule for it. Rules act on
allow-listed fields only: a SMART JSON verdict field, one ``ethtool`` counter
line, one GPU row or utilization column, or -- for the whole-field variants
(scan failed, malformed JSON, no BMC, no statistics) -- the reviewed stdout and
exit status. A failing real query is passed through as it failed; a rule may be
bounded to N calls; the plan carries a deadline after which everything passes
through. Each catalogued call is journaled with the sha256 of the real and the
served output so the runner can read back what the product actually saw.

Standard library only, Python 3.9+, runs as ``python -I`` from the unit's
private PATH; the same file is imported by the probe and the runner to validate
a plan before it is installed. ``proof_scope`` is always ``live-replay``.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPLAY_ROOT = "/run/gfrs-live"
TOOL_ENV = "GFRS_LIVE_TOOL"
DIR_ENV = "GFRS_LIVE_REPLAY_DIR"
PROOF_SCOPE = "live-replay"
HOST_UNIT = "gpu-fault-host-collector.service"
NODE_AGENT_UNIT = "gpu-fault-node-agent.service"
#: The only unit whose PATH may carry a wrapper. The Node Agent is absent on purpose.
ALLOWED_UNITS = (HOST_UNIT,)
MAX_RULES = 16
MAX_STDOUT_BYTES = 64 * 1024
MAX_STDERR_BYTES = 4 * 1024
MAX_DELAY_SECONDS = 120
MAX_CALLS = 1000
MAX_PLAN_SECONDS = 6 * 3600
REAL_TIMEOUT_SECONDS = 90
GPU_UUID = re.compile(r"^GPU-[0-9a-fA-F]{8}(-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}$")
COUNTER_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
COUNTER_VALUE = re.compile(r"^(-?[0-9]{1,20}|[A-Za-z_-]{1,16})$")
UTILIZATION_VALUE = re.compile(r"^([0-9]{1,3}|\[(N/A|Unknown Error|Not Supported)\])$")
DEVICE_NODE = re.compile(r"^/dev/[a-z][a-z0-9/_-]{0,31}$")
INTERFACE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,15}$")

#: Tools the wrapper refuses by name, and why. Listed so a lane gets the exact
#: reason instead of an "unknown tool".
REFUSED_TOOLS = {
    "py-spy": (
        "py-spy is run only by the Node Agent (GPU_FAULT_PYTHON_STACK_TOOL, an "
        "absolute path), and the Node Agent unit is never in a replay allow-list"
    ),
}

#: Rule keys every action accepts.
COMMON_KEYS = frozenset({"query", "action", "max_calls", "delay_seconds"})
#: Per action: the keys it requires, the optional keys it may carry.
ACTION_KEYS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "passthrough": (frozenset(), frozenset()),
    "replace_output": (frozenset({"stdout"}), frozenset({"exit_code", "stderr"})),
    "exit": (frozenset({"exit_code"}), frozenset({"stderr"})),
    "set_json": (frozenset({"field", "value"}), frozenset()),
    "delete_json": (frozenset({"field"}), frozenset()),
    "set_counter": (frozenset({"counter", "value"}), frozenset()),
    "delete_counter": (frozenset({"counter"}), frozenset()),
    "drop_row": (frozenset({"uuid"}), frozenset()),
    "set_column": (frozenset({"uuid", "column", "value"}), frozenset()),
}
#: Actions that edit the real output and therefore need the real binary to run.
EDIT_ACTIONS = frozenset(
    {
        "set_json",
        "delete_json",
        "set_counter",
        "delete_counter",
        "drop_row",
        "set_column",
    }
)


def _query(
    *,
    argv: tuple[str, ...] = (),
    prefix: tuple[str, ...] = (),
    arity: int | None = None,
    selector: str = "",
    selector_index: int = -1,
    actions: tuple[str, ...],
    json_fields: dict[str, str] | None = None,
    json_deletable: tuple[str, ...] = (),
    columns: dict[str, int] | None = None,
) -> dict[str, Any]:
    return {
        "argv": argv,
        "prefix": prefix,
        "arity": arity,
        "selector": selector,
        "selector_index": selector_index,
        "actions": actions,
        "json_fields": json_fields or {},
        "json_deletable": json_deletable,
        "columns": columns or {},
    }


#: The per-tool, per-query, per-field allow-list. ``argv`` is an exact match,
#: ``prefix`` + ``arity`` a positional one; ``selector`` names the optional rule
#: key that narrows a rule to one device/interface (the argv item at
#: ``selector_index``). Only the listed actions and fields exist for a query.
TOOLS: dict[str, dict[str, Any]] = {
    "smartctl": {
        "missing_real_ok": False,
        "queries": {
            "scan": _query(
                argv=("--scan-open",),
                actions=("passthrough", "replace_output", "exit"),
            ),
            "health": _query(
                prefix=("-H", "-j"),
                arity=3,
                selector="device",
                selector_index=2,
                actions=(
                    "passthrough",
                    "set_json",
                    "delete_json",
                    "replace_output",
                    "exit",
                ),
                json_fields={
                    "smart_status.passed": "bool",
                    "smartctl.exit_status": "int",
                },
                json_deletable=("smart_status",),
            ),
        },
    },
    "ipmitool": {
        # EC2 has no BMC: the real ipmitool fails or is absent, so the sensor
        # table itself is the reviewed field.
        "missing_real_ok": True,
        "queries": {
            "sensor": _query(
                argv=("sensor",),
                actions=("passthrough", "replace_output", "exit"),
            ),
        },
    },
    "ethtool": {
        "missing_real_ok": False,
        "queries": {
            "stats": _query(
                prefix=("-S",),
                arity=2,
                selector="interface",
                selector_index=1,
                actions=(
                    "passthrough",
                    "set_counter",
                    "delete_counter",
                    "replace_output",
                    "exit",
                ),
            ),
        },
    },
    "nvidia-smi": {
        "missing_real_ok": False,
        "queries": {
            # collectors/host/gpu_rank.py GPU_QUERY_ARGV: utilization + inventory.
            "utilization": _query(
                argv=(
                    "--query-gpu=uuid,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ),
                actions=(
                    "passthrough",
                    "drop_row",
                    "set_column",
                    "replace_output",
                    "exit",
                ),
                columns={"uuid": 0, "utilization.gpu": 1},
            ),
            # collectors/host/gpu_rank.py COMPUTE_APPS_ARGV.
            "compute_apps": _query(
                argv=(
                    "--query-compute-apps=pid,gpu_uuid",
                    "--format=csv,noheader,nounits",
                ),
                actions=("passthrough", "replace_output", "exit"),
            ),
            # collectors/gpu/discovery.py software version discovery.
            "driver_version": _query(
                argv=("--query-gpu=driver_version", "--format=csv,noheader"),
                actions=("passthrough", "replace_output", "exit"),
            ),
            # GPU_FAULT_NVSWITCH_TOPOLOGY_COMMAND when it is an nvidia-smi topo dump.
            "topology": _query(
                prefix=("topo",),
                actions=("passthrough", "replace_output", "exit"),
            ),
        },
    },
}


class PlanError(ValueError):
    pass


# --------------------------------------------------------------------------- #
# plan validation (shared by the probe, the runner and the wrapper itself)
# --------------------------------------------------------------------------- #
def checked_tool(tool: Any) -> str:
    if tool in REFUSED_TOOLS:
        raise PlanError(f"{tool} may not be wrapped: {REFUSED_TOOLS[tool]}")
    if not isinstance(tool, str) or tool not in TOOLS:
        raise PlanError(f"{tool!r} is not a replayable tool; known: {sorted(TOOLS)}")
    return tool


def checked_unit(unit: Any) -> str:
    if unit == NODE_AGENT_UNIT:
        raise PlanError("the Node Agent unit is never in a replay allow-list")
    if unit not in ALLOWED_UNITS:
        raise PlanError(f"{unit!r} may not carry a wrapper; allowed: {ALLOWED_UNITS}")
    return str(unit)


def _bounded_int(value: Any, low: int, high: int, label: str) -> int:
    if type(value) is not int or not low <= value <= high:
        raise PlanError(f"{label} must be an integer in {low}..{high}")
    return value


def _bounded_text(value: Any, limit: int, label: str) -> str:
    if not isinstance(value, str) or len(value.encode("utf-8")) > limit:
        raise PlanError(f"{label} must be a string of at most {limit} bytes")
    return value


def _matching(value: Any, pattern: re.Pattern[str], label: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise PlanError(f"{label} {value!r} is not allow-listed")
    return value


def validate_rule(tool: str, rule: Any, index: int) -> dict[str, Any]:
    if not isinstance(rule, dict):
        raise PlanError(f"rule {index} must be an object")
    query_name = rule.get("query")
    queries = TOOLS[tool]["queries"]
    if query_name not in queries:
        raise PlanError(f"rule {index}: {tool} has no query {query_name!r}")
    query = queries[query_name]
    action = rule.get("action")
    if action not in query["actions"]:
        raise PlanError(
            f"rule {index}: {tool} {query_name} allows only {list(query['actions'])}"
        )
    required, optional = ACTION_KEYS[action]
    allowed = COMMON_KEYS | required | optional
    if query["selector"]:
        allowed = allowed | {query["selector"]}
    extra = sorted(set(rule) - allowed)
    missing = sorted(required - set(rule))
    if extra or missing:
        raise PlanError(
            f"rule {index}: unexpected keys {extra}, missing keys {missing}"
        )
    if "max_calls" in rule:
        _bounded_int(rule["max_calls"], 0, MAX_CALLS, f"rule {index} max_calls")
    if "delay_seconds" in rule:
        _bounded_int(
            rule["delay_seconds"], 0, MAX_DELAY_SECONDS, f"rule {index} delay_seconds"
        )
    if query["selector"] in rule:
        pattern = DEVICE_NODE if query["selector"] == "device" else INTERFACE_NAME
        _matching(rule[query["selector"]], pattern, f"rule {index} {query['selector']}")
    if action == "replace_output":
        _bounded_text(rule["stdout"], MAX_STDOUT_BYTES, f"rule {index} stdout")
        _bounded_int(rule.get("exit_code", 0), 0, 255, f"rule {index} exit_code")
        _bounded_text(rule.get("stderr", ""), MAX_STDERR_BYTES, f"rule {index} stderr")
    elif action == "exit":
        _bounded_int(rule["exit_code"], 1, 255, f"rule {index} exit_code")
        _bounded_text(rule.get("stderr", ""), MAX_STDERR_BYTES, f"rule {index} stderr")
    elif action == "set_json":
        kind = query["json_fields"].get(rule["field"])
        if kind is None:
            raise PlanError(
                f"rule {index}: JSON field {rule['field']!r} is not allow-listed"
            )
        if kind == "bool" and type(rule["value"]) is not bool:
            raise PlanError(f"rule {index}: {rule['field']} takes a boolean")
        if kind == "int":
            _bounded_int(rule["value"], 0, 255, f"rule {index} {rule['field']}")
    elif action == "delete_json":
        if rule["field"] not in query["json_deletable"]:
            raise PlanError(
                f"rule {index}: JSON field {rule['field']!r} may not be deleted"
            )
    elif action in {"set_counter", "delete_counter"}:
        _matching(rule["counter"], COUNTER_NAME, f"rule {index} counter")
        if action == "set_counter":
            _matching(rule["value"], COUNTER_VALUE, f"rule {index} value")
    elif action in {"drop_row", "set_column"}:
        _matching(rule["uuid"], GPU_UUID, f"rule {index} uuid")
        if action == "set_column":
            if rule["column"] not in query["columns"] or rule["column"] == "uuid":
                raise PlanError(
                    f"rule {index}: column {rule['column']!r} is not allow-listed"
                )
            _matching(rule["value"], UTILIZATION_VALUE, f"rule {index} value")
    return dict(rule)


def validate_plan(tool: str, plan: Any) -> dict[str, Any]:
    """The reviewed plan of ``tool`` or a ``PlanError`` naming the first defect."""

    tool = checked_tool(tool)
    if not isinstance(plan, dict) or set(plan) != {"tool", "not_after", "rules"}:
        raise PlanError("a plan has exactly the keys tool, not_after and rules")
    if plan["tool"] != tool:
        raise PlanError(f"plan is for {plan['tool']!r}, not {tool!r}")
    deadline = plan["not_after"]
    if type(deadline) not in (int, float) or deadline <= 0:
        raise PlanError("not_after must be a positive epoch time")
    rules = plan["rules"]
    if not isinstance(rules, list) or not 1 <= len(rules) <= MAX_RULES:
        raise PlanError(f"a plan carries 1..{MAX_RULES} rules")
    return {
        "tool": tool,
        "not_after": deadline,
        "rules": [validate_rule(tool, rule, index) for index, rule in enumerate(rules)],
    }


def build_plan(
    tool: str, rules: list[dict[str, Any]], *, not_after: float
) -> dict[str, Any]:
    """A validated plan document for ``tool`` (the runner's constructor)."""

    return validate_plan(tool, {"tool": tool, "not_after": not_after, "rules": rules})


# --------------------------------------------------------------------------- #
# matching and rewriting (pure)
# --------------------------------------------------------------------------- #
def match_query(tool: str, argv: list[str]) -> str | None:
    """The catalogued query name ``argv`` is, or None for anything else."""

    for name, query in TOOLS[tool]["queries"].items():
        if query["argv"] and tuple(argv) == query["argv"]:
            return name
        prefix = query["prefix"]
        if prefix and tuple(argv[: len(prefix)]) == prefix:
            arity = query["arity"]
            if arity is None or len(argv) == arity:
                return name
    return None


def rule_applies(
    tool: str, rule: dict[str, Any], query_name: str, argv: list[str]
) -> bool:
    if rule["query"] != query_name:
        return False
    query = TOOLS[tool]["queries"][query_name]
    selector = query["selector"]
    if selector and selector in rule:
        index = query["selector_index"]
        return len(argv) > index and argv[index] == rule[selector]
    return True


def _set_path(payload: dict[str, Any], field: str, value: Any) -> None:
    node = payload
    parts = field.split(".")
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def _delete_path(payload: dict[str, Any], field: str) -> None:
    node: Any = payload
    parts = field.split(".")
    for part in parts[:-1]:
        node = node.get(part) if isinstance(node, dict) else None
    if isinstance(node, dict):
        node.pop(parts[-1], None)


def apply_rule(
    rule: dict[str, Any], stdout: str, returncode: int, stderr: str
) -> tuple[str, int, str, str | None]:
    """``(stdout, returncode, stderr, unchanged_reason)`` after one rule."""

    action = rule["action"]
    if action == "passthrough":
        return stdout, returncode, stderr, None
    if action == "replace_output":
        return (
            rule["stdout"],
            int(rule.get("exit_code", 0)),
            rule.get("stderr", ""),
            None,
        )
    if action == "exit":
        return "", int(rule["exit_code"]), rule.get("stderr", ""), None
    if returncode != 0:
        return (
            stdout,
            returncode,
            stderr,
            "the real query failed; passed through as it failed",
        )
    if action in {"set_json", "delete_json"}:
        try:
            payload = json.loads(stdout)
        except ValueError:
            return stdout, returncode, stderr, "the real output is not JSON"
        if not isinstance(payload, dict):
            return stdout, returncode, stderr, "the real output is not a JSON object"
        if action == "set_json":
            _set_path(payload, rule["field"], rule["value"])
        else:
            _delete_path(payload, rule["field"])
        return json.dumps(payload, indent=2) + "\n", returncode, stderr, None
    if action in {"set_counter", "delete_counter"}:
        lines = stdout.splitlines()
        kept: list[str] = []
        seen = False
        for line in lines:
            name, separator, _value = line.partition(":")
            if separator and name.strip() == rule["counter"]:
                seen = True
                if action == "set_counter":
                    kept.append(f"{name}: {rule['value']}")
                continue
            kept.append(line)
        if action == "set_counter" and not seen:
            kept.append(f"     {rule['counter']}: {rule['value']}")
        return "\n".join(kept) + ("\n" if kept else ""), returncode, stderr, None
    if action in {"drop_row", "set_column"}:
        rows = stdout.splitlines()
        kept = []
        seen = False
        for row in rows:
            fields = [item.strip() for item in row.split(",")]
            if fields and fields[0] == rule["uuid"]:
                seen = True
                if action == "drop_row":
                    continue
                index = TOOLS["nvidia-smi"]["queries"]["utilization"]["columns"][
                    rule["column"]
                ]
                if index < len(fields):
                    fields[index] = rule["value"]
                    kept.append(", ".join(fields))
                    continue
            kept.append(row)
        if not seen:
            return stdout, returncode, stderr, f"no row for {rule['uuid']}"
        return "\n".join(kept) + ("\n" if kept else ""), returncode, stderr, None
    return stdout, returncode, stderr, f"unknown action {action}"


# --------------------------------------------------------------------------- #
# the wrapper process
# --------------------------------------------------------------------------- #
def real_binary(tool: str, path_value: str, *, own_dir: str = "") -> str | None:
    """The first ``tool`` on PATH outside the replay root and the shim's directory."""

    for entry in path_value.split(os.pathsep):
        if not entry:
            continue
        resolved = os.path.realpath(entry)
        if resolved.startswith(REPLAY_ROOT) or (own_dir and resolved == own_dir):
            continue
        candidate = os.path.join(entry, tool)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            if not os.path.realpath(candidate).startswith(REPLAY_ROOT):
                return candidate
    return None


def plan_path(run_dir: str, tool: str) -> Path:
    return Path(run_dir) / "fixtures" / f"plan-{tool}.json"


def audit_dir(run_dir: str) -> Path:
    return Path(run_dir) / "audit"


def load_plan(run_dir: str, tool: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        document = json.loads(plan_path(run_dir, tool).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return None, f"no readable plan: {type(error).__name__}"
    try:
        return validate_plan(tool, document), None
    except PlanError as error:
        return None, f"plan rejected: {error}"


def consume_call(run_dir: str, tool: str, index: int, limit: int) -> bool:
    """Count one application of rule ``index``; False once ``limit`` is reached.

    The counter lives in the run's audit directory (the one path the unit may
    write) under a file lock, because two collector threads may share a tick.
    A counter that cannot be kept means the bounded rule is *not* applied:
    fail towards pass-through, never towards an unbounded replay.
    """

    directory = audit_dir(run_dir)
    try:
        directory.mkdir(mode=0o755, parents=True, exist_ok=True)
        with open(directory / "calls.lock", "a+", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = directory / "calls.json"
            counts: dict[str, Any] = {}
            if path.is_file():
                counts = json.loads(path.read_text(encoding="utf-8") or "{}")
            key = f"{tool}/{index}"
            used = int(counts.get(key, 0))
            if limit and used >= limit:
                return False
            counts[key] = used + 1
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(counts, sort_keys=True), encoding="utf-8")
            os.replace(temporary, path)
            return True
    except (OSError, ValueError):
        return False


def journal(run_dir: str, tool: str, record: dict[str, Any]) -> None:
    try:
        directory = audit_dir(run_dir)
        directory.mkdir(mode=0o755, parents=True, exist_ok=True)
        with open(directory / f"{tool}.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError:
        pass


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def serve(
    tool: str,
    argv: list[str],
    real: str | None,
    run_dir: str,
    *,
    now: float | None = None,
) -> tuple[int, str, str]:
    """``(returncode, stdout, stderr)`` for one catalogued call."""

    now = time.time() if now is None else now
    query_name = match_query(tool, argv)
    plan, problem = load_plan(run_dir, tool)
    applicable: list[tuple[int, dict[str, Any]]] = []
    if plan is not None and query_name is not None:
        if now > plan["not_after"]:
            problem = "plan deadline passed; passed through"
        else:
            applicable = [
                (index, rule)
                for index, rule in enumerate(plan["rules"])
                if rule_applies(tool, rule, query_name, argv)
            ]
            if not applicable:
                problem = "no rule for this call; passed through"

    def run_real() -> tuple[str, str, int]:
        try:
            completed = subprocess.run(
                [str(real), *argv],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=REAL_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return "", f"{tool}: timed out\n", 124
        except OSError as error:
            return "", f"{tool}: {error}\n", 126
        return completed.stdout, completed.stderr, completed.returncode

    # The real binary runs lazily: a whole-output rule (exit, replace_output)
    # that fires first does not need it; an edit or passthrough rule does, and a
    # call no rule fires on is a plain pass-through of the real answer.
    real_returncode: int | None = None
    stdout, stderr, returncode = "", "", 0
    ran_real = served_whole = False
    applied: list[int] = []
    reasons: list[str] = []
    for index, rule in applicable:
        action = rule["action"]
        whole = action in {"exit", "replace_output"}
        if not whole:
            if real is None:
                reasons.append(
                    f"rule {index}: no real {tool} to forward to; not applied"
                )
                continue
            if not ran_real:
                stdout, stderr, returncode = run_real()
                ran_real, real_returncode = True, returncode
            if action in EDIT_ACTIONS and returncode != 0:
                reasons.append(
                    f"rule {index}: the real query failed; passed through as it failed"
                )
                continue
        limit = int(rule.get("max_calls", 0))
        if not consume_call(run_dir, tool, index, limit):
            reasons.append(f"rule {index} exhausted or uncountable; not applied")
            continue
        delay = int(rule.get("delay_seconds", 0))
        if delay:
            time.sleep(delay)
        stdout, returncode, stderr, reason = apply_rule(
            rule, stdout, returncode, stderr
        )
        if reason is None:
            applied.append(index)
            served_whole = served_whole or whole
        else:
            reasons.append(f"rule {index}: {reason}")
        if whole:
            break
    if not served_whole and not ran_real:
        if real is None:
            stdout, stderr, returncode = "", f"{tool}: command not found\n", 127
        else:
            stdout, stderr, returncode = run_real()
            ran_real, real_returncode = True, returncode
    if problem:
        reasons.append(problem)
    journal(
        run_dir,
        tool,
        {
            "at": now,
            "tool": tool,
            "query": query_name,
            "argv": argv,
            "real": real,
            "real_returncode": real_returncode,
            "served_returncode": returncode,
            "served_sha256": _sha256(stdout),
            "applied_rules": applied,
            "unchanged_reason": reasons,
            "proof_scope": PROOF_SCOPE,
            "physical_fault_injected": False,
        },
    )
    return returncode, stdout, stderr


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    tool = os.environ.get(TOOL_ENV, "")
    run_dir = os.environ.get(DIR_ENV, "")
    own_dir = os.path.realpath(os.path.join(run_dir, "bin")) if run_dir else ""
    if tool not in TOOLS or not run_dir:
        sys.stderr.write("host command replay wrapper: not configured\n")
        return 125
    real = real_binary(tool, os.environ.get("PATH", ""), own_dir=own_dir)
    if match_query(tool, arguments) is None:
        if real is None:
            sys.stderr.write(f"{tool}: command not found\n")
            return 127
        os.execv(real, [real, *arguments])
    returncode, stdout, stderr = serve(tool, arguments, real, run_dir)
    sys.stdout.write(stdout)
    sys.stdout.flush()
    if stderr:
        sys.stderr.write(stderr)
        sys.stderr.flush()
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
