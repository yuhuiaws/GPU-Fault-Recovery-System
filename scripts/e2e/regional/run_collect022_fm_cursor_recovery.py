#!/usr/bin/env python3
"""Manual, guarded deployed-host FM cursor exercise with no forwarding or GPU actions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional.acceptance_runner_common import write_json_atomic  # noqa: E402
from scripts.e2e.regional.collector_action_guard import (  # noqa: E402
    bounded_collector_case,
    require_action_time,
)
from scripts.e2e.regional.host_probe_fixture import HostProbeFixture, HostProbeSettings  # noqa: E402
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    CaseRunner,
    add_live_arguments,
    run_standard_case,
)
from scripts.e2e.regional.probes import collect022_fm_cursor_probe as probe  # noqa: E402
from scripts.e2e.regional.regional_commands import RegionalFixtureError  # noqa: E402
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalLiveFixture,
    RegionalLiveSettings,
    predecessor_evidence,
    required,
    run_case_main,
    settings_from_arguments,
)
from gpu_fault.collectors.logs.fabric_manager import (  # noqa: E402
    file_evidence_ref,
    file_record_id,
)

CASE_ID = "GF-REGIONAL-COLLECT-022"
PREDECESSOR_CASE_ID = "GF-REGIONAL-COLLECT-005"
CONFIRMATION = "COLLECT022_PRIVATE_FM_CURSOR_RECOVERY"
PROBE_SCRIPT = Path(__file__).with_name("probes") / "collect022_fm_cursor_probe.py"
PROBE_TIMEOUT_SECONDS = 90
PRIVATE_WINDOW_SECONDS = 1200
NODE_FIELDS = (
    "uid",
    "boot_id",
    "ready",
    "unschedulable",
    "taints",
    "ownership_annotations",
)


@dataclass(frozen=True)
class Settings:
    regional: RegionalLiveSettings
    node: str
    host_probe_image: str
    predecessor_path: Path

    def environment(self) -> dict[str, str]:
        return {
            **self.regional.environment(),
            "GPU_FAULT_COLLECT022_NODE": self.node,
            "GPU_FAULT_HOST_PROBE_IMAGE": self.host_probe_image,
            "GPU_FAULT_PREDECESSOR_EVIDENCE": str(self.predecessor_path),
        }


def configure(arguments: argparse.Namespace) -> Settings:
    image = required(
        arguments.host_probe_image or os.getenv("GPU_FAULT_HOST_PROBE_IMAGE", ""),
        "host probe image",
    )
    if re.fullmatch(r".+@sha256:[0-9a-f]{64}", image) is None:
        raise RegionalFixtureError(
            "host probe image must use an immutable SHA-256 digest"
        )
    predecessor = (
        Path(arguments.predecessor_evidence)
        if arguments.predecessor_evidence
        else arguments.run_dir
        / "cases"
        / PREDECESSOR_CASE_ID
        / f"{PREDECESSOR_CASE_ID}.json"
    )
    return Settings(
        regional=settings_from_arguments(arguments),
        node=required(arguments.node, "target node"),
        host_probe_image=image,
        predecessor_path=predecessor.expanduser().resolve(),
    )


def read_only_preflight(settings: Settings, case_dir: Path) -> dict[str, Any]:
    regional = RegionalLiveFixture(settings.regional)
    identity = regional.evidence_identity()
    node = regional.node_snapshot(settings.node)
    agent = regional.store_snapshot(node=settings.node).get("agent") or {}
    predecessor = predecessor_evidence(
        settings.predecessor_path,
        PREDECESSOR_CASE_ID,
        **identity,
    )
    errors = []
    if predecessor.get("valid") is not True:
        errors.append("COLLECT-005 predecessor is not current bound PASS evidence")
    if (
        any(field not in node for field in NODE_FIELDS)
        or not node.get("uid")
        or not node.get("boot_id")
    ):
        errors.append("target node identity/state is incomplete")
    if (
        node.get("ready") != "True"
        or node.get("unschedulable")
        or node.get("taints")
        or node.get("ownership_annotations")
    ):
        errors.append("target node is not clean, Ready and schedulable")
    if agent.get("lifecycle_state") != "ACTIVE":
        errors.append("target Node Agent is not ACTIVE")
    if type(agent.get("generation")) is not int or agent["generation"] <= 0:
        errors.append("target Node Agent generation is unknown")
    if regional.business_workloads(settings.node):
        errors.append("target node has a business workload")
    result = {
        **identity,
        "node": node,
        "predecessor": predecessor,
        "agent_generation": agent.get("generation"),
        "cpu_blast": regional.cpu_blast_snapshot(),
        "errors": errors,
    }
    write_json_atomic(case_dir / "preflight.json", result)
    return result


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "risk": "live-non-destructive",
        "automation": "manual",
        "predecessor": preflight["predecessor"],
        "preflight_identity": {
            "release_id": preflight["release_id"],
            "cluster_id": preflight["cluster_id"],
            "node_uid": preflight["node"]["uid"],
            "boot_id": preflight["node"]["boot_id"],
            "agent_generation": preflight["agent_generation"],
        },
        "target_node": settings.node,
        "phases": list(probe.STEPS),
        "mutation": "Only nonce-owned private FM log/cursor files; isolated reader uses a non-forwarding sink.",
        "forbidden_surfaces": [
            "existing collector logs or state",
            "service mutation",
            "CPU ingestion",
            "GPU operations",
            "credentials or environment-based sinks",
        ],
        "stop_conditions": [
            "predecessor/plan/runtime/node/service identity drift",
            "expired deadline or prior unresolved phase",
            "wrong marker count, replay, cursor or rotation identity",
            "symlink, hardlink, unknown entry or ownership uncertainty",
            "any private-file or HostProbe cleanup residual",
        ],
        "cleanup": "finally: remove only nonce/inode-bound private files, then UID-journal-owned HostProbe resources",
        "preflight": preflight,
    }


def initialization_errors(
    value: dict[str, Any], *, nonce: str, settings: Settings, boot_id: str
) -> list[str]:
    expected = {
        "initialized": True,
        "schema_version": 1,
        "case_id": CASE_ID,
        "nonce": nonce,
        "cluster_id": settings.regional.cluster_id,
        "node_id": settings.node,
        "boot_id": boot_id,
    }
    runtime = value.get("runtime") or {}
    root = value.get("root_identity") or {}
    if (
        any(value.get(key) != item for key, item in expected.items())
        or value.get("initialized") is not True
        or type(value.get("schema_version")) is not int
        or not runtime.get("python_prefix")
        or not runtime.get("package_version")
        or re.fullmatch(
            r"[0-9a-f]{64}", str(runtime.get("collector_module_sha256") or "")
        )
        is None
        or any(
            type(root.get(key)) is not int or root[key] <= 0
            for key in ("device", "inode")
        )
        or not (value.get("service") or {}).get("InvocationID")
    ):
        return [
            "private initialization identity or deployed runtime proof is incomplete"
        ]
    return []


def truncation_errors(
    value: dict[str, Any], previous: list[dict[str, Any]]
) -> list[str]:
    proof = value.get("truncation") or {}
    if not isinstance(proof, dict):
        return ["private truncation proof is missing"]
    nonce = value.get("nonce")
    if not isinstance(nonce, str) or re.fullmatch(r"[0-9a-f]{32}", nonce) is None:
        return ["private truncation nonce is invalid"]
    baseline = proof.get("before") or {}
    truncated = proof.get("truncated") or {}
    appended = value.get("before") or {}
    if not all(
        isinstance(snapshot, dict)
        and isinstance(snapshot.get("logs"), dict)
        and isinstance(snapshot.get("cursor"), dict)
        and isinstance(snapshot["cursor"].get("files"), dict)
        for snapshot in (baseline, truncated, appended)
    ):
        return ["private truncation snapshots are incomplete"]
    name = probe.TRUNCATED_LOG
    path = str(probe.PRIVATE_BASE / f"c022-{nonce}" / name)
    metadata = baseline["logs"].get(name) or {}
    checkpoint = baseline["cursor"]["files"].get(path) or {}
    saved_offset = checkpoint.get("offset")
    size = (appended["logs"].get(name) or {}).get("size")
    if (
        type(saved_offset) is not int
        or type(size) is not int
        or not 0 < size < saved_offset
        or saved_offset != metadata.get("size")
    ):
        return ["private log did not shrink below its saved checkpoint"]
    empty_logs = {**baseline["logs"], name: {**metadata, "size": 0}}
    appended_logs = {**baseline["logs"], name: {**metadata, "size": size}}
    events = value.get("events") or []
    if (
        not previous
        or previous[-1].get("step") != "rotated-restart"
        or baseline != previous[-1].get("after")
        or truncated != {**baseline, "logs": empty_logs}
        or appended != {**baseline, "logs": appended_logs}
        or (value.get("after") or {}).get("logs") != appended_logs
        or len(events) != 1
        or any(
            (event.get("fields") or {}).get("path") != path
            or (event.get("fields") or {}).get("offset") != "0"
            for event in events
        )
    ):
        return ["private truncation identity, zero-length proof or delivery changed"]
    reused = proof.get("reused_event")
    rotations = [item for item in previous if item.get("step") == "rotate"]
    if (
        not isinstance(reused, dict)
        or len(rotations) != 1
        or not isinstance(rotations[0].get("events"), list)
        or sum(event == reused for event in rotations[0]["events"]) != 1
        or reused.get("message") != probe.line(nonce, "N").decode().rstrip("\n")
        or events[0].get("message") != probe.line(nonce, "T").decode().rstrip("\n")
    ):
        return ["private truncation did not reuse the actually emitted N record"]
    old_fields = reused.get("fields")
    new_fields = events[0].get("fields")
    if not isinstance(old_fields, dict) or not isinstance(new_fields, dict):
        return ["private truncation record identity fields are missing"]
    original_generation = checkpoint.get("generation")
    if (
        type(original_generation) is not int
        or original_generation < 0
        or any(
            old_fields.get(key) != str(checkpoint.get(key))
            for key in ("device", "inode", "generation")
        )
        or old_fields.get("path") != path
        or old_fields.get("offset") != "0"
        or any(
            new_fields.get(key) != old_fields.get(key)
            for key in ("path", "device", "inode", "offset")
        )
        or new_fields.get("generation") != str(original_generation + 1)
        or events[0].get("record_id") == reused.get("record_id")
        or events[0].get("evidence_ref") == reused.get("evidence_ref")
    ):
        return ["private truncation did not advance the emitted record generation"]
    after_files = (value.get("after") or {}).get("cursor", {}).get("files", {})
    expected_files = {
        **baseline["cursor"]["files"],
        path: {
            **checkpoint,
            "offset": size,
            "generation": original_generation + 1,
        },
    }
    if after_files != expected_files:
        return ["private truncation changed another cursor or lost its generation"]
    return []


def file_record_errors(
    event: dict[str, Any],
    *,
    initial: dict[str, Any],
    path: str,
    checkpoint: dict[str, Any],
) -> list[str]:
    fields = event.get("fields")
    if not isinstance(fields, dict) or any(
        not isinstance(fields.get(key), str)
        or re.fullmatch(r"[0-9]{1,40}", fields[key]) is None
        for key in ("device", "inode", "offset", "generation")
    ):
        return ["local record has malformed generation or position"]
    device, inode, offset, generation = (
        int(fields[key]) for key in ("device", "inode", "offset", "generation")
    )
    if (
        fields["generation"] != str(checkpoint.get("generation"))
        or event.get("record_id")
        != file_record_id(
            initial["cluster_id"],
            initial["node_id"],
            initial["boot_id"],
            device,
            inode,
            offset,
            generation,
        )
        or event.get("evidence_ref")
        != file_evidence_ref(
            initial["node_id"], path, device, inode, offset, generation
        )
    ):
        return ["local record ID or evidence ref differs from its file generation"]
    return []


def phase_errors(
    value: dict[str, Any],
    *,
    step: str,
    initial: dict[str, Any],
    previous: list[dict[str, Any]],
) -> list[str]:
    errors = []
    if (
        any(
            value.get(key) != initial.get(key)
            for key in (
                "case_id",
                "nonce",
                "cluster_id",
                "node_id",
                "boot_id",
                "runtime",
                "service",
                "root_identity",
            )
        )
        or value.get("step") != step
        or type(value.get("schema_version")) is not int
        or value["schema_version"] != 1
    ):
        errors.append("phase identity differs from the initialized private reader")
    process = str(value.get("process_identity") or "")
    if re.fullmatch(r"[1-9][0-9]*:[0-9]+", process) is None or process in {
        item.get("process_identity") for item in previous
    }:
        errors.append("phase did not run in a distinct identified reader process")
    events = value.get("events")
    if not isinstance(events, list):
        return [*errors, "phase has no explicit local event list"]
    tags = probe.TAGS[step]
    expected_markers = sorted(f"c022-{initial['nonce']}-{tag}" for tag in tags)
    observed_markers = []
    after = value.get("after") or {}
    logs = after.get("logs") or {}
    cursor = after.get("cursor") or {}
    expected_logs = (
        {"active.log", "rotated.log"}
        if step in {"rotate", "rotated-restart", "truncate", "truncated-restart"}
        else {"active.log"}
    )
    private_path = probe.PRIVATE_BASE / f"c022-{initial['nonce']}"
    if (
        set(logs) != expected_logs
        or cursor.get("valid") is not True
        or cursor.get("present") is not True
    ):
        errors.append("phase did not persist a complete private cursor")
    files = cursor.get("files") or {}
    if set(files) != {str(private_path / name) for name in logs}:
        errors.append("cursor file set differs from the private logs")
    for name, metadata in logs.items():
        checkpoint = files.get(str(private_path / name)) or {}
        if (
            any(
                type(metadata.get(key)) is not int
                for key in ("device", "inode", "size")
            )
            or metadata.get("size", 0) <= 0
            or any(
                checkpoint.get(key) != metadata.get(key) for key in ("device", "inode")
            )
            or checkpoint.get("offset") != metadata.get("size")
            or type(checkpoint.get("generation")) is not int
            or checkpoint.get("generation", -1) < 0
        ):
            errors.append("cursor did not reach the exact current file identity/EOF")
    for event in events:
        if not isinstance(event, dict):
            errors.append("local record is not an object")
            continue
        match = re.search(r"\bmarker=(\S+)$", str(event.get("message") or ""))
        observed_markers.append(match.group(1) if match else "")
        fields = event.get("fields") or {}
        path = str(fields.get("path") or "")
        metadata = logs.get(Path(path).name) or {}
        if (
            event.get("source") != "file"
            or event.get("cluster_id") != initial["cluster_id"]
            or event.get("node_id") != initial["node_id"]
            or re.fullmatch(r"fm-file-[0-9a-f]{64}", str(event.get("record_id") or ""))
            is None
            or path != str(private_path / Path(path).name)
            or any(
                str(metadata.get(key)) != fields.get(key) for key in ("device", "inode")
            )
            or not str(fields.get("offset") or "").isdecimal()
            or int(fields["offset"]) >= metadata.get("size", 0)
        ):
            errors.append("local record is not bound to the private file identity")
        errors.extend(
            file_record_errors(
                event,
                initial=initial,
                path=path,
                checkpoint=files.get(path) or {},
            )
        )
    stats = value.get("stats") or {}
    if (
        sorted(observed_markers) != expected_markers
        or value.get("sink") != "non-forwarding"
        or any(
            type(stats.get(key)) is not int or stats[key] != len(tags)
            for key in ("observed", "delivered")
        )
        or any(
            type(stats.get(key)) is not int or stats[key] != 0
            for key in ("buffered", "skipped", "duplicates")
        )
    ):
        errors.append(
            "local delivery count/markers differ or historical records replayed"
        )
    before_cursor = (value.get("before") or {}).get("cursor") or {}
    if step == "loss" and before_cursor.get("present") is not False:
        errors.append("missing-checkpoint phase did not remove the private checkpoint")
    if step == "corrupt" and (
        before_cursor.get("present") is not True
        or before_cursor.get("valid") is not False
    ):
        errors.append(
            "corrupt-checkpoint phase did not present malformed private state"
        )
    if step == "rotate" and previous:
        old = previous[-1]["after"]["logs"]["active.log"]
        rotated, active = logs.get("rotated.log") or {}, logs.get("active.log") or {}
        if (
            any(rotated.get(key) != old.get(key) for key in ("device", "inode"))
            or rotated.get("size", 0) <= old["size"]
            or active.get("inode") == old["inode"]
        ):
            errors.append(
                "rotation did not preserve the unread old inode and create a new active file"
            )
    if step == "truncate":
        errors.extend(truncation_errors(value, previous))
    old_ids = {event["record_id"] for item in previous for event in item["events"]}
    ids = [event.get("record_id") for event in events]
    if len(set(ids)) != len(ids) or old_ids.intersection(ids):
        errors.append("local reader replayed a prior record identity")
    return errors


def record_intent(path: Path, value: dict[str, Any]) -> None:
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


@bounded_collector_case
def execute_case(
    settings: Settings,
    run_dir: Path,
    attempt: int,
    maintenance_window_end: datetime,
) -> int:
    case_dir = run_dir / "cases" / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    preflight = read_only_preflight(settings, case_dir)
    if preflight["errors"]:
        raise RegionalFixtureError(
            "preflight failed: " + "; ".join(preflight["errors"])
        )
    planned = json.loads((case_dir / "plan.json").read_text())["details"][
        "preflight_identity"
    ]
    if planned != plan_details(settings, preflight)["preflight_identity"]:
        raise RegionalFixtureError("COLLECT-022 approved plan identity changed")
    require_action_time(180)
    regional = RegionalLiveFixture(settings.regional)
    nonce = uuid4().hex
    host = HostProbeFixture(
        HostProbeSettings(
            kubeconfig=settings.regional.gpu_kubeconfig,
            context=settings.regional.gpu_context,
            namespace=settings.regional.namespace,
            node=settings.node,
            image=settings.host_probe_image,
            case_id=CASE_ID,
            run_id=f"c022-{nonce}",
            probe_script=PROBE_SCRIPT,
            state_directory=case_dir / "host-probes",
            active_deadline_seconds=1800,
        )
    )
    intent_path = case_dir / f"private-cursor-intent-a{attempt}.json"
    intent = {
        "schema_version": 1,
        "nonce": nonce,
        "attempt": attempt,
        "state": "PREPARING",
        "node_uid": preflight["node"]["uid"],
        "identity": planned,
    }
    record_intent(intent_path, intent)
    common = (
        "--nonce",
        nonce,
        "--cluster-id",
        settings.regional.cluster_id,
        "--node-id",
        settings.node,
    )
    deadline = (
        "--expires-at",
        str(
            min(
                maintenance_window_end.timestamp(),
                datetime.now(timezone.utc).timestamp() + PRIVATE_WINDOW_SECONDS,
            )
        ),
    )
    initialized = False
    receipts: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "schema_version": 2,
        "report_type": "fault-acceptance",
        "case_id": CASE_ID,
        "attempt": attempt,
        "verdict": "FAIL",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "predecessor": preflight["predecessor"],
        **regional.evidence_identity(),
        "nonce": nonce,
        "phases": receipts,
        "limitations": [
            "Deployed-host private file/cursor proof only, not hardware fault evidence.",
            "The local recording sink never sends records to CPU ingestion.",
        ],
    }
    try:
        host.create()
        require_action_time(PROBE_TIMEOUT_SECONDS)
        initialized = True
        intent["state"] = "INITIALIZATION_ATTEMPTED"
        write_json_atomic(intent_path, intent)
        initial = host.execute(
            "init", *common, *deadline, timeout=PROBE_TIMEOUT_SECONDS
        )
        result["initialization"] = initial
        errors = initialization_errors(
            initial,
            nonce=nonce,
            settings=settings,
            boot_id=preflight["node"]["boot_id"],
        )
        if errors:
            raise RegionalFixtureError("; ".join(errors))
        for step in probe.STEPS:
            require_action_time(PROBE_TIMEOUT_SECONDS)
            receipt = host.execute(
                "step",
                *common,
                *deadline,
                "--step",
                step,
                timeout=PROBE_TIMEOUT_SECONDS,
            )
            write_json_atomic(case_dir / f"phase-{step}.json", receipt)
            errors = phase_errors(
                receipt, step=step, initial=initial, previous=receipts
            )
            receipts.append(receipt)
            if errors:
                raise RegionalFixtureError("; ".join(errors))
        result["verdict"] = "PASS"
    except BaseException as exc:
        result["verdict"] = "FAIL"
        result["error"] = f"{type(exc).__name__}: {exc}"
        if not isinstance(exc, Exception):
            raise
    finally:
        cleanup_errors = []
        if initialized:
            try:
                residuals = host.execute(
                    "cleanup", *common, timeout=PROBE_TIMEOUT_SECONDS
                )
                result["private_residuals"] = residuals
                if not {"private_root", "creation_unresolved"} <= set(residuals) or any(
                    value is not False for value in residuals.values()
                ):
                    cleanup_errors.append("private cursor cleanup is unconfirmed")
            except Exception as exc:
                result["private_residuals"] = {
                    "private_root": True,
                    "creation_unresolved": True,
                }
                cleanup_errors.append(
                    f"private cursor cleanup failed: {type(exc).__name__}: {exc}"
                )
        try:
            residuals = host.cleanup()
            result["probe_residuals"] = residuals
            if not residuals or any(value is not False for value in residuals.values()):
                cleanup_errors.append("HostProbe cleanup has unresolved residuals")
        except Exception as exc:
            result["probe_residuals"] = {"cleanup_error": True}
            cleanup_errors.append(
                f"HostProbe cleanup failed: {type(exc).__name__}: {exc}"
            )
        try:
            after = regional.node_snapshot(settings.node)
            if any(after.get(key) != preflight["node"].get(key) for key in NODE_FIELDS):
                cleanup_errors.append("target node identity/state changed")
            if regional.cpu_blast_snapshot() != preflight["cpu_blast"]:
                cleanup_errors.append("CPU cluster state changed")
        except Exception as exc:
            cleanup_errors.append(f"final read failed: {type(exc).__name__}: {exc}")
        if cleanup_errors:
            result["verdict"] = "FAIL"
            result["cleanup_errors"] = cleanup_errors
        intent["state"] = "CLEANUP_REQUIRED" if cleanup_errors else "CLOSED"
        write_json_atomic(intent_path, intent)
        result["ended_at"] = datetime.now(timezone.utc).isoformat()
        write_json_atomic(case_dir / f"{CASE_ID}.json", result)
    return 0 if result["verdict"] == "PASS" else 1


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    add_live_arguments(value, confirmation=CONFIRMATION)
    value.add_argument("--cpu-kubeconfig", default="")
    value.add_argument("--gpu-kubeconfig", default="")
    value.add_argument("--gpu-context", default="")
    value.add_argument("--namespace", default="gpu-fault-system")
    value.add_argument("--cluster-id", default="")
    value.add_argument("--region", default="")
    value.add_argument("--node", default="")
    value.add_argument("--host-probe-image", default="")
    value.add_argument("--predecessor-evidence", default="")
    return value


def main() -> int:
    return run_standard_case(
        CaseRunner(
            case_id=CASE_ID,
            confirmation=CONFIRMATION,
            parser=parser,
            configure=configure,
            read_only_preflight=read_only_preflight,
            plan_details=plan_details,
            execute_case=execute_case,
        )
    )


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))
