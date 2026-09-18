"""Run only inside HA011's new, private production-image Pod."""

from __future__ import annotations

import json
import logging
import os
import re
import stat
import sys
import time
from contextlib import ExitStack, closing
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable

from gpu_fault.processor import ProcessorRequest, ProcessorRequestStatus
from gpu_fault.store import PostgresStore
from gpu_fault.store.shared.errors import StaleFencingTokenError
from scripts.e2e.regional.ha011_contracts import (
    BACKLOG,
    BOUNDARY,
    CASE_ID,
    FAILURE_TYPES,
    QUEUE_PATH,
    ROLES,
    SPOOL_PATH,
    TOKEN,
    ProofError,
    digest,
    evidence_errors,
)
from scripts.e2e.regional.probes.ha011_processes import WorkerProcess
from scripts.e2e.regional.probes.ha011_workers import Emitter, run_worker

PASSWORD_FILE = Path("/private/password")
SERVICE_ACCOUNT = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")
ARM_FILE = Path("/arm/armed.json")
# Kubelet supplies the default API service even when service links are disabled.
SERVICE_DISCOVERY_ENV = frozenset(
    {
        "KUBERNETES_SERVICE_HOST",
        "KUBERNETES_SERVICE_PORT",
        "KUBERNETES_SERVICE_PORT_HTTPS",
        "KUBERNETES_PORT",
        "KUBERNETES_PORT_443_TCP",
        "KUBERNETES_PORT_443_TCP_ADDR",
        "KUBERNETES_PORT_443_TCP_PORT",
        "KUBERNETES_PORT_443_TCP_PROTO",
    }
)


def arm(pod_uid: str, isolation_id: str, intent_sha256: str) -> dict[str, Any]:
    observed_id, observed_uid = isolated_identity()
    if (observed_id, observed_uid) != (isolation_id, pod_uid) or not re.fullmatch(
        r"[a-f0-9]{64}", intent_sha256
    ):
        raise ProofError("arm request does not match the live owned Pod")
    receipt = {
        "armed": True,
        "pod_uid": pod_uid,
        "isolation_id": isolation_id,
        "intent_sha256": intent_sha256,
    }
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=ARM_FILE.parent
    ) as temporary:
        json.dump(receipt, temporary)
        temporary.flush()
        os.fsync(temporary.fileno())
        os.link(temporary.name, ARM_FILE)
    return receipt


def wait_for_arm(*, timeout: float = 60) -> dict[str, Any]:
    isolation_id, pod_uid = isolated_identity()
    deadline = time.monotonic() + timeout
    while not ARM_FILE.exists():
        if time.monotonic() >= deadline:
            raise ProofError(
                "owned start barrier expired before arm; no database work started"
            )
        time.sleep(0.1)
    status = ARM_FILE.lstat()
    if not stat.S_ISREG(status.st_mode) or status.st_size > 4096:
        raise ProofError("owned start barrier is not a bounded regular receipt")
    receipt = json.loads(ARM_FILE.read_text(encoding="utf-8"))
    if (
        not isinstance(receipt, dict)
        or set(receipt) != {"armed", "pod_uid", "isolation_id", "intent_sha256"}
        or receipt["armed"] is not True
        or receipt["pod_uid"] != pod_uid
        or receipt["isolation_id"] != isolation_id
        or not isinstance(receipt["intent_sha256"], str)
        or not re.fullmatch(r"[a-f0-9]{64}", receipt["intent_sha256"])
    ):
        raise ProofError("owned start barrier receipt is not bound to this Pod")
    return receipt


def isolated_identity() -> tuple[str, str]:
    isolation_id = os.environ.get("HA011_ISOLATION_ID", "")
    pod_uid = os.environ.get("POD_UID", "")
    if (
        not TOKEN.fullmatch(isolation_id)
        or not pod_uid
        or os.environ.get("POD_NAMESPACE") != f"gf-regional-ha011-{isolation_id}"
        or os.environ.get("HA011_BOUNDARY") != BOUNDARY
        or SERVICE_ACCOUNT.exists()
    ):
        raise ProofError(
            "probe requires its own namespace, Pod identity "
            "and no service-account token"
        )
    allowed = {
        "AWS_CONFIG_FILE": "/dev/null",
        "AWS_SHARED_CREDENTIALS_FILE": "/dev/null",
        "AWS_EC2_METADATA_DISABLED": "true",
        "KUBECONFIG": "/dev/null",
    }
    for key, value in os.environ.items():
        if (
            key.startswith(("GPU_FAULT_", "PG", "AWS_", "KUBE"))
            and key not in SERVICE_DISCOVERY_ENV
            and allowed.get(key) != value
        ):
            raise ProofError(
                "inherited credential or runtime configuration is forbidden"
            )
        if key.lower() in {"http_proxy", "https_proxy", "all_proxy"} and value:
            raise ProofError("proxy routing is forbidden in the isolated probe")
    return isolation_id, pod_uid


def private_store(*, initialize: bool) -> PostgresStore:
    import psycopg
    from psycopg.conninfo import make_conninfo

    isolated_identity()
    wait_for_arm()
    password = PASSWORD_FILE.read_text(encoding="utf-8").strip()
    if len(password) < 32 or any(char in password for char in "\r\n\0"):
        raise ProofError("isolated database credential is missing or malformed")
    url = make_conninfo(
        host="127.0.0.1",
        port=5432,
        dbname="postgres",
        user="postgres",
        password=password,
        connect_timeout=3,
        options="-c statement_timeout=10000",
    )
    deadline = time.monotonic() + 90
    while True:
        try:
            with psycopg.connect(url, autocommit=True) as connection:
                if not 160000 <= connection.info.server_version < 170000:
                    raise ProofError("isolated probe requires PostgreSQL 16")
                aurora = connection.execute(
                    "SELECT EXISTS (SELECT 1 FROM pg_proc "
                    "WHERE proname='aurora_version')"
                ).fetchone()
                if aurora != (False,):
                    raise ProofError("the isolated probe refuses Aurora")
                if initialize:
                    existing = connection.execute(
                        "SELECT EXISTS (SELECT 1 FROM pg_tables "
                        "WHERE schemaname='public' AND tablename LIKE 'gpu\\_fault%')"
                    ).fetchone()
                    if existing != (False,):
                        raise ProofError(
                            "schema setup refuses an existing runtime database"
                        )
            break
        except psycopg.OperationalError:
            if time.monotonic() >= deadline:
                raise ProofError("isolated database startup deadline expired") from None
            time.sleep(0.2)
    # Schema setup is explicit and occurs only in the probe parent on an empty DB.
    # Actual runtime children always open with initialize_schema=False.
    return PostgresStore(url, initialize_schema=initialize, pool_max_size=4)


def worker_entry(
    role: str,
    target: str,
    connection: Any,
    stop: Any,
    release: Any,
    *,
    store_factory: Callable[[], Any] | None = None,
) -> None:
    logging.disable(logging.CRITICAL)
    try:
        supplied = (
            private_store(initialize=False)
            if store_factory is None
            else store_factory()
        )
        with closing(supplied) as store:
            run_worker(store, role, target, connection, stop, release)
    except BaseException:
        Emitter(connection).emit("error")
        raise SystemExit(1) from None
    finally:
        connection.close()


def sample(isolation_id: str, role: str, number: int) -> ProcessorRequest:
    payload = {
        "node_id": f"acceptance-{isolation_id}-{number}",
        "summary": True,
        "lines": [],
    }
    path = SPOOL_PATH if role == "spool" else QUEUE_PATH
    return ProcessorRequest.from_http(
        method="POST",
        path=path,
        query="",
        body=json.dumps(payload).encode(),
        content_type="application/json",
        cluster_id=f"ha011-{isolation_id}",
    ).model_copy(update={"request_id": f"{isolation_id}-{role}-{number}"})


def admit(store: Any, role: str, requests: list[ProcessorRequest]) -> None:
    if role == "spool":
        result = store.try_spool_telemetry_requests(
            requests, max_depth=16, max_cluster_depth=16, now=datetime.now(timezone.utc)
        )
        if any(item is None or reason is not None for item, reason in result) or len(
            result
        ) != len(requests):
            raise ProofError(
                "isolated spool admission lost or coalesced acceptance work"
            )
    else:
        for request in requests:
            store.enqueue_processor_request(request)


def depth(store: Any, role: str) -> int:
    stats = (
        store.telemetry_spool_stats()
        if role == "spool"
        else store.processor_queue_stats()
    )
    return int(stats["depth"])


def live_claim(store: Any, role: str, item: Any) -> bool:
    if role == "spool":
        stats = store.telemetry_spool_stats()
        return bool(stats["leased"] == 1 and stats["depth"] >= 1)
    current = store.get_processor_request(item.request_id)
    return bool(
        current.status is ProcessorRequestStatus.LEASED
        and current.lease_token == item.lease_token
        and current.lease_expires_at is not None
        and current.lease_expires_at > datetime.now(timezone.utc)
    )


def refuse_old_completion(store: Any, role: str, claim: dict[str, Any]) -> bool:
    item = claim["item"]
    if role == "spool":
        count = store.complete_telemetry_spool([item])
        return type(count) is int and count == 0
    try:
        store.complete_active_processor_request(
            item.request_id,
            claim["owner"],
            item.leader_epoch,
            item.lease_token,
            response_status=200,
            response_content_type="application/json",
            response_body_base64="e30=",
            path=item.path,
        )
    except StaleFencingTokenError:
        return True
    return False


def work_identity(role: str, item: Any) -> str:
    queued = isinstance(item, ProcessorRequest)
    return digest(
        {
            "request_id": item.request_id,
            "cluster_id": item.cluster_id,
            "path": item.path,
            "key": (item.spool_key() if role == "spool" else item.ordering_key())
            if queued
            else item.spool_key,
            "payload": json.loads(item.body()) if queued else item.payload,
        }
    )


def exercise_role(
    store: Any,
    role: str,
    isolation_id: str,
    *,
    factory: Callable[..., Any] | None = None,
    budget_seconds: float = 100,
) -> dict[str, Any]:
    requests = [sample(isolation_id, role, number) for number in range(BACKLOG + 1)]
    target = requests[0].request_id
    deadline = time.monotonic() + budget_seconds
    make_worker = WorkerProcess if factory is None else factory
    admit(store, role, requests[:1])
    with ExitStack() as cleanup:

        def start() -> Any:
            worker = make_worker(worker_entry, role=role, request_id=target)
            cleanup.callback(worker.close)
            worker.wait("ready", deadline)
            return worker

        old = start()
        old_claim = old.wait("claim", deadline, target)
        old_busy = old.wait("busy", deadline, target)
        contender = start()
        contender.wait("idle", deadline)
        contender.finish()
        early_claims = sum(event["kind"] == "claim" for event in contender.events)
        admit(store, role, requests[1:])
        backlog_at_crash = depth(store, role)
        exitcode = old.crash()
        replacement = start()
        new_claim = replacement.wait("claim", deadline, target)
        new_busy = replacement.wait("busy", deadline, target)
        before = live_claim(store, role, new_claim["item"]) and replacement.alive()
        if not before or replacement.release.is_set():
            raise ProofError("replacement is not still busy before the stale callback")
        refused = refuse_old_completion(store, role, old_claim)
        after = live_claim(store, role, new_claim["item"]) and replacement.alive()
        if not refused or not after or replacement.release.is_set():
            raise ProofError(
                "old completion was not fenced while replacement owned live work"
            )
        replacement.release.set()
        for request in requests:
            replacement.wait("complete", deadline, request.request_id)
        replacement.finish()
        completed = [
            event["request_id"]
            for event in replacement.events
            if event["kind"] == "complete"
        ]
        if sorted(completed) != sorted(request.request_id for request in requests):
            raise ProofError(
                "backlog completion identity is missing, duplicate, or foreign"
            )
        claims = [
            event["item"] for event in replacement.events if event["kind"] == "claim"
        ]
        expected = {
            request.request_id: work_identity(role, request) for request in requests
        }
        if sorted(item.request_id for item in claims) != sorted(expected) or any(
            work_identity(role, item) != expected[item.request_id] for item in claims
        ):
            raise ProofError(
                "replacement claims were duplicated or changed durable backlog payloads"
            )
        return {
            "same_durable_work": work_identity(role, old_claim["item"])
            == work_identity(role, new_claim["item"])
            and (
                role != "spool"
                or old_claim["item"].revision == new_claim["item"].revision
            ),
            "work_sha256": work_identity(role, new_claim["item"]),
            "fence_changed": bool(getattr(old_claim["item"], "lease_token", None))
            and getattr(old_claim["item"], "lease_token", None)
            != getattr(new_claim["item"], "lease_token", None),
            "owners": [old_claim["owner"], new_claim["owner"]],
            "early_claim_count": early_claims,
            "old_exitcode": exitcode,
            "old_cpu_seconds": old_busy["cpu_seconds"],
            "replacement_cpu_seconds": new_busy["cpu_seconds"],
            "late_completion_refused": refused,
            "replacement_live_before_late": before,
            "replacement_live_after_late": after,
            "backlog_at_crash": backlog_at_crash,
            "completed_count": len(completed),
            "final_depth": depth(store, role),
            "owned_processes_stopped": not any(
                worker.alive() for worker in (old, contender, replacement)
            ),
        }


def main() -> int:
    logging.disable(logging.CRITICAL)
    stage = "identity"
    try:
        isolation_id, pod_uid = isolated_identity()
        stage = "arm"
        receipt = wait_for_arm()
        stage = "database"
        with closing(private_store(initialize=True)) as setup:
            if depth(setup, "processor") or depth(setup, "spool"):
                raise ProofError("isolated schema setup did not produce empty queues")
        with closing(private_store(initialize=False)) as store:
            roles = {}
            for role in ROLES:
                stage = role
                roles[role] = exercise_role(store, role, isolation_id)
        stage = "evidence"
        result = {
            "case_id": CASE_ID,
            "validation_scope": BOUNDARY,
            "isolation_id": isolation_id,
            "pod_uid": pod_uid,
            "postgres_major": 16,
            "roles": roles,
            "arm_intent_sha256": receipt["intent_sha256"],
            "business_worker_targeted": False,
            "cpu_saturation_tested": False,
        }
        errors = evidence_errors(
            result,
            isolation_id=isolation_id,
            pod_uid=pod_uid,
            intent_sha256=receipt["intent_sha256"],
        )
        if errors:
            raise ProofError(
                "isolated runtime evidence did not satisfy the case contract"
            )
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        # Never expose credentials, connection strings or captured lease objects.
        print(
            json.dumps(
                {
                    "case_id": CASE_ID,
                    "stage": stage,
                    "error_type": (
                        type(exc).__name__
                        if type(exc).__name__ in FAILURE_TYPES
                        else "OtherError"
                    ),
                    "verdict": "FAIL",
                }
            )
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
