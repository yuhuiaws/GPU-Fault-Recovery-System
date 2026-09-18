"""CPU-only DESTR008 withdrawal watchdog, controlled by one UID-bound ConfigMap."""

from __future__ import annotations

import argparse
import logging
import os
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NoReturn, Protocol, cast
from urllib.parse import urlsplit

from kubernetes import client, config
from kubernetes.client.exceptions import ApiException
from urllib3.exceptions import HTTPError

from gpu_fault.settings import StoreSettings
from gpu_fault.store.postgres.pool import STORE_URL_FILE_ENV, StoreCredentials
from gpu_fault.store.postgres.store import PostgresStore
from gpu_fault.store.shared.errors import StaleWriteError, operation_should_retry

if TYPE_CHECKING or __package__:
    from .destr008_cancellation_protocol import (
        DRAIN_SECONDS,
        MAX_CLEANUP_SECONDS,
        QUIET_SECONDS,
        CleanupAttempt,
        Control,
        Plan,
        ProbeError,
        Receipt,
        decode,
        digest,
        encode,
        receipt,
        receipt_base,
        revoke,
        source_sha256,
        validate_control,
        validate_receipt,
    )
    from .destr008_cancellation_store import (
        CancellationStore,
        read_inventory,
        withdraw,
    )
else:
    from destr008_cancellation_protocol import (
        DRAIN_SECONDS,
        MAX_CLEANUP_SECONDS,
        QUIET_SECONDS,
        CleanupAttempt,
        Control,
        Plan,
        ProbeError,
        Receipt,
        decode,
        digest,
        encode,
        receipt,
        receipt_base,
        revoke,
        source_sha256,
        validate_control,
        validate_receipt,
    )
    from destr008_cancellation_store import (
        CancellationStore,
        read_inventory,
        withdraw,
    )

POLL_SECONDS = 2
RETRY_ATTEMPTS = 4
TRANSIENT_API_STATUSES = {409, 422, 429, 500, 502, 503, 504}
UNRESOLVED_CODES = {
    "INVENTORY_CHANGED",
    "RECORD_MISSING",
    "ROOT_MISSING",
    "DESCENDANT_UNRESOLVED",
    "API_UNAVAILABLE",
    "STORE_UNAVAILABLE",
    "CPU_STORE_UNAVAILABLE",
    "RECEIPT_UNAVAILABLE",
}


class ConfigMapApi(Protocol):
    def read_namespaced_config_map(
        self, name: str, namespace: str, **kwargs: Any
    ) -> Any: ...

    def patch_namespaced_config_map(
        self, name: str, namespace: str, body: list[dict[str, Any]], **kwargs: Any
    ) -> Any: ...


@dataclass(frozen=True)
class Envelope:
    plan: Plan
    version: str
    data: dict[str, str]


@dataclass(frozen=True)
class Snapshot:
    envelope: Envelope
    control: Control
    status: Receipt | None


class ControlPort(Protocol):
    uid: str

    def read(self) -> Snapshot: ...

    def write(
        self, snapshot: Snapshot, control: Control, status: Receipt
    ) -> Snapshot: ...

    def fail(self, *, code: str, now: int, monitoring: bool) -> Receipt: ...


class KubernetesControlMap:
    def __init__(
        self,
        api: ConfigMapApi,
        *,
        namespace: str,
        name: str,
        uid: str,
        plan_sha256: str,
        probe_sha256: str,
    ) -> None:
        if (
            any(
                not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,252}", value)
                for value in (namespace, name, uid)
            )
            or not re.fullmatch(r"[0-9a-f]{64}", plan_sha256)
            or not re.fullmatch(r"[0-9a-f]{64}", probe_sha256)
        ):
            raise ProbeError("CONFIGMAP_BINDING")
        self.api = api
        self.namespace = namespace
        self.name = name
        self.uid = uid
        self.plan_sha256 = plan_sha256
        self.probe_sha256 = probe_sha256

    def envelope(self, value: Any) -> Envelope:
        metadata = getattr(value, "metadata", None)
        data = getattr(value, "data", None)
        version = getattr(metadata, "resource_version", None)
        if (
            getattr(value, "api_version", None) != "v1"
            or getattr(value, "kind", None) != "ConfigMap"
            or getattr(metadata, "name", None) != self.name
            or getattr(metadata, "namespace", None) != self.namespace
            or getattr(metadata, "uid", None) != self.uid
            or getattr(metadata, "deletion_timestamp", None) is not None
            or not isinstance(version, str)
            or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", version)
            or getattr(value, "binary_data", None) not in (None, {})
            or getattr(value, "immutable", None) not in (None, False)
            or not isinstance(data, dict)
            or set(data) != {"plan.json", "control.json", "status.json"}
        ):
            raise ProbeError("CONFIGMAP_BINDING")
        plan = decode(Plan, data["plan.json"])
        if digest(plan) != self.plan_sha256 or plan.probe_sha256 != self.probe_sha256:
            raise ProbeError("PLAN_SOURCE")
        return Envelope(plan, version, dict(data))

    def read_envelope(self) -> Envelope:
        return self.envelope(
            self.api.read_namespaced_config_map(
                self.name, self.namespace, _request_timeout=(3, 5)
            )
        )

    def snapshot(self, envelope: Envelope) -> Snapshot:
        return Snapshot(
            envelope,
            decode(Control, envelope.data["control.json"]),
            None
            if envelope.data["status.json"] == "null"
            else decode(Receipt, envelope.data["status.json"]),
        )

    def read(self) -> Snapshot:
        return self.snapshot(self.read_envelope())

    def patch(
        self, envelope: Envelope, control: Control | None, status: Receipt
    ) -> Envelope:
        patch: list[dict[str, Any]] = [
            {"op": "test", "path": "/metadata/uid", "value": self.uid},
            {
                "op": "test",
                "path": "/metadata/resourceVersion",
                "value": envelope.version,
            },
            *[
                {"op": "test", "path": f"/data/{key}", "value": value}
                for key, value in envelope.data.items()
            ],
        ]
        if control is not None:
            patch.append(
                {
                    "op": "replace",
                    "path": "/data/control.json",
                    "value": encode(control),
                }
            )
        patch.append(
            {"op": "replace", "path": "/data/status.json", "value": encode(status)}
        )
        updated = self.envelope(
            self.api.patch_namespaced_config_map(
                self.name, self.namespace, patch, _request_timeout=(3, 5)
            )
        )
        if (
            updated.version == envelope.version
            or updated.data["status.json"] != encode(status)
            or updated.data["control.json"]
            != (envelope.data["control.json"] if control is None else encode(control))
        ):
            raise ProbeError("CAS_NOT_ACKNOWLEDGED")
        return updated

    def write(self, snapshot: Snapshot, control: Control, status: Receipt) -> Snapshot:
        return self.snapshot(self.patch(snapshot.envelope, control, status))

    def fail(self, *, code: str, now: int, monitoring: bool) -> Receipt:
        envelope = self.read_envelope()
        control: Control | None = None
        try:
            decoded_control = decode(Control, envelope.data["control.json"])
            validate_control(envelope.plan, decoded_control, now)
            control = decoded_control
        except ProbeError:
            pass
        previous: Receipt | None = None
        try:
            decoded_status = decode(Receipt, envelope.data["status.json"])
            binding = receipt_base(
                envelope.plan, uid=self.uid, now=decoded_status.observed_at
            )
            if all(
                getattr(decoded_status, key) == value for key, value in binding.items()
            ):
                previous = decoded_status
        except ProbeError:
            pass
        if control is not None:
            if previous is not None and previous.revocation is not None:
                control = control.model_copy(update={"revocation": previous.revocation})
            control = revoke(control, now=now, reason="FAILURE")
        preserved: dict[str, Any] = {}
        if control is None and previous is not None and previous.revocation is not None:
            preserved = {
                "producer": previous.producer,
                "revocation": previous.revocation,
                "producer_revoked": True,
            }
        failed = receipt(
            envelope.plan,
            control,
            previous,
            uid=self.uid,
            now=now,
            state="FAILED",
            error_code=code,
            monitoring=monitoring,
            **preserved,
        )
        self.patch(envelope, control, failed)
        return failed


def transient(error: Exception) -> bool:
    return (
        (isinstance(error, ApiException) and error.status in TRANSIENT_API_STATUSES)
        or isinstance(error, (HTTPError, TimeoutError, StaleWriteError))
        or operation_should_retry(error)
    )


class Watchdog:
    def __init__(
        self,
        port: ControlPort,
        store: CancellationStore,
        *,
        sleep: Callable[[float], None] = time.sleep,
        cleanup_only: bool = False,
        cleanup_seconds: int = MAX_CLEANUP_SECONDS,
        cleanup_attempt_id: str | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if (
            type(cleanup_only) is not bool
            or type(cleanup_seconds) is not int
            or not QUIET_SECONDS <= cleanup_seconds <= MAX_CLEANUP_SECONDS
            or (
                cleanup_only
                and (
                    not isinstance(cleanup_attempt_id, str)
                    or not re.fullmatch(
                        r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,252}", cleanup_attempt_id
                    )
                )
            )
            or (not cleanup_only and cleanup_attempt_id is not None)
        ):
            raise ProbeError("CLEANUP_ARGUMENTS")
        self.port = port
        self.store = store
        self.sleep = sleep
        self.cleanup_only = cleanup_only
        self.cleanup_seconds = cleanup_seconds
        self.cleanup_attempt_id = cleanup_attempt_id
        self.monotonic = monotonic
        self.cleanup: CleanupAttempt | None = None
        self.cleanup_committed = False
        self.cleanup_monotonic_deadline = 0.0
        self.tick_at = 0
        self.tick_started = self.monotonic()

    def current_at(self) -> int:
        return self.tick_at + max(0, int(self.monotonic() - self.tick_started))

    def checkpoint(self) -> None:
        if self.cleanup is not None and (
            self.current_at() >= self.cleanup.deadline_at
            or self.monotonic() >= self.cleanup_monotonic_deadline
        ):
            raise ProbeError("CLEANUP_TIMEOUT")

    def prepare_cleanup(self, snapshot: Snapshot, now: int) -> Snapshot:
        previous = snapshot.status
        if self.cleanup is None:
            old = previous.cleanup if previous is not None else None
            if old is not None and old.attempt_id == self.cleanup_attempt_id:
                if old.deadline_at - old.started_at != self.cleanup_seconds:
                    raise ProbeError("CLEANUP_BINDING")
                self.cleanup = old
            else:
                self.cleanup = CleanupAttempt.model_validate(
                    {
                        "attempt_id": self.cleanup_attempt_id,
                        "started_at": now,
                        "deadline_at": now + self.cleanup_seconds,
                    }
                )
            self.cleanup_monotonic_deadline = self.monotonic() + max(
                0, self.cleanup.deadline_at - now
            )
        if not self.cleanup_committed:
            control = revoke(snapshot.control, now=now, reason="FAILURE")
            status = receipt(
                snapshot.envelope.plan,
                control,
                previous,
                uid=self.port.uid,
                now=now,
                state="REVOKED",
                cleanup=self.cleanup,
                quiet_since=None,
            )
            snapshot = self.port.write(snapshot, control, status)
            self.cleanup_committed = True
        elif previous is None or previous.cleanup != self.cleanup:
            raise ProbeError("CLEANUP_ATTEMPT_CHANGED")
        self.checkpoint()
        return snapshot

    def advance(self, now: int) -> Receipt:
        snapshot = self.port.read()
        now = self.current_at()
        plan, control, previous = (
            snapshot.envelope.plan,
            snapshot.control,
            snapshot.status,
        )
        validate_control(plan, control, now)
        if previous is not None:
            validate_receipt(
                plan,
                control,
                previous,
                uid=self.port.uid,
                now=now,
                cleanup_only=self.cleanup_only,
            )
            if not previous.monitoring and not self.cleanup_only:
                return previous
        elif control.producer.state != "NOT_STARTED" and not self.cleanup_only:
            raise ProbeError("UNARMED_SUBMISSION")
        if self.cleanup_only:
            snapshot = self.prepare_cleanup(snapshot, now)
            control, previous = snapshot.control, snapshot.status
        if control.revocation is None:
            if now < plan.deadline_at and control.close_request is None:
                armed = receipt(
                    plan, control, previous, uid=self.port.uid, now=now, state="ARMED"
                )
                return cast(Receipt, self.port.write(snapshot, control, armed).status)
            control = revoke(
                control,
                now=now,
                reason="DEADLINE" if now >= plan.deadline_at else "PARENT_CLOSE",
            )
            revoked = receipt(
                plan, control, previous, uid=self.port.uid, now=now, state="REVOKED"
            )
            # This CAS must finish before the first Store read, including on resume.
            snapshot = self.port.write(snapshot, control, revoked)
            previous = snapshot.status
        self.checkpoint()
        before = read_inventory(
            self.store, plan, control, previous, checkpoint=self.checkpoint
        )
        if (
            not self.cleanup_only
            and control.close_request is not None
            and control.close_request.reason == "NEGATIVE_TERMINAL"
            and before.root is not None
            and before.pairs[before.root.workflow_request_id][1].status.value
            == "SUCCEEDED"
        ):
            raise ProbeError("NOT_NEGATIVE_TERMINAL")
        self.checkpoint()
        observed = before.proof()
        now = self.current_at()
        progress = receipt(
            plan,
            control,
            previous,
            uid=self.port.uid,
            now=now,
            state="REVOKED",
            **observed,
            quiet_since=(
                previous.quiet_since
                if previous is not None
                and previous.inventory_sha256 == observed["inventory_sha256"]
                else None
            ),
        )
        # Persist observed IDs before mutation so a restarted Job cannot use a
        # disappeared command or workflow as evidence that it stopped.
        snapshot = self.port.write(snapshot, control, progress)
        previous = snapshot.status
        withdraw(
            self.store,
            plan,
            control,
            before,
            now=now,
            checkpoint=self.checkpoint,
        )
        # Never report a cancellation request as physical completion. A second
        # typed scan sees actual leases, new descendants, and in-flight creators.
        after = read_inventory(
            self.store, plan, control, previous, checkpoint=self.checkpoint
        )
        self.checkpoint()
        now = self.current_at()
        proof = after.proof()
        ready = (
            proof["source_complete"]
            and proof["commands_active"] == 0
            and proof["workflows_active"] == 0
            and proof["pending_creation"] is False
        )
        quiet_since = None
        if ready:
            quiet_since = (
                previous.quiet_since
                if previous is not None
                and previous.inventory_sha256 == proof["inventory_sha256"]
                and previous.quiet_since is not None
                else now
            )
        state = "REVOKED"
        error = None
        if quiet_since is not None and now - quiet_since >= QUIET_SECONDS:
            state = "QUIESCENT"
        elif not proof["source_complete"]:
            state, error = "FAILED", "SOURCE_UNRESOLVED"
        elif (
            not self.cleanup_only
            and control.revocation is not None
            and now - control.revocation.at >= DRAIN_SECONDS
        ):
            state, error = "FAILED", "DRAIN_UNRESOLVED"
        status = receipt(
            plan,
            control,
            previous,
            uid=self.port.uid,
            now=now,
            state=state,
            **proof,
            quiet_since=quiet_since,
            error_code=error,
            monitoring=state != "QUIESCENT",
        )
        return cast(Receipt, self.port.write(snapshot, control, status).status)

    def tick(self, now: int) -> Receipt:
        self.tick_at = now
        self.tick_started = self.monotonic()
        for attempt in range(RETRY_ATTEMPTS):
            try:
                return self.advance(now)
            except Exception as error:
                if transient(error) and attempt + 1 < RETRY_ATTEMPTS:
                    self.sleep(0.25 * 2**attempt)
                    continue
                code = (
                    error.code
                    if isinstance(error, ProbeError)
                    else (
                        "API_UNAVAILABLE"
                        if isinstance(error, (ApiException, HTTPError, TimeoutError))
                        else "STORE_UNAVAILABLE"
                        if transient(error)
                        else "STORE_ERROR"
                    )
                )
                return persist_failure(
                    self.port,
                    code=code,
                    now=self.current_at(),
                    monitoring=code in UNRESOLVED_CODES,
                    sleep=self.sleep,
                )
        raise ProbeError("RETRY_EXHAUSTED")


def persist_failure(
    port: ControlPort,
    *,
    code: str,
    now: int,
    monitoring: bool,
    sleep: Callable[[float], None] = time.sleep,
) -> Receipt:
    for attempt in range(RETRY_ATTEMPTS):
        try:
            return port.fail(code=code, now=now, monitoring=monitoring)
        except Exception as error:
            if not transient(error) or attempt + 1 == RETRY_ATTEMPTS:
                raise ProbeError("RECEIPT_UNAVAILABLE") from None
            sleep(0.25 * 2**attempt)
    raise ProbeError("RETRY_EXHAUSTED")


def cpu_store() -> CancellationStore:
    from psycopg.conninfo import conninfo_to_dict

    settings = StoreSettings.from_mapping(os.environ)
    if settings.kind != "postgres" or settings.postgres_auto_schema_init:
        raise ProbeError("CPU_STORE_CONFIGURATION")
    credentials = StoreCredentials(
        settings.url, path=os.environ.get(STORE_URL_FILE_ENV) or None
    )
    try:
        connection = conninfo_to_dict(credentials.conninfo())
    except Exception:
        raise ProbeError("CPU_STORE_CONFIGURATION") from None
    if connection.get("sslmode") != "verify-full" or not connection.get("host"):
        raise ProbeError("CPU_STORE_TLS_REQUIRED")
    # This Job has its own small pool, independent of control-worker capacity.
    return cast(
        CancellationStore,
        PostgresStore(
            settings.url,
            initialize_schema=False,
            pool_min_size=0,
            pool_max_size=2,
            pool_timeout_seconds=2,
        ),
    )


def read_initial(
    port: ControlPort, *, sleep: Callable[[float], None] = time.sleep
) -> Snapshot:
    for attempt in range(RETRY_ATTEMPTS):
        try:
            return port.read()
        except Exception as error:
            if not transient(error):
                raise
            if attempt + 1 == RETRY_ATTEMPTS:
                raise ProbeError("API_UNAVAILABLE") from None
            sleep(0.25 * 2**attempt)
    raise ProbeError("RETRY_EXHAUSTED")


def run(
    port: ControlPort,
    store: CancellationStore,
    *,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
    emit: Callable[[str], None] = print,
    cleanup_only: bool = False,
    cleanup_seconds: int = MAX_CLEANUP_SECONDS,
    cleanup_attempt_id: str | None = None,
) -> int:
    watchdog = Watchdog(
        port,
        store,
        sleep=sleep,
        cleanup_only=cleanup_only,
        cleanup_seconds=cleanup_seconds,
        cleanup_attempt_id=cleanup_attempt_id,
    )
    last_state = None
    while True:
        status = watchdog.tick(int(clock()))
        if status.state != last_state or not status.monitoring:
            emit(encode(status))
            last_state = status.state
        if not status.monitoring:
            return 0 if status.state == "QUIESCENT" else 1
        sleep(POLL_SECONDS)


class ProbeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise ProbeError("CLI_ARGUMENTS")


def unpersisted_failure(code: str) -> int:
    print(
        encode(
            {
                "state": "FAILED",
                "error_code": code,
                "receipt_persisted": False,
                "fence_release_authorized": False,
            }
        ),
        flush=True,
    )
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = ProbeArgumentParser(description=__doc__)
    for key in ("namespace", "configmap", "uid", "plan-sha256"):
        parser.add_argument(f"--{key}", required=True)
    parser.add_argument("--cleanup-only", action="store_true")
    parser.add_argument("--cleanup-seconds", type=int, default=MAX_CLEANUP_SECONDS)
    parser.add_argument("--cleanup-attempt-id")
    try:
        options = parser.parse_args(argv)
        if not options.cleanup_only and (
            options.cleanup_attempt_id is not None
            or options.cleanup_seconds != MAX_CLEANUP_SECONDS
        ):
            raise ProbeError("CLEANUP_ARGUMENTS")
    except ProbeError as error:
        return unpersisted_failure(error.code)
    # Upstream database/network exception logging may carry credential material.
    # This dedicated process emits only its sanitized, durably written receipts.
    logging.disable(logging.CRITICAL)
    port: KubernetesControlMap | None = None
    store: CancellationStore | None = None
    try:
        if os.environ.get("KUBECONFIG") not in {None, "", "/dev/null"}:
            raise ProbeError("KUBECONFIG_FORBIDDEN")
        configuration = client.Configuration()
        config.load_incluster_config(client_configuration=configuration)
        if (
            configuration.verify_ssl is not True
            or urlsplit(configuration.host).scheme != "https"
        ):
            raise ProbeError("CPU_API_TLS_REQUIRED")
        with client.ApiClient(configuration) as api_client:
            port = KubernetesControlMap(
                client.CoreV1Api(api_client),
                namespace=options.namespace,
                name=options.configmap,
                uid=options.uid,
                plan_sha256=options.plan_sha256,
                probe_sha256=source_sha256(),
            )
            try:
                initial = read_initial(port)
                now = int(time.time())
                validate_control(initial.envelope.plan, initial.control, now)
                if initial.status is not None:
                    validate_receipt(
                        initial.envelope.plan,
                        initial.control,
                        initial.status,
                        uid=port.uid,
                        now=now,
                        cleanup_only=options.cleanup_only,
                    )
                    if not initial.status.monitoring and not options.cleanup_only:
                        print(encode(initial.status), flush=True)
                        return 0 if initial.status.state == "QUIESCENT" else 1
                store = cpu_store()
                if options.cleanup_only:
                    return run(
                        port,
                        store,
                        cleanup_only=True,
                        cleanup_seconds=options.cleanup_seconds,
                        cleanup_attempt_id=options.cleanup_attempt_id,
                    )
                return run(port, store)
            except Exception as error:
                code = (
                    error.code
                    if isinstance(error, ProbeError)
                    else "CPU_STORE_UNAVAILABLE"
                    if transient(error)
                    else "CPU_STORE_ERROR"
                )
                failed = persist_failure(
                    port,
                    code=code,
                    now=int(time.time()),
                    monitoring=code in UNRESOLVED_CODES,
                )
                print(encode(failed), flush=True)
                return 1
            finally:
                if store is not None:
                    store.close()
    except Exception as error:
        code = error.code if isinstance(error, ProbeError) else "CPU_BRIDGE_ERROR"
        return unpersisted_failure(code)


if __name__ == "__main__":
    raise SystemExit(main())
