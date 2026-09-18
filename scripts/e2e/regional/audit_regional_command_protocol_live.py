"""Live audit of the regional remote-command protocol (GF-REGIONAL-CMD-001..016).

Runs inside an API Pod against ``http://127.0.0.1:8080`` and the live store.
Every command it seeds carries a test-only ``execution_owner`` unless the case
asserts something about a real owner (CMD-003 default owner, CMD-004 owner
filter), and every command it leases it hands back as ``WAITING`` before the
case ends, so nothing it touches is left leased against the production queue.

The audit refuses to start unless the target cluster has zero open remote
commands and the production executor cannot claim: either the operator asserts
``--isolated-cluster`` or reports ``--executor-ready-replicas 0`` (read from the
executor Deployment outside the Pod, where kubectl exists). Each selected case
is judged on its own; a failed case or cleanup stops the remaining selection.
Given ``--run-dir``, each executed case writes
``cases/<id>/<id>.json`` so the CMD cases can carry evidence and satisfy the
predecessor chain.

Hot-registered perf-cap-000/001 callers may supply a short-lived, run-bound
credential envelope on protected stdin. Execute the emitted source with
``python3 -c`` in that mode: ``python3 -`` already consumes stdin as source.
The CPU-local deadline includes a separate, cumulative owned-cleanup budget.
The envelope must cover the remaining work and hard cleanup deadline plus a
one-second guard. The 30-second cleanup default is for a single case, not a
qualification of the full 15-case selection. Multi-case callers must explicitly
set ``--cleanup-seconds`` (at most 120) for all case cleanups and final Store close;
that allowance is never renewed per case.
A hard watchdog exit is not cleanup evidence and requires independent recovery.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import select
import signal
import stat
import subprocess
import sys
import time
import urllib.parse
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, Protocol, cast
from uuid import uuid4

from pydantic import ConfigDict, Field, SecretStr

from gpu_fault.execution import (
    ProductionExecutorConfig,
    ProductionWorkflowExecutor,
    WorkflowStepContext,
)
from gpu_fault.execution.restart_budget_preflight import reserve_restart_budgets
from gpu_fault.models import (
    FaultIncident,
    RestartBudgetState,
    StrictModel,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
)
from gpu_fault.regional import (
    TOKEN_SLOT_CURRENT,
    RegionalRemoteWorkflowAdapter,
    RemoteActionCommand,
    regional_registry_content_sha256,
)
from gpu_fault.regional_compatibility import CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION
from gpu_fault.settings import StoreSettings
from gpu_fault.store import NotFoundError, PostgresStore
from gpu_fault.store.contracts import ControlPlaneStore
from gpu_fault.store.postgres.pool import StoreCredentials
from gpu_fault.store.shared.remote_commands import STALE_FENCE_STATUS_SOURCE

try:
    ROOT: Path | None = Path(__file__).resolve().parents[3]
except (NameError, IndexError):  # A bundled Pod probe has no checkout.
    ROOT = None
if ROOT is not None and str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __package__:
    from .command_protocol_probes import (
        ProtocolAuditError,
        anonymous_submission_checks,
        build_audit_parser,
        check_result_payloads,
        claim_records,
        cleanup_audit_records,
        expect,
        make_audit_pair,
        make_submission_probe_record,
        request_json,
    )
else:
    from command_protocol_probes import (  # type: ignore[no-redef]
        ProtocolAuditError,
        anonymous_submission_checks,
        build_audit_parser,
        check_result_payloads,
        claim_records,
        cleanup_audit_records,
        expect,
        make_audit_pair,
        make_submission_probe_record,
        request_json,
    )

AUDITED_CASE_IDS = tuple(
    [
        *(f"GF-REGIONAL-CMD-{number:03d}" for number in range(1, 11)),
        *(f"GF-REGIONAL-CMD-{number:03d}" for number in range(12, 17)),
    ]
)
KUBERNETES_OWNER = "gpu-fault-kubernetes-adapter"
NODE_AGENT_OWNER = "gpu-fault-node-agent"
HYPERPOD_OWNER = "gpu-fault-hyperpod-adapter"
REMOTE_COMMAND_ID = re.compile(r"^remote-[0-9a-f]{24}$")

PERF_AUDIT_CLUSTER_IDS = frozenset({"perf-cap-000", "perf-cap-001"})
CREDENTIAL_INPUT_LIMIT = 16 * 1024


class CommandAuditStore(ControlPlaneStore, Protocol):
    """Additional Store operations required by owned CMD fixtures and cleanup."""

    def ensure_remote_command(
        self, command: RemoteActionCommand
    ) -> RemoteActionCommand: ...

    def reserve_job_restart(
        self, cluster_id: str, job_id: str, restart_budget: int, reservation_id: str
    ) -> tuple[RestartBudgetState, bool]: ...

    def get_restart_budget(
        self, cluster_id: str, job_id: str
    ) -> RestartBudgetState: ...

    def _restart_budget_key(self, cluster_id: str, job_id: str) -> str: ...

    def _delete(self, kind: str, key: str) -> None: ...

    def close(self) -> None: ...


class AuditCredential(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)

    cluster_id: Literal["perf-cap-000", "perf-cap-001"]
    token: SecretStr = Field(min_length=32, max_length=256, repr=False)


class AuditCredentialEnvelope(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)

    schema_version: int = Field(ge=1, le=1)
    purpose: Literal["regional-command-audit"]
    cluster_id: Literal["perf-cap-000", "perf-cap-001"]
    other_cluster_id: Literal["perf-cap-000", "perf-cap-001"]
    synthetic_run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$")
    expires_at: str = Field(min_length=1, max_length=64)
    registry_generation: int = Field(ge=1)
    registry_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    credentials: list[AuditCredential] = Field(min_length=2, max_length=2, repr=False)


def unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def decode_credentials(
    raw: bytes,
    *,
    cluster_id: str,
    other_cluster_id: str,
    synthetic_run_id: str,
) -> AuditCredentialEnvelope:
    """Only this boundary handles untrusted credential-bearing validation errors."""
    try:
        if not raw or len(raw) > CREDENTIAL_INPUT_LIMIT:
            raise ValueError("size")
        value = AuditCredentialEnvelope.model_validate(
            json.loads(raw, object_pairs_hook=unique_json_object)
        )
        expires = datetime.fromisoformat(value.expires_at)
        now = datetime.now(timezone.utc)
        if (
            {cluster_id, other_cluster_id} != PERF_AUDIT_CLUSTER_IDS
            or value.cluster_id != cluster_id
            or value.other_cluster_id != other_cluster_id
            or not synthetic_run_id
            or value.synthetic_run_id != synthetic_run_id
            or {item.cluster_id for item in value.credentials} != PERF_AUDIT_CLUSTER_IDS
            or expires.tzinfo is None
            or not now < expires <= now + timedelta(minutes=15)
            or any(
                re.fullmatch(r"[A-Za-z0-9_-]{32,256}", item.token.get_secret_value())
                is None
                for item in value.credentials
            )
        ):
            raise ValueError("scope or expiration")
        return value
    except Exception:
        # Never format ValidationError, JSON input, SecretStr input, or its context.
        raise ProtocolAuditError("credential envelope rejected") from None


def read_credentials(
    *, cluster_id: str, other_cluster_id: str, synthetic_run_id: str
) -> AuditCredentialEnvelope:
    if sys.argv[0] == "-":
        raise ProtocolAuditError(
            "credential stdin requires a script file or python -c; "
            "python - consumes stdin as source"
        )
    try:
        metadata = os.fstat(sys.stdin.fileno())
        mode = metadata.st_mode
        protected_file = (
            stat.S_ISREG(mode)
            and metadata.st_uid == os.geteuid()
            and stat.S_IMODE(mode) & 0o077 == 0
        )
        if sys.stdin.isatty() or not (
            protected_file or stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode)
        ):
            raise ValueError("unprotected stdin")
        raw = sys.stdin.buffer.read(CREDENTIAL_INPUT_LIMIT + 1)
    except Exception:
        raise ProtocolAuditError("protected credential stdin is unavailable") from None
    return decode_credentials(
        raw,
        cluster_id=cluster_id,
        other_cluster_id=other_cluster_id,
        synthetic_run_id=synthetic_run_id,
    )


def current_credential_registry(
    store: Any, envelope: AuditCredentialEnvelope
) -> dict[str, dict[str, Any]]:
    """Read the actual head, verify its content, then recheck the head without writes."""
    try:
        now = datetime.now(timezone.utc)
        expires = datetime.fromisoformat(envelope.expires_at)
        if expires.tzinfo is None or expires <= now:
            raise ValueError("expired input")
        head = store.get_regional_registry_head()
        revision = store.get_regional_registry_revision(head.generation)
        if (
            head.generation != envelope.registry_generation
            or revision.generation != head.generation
            or head.content_sha256 != envelope.registry_content_sha256
            or revision.content_sha256 != head.content_sha256
            or regional_registry_content_sha256(revision.registrations)
            != head.content_sha256
        ):
            raise ValueError("registry drift")
        registrations = {item.cluster_id: item for item in revision.registrations}
        if len(registrations) != len(revision.registrations):
            raise ValueError("duplicate registration")
        selected = {}
        for credential in envelope.credentials:
            item = registrations[credential.cluster_id]
            if (
                not item.synthetic
                or not item.is_active(now)
                or item.synthetic_run_id != envelope.synthetic_run_id
                or item.synthetic_expires_at < expires
                or item.hyperpod_cluster_name != item.cluster_id
                or item.eks_cluster_arn
                != f"arn:aws:eks:{item.region}:000000000000:cluster/{item.cluster_id}"
                or item.agent_endpoint_allowed_cidrs != ["127.0.0.1/32"]
                or item.matched_token_slot(credential.token.get_secret_value(), now)
                != TOKEN_SLOT_CURRENT
            ):
                raise ValueError("credential scope")
            # A current synthetic credential must not authenticate any other identity.
            if (
                sum(
                    digest == item.token_sha256
                    for registration in revision.registrations
                    for digest in (
                        registration.token_sha256,
                        registration.retiring_token_sha256,
                    )
                )
                != 1
            ):
                raise ValueError("shared credential")
            agents = store.list_agents(item.cluster_id)
            if not isinstance(agents, list) or agents:
                raise ValueError("synthetic identity has agents or unknown inventory")
            selected[item.cluster_id] = item.model_dump(mode="json")
        if set(selected) != PERF_AUDIT_CLUSTER_IDS:
            raise ValueError("selection")
        latest = store.get_regional_registry_head()
        if (
            latest.generation != head.generation
            or latest.content_sha256 != head.content_sha256
        ):
            raise ValueError("registry changed during validation")
        return selected
    except Exception:
        raise ProtocolAuditError("durable credential scope rejected") from None


def open_credential_audit_store() -> PostgresStore:
    """No ApplicationContext, registry bootstrap, physical adapters, or schema DDL."""
    try:
        path = os.getenv("GPU_FAULT_STORE_URL_FILE")
        credentials = StoreCredentials(os.getenv("GPU_FAULT_STORE_URL", ""), path=path)
        url = credentials.conninfo()
        if not url or (path is not None and credentials.source != "file"):
            raise ValueError("current Store credentials unavailable")
        settings = StoreSettings.from_mapping(
            {**os.environ, "GPU_FAULT_STORE_URL": url}
        )
        if settings.kind != "postgres":
            raise ValueError("credential audit requires the CPU PostgreSQL Store")
        # Match ApplicationContext._base_context's CPU-wheel settings contract.
        # PostgresStore itself reads the CPU state modes and remaining PG settings.
        return PostgresStore(
            settings.url,
            pool_min_size=settings.postgres_pool_min_size,
            pool_max_size=settings.postgres_pool_max_size,
            pool_timeout_seconds=settings.postgres_pool_timeout_seconds,
            initialize_schema=False,
        )
    except Exception:
        raise ProtocolAuditError("audit Store initialization rejected") from None


class AuditStopped(BaseException):
    """Abort through library/helper Exception handlers; only the driver handles it."""


# This CPU-local child receives only a pidfd and monotonic deadlines, never tokens.
# A separate interpreter can enforce SIGKILL even when the producer holds its GIL.
AUDIT_WATCHDOG_SOURCE = """\
import os, select, signal, sys, time
pidfd = int(sys.argv[1])
stopfd = sys.stdin.fileno()
signal.signal(signal.SIGHUP, signal.SIG_IGN)
signal.signal(signal.SIGINT, signal.SIG_IGN)
print("ARMED", flush=True)
for deadline, signum in (
    (float(sys.argv[2]), signal.SIGUSR1),
    (float(sys.argv[3]), signal.SIGKILL),
):
    while time.monotonic() < deadline:
        ready, _, _ = select.select(
            [pidfd] + ([stopfd] if stopfd is not None else []),
            [], [], max(0, deadline - time.monotonic()),
        )
        if pidfd in ready:
            raise SystemExit(0)
        if stopfd in ready:
            if os.read(stopfd, 1) == b"D":
                raise SystemExit(0)
            stopfd = None
            if signum == signal.SIGUSR1:
                break
    try:
        signal.pidfd_send_signal(pidfd, signum)
    except ProcessLookupError:
        break
"""


class AuditDeadline:
    def __init__(
        self, overall_seconds: float = 300, cleanup_seconds: float = 30
    ) -> None:
        if any(
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not 0 < value <= maximum
            for value, maximum in ((overall_seconds, 900), (cleanup_seconds, 120))
        ):
            raise ProtocolAuditError("invalid CPU audit deadline or cleanup budget")
        self.work_until = time.monotonic() + overall_seconds
        self.hard_until = self.work_until + cleanup_seconds
        self.cleanup_remaining = cleanup_seconds
        self.stopped = False
        self.cleaning = False
        self.watchdog: subprocess.Popen[bytes] | None = None
        self.handlers: dict[int, Any] = {}

    def require_credential_lifetime(self, envelope: AuditCredentialEnvelope) -> None:
        self.check_work()
        # Cover SIGKILL, not only the cooperative stop: a GIL-blocked producer
        # might remain in flight for the cleanup grace period.
        remaining = self.hard_until - time.monotonic()
        try:
            expires = datetime.fromisoformat(envelope.expires_at)
            covered = expires.tzinfo is not None and expires > datetime.now(
                timezone.utc
            ) + timedelta(seconds=remaining + 1)
        except Exception:
            covered = False
        if not covered:
            raise ProtocolAuditError(
                "credential lifetime must cover CPU work and cleanup deadlines"
            ) from None
        self.check_work()

    def stop(self, _signum: int = 0, _frame: object = None) -> None:
        self.stopped = True
        if _signum and not self.cleaning:
            raise AuditStopped("CPU audit stopped")

    def check_work(self) -> None:
        if (
            self.stopped
            or self.cleaning
            or time.monotonic() >= self.work_until
            or (self.watchdog is not None and self.watchdog.poll() is not None)
        ):
            self.stopped = True
            raise AuditStopped("CPU audit deadline or supervision lost")

    @contextmanager
    def cleanup(self) -> Iterator[None]:
        remaining = min(self.cleanup_remaining, self.hard_until - time.monotonic())
        if remaining <= 0 or self.cleaning:
            self.stopped = True
            raise AuditStopped("CPU audit cleanup budget expired")
        started = time.monotonic()
        self.cleaning = True
        if self.watchdog is not None:
            # Exhausted cleanup cannot safely continue; native termination does
            # not depend on Python processing an exception or releasing its GIL.
            signal.signal(signal.SIGALRM, signal.SIG_DFL)
            signal.setitimer(signal.ITIMER_REAL, remaining)
        try:
            yield
            if time.monotonic() - started >= remaining:
                self.stopped = True
                raise AuditStopped("CPU audit cleanup budget expired")
        finally:
            if self.watchdog is not None:
                signal.setitimer(signal.ITIMER_REAL, 0)
            self.cleanup_remaining = max(
                0, self.cleanup_remaining - (time.monotonic() - started)
            )
            self.cleaning = False

    def __enter__(self) -> AuditDeadline:
        if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
            raise ProtocolAuditError("CPU audit requires Linux pidfd supervision")
        if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
            raise ProtocolAuditError("CPU audit cannot replace an existing timer")
        for signum in (signal.SIGUSR1, signal.SIGHUP, signal.SIGTERM, signal.SIGINT):
            self.handlers[signum] = signal.signal(signum, self.stop)
        self.handlers[signal.SIGALRM] = signal.signal(signal.SIGALRM, signal.SIG_DFL)
        try:
            pidfd = os.pidfd_open(os.getpid())
            try:
                self.watchdog = subprocess.Popen(
                    [
                        sys.executable,
                        "-I",
                        "-c",
                        AUDIT_WATCHDOG_SOURCE,
                        str(pidfd),
                        str(self.work_until),
                        str(self.hard_until),
                    ],
                    pass_fds=(pidfd,),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    env={},
                    start_new_session=True,
                )
            finally:
                os.close(pidfd)
            output = self.watchdog.stdout
            assert output is not None
            ready, _, _ = select.select(
                [output], [], [], max(0, min(5, self.work_until - time.monotonic()))
            )
            if not ready or os.read(output.fileno(), 6) != b"ARMED\n":
                raise ProtocolAuditError("CPU audit watchdog did not arm")
            output.close()
            self.check_work()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_exc: Any) -> None:
        self.stop()
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if self.watchdog is not None:
                try:
                    if self.watchdog.stdin is not None:
                        try:
                            self.watchdog.stdin.write(b"D")
                            self.watchdog.stdin.flush()
                        except BrokenPipeError:
                            pass
                        finally:
                            self.watchdog.stdin.close()
                    self.watchdog.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    self.watchdog.kill()
                    self.watchdog.wait(timeout=1)
                finally:
                    if self.watchdog.stdout is not None:
                        self.watchdog.stdout.close()
        finally:
            for signum, handler in self.handlers.items():
                signal.signal(signum, handler)


def probe_source() -> str:
    """Emit credential-free code; use python -c when stdin carries credentials."""
    helper = (
        Path(__file__)
        .with_name("command_protocol_probes.py")
        .read_text(encoding="utf-8")
    )
    audit = Path(__file__).read_text(encoding="utf-8")
    return (
        "import sys, types\n"
        "protocol_helpers = types.ModuleType('command_protocol_probes')\n"
        "sys.modules[protocol_helpers.__name__] = protocol_helpers\n"
        f"exec(compile({helper!r}, 'command_protocol_probes.py', 'exec'), protocol_helpers.__dict__)\n"
        f"exec(compile({audit!r}, 'audit_regional_command_protocol_live.py', 'exec'))\n"
    )


class LiveProtocolAudit:
    credential_envelope: AuditCredentialEnvelope | None = None
    deadline: AuditDeadline | None = None

    def __init__(
        self,
        *,
        cluster_id: str,
        other_cluster_id: str,
        executor_sha256: str,
        executor_digest: str,
        run_dir: Path | None = None,
        isolated_cluster: bool = False,
        executor_ready_replicas: int | None = None,
        release_id: str = "",
        credential_envelope: AuditCredentialEnvelope | None = None,
        deadline: AuditDeadline | None = None,
    ) -> None:
        self.cluster_id = cluster_id
        self.other_cluster_id = other_cluster_id
        self.executor_sha256 = executor_sha256
        self.executor_digest = executor_digest
        self.run_dir = run_dir
        self.isolated_cluster = isolated_cluster
        self.executor_ready_replicas = executor_ready_replicas
        self.release_id = release_id
        self.credential_envelope = credential_envelope
        self.deadline = deadline
        self.run_id = f"cmd-audit-{uuid4().hex[:12]}"
        # A step owner no deployed executor advertises, so a seeded command can
        # only ever be claimed by this audit's own claim calls.
        self.test_owner = f"{self.run_id}-owner"
        if credential_envelope is None:
            raw_registry = json.loads(os.environ["GPU_FAULT_REGIONAL_CLUSTERS_JSON"])
            self.registry = {item["cluster_id"]: item for item in raw_registry}
            self.tokens = {
                key: str(item["token"]) for key, item in self.registry.items()
            }
            self.store = cast(CommandAuditStore, open_credential_audit_store())
        else:
            expect(
                credential_envelope.cluster_id == cluster_id
                and credential_envelope.other_cluster_id == other_cluster_id
                and {cluster_id, other_cluster_id} == PERF_AUDIT_CLUSTER_IDS
                and not isolated_cluster
                and executor_ready_replicas is None,
                "credential input cannot substitute executor isolation assertions",
            )
            self.store = cast(CommandAuditStore, open_credential_audit_store())
            try:
                self.registry = current_credential_registry(
                    self.store, credential_envelope
                )
                self.tokens = {
                    item.cluster_id: item.token.get_secret_value()
                    for item in credential_envelope.credentials
                }
            except BaseException:
                with self.cleanup_budget():
                    self.store.close()
                raise
        self.created_commands: set[str] = set()
        self.created_workflows: set[str] = set()
        self.created_incidents: set[str] = set()
        self.created_event_links: dict[str, str] = {}
        self.created_auxiliary: set[tuple[str, str]] = set()
        self.results: dict[str, Any] = {}
        self.preflight_result: dict[str, Any] = {}

    def close(self) -> None:
        if self.deadline is not None:
            self.deadline.stop()
        self._cleanup_since(set(), set(), set())
        with self.cleanup_budget():
            self.store.close()

    def cleanup_budget(self) -> AbstractContextManager[None]:
        return self.deadline.cleanup() if self.deadline is not None else nullcontext()

    def check_work(self) -> None:
        if self.deadline is not None:
            self.deadline.check_work()
        if self.credential_envelope is not None:
            self.registry = current_credential_registry(
                self.store, self.credential_envelope
            )
        if self.deadline is not None:
            self.deadline.check_work()

    def record(self, case_id: str, **details: Any) -> None:
        self.results[case_id] = details

    # ------------------------------------------------------------------ #
    # isolation preflight
    # ------------------------------------------------------------------ #
    def open_commands_in_cluster(self) -> int:
        stats = self.store.remote_command_stats()
        open_by_cluster = stats.get("open_by_cluster")
        if not isinstance(open_by_cluster, dict):
            raise ProtocolAuditError("remote-command queue state is unknown")
        expect(
            all(
                type(count) is int and count >= 0 for count in open_by_cluster.values()
            ),
            "remote-command queue counts are invalid",
        )
        counts = cast(dict[str, int], open_by_cluster)
        if self.credential_envelope is not None:
            return sum(
                counts.get(cluster_id, 0)
                for cluster_id in (self.cluster_id, self.other_cluster_id)
            )
        return counts.get(self.cluster_id, 0)

    def preflight(self) -> dict[str, Any]:
        """Refuse to run against a queue a production executor could still drain.

        The claim route hands out real leases: a claim the audit makes before
        seeding would take a production command for the lease window, and a
        seeded command with a real adapter owner would be taken by a production
        executor. Both need the target cluster's queue empty *and* the executor
        unable to claim -- asserted by the operator (``--isolated-cluster``) or
        observed as zero Ready replicas of the executor Deployment.
        """

        self.check_work()
        open_commands = self.open_commands_in_cluster()
        errors = []
        if open_commands:
            errors.append(
                f"target cluster has {open_commands} open remote command(s); "
                "the audit leases and seeds against this queue"
            )
        if (
            self.credential_envelope is None
            and not self.isolated_cluster
            and self.executor_ready_replicas != 0
        ):
            errors.append(
                "production executor may still claim: pass --isolated-cluster or "
                "--executor-ready-replicas 0 after scaling the executor "
                "Deployment down (observed "
                f"{self.executor_ready_replicas!r} ready replicas)"
            )
        result: dict[str, Any] = {
            "open_remote_commands": open_commands,
            "isolated_cluster": self.isolated_cluster,
            "executor_ready_replicas": self.executor_ready_replicas,
            "test_owner": self.test_owner,
            "errors": errors,
        }
        self.preflight_result = result
        if self.credential_envelope is not None:
            result["synthetic_registry_scope"] = {
                "generation": self.credential_envelope.registry_generation,
                "content_sha256": self.credential_envelope.registry_content_sha256,
                "synthetic_run_id": self.credential_envelope.synthetic_run_id,
                "dummy_eks_identities": True,
                "registered_agents": 0,
                "physical_executor_absence": "NOT_OBSERVED",
            }
        if errors:
            raise ProtocolAuditError("preflight refused: " + "; ".join(errors))
        return result

    # ------------------------------------------------------------------ #
    # transport
    # ------------------------------------------------------------------ #
    def _request(
        self,
        method: str,
        path: str,
        *,
        cluster_id: str | None,
        token: str | None,
        payload: Any = None,
    ) -> tuple[int, Any]:
        self.check_work()
        return request_json(
            method,
            path,
            cluster_id=cluster_id,
            token=token,
            payload=payload,
        )

    def claim(
        self,
        *,
        cluster_id: str | None = None,
        token: str | None = None,
        executor_id: str | None = None,
        owners: list[str] | None = None,
        max_commands: Any = 1,
        lease_seconds: Any = 60,
    ) -> tuple[int, Any]:
        target = cluster_id or self.cluster_id
        return self._request(
            "POST",
            "/v1/regional/executors/claim",
            cluster_id=target,
            token=token or self.tokens[target],
            payload={
                "executor_id": executor_id or f"{self.run_id}-executor",
                "executor_protocol_version": CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION,
                "executor_artifact_sha256": self.executor_sha256,
                "executor_compatibility_digest": self.executor_digest,
                "execution_owners": ([self.test_owner] if owners is None else owners),
                "max_commands": max_commands,
                "lease_seconds": lease_seconds,
            },
        )

    def complete(
        self,
        command_id: str,
        *,
        cluster_id: str | None = None,
        token: str | None = None,
        payload: dict[str, Any],
    ) -> tuple[int, Any]:
        target = cluster_id or self.cluster_id
        return self._request(
            "POST",
            f"/v1/regional/executors/{command_id}/result",
            cluster_id=target,
            token=token or self.tokens[target],
            payload=payload,
        )

    def hand_back(self, commands: list[dict[str, Any]]) -> int:
        """Return every leased command as WAITING so no lease outlives its case."""

        returned = 0
        for command in commands:
            status, result = self.complete(
                str(command["command_id"]),
                payload={
                    "lease_token": self._lease_token(command),
                    "status": "WAITING",
                    "details": {"audit": self.run_id, "returned": True},
                },
            )
            expect(
                status == 200 and result.get("status") == "WAITING",
                f"could not hand back {command['command_id']} as WAITING",
            )
            returned += 1
        return returned

    # ------------------------------------------------------------------ #
    # seeding
    # ------------------------------------------------------------------ #
    def _pair(
        self,
        suffix: str,
        *,
        owner: str,
        cluster_id: str,
        fencing_token: int,
        step: WorkflowStepSpec | None = None,
        not_before: datetime | None = None,
    ) -> tuple[FaultIncident, WorkflowRequest, WorkflowStepSpec]:
        self.check_work()
        return make_audit_pair(
            self.run_id,
            suffix,
            owner=owner,
            cluster_id=cluster_id,
            fencing_token=fencing_token,
            step=step,
            not_before=not_before,
        )

    def persist_pair(
        self,
        suffix: str,
        *,
        owner: str,
        step: WorkflowStepSpec | None = None,
    ) -> WorkflowRequest:
        """Persist an incident/workflow pair the production dispatcher cannot take.

        ``not_before`` an hour out keeps the dispatcher's eligibility filter off
        the record for the life of the audit; only this process executes it.
        """

        incident, workflow, _ = self._pair(
            suffix,
            owner=owner,
            cluster_id=self.cluster_id,
            fencing_token=1,
            step=step,
            not_before=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        self.created_incidents.add(incident.incident_id)
        self.created_workflows.add(workflow.request_id)
        self.created_event_links[incident.event_id] = incident.incident_id
        self.store.save_incident_and_workflow(incident, workflow)
        return workflow

    def seed(
        self,
        suffix: str,
        *,
        owner: str | None = None,
        cluster_id: str | None = None,
        created_at: datetime | None = None,
        fencing_token: int = 1,
        persist_workflow: bool = False,
    ) -> RemoteActionCommand:
        target = cluster_id or self.cluster_id
        incident, workflow, step = self._pair(
            suffix,
            owner=owner or self.test_owner,
            cluster_id=target,
            fencing_token=fencing_token,
            not_before=(
                datetime.now(timezone.utc) + timedelta(hours=1)
                if persist_workflow
                else None
            ),
        )
        if persist_workflow:
            self.created_incidents.add(incident.incident_id)
            self.created_workflows.add(workflow.request_id)
            self.created_event_links[incident.event_id] = incident.incident_id
            self.store.save_incident_and_workflow(incident, workflow)
        command = RemoteActionCommand(
            command_id=f"{self.run_id}-{suffix}-command",
            cluster_id=target,
            workflow_request_id=workflow.request_id,
            incident_id=incident.incident_id,
            step_index=0,
            fencing_token=fencing_token,
            idempotency_key=f"{self.run_id}/{suffix}",
            step=step,
            workflow=workflow,
            incident=incident,
            created_at=created_at or datetime.now(timezone.utc),
            updated_at=created_at or datetime.now(timezone.utc),
        )
        self.created_commands.add(command.command_id)
        stored = self.store.ensure_remote_command(command)
        return stored

    @staticmethod
    def _lease_token(command: dict[str, Any]) -> str:
        token = command.get("lease_token")
        expect(isinstance(token, str) and token, "claimed command has no lease token")
        return str(token)

    def _node_agent_record_absent(self) -> bool:
        try:
            self.store.get_agent(self.cluster_id, f"{self.run_id}-nonexistent-node")
        except NotFoundError:
            return True
        return False

    @staticmethod
    def claim_records(
        status: int, body: Any, expected_ids: set[str], *, count: int | None = None
    ) -> list[dict[str, Any]]:
        return claim_records(status, body, expected_ids, count=count)

    # ------------------------------------------------------------------ #
    # cases
    # ------------------------------------------------------------------ #
    def run_001(self) -> None:
        # A backlog larger than the cap proves the bound, including a nonempty
        # success response. Every granted lease goes straight back as WAITING.
        seeded = [self.seed(f"cmd001-{index}").command_id for index in range(26)]
        observed: dict[str, int] = {}
        returned = 0
        for value in (0, 1, 25, 26, -1, "5", 5.5):
            status, body = self.claim(max_commands=value)
            observed[repr(value)] = status
            if value in {1, 25, "5"}:
                commands = self.claim_records(
                    status, body, set(seeded), count=int(value)
                )
                returned += self.hand_back(commands)
            else:
                expect(status == 422, f"max_commands={value!r} answered {status}")
        self.record(
            "GF-REGIONAL-CMD-001",
            statuses=observed,
            seeded=seeded,
            leased_and_returned=returned,
        )

    def run_002(self) -> None:
        invalid = {}
        for value in (9, 7201, 0):
            status, _ = self.claim(lease_seconds=value)
            invalid[str(value)] = status
            expect(status == 422, f"lease_seconds={value} answered {status}")
        command = self.seed("cmd002")
        leases = {}
        for value in (10, 600, 601, 7200):
            before = datetime.now(timezone.utc)
            status, body = self.claim(
                executor_id=f"{self.run_id}-lease-{value}",
                lease_seconds=value,
            )
            after = datetime.now(timezone.utc)
            claimed = self.claim_records(status, body, {command.command_id})[0]
            expires = datetime.fromisoformat(
                claimed["lease_expires_at"].replace("Z", "+00:00")
            )
            lower = (expires - after).total_seconds()
            upper = (expires - before).total_seconds()
            expect(
                value - 3 <= lower <= value + 3 and value - 3 <= upper <= value + 3,
                f"lease_seconds={value} produced a deadline {lower:.1f}s out",
            )
            leases[str(value)] = round(lower, 3)
            status, result = self.complete(
                command.command_id,
                payload={
                    "lease_token": self._lease_token(claimed),
                    "status": "WAITING",
                    "details": {"lease_seconds": value},
                },
            )
            expect(
                status == 200 and result.get("status") == "WAITING",
                "WAITING hand-back was refused",
            )
        status, body = self.claim(lease_seconds=10)
        leased = self.claim_records(status, body, {command.command_id})[0]
        time.sleep(11)
        expired_status, _ = self.complete(
            command.command_id,
            payload={
                "lease_token": self._lease_token(leased),
                "status": "SUCCEEDED",
            },
        )
        expect(expired_status == 409, f"expired lease answered {expired_status}")
        self.record(
            "GF-REGIONAL-CMD-002",
            invalid=invalid,
            observed_lease_seconds=leases,
            expired_result_status=expired_status,
        )

    def run_003(self) -> None:
        values = [
            ([""], 422),
            ([" gpu-fault-kubernetes-adapter"], 422),
            (["gpu-fault-kubernetes-adapter "], 422),
            (["a", "a"], 422),
            ([f"owner-{index}" for index in range(33)], 422),
            ([f"owner-{index}" for index in range(32)], 200),
        ]
        observed = {}
        for owners, expected in values:
            status, body = self.claim(owners=owners)
            observed[str(len(owners)) + ":" + repr(owners[:2])] = status
            expect(status == expected, f"owners {owners[:2]!r} answered {status}")
            if expected == 200:
                self.claim_records(status, body, set())
        protected = self.seed("cmd003", owner=NODE_AGENT_OWNER)
        status, body = self.claim(owners=[])
        self.claim_records(status, body, set())
        # The other half of the default: an empty list means the kubernetes
        # adapter, so a kubernetes-adapter command IS returned.
        default_owned = self.seed("cmd003-default", owner=KUBERNETES_OWNER)
        status, body = self.claim(owners=[], max_commands=25)
        claimed = self.claim_records(status, body, {default_owned.command_id})
        claimed_ids = [item["command_id"] for item in claimed]
        self.hand_back(claimed)
        expect(
            status == 200 and claimed_ids == [default_owned.command_id],
            f"an empty owner list leased {claimed_ids!r}, expected the "
            "kubernetes-adapter command alone",
        )
        self.record(
            "GF-REGIONAL-CMD-003",
            statuses=observed,
            empty_owner_claimed=[],
            protected_command=protected.command_id,
            empty_owner_default_claimed=claimed_ids,
        )

    def run_004(self) -> None:
        commands = {
            owner: self.seed(f"cmd004-{index}", owner=owner)
            for index, owner in enumerate(
                (KUBERNETES_OWNER, NODE_AGENT_OWNER, HYPERPOD_OWNER)
            )
        }
        status, body = self.claim(owners=[NODE_AGENT_OWNER], max_commands=5)
        first = self.claim_records(
            status, body, {commands[NODE_AGENT_OWNER].command_id}
        )
        first_owners = [item["step"]["execution_owner"] for item in first]
        self.hand_back(first)
        expect(
            status == 200 and first_owners == [NODE_AGENT_OWNER],
            f"node-agent claim leased {first_owners!r}",
        )
        status, remaining = self.claim(
            owners=[KUBERNETES_OWNER, HYPERPOD_OWNER],
            max_commands=5,
        )
        second = self.claim_records(
            status,
            remaining,
            {
                commands[KUBERNETES_OWNER].command_id,
                commands[HYPERPOD_OWNER].command_id,
            },
        )
        owners = [item["step"]["execution_owner"] for item in second]
        self.hand_back(second)
        expect(
            status == 200 and set(owners) == {KUBERNETES_OWNER, HYPERPOD_OWNER},
            f"adapter claim leased {owners!r}",
        )
        self.record(
            "GF-REGIONAL-CMD-004",
            command_ids={key: value.command_id for key, value in commands.items()},
            claimed_owners=[NODE_AGENT_OWNER, *owners],
        )

    def run_005(self) -> None:
        now = datetime.now(timezone.utc)
        expected = []
        for index in range(5):
            command = self.seed(
                f"cmd005-{index}",
                created_at=now - timedelta(seconds=(5 - index) * 5),
            )
            expected.append(command.command_id)
        # Two commands created the same instant: the store breaks the tie on
        # command_id, so the one seeded second must still come out first.
        tie = now - timedelta(seconds=1)
        tie_b = self.seed("cmd005-tie-b", created_at=tie)
        tie_a = self.seed("cmd005-tie-a", created_at=tie)
        expected.extend(sorted([tie_b.command_id, tie_a.command_id]))
        status, body = self.claim(max_commands=25)
        claimed = self.claim_records(status, body, set(expected))
        actual = [item["command_id"] for item in claimed]
        self.hand_back(claimed)
        expect(
            status == 200 and actual == expected,
            f"claim order {actual!r} is not FIFO with an id tie-break {expected!r}",
        )
        self.record(
            "GF-REGIONAL-CMD-005",
            expected=expected,
            actual=actual,
            tie_break_sample={
                "created_at": tie.isoformat(),
                "seeded_order": [tie_b.command_id, tie_a.command_id],
                "claimed_order": actual[-2:],
            },
        )

    def run_006(self) -> None:
        command = self.seed("cmd006")
        status, first = self.claim(executor_id="cmd006-a1", lease_seconds=10)
        one = self.claim_records(status, first, {command.command_id})[0]
        token_one = self._lease_token(one)
        status, result = self.complete(
            command.command_id,
            payload={
                "lease_token": token_one,
                "status": "WAITING",
                "details": {"round": 1},
            },
        )
        expect(
            status == 200 and result.get("status") == "WAITING",
            "first WAITING hand-back was refused",
        )
        status, second = self.claim(executor_id="cmd006-a2", lease_seconds=10)
        two = self.claim_records(status, second, {command.command_id})[0]
        token_two = self._lease_token(two)
        expect(token_one != token_two, "reclaim reused the previous lease token")
        stale_one, _ = self.complete(
            command.command_id,
            payload={"lease_token": token_one, "status": "SUCCEEDED"},
        )
        expect(stale_one == 409, f"stale token answered {stale_one}")
        time.sleep(15)
        stale_two, _ = self.complete(
            command.command_id,
            payload={"lease_token": token_two, "status": "SUCCEEDED"},
        )
        expect(stale_two == 409, f"expired token answered {stale_two}")
        status, third = self.claim(executor_id="cmd006-a3", lease_seconds=60)
        three = self.claim_records(status, third, {command.command_id})[0]
        final_status, final = self.complete(
            command.command_id,
            payload={
                "lease_token": self._lease_token(three),
                "status": "SUCCEEDED",
            },
        )
        expect(
            final_status == 200 and final["status"] == "SUCCEEDED",
            f"final result answered {final_status}",
        )
        expect(
            final["last_lease_owner"] == "cmd006-a3",
            f"final lease owner is {final.get('last_lease_owner')!r}",
        )
        self.record(
            "GF-REGIONAL-CMD-006",
            stale_token_one=stale_one,
            expired_token_two=stale_two,
            final_owner=final["last_lease_owner"],
        )

    def run_007(self) -> None:
        command = self.seed("cmd007")
        status, claim = self.claim(executor_id="cmd007", lease_seconds=60)
        leased = self.claim_records(status, claim, {command.command_id})[0]
        token = self._lease_token(leased)
        first_status, first = self.complete(
            command.command_id,
            payload={"lease_token": token, "status": "SUCCEEDED"},
        )
        expect(
            first_status == 200
            and first.get("status") == "SUCCEEDED"
            and first.get("error") is None,
            f"terminal result did not confirm SUCCEEDED: HTTP {first_status}",
        )
        variants: list[dict[str, Any]] = [
            {"lease_token": token, "status": "SUCCEEDED"},
            {"lease_token": "wrong-token", "status": "SUCCEEDED"},
            {
                "lease_token": "wrong-token",
                "status": "FAILED",
                "error": "must not overwrite",
            },
        ]
        replay_statuses = []
        for payload in variants:
            status, body = self.complete(command.command_id, payload=payload)
            replay_statuses.append(status)
            expect(
                status == 200 and body == first,
                f"terminal replay answered {status} with a changed record",
            )
        self.record(
            "GF-REGIONAL-CMD-007",
            first_status=first_status,
            replay_statuses=replay_statuses,
            final=first,
        )

    def run_008(self) -> None:
        command = self.seed("cmd008")
        status, claim = self.claim(executor_id="cmd008", lease_seconds=60)
        leased = self.claim_records(status, claim, {command.command_id})[0]
        check_result_payloads(self, command.command_id, self._lease_token(leased))

    def run_009(self) -> None:
        missing_id = f"{self.run_id}-does-not-exist"
        missing_status, missing = self.complete(
            missing_id,
            payload={"lease_token": "missing", "status": "SUCCEEDED"},
        )
        other = self.seed("cmd009-other", cluster_id=self.other_cluster_id)
        cross_status, cross = self.complete(
            other.command_id,
            payload={"lease_token": "missing", "status": "SUCCEEDED"},
        )
        expect(
            missing_status == cross_status == 404,
            f"unknown answered {missing_status}, foreign answered {cross_status}",
        )
        expect(
            missing.get("detail")
            == f"resource not found: {self.cluster_id}/{missing_id}"
            and cross.get("detail")
            == f"resource not found: {self.cluster_id}/{other.command_id}",
            "a foreign command is distinguishable from an unknown one: "
            f"{missing.get('detail')!r} != {cross.get('detail')!r}",
        )
        expect(
            self.store.get_remote_command(other.command_id) == other,
            "the foreign command changed",
        )
        self.record(
            "GF-REGIONAL-CMD-009",
            missing={"status": missing_status, "detail": missing["detail"]},
            cross_cluster={"status": cross_status, "detail": cross["detail"]},
        )

    def run_010(self) -> None:
        node_absent_before = self._node_agent_record_absent()
        command = self.seed("cmd010", persist_workflow=True)
        status, claim = self.claim(executor_id="cmd010", lease_seconds=60)
        expect(
            status == 200 and len(claim["commands"]) == 1,
            "the seeded command was not leased",
        )
        leased = claim["commands"][0]
        workflow = self.store.get_workflow(command.workflow_request_id)
        incident = self.store.get_incident(command.incident_id)
        fenced_at = datetime.now(timezone.utc)
        workflow = workflow.model_copy(
            update={"fencing_token": 2, "updated_at": fenced_at}
        )
        incident = incident.model_copy(
            update={"fencing_token": 2, "updated_at": fenced_at}
        )
        self.store.save_incident_and_workflow(incident, workflow)
        stale_status, stale = self.complete(
            command.command_id,
            payload={
                "lease_token": self._lease_token(leased),
                "status": "SUCCEEDED",
            },
        )
        # The full batch: FIFO must not be allowed to hide the fenced command
        # behind other backlog, so ask for everything and look for its id.
        status, reclaimed = self.claim(executor_id="cmd010-reclaim", max_commands=25)
        reclaimed_ids = [
            item["command_id"] for item in self.claim_records(status, reclaimed, set())
        ]
        # The late outcome is retained for audit, never adopted as effective success.
        expect(
            stale_status == 200 and stale.get("status") == "FAILED",
            f"stale fencing result did not terminalize FAILED: HTTP {stale_status}",
        )
        expect(
            stale.get("status_source") == STALE_FENCE_STATUS_SOURCE
            and (stale.get("result_details") or {}).get("post_stale_fence_status")
            == "SUCCEEDED"
            and (stale.get("result_details") or {}).get("post_stale_fence_error")
            is None,
            "stale completion did not preserve the late result as fenced evidence",
        )
        expect(
            command.command_id not in reclaimed_ids,
            "the fenced command was leased again",
        )
        after_workflow = self.store.get_workflow(command.workflow_request_id)
        after_incident = self.store.get_incident(command.incident_id)
        after_command = self.store.get_remote_command(command.command_id)
        expect(
            after_workflow.fencing_token == 2
            and after_workflow.status is WorkflowStatus.RUNNING
            and after_incident.fencing_token == 2,
            "the fenced workflow/incident changed under the stale result",
        )
        expect(
            after_command.status.value == "FAILED"
            and after_command.status_source == STALE_FENCE_STATUS_SOURCE
            and after_command.lease_owner is None
            and after_command.lease_token is None
            and after_command.lease_expires_at is None
            and after_command.last_lease_owner == "cmd010"
            and after_command.fencing_token == 1,
            f"the fenced command did not settle FAILED under stale-fence: "
            f"{after_command.status.value} source={after_command.status_source!r} "
            f"owner={after_command.lease_owner!r}",
        )
        late_status = after_command.result_details.get("post_stale_fence_status")
        expect(
            late_status == "SUCCEEDED",
            f"the executor's reported outcome was not preserved for audit: {late_status!r}",
        )
        node_absent_after = self._node_agent_record_absent()
        expect(
            node_absent_before and node_absent_after,
            "a node agent record appeared for the probe node",
        )
        not_before = after_workflow.not_before
        self.record(
            "GF-REGIONAL-CMD-010",
            complete_status=stale_status,
            reclaimed=False,
            reclaim_batch_ids=reclaimed_ids,
            workflow_after={
                "status": after_workflow.status.value,
                "fencing_token": after_workflow.fencing_token,
                "not_before": not_before.isoformat() if not_before else None,
            },
            command_after={
                "status": after_command.status.value,
                "status_source": after_command.status_source,
                "lease_owner": after_command.lease_owner,
                "last_lease_owner": after_command.last_lease_owner,
                "fencing_token": after_command.fencing_token,
                "post_stale_fence_status": late_status,
            },
            node_agent_record_absent={
                "before": node_absent_before,
                "after": node_absent_after,
            },
        )

    def _adapter_context(
        self,
        suffix: str,
        *,
        fencing_token: int = 1,
        step: WorkflowStepSpec | None = None,
    ) -> WorkflowStepContext:
        incident, workflow, selected = self._pair(
            suffix,
            owner=NODE_AGENT_OWNER,
            cluster_id=self.cluster_id,
            fencing_token=fencing_token,
            step=step,
        )
        incident = incident.model_copy(update={"workflow_request_id": None})
        self.created_incidents.add(incident.incident_id)
        self.created_workflows.add(workflow.request_id)
        return WorkflowStepContext(
            workflow=workflow,
            incident=incident,
            step=selected,
            step_index=0,
            request=WorkflowExecutionRequest(expected_fencing_token=fencing_token),
            idempotency_key=f"{self.run_id}/{suffix}",
        )

    def run_012(self) -> None:
        adapter = RegionalRemoteWorkflowAdapter(
            self.store,
            owners={NODE_AGENT_OWNER},
        )
        first_context = self._adapter_context("cmd012", fencing_token=1)
        first = adapter.execute(first_context)
        first_id = str(first.details["remote_command_id"])
        self.created_commands.add(first_id)
        created_at = self.store.get_remote_command(first_id).created_at
        retry_ids = []
        created_at_after_retries = []
        for _attempt in range(2):
            retry = adapter.execute(first_context)
            retry_ids.append(str(retry.details["remote_command_id"]))
            created_at_after_retries.append(
                self.store.get_remote_command(first_id).created_at
            )
        second_context = self._adapter_context("cmd012", fencing_token=2)
        second = adapter.execute(second_context)
        second_id = str(second.details["remote_command_id"])
        self.created_commands.add(second_id)
        expect(
            all(item == first_id for item in retry_ids),
            f"retries minted new command ids: {retry_ids!r}",
        )
        expect(
            all(item == created_at for item in created_at_after_retries),
            "a retry rewrote the persisted command's created_at",
        )
        expect(second_id != first_id, "a new fencing token reused the command id")
        expect(
            REMOTE_COMMAND_ID.fullmatch(first_id) is not None
            and REMOTE_COMMAND_ID.fullmatch(second_id) is not None,
            f"command ids are not remote-<24 hex>: {first_id!r}, {second_id!r}",
        )
        self.record(
            "GF-REGIONAL-CMD-012",
            first_id=first_id,
            duplicate_ids=retry_ids,
            created_at_unchanged=True,
            next_fencing_id=second_id,
            limitation=(
                "identity reuse is exercised as three adapter.execute calls on "
                "one step context; the dispatcher's own retry path is "
                "NOT_EXERCISED here"
            ),
        )

    def run_013(self) -> None:
        command = self.seed("cmd013")
        previous: dict[str, int] = {}
        for round_number in range(1, 21):
            status, claim = self.claim(
                executor_id=f"cmd013-{round_number}",
                lease_seconds=60,
            )
            leased = self.claim_records(status, claim, {command.command_id})[0]
            if round_number > 1:
                expect(
                    leased["result_details"] == previous,
                    f"round {round_number} lost the previous WAITING details",
                )
            previous = {"probe_round": round_number}
            status, result = self.complete(
                command.command_id,
                payload={
                    "lease_token": self._lease_token(leased),
                    "status": "WAITING",
                    "details": previous,
                },
            )
            expect(
                status == 200 and result.get("status") == "WAITING",
                f"round {round_number} WAITING answered {status}",
            )
        status, claim = self.claim(executor_id="cmd013-final", lease_seconds=60)
        leased = self.claim_records(status, claim, {command.command_id})[0]
        expect(
            leased["result_details"] == {"probe_round": 20},
            "the final claim lost the round-20 details",
        )
        final_status, final = self.complete(
            command.command_id,
            payload={
                "lease_token": self._lease_token(leased),
                "status": "SUCCEEDED",
            },
        )
        expect(
            final_status == 200 and final["status"] == "SUCCEEDED",
            f"final result answered {final_status}",
        )
        self.record(
            "GF-REGIONAL-CMD-013",
            waiting_rounds=20,
            final_status=final["status"],
            limitation="no maximum WAITING round count",
        )

    def run_014(self) -> None:
        """The delegating step's negative claim, read back from the store.

        The adapter's in-memory outcome is what the pytest proxy checks; the
        live case has to prove the executor *persisted* it, so the workflow is
        written (dispatcher-proof via ``not_before``) and executed here, and the
        step execution is read from the store.
        """

        workflow = self.persist_pair("cmd014", owner=NODE_AGENT_OWNER)
        adapter = RegionalRemoteWorkflowAdapter(
            self.store,
            owners={NODE_AGENT_OWNER},
        )
        executor = ProductionWorkflowExecutor(
            self.store,
            [adapter],
            ProductionExecutorConfig(
                enabled=True,
                executor_id=f"{self.run_id}-cmd014-executor",
                allowed_operations=frozenset(
                    {WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT}
                ),
            ),
        )
        result = executor.execute(
            workflow.request_id,
            WorkflowExecutionRequest(expected_fencing_token=1),
        )
        persisted = self.store.get_workflow(workflow.request_id)
        executions = [
            item for item in persisted.step_executions if item.step_index == 0
        ]
        expect(len(executions) == 1, "no persisted step execution for step 0")
        execution = executions[0]
        details = dict(execution.details)
        command_id = str(details.get("remote_command_id") or "")
        if command_id:
            self.created_commands.add(command_id)
        expect(
            execution.status.value == "WAITING",
            f"persisted step status is {execution.status.value}",
        )
        expected = {
            "remote_command_id": command_id,
            "remote_cluster_id": self.cluster_id,
            "remote_status": "PENDING",
            "mutation_submitted_by_control_plane": False,
        }
        expect(
            REMOTE_COMMAND_ID.fullmatch(command_id) is not None
            and all(details.get(key) == value for key, value in expected.items()),
            f"persisted step details {details!r} do not carry {expected!r}",
        )
        self.record(
            "GF-REGIONAL-CMD-014",
            workflow_status=result.status.value,
            persisted_step_status=execution.status.value,
            details=details,
        )

    def run_015(self) -> None:
        """Restart budget is decided once, at claim time; dispatch only reads it.

        Three refusals, none of which mints a remote command or touches a
        workload: an occupied budget refuses the claim preflight
        (``reserve_restart_budgets`` -> RESTART_BUDGET_EXHAUSTED); a dispatch
        that holds no reservation fails closed at the adapter
        (``issue_restart_authorization`` -> RESTART_RESERVATION_MISSING) instead
        of reserving there; and a step without ``source_gpu_count`` is refused
        at the gate before either site reads the budget row.
        """

        adapter = RegionalRemoteWorkflowAdapter(
            self.store,
            owners={KUBERNETES_OWNER},
        )
        job_id = f"{self.run_id}-budget"
        priming_reservation = f"{self.run_id}-existing-reservation"
        self.created_auxiliary.add(
            ("restart_budget", self.store._restart_budget_key(self.cluster_id, job_id))
        )
        try:
            state, reserved = self.store.reserve_job_restart(
                self.cluster_id,
                job_id,
                1,
                priming_reservation,
            )
            expect(
                reserved and state.restart_count == 1,
                "the priming restart reservation was refused",
            )
            full = WorkflowStepSpec(
                operation=WorkflowOperation.RESTART_WORKLOAD,
                execution_owner=KUBERNETES_OWNER,
                workload_ids=["training/job/nonexistent-cmd015"],
                parameters={
                    "cluster_id": self.cluster_id,
                    "job_id": job_id,
                    "source_attempt_id": f"{job_id}-a1",
                    "source_gpu_count": 1,
                    "restart_budget": 1,
                },
            )
            # (a) The claim preflight against the occupied budget: the only
            # site that reserves, and so the only site that can say exhausted.
            claim = self._adapter_context("cmd015-exhausted", step=full)
            failure = reserve_restart_budgets(
                self.store,
                claim.workflow,
                claim.incident,
                claim.workflow.official_steps,
            )
            expect(
                failure is not None,
                "the claim preflight admitted an exhausted budget",
            )
            assert failure is not None
            exhausted = failure.outcome
            exhausted_details = exhausted.details or {}
            expect(
                failure.step_index == 0
                and exhausted.status.value == "FAILED"
                and exhausted_details.get("reason") == "RESTART_BUDGET_EXHAUSTED"
                and "restart budget exhausted" in str(exhausted.error)
                and "1/1" in str(exhausted.error),
                f"exhausted budget answered {exhausted.status.value}: "
                f"{exhausted.error!r}",
            )
            state = self.store.get_restart_budget(self.cluster_id, job_id)
            expect(
                state.restart_count == 1
                and state.reservation_ids == [priming_reservation],
                f"the refused claim changed the budget row: {state.reservation_ids!r}",
            )
            # (b) Dispatch without a reservation: the adapter reads the
            # preflight's reservation back and fails closed when there is none.
            unreserved = adapter.execute(
                self._adapter_context("cmd015-unreserved", step=full)
            )
            expect(
                unreserved.status.value == "FAILED"
                and unreserved.details.get("reason") == "RESTART_RESERVATION_MISSING"
                and "restart reservation missing" in str(unreserved.error)
                and unreserved.details.get("restart_count") == 1
                and unreserved.details.get("restart_budget") == 1,
                f"unreserved dispatch answered {unreserved.status.value}: "
                f"{unreserved.error!r} {unreserved.details!r}",
            )
            # (c) The gate refuses an incomplete safety context before any
            # budget read, in the claim preflight's words.
            missing = full.model_copy(
                update={
                    "parameters": {
                        key: value
                        for key, value in full.parameters.items()
                        if key != "source_gpu_count"
                    }
                }
            )
            incomplete = adapter.execute(
                self._adapter_context("cmd015-missing", step=missing)
            )
            expect(
                incomplete.status.value == "FAILED"
                and incomplete.details.get("reason") == "RESTART_SAFETY_CONTEXT_MISSING"
                and "restart safety context is missing: source_gpu_count"
                in str(incomplete.error),
                f"missing context answered {incomplete.status.value}: "
                f"{incomplete.error!r}",
            )
            command_incidents = {
                f"{self.run_id}-cmd015-exhausted-incident",
                f"{self.run_id}-cmd015-unreserved-incident",
                f"{self.run_id}-cmd015-missing-incident",
            }
            leaked = [
                command.command_id
                for command in self.store.list_remote_commands()
                if command.incident_id in command_incidents
            ]
            for command_id in leaked:
                self.created_commands.add(command_id)
            expect(not leaked, f"refused restarts still minted commands: {leaked!r}")
        finally:
            with self.cleanup_budget():
                self.store._delete(
                    "restart_budget",
                    self.store._restart_budget_key(self.cluster_id, job_id),
                )
        self.record(
            "GF-REGIONAL-CMD-015",
            exhausted_error=exhausted.error,
            exhausted_details=dict(exhausted_details),
            unreserved_error=unreserved.error,
            unreserved_details=dict(unreserved.details),
            missing_context_error=incomplete.error,
            commands_created=0,
            restart_budget_deleted=True,
        )

    def run_016(self) -> None:
        run_hyperpod_submission_case(self)

    # ------------------------------------------------------------------ #
    # driver
    # ------------------------------------------------------------------ #
    def _cleanup_since(
        self,
        commands_before: set[str],
        workflows_before: set[str],
        incidents_before: set[str],
    ) -> None:
        with self.cleanup_budget():
            cleanup_audit_records(
                self.store,
                cluster_ids={self.cluster_id, self.other_cluster_id},
                created_commands=self.created_commands,
                created_workflows=self.created_workflows,
                created_incidents=self.created_incidents,
                created_event_links=self.created_event_links,
                created_auxiliary=self.created_auxiliary,
                commands_before=commands_before,
                workflows_before=workflows_before,
                incidents_before=incidents_before,
            )

    def write_case_evidence(self, case_id: str) -> Path | None:
        if self.run_dir is None:
            return None
        document = case_evidence_document(
            case_id=case_id,
            details=self.results.get(case_id) or {},
            run_id=self.run_id,
            cluster_id=self.cluster_id,
            other_cluster_id=self.other_cluster_id,
            preflight=self.preflight_result,
            release_id=self.release_id,
        )
        return write_case_evidence(self.run_dir, document)

    def run(self, case_ids: tuple[str, ...] | None = None) -> dict[str, Any]:
        """Run selected cases, preserving FAIL evidence and stopping on failure."""

        selected = AUDITED_CASE_IDS if case_ids is None else case_ids
        expect(
            bool(selected)
            and len(set(selected)) == len(selected)
            and all(case_id in AUDITED_CASE_IDS for case_id in selected),
            "audit case selection is invalid",
        )
        self.preflight()
        for case_id in selected:
            method = getattr(self, f"run_{case_id.rsplit('-', 1)[1]}")
            commands_before = set(self.created_commands)
            workflows_before = set(self.created_workflows)
            incidents_before = set(self.created_incidents)
            try:
                self.check_work()
                method()
                self.check_work()
            except (Exception, AuditStopped) as exc:
                error = (
                    type(exc).__name__
                    if self.credential_envelope is not None
                    else f"{type(exc).__name__}: {exc}"
                )
                self.results[case_id] = {
                    **dict(self.results.get(case_id) or {}),
                    "verdict": "FAIL",
                    "error": error,
                }
                print(f"FAIL {case_id}: {error}", flush=True)
            else:
                self.results[case_id] = {
                    **dict(self.results.get(case_id) or {}),
                    "verdict": "PASS",
                }
                print(f"PASS {case_id}", flush=True)
            finally:
                try:
                    self._cleanup_since(
                        commands_before, workflows_before, incidents_before
                    )
                except (Exception, AuditStopped) as exc:
                    self.results[case_id]["verdict"] = "FAIL"
                    self.results[case_id]["cleanup_error"] = (
                        f"cleanup failed: {type(exc).__name__}"
                    )
            self.write_case_evidence(case_id)
            if self.results[case_id]["verdict"] != "PASS":
                break
        verdicts = {
            case_id: self.results[case_id]["verdict"] for case_id in self.results
        }
        return {
            "run_id": self.run_id,
            "cluster_id": self.cluster_id,
            "other_cluster_id": self.other_cluster_id,
            "release_id": self.release_id or None,
            "preflight": self.preflight_result,
            "verdict": (
                "PASS"
                if verdicts and all(value == "PASS" for value in verdicts.values())
                else "FAIL"
            ),
            "verdicts": verdicts,
            "results": self.results,
            "not_run": [case_id for case_id in selected if case_id not in self.results],
        }


def case_evidence_document(
    *,
    case_id: str,
    details: dict[str, Any],
    run_id: str,
    cluster_id: str,
    other_cluster_id: str,
    preflight: dict[str, Any],
    release_id: str = "",
) -> dict[str, Any]:
    """One CMD case's evidence document, from the case's recorded details."""

    body = dict(details)
    document: dict[str, Any] = {
        "schema_version": 1,
        "report_type": "fault-acceptance",
        "case_id": case_id,
        "verdict": body.pop("verdict", "FAIL"),
        "run_id": run_id,
        "cluster_id": cluster_id,
        "other_cluster_id": other_cluster_id,
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "preflight": preflight,
        "details": body,
    }
    if release_id:
        document["release_id"] = release_id
    if "error" in body:
        document["error"] = body["error"]
    return document


def _local_write_json(path: Path, document: dict[str, Any]) -> None:
    """The in-Pod fallback: same path layout, all-or-nothing rename, no scope."""

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    temporary.replace(path)


def write_case_evidence(run_dir: Path, document: dict[str, Any]) -> Path:
    """Write ``cases/<id>/<id>.json`` under ``run_dir``.

    In a checkout the shared writer is used, so the acceptance scope guard sees
    the document like any other case's. Inside the Pod there is no checkout;
    the same layout is written directly and the operator copies it out (or
    re-derives it from the printed summary with ``--write-evidence``).
    """

    case_id = str(document["case_id"])
    try:
        from scripts.e2e.regional.acceptance_runner_common import (
            write_json_atomic,
        )
        from scripts.e2e.regional.regional_case_contract import (
            case_evidence_path,
        )
    except ImportError:
        path = run_dir / "cases" / case_id / f"{case_id}.json"
        _local_write_json(path, document)
        return path
    path = case_evidence_path(run_dir, case_id)
    write_json_atomic(path, document)
    return path


def write_evidence_from_summary(
    summary: dict[str, Any],
    run_dir: Path,
    *,
    release_id: str = "",
) -> list[Path]:
    """Operator-side: turn the printed run summary into per-case evidence files."""

    results = summary.get("results")
    if not isinstance(results, dict) or not results:
        raise ProtocolAuditError("summary carries no per-case results")
    written = []
    for case_id in AUDITED_CASE_IDS:
        details = results.get(case_id)
        if not isinstance(details, dict):
            continue
        document = case_evidence_document(
            case_id=case_id,
            details=details,
            run_id=str(summary.get("run_id") or ""),
            cluster_id=str(summary.get("cluster_id") or ""),
            other_cluster_id=str(summary.get("other_cluster_id") or ""),
            preflight=dict(summary.get("preflight") or {}),
            release_id=release_id or str(summary.get("release_id") or ""),
        )
        written.append(write_case_evidence(run_dir, document))
    return written


def run_hyperpod_submission_case(audit: LiveProtocolAudit) -> None:
    record = make_submission_probe_record(audit)
    key = record["idempotency_key"]
    cluster_name = record["cluster_name"]
    other_name = audit.registry[audit.other_cluster_id]["hyperpod_cluster_name"]
    node = record["requested_node_identifiers"][0]
    request = {"cluster_id": audit.cluster_id, "record": record}
    first_status, first = audit._request(
        "POST",
        "/v1/regional/executors/hyperpod-submissions/reserve",
        cluster_id=audit.cluster_id,
        token=audit.tokens[audit.cluster_id],
        payload=request,
    )
    second_status, second = audit._request(
        "POST",
        "/v1/regional/executors/hyperpod-submissions/reserve",
        cluster_id=audit.cluster_id,
        token=audit.tokens[audit.cluster_id],
        payload=request,
    )
    cross_status, _ = audit._request(
        "POST",
        "/v1/regional/executors/hyperpod-submissions/reserve",
        cluster_id=audit.other_cluster_id,
        token=audit.tokens[audit.other_cluster_id],
        payload={
            "cluster_id": audit.other_cluster_id,
            "record": record,
        },
    )
    mismatch_status, _ = audit._request(
        "POST",
        "/v1/regional/executors/hyperpod-submissions/reserve",
        cluster_id=audit.cluster_id,
        token=audit.tokens[audit.cluster_id],
        payload={
            "cluster_id": audit.other_cluster_id,
            "record": {
                **record,
                "cluster_name": other_name,
            },
        },
    )
    query = urllib.parse.urlencode(
        {
            "cluster_name": cluster_name,
            "idempotency_key": key,
        }
    )
    get_status, current = audit._request(
        "GET",
        f"/v1/regional/executors/hyperpod-submissions?{query}",
        cluster_id=audit.cluster_id,
        token=audit.tokens[audit.cluster_id],
    )
    missing_query = urllib.parse.urlencode(
        {
            "cluster_name": cluster_name,
            "idempotency_key": f"{key}-missing",
        }
    )
    missing_status, missing = audit._request(
        "GET",
        f"/v1/regional/executors/hyperpod-submissions?{missing_query}",
        cluster_id=audit.cluster_id,
        token=audit.tokens[audit.cluster_id],
    )
    # A GET on a key nobody reserved must not create the record it looked for.
    missing_again_status, missing_again = audit._request(
        "GET",
        f"/v1/regional/executors/hyperpod-submissions?{missing_query}",
        cluster_id=audit.cluster_id,
        token=audit.tokens[audit.cluster_id],
    )
    outcome_record = {**record, "state": "SUBMITTED"}
    outcome_status, outcome = audit._request(
        "POST",
        "/v1/regional/executors/hyperpod-submissions/outcome",
        cluster_id=audit.cluster_id,
        token=audit.tokens[audit.cluster_id],
        payload={
            "cluster_id": audit.cluster_id,
            "record": outcome_record,
        },
    )
    unreserved_key = f"{key}-unreserved"
    unreserved_status, unreserved = audit._request(
        "POST",
        "/v1/regional/executors/hyperpod-submissions/outcome",
        cluster_id=audit.cluster_id,
        token=audit.tokens[audit.cluster_id],
        payload={
            "cluster_id": audit.cluster_id,
            "record": {
                **outcome_record,
                "idempotency_key": unreserved_key,
            },
        },
    )
    unreserved_query = urllib.parse.urlencode(
        {"cluster_name": cluster_name, "idempotency_key": unreserved_key}
    )
    unreserved_get_status, unreserved_after = audit._request(
        "GET",
        f"/v1/regional/executors/hyperpod-submissions?{unreserved_query}",
        cluster_id=audit.cluster_id,
        token=audit.tokens[audit.cluster_id],
    )
    changed_status, changed = audit._request(
        "POST",
        "/v1/regional/executors/hyperpod-submissions/outcome",
        cluster_id=audit.cluster_id,
        token=audit.tokens[audit.cluster_id],
        payload={
            "cluster_id": audit.cluster_id,
            "record": {
                **outcome_record,
                "requested_node_identifiers": [f"{node}-changed"],
            },
        },
    )
    anonymous = anonymous_submission_checks(
        audit._request,
        cluster_id=audit.cluster_id,
        reserve_payload=request,
        outcome_record=outcome_record,
        query=query,
    )
    expect(
        first_status == second_status == 200,
        f"reserve answered {first_status} then {second_status}",
    )
    expect(first["reserved"] is True, "the first reserve did not reserve")
    expect(
        second["reserved"] is False and second["record"]["state"] == "INTENDED",
        "the duplicate reserve did not return the INTENDED record",
    )
    expect(
        cross_status == mismatch_status == 403,
        f"cross-cluster answered {cross_status}, header/body mismatch "
        f"{mismatch_status}",
    )
    expect(
        get_status == 200 and current["state"] == "INTENDED",
        f"GET answered {get_status}",
    )
    expect(
        missing_status == 200 and missing is None,
        f"GET of a missing key answered {missing_status}: {missing!r}",
    )
    expect(
        missing_again_status == 200 and missing_again is None,
        "a GET of a missing key created a record",
    )
    expect(
        outcome_status == 200 and outcome["record"]["state"] == "SUBMITTED",
        f"outcome answered {outcome_status}",
    )
    expect(
        unreserved_status == 409 and "unreserved" in unreserved["detail"],
        f"unreserved outcome answered {unreserved_status}",
    )
    expect(
        unreserved_get_status == 200 and unreserved_after is None,
        "a refused unreserved outcome left a record behind",
    )
    expect(
        changed_status == 409 and "does not match" in changed["detail"],
        f"changed request answered {changed_status}",
    )
    audit.record(
        "GF-REGIONAL-CMD-016",
        reserve=[first_status, second_status],
        cross_cluster=cross_status,
        header_body_mismatch=mismatch_status,
        get=get_status,
        missing_get=missing_status,
        missing_get_repeat=missing_again_status,
        missing_get_created_record=missing_again is not None,
        outcome=outcome_status,
        unreserved=unreserved_status,
        unreserved_record_after=unreserved_after,
        changed_request=changed_status,
        anonymous=anonymous,
        cloudtrail={
            "status": "NOT_EVALUATED",
            "reason": (
                "the audit runs inside the API Pod without AWS credentials; "
                "the operator confirms no HyperPod UpdateClusterSoftware / "
                "BatchDeleteClusterNodes event for the probe key in CloudTrail"
            ),
            "idempotency_key": key,
        },
    )


def parser() -> argparse.ArgumentParser:
    result = build_audit_parser(
        AUDITED_CASE_IDS,
        description=__doc__.split("\n\n")[0],
    )
    result.add_argument(
        "--credentials-stdin",
        action="store_true",
        help="read only the two selected synthetic credentials from protected stdin",
    )
    result.add_argument("--synthetic-run-id", default="")
    result.add_argument("--overall-seconds", type=float, default=300)
    result.add_argument(
        "--cleanup-seconds",
        type=float,
        default=None,
        help=(
            "cumulative cleanup budget, at most 120 seconds; defaults to 30 for "
            "one case, required explicitly for multiple cases, never renewed"
        ),
    )
    return result


def run_with_deadline(arguments: argparse.Namespace) -> dict[str, Any]:
    audit: LiveProtocolAudit | None = None
    summary: dict[str, Any] = {"verdict": "FAIL", "not_run": list(arguments.case)}
    cleanup_seconds = arguments.cleanup_seconds
    if cleanup_seconds is None:
        expect(
            len(arguments.case) == 1,
            "multi-case audits require an explicit cumulative --cleanup-seconds",
        )
        cleanup_seconds = 30
    with AuditDeadline(arguments.overall_seconds, cleanup_seconds) as deadline:
        try:
            credentials = (
                read_credentials(
                    cluster_id=arguments.cluster_id,
                    other_cluster_id=arguments.other_cluster_id,
                    synthetic_run_id=arguments.synthetic_run_id,
                )
                if arguments.credentials_stdin
                else None
            )
            if credentials is not None:
                deadline.require_credential_lifetime(credentials)
            audit = LiveProtocolAudit(
                cluster_id=arguments.cluster_id,
                other_cluster_id=arguments.other_cluster_id,
                executor_sha256=arguments.executor_sha256,
                executor_digest=arguments.executor_digest,
                run_dir=arguments.run_dir,
                isolated_cluster=arguments.isolated_cluster,
                executor_ready_replicas=arguments.executor_ready_replicas,
                release_id=arguments.release_id,
                credential_envelope=credentials,
                deadline=deadline,
            )
            summary = audit.run(tuple(arguments.case))
        except (Exception, AuditStopped) as exc:
            summary["verdict"] = "FAIL"
            summary["error"] = f"audit aborted: {type(exc).__name__}"
        finally:
            deadline.stop()
            if audit is not None:
                try:
                    audit.close()
                except (Exception, AuditStopped) as exc:
                    summary["verdict"] = "FAIL"
                    summary["cleanup_error"] = f"owned cleanup: {type(exc).__name__}"
                    summary["recovery_required"] = True
    return summary


def main() -> int:
    arguments = parser().parse_args()
    if arguments.emit_probe:
        print(probe_source(), end="")
        return 0
    if arguments.write_evidence is not None:
        if arguments.credentials_stdin or arguments.synthetic_run_id:
            raise SystemExit("credential input is only valid for a live audit")
        if arguments.run_dir is None:
            raise SystemExit("--write-evidence needs --run-dir")
        summary = json.loads(arguments.write_evidence.read_text(encoding="utf-8"))
        written = write_evidence_from_summary(
            summary, arguments.run_dir, release_id=arguments.release_id
        )
        print(json.dumps({"written": [str(path) for path in written]}, indent=2))
        return 0 if summary.get("verdict") == "PASS" else 1
    for name in (
        "cluster_id",
        "other_cluster_id",
        "executor_sha256",
        "executor_digest",
    ):
        if not getattr(arguments, name):
            raise SystemExit(f"--{name.replace('_', '-')} is required to run the audit")
    if not arguments.case:
        raise SystemExit("an explicit --case is required to run the live audit")
    if bool(arguments.synthetic_run_id) != arguments.credentials_stdin:
        raise SystemExit(
            "--credentials-stdin requires --synthetic-run-id and vice versa"
        )
    try:
        summary = run_with_deadline(arguments)
    except (Exception, AuditStopped) as exc:
        summary = {"verdict": "FAIL", "error": f"audit refused: {type(exc).__name__}"}
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
