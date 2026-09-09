from __future__ import annotations

import gzip
import hashlib
import hmac
import json
import os
import re
import shutil
import subprocess
import tarfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from gpu_fault.dcgm_diagnostic_analysis import (
    build_dcgm_recommendations,
    dcgm_failures_are_configuration_only,
    extract_dcgm_diagnostic_findings,
    normalize_dcgm_status,
)
from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent.protocol import (
    NodeActionCommand,
)

FIELD_DIAGNOSTIC_FAILURE_PATTERN = re.compile(
    r"\b(?:FAIL|FAILED|FAILURE|FATAL|ERROR)\b",
    re.IGNORECASE,
)

# The phrases a *passing* Field Diagnostic uses to report a clean count. The
# first alternative is the count ahead of the noun ("0 errors", "no failures");
# the second is the count behind it, which is how NVIDIA's own tool writes it
# ("Error count: 0", "NVLink error counters: 0", "ERROR: none detected") and
# which used to fail every healthy GPU with "reported failure despite exit
# status 0". Only a zero or "none" counts: "Error count: 5" is a failure.
BENIGN_DIAGNOSTIC_COUNT_PATTERN = re.compile(
    r"""
      \b(?:0|no|none)\s+(?:errors?|failures?|faults?)
        (?:\s+(?:detected|found|reported))?\b
    | \b(?:errors?|failures?|faults?)
        (?:\s+(?:count|counts|counters?))?
        \s*[:=]\s*
        (?:0+|none(?:\s+(?:detected|found|reported))?)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Every family of evidence this agent leaves in ``diagnostic_output_dir``.
DIAGNOSTIC_EVIDENCE_PATTERNS = (
    "gpu-diagnostic-*.tar.gz",
    "dcgm-quick-diagnostic-*.json",
)


class DiagnosticOperationsMixin:
    # Attributes supplied by the composed concrete implementation.
    field_diagnostic_enabled: bool

    _capture_efa_rdma_state: Callable[..., Any]
    _capture_hung_process_state: Callable[..., Any]
    _verify_no_clients: Callable[..., Any]
    diagnostic_max_archives: int
    diagnostic_output_dir: Path
    diagnostic_retention_seconds: int
    diagnostic_s3_uri: str | None
    field_diagnostic_command: tuple[str, ...]
    field_diagnostic_sha256: str | None
    field_diagnostic_timeout_seconds: int
    health_snapshot_request_dir: Path
    memory_field_diagnostic_command: tuple[str, ...]
    memory_field_diagnostic_sha256: str | None
    now: Callable[..., Any]
    runner: Callable[..., Any]

    def _trigger_health_snapshot(self) -> dict[str, Any]:
        self.health_snapshot_request_dir.mkdir(parents=True, exist_ok=True)
        requested_at = self.now().isoformat()
        for channel in ("gpu", "host"):
            (self.health_snapshot_request_dir / f"{channel}.request").write_text(
                requested_at + "\n", encoding="ascii"
            )
        services = [
            "gpu-fault-metrics-collector.service",
            "gpu-fault-host-collector.service",
        ]
        for service in services:
            self._run_checked(
                ["systemctl", "restart", service],
                timeout=120,
            )
            self._run_checked(
                [
                    "systemctl",
                    "is-active",
                    "--quiet",
                    service,
                ],
                timeout=15,
            )
        return {
            "snapshot_triggered": True,
            "triggered_at": requested_at,
            "restarted_collectors": services,
        }

    @staticmethod
    def _validate_field_diagnostic_config(
        command: tuple[str, ...],
        expected_sha256: str | None,
    ) -> None:
        if not command or not Path(command[0]).is_absolute():
            raise ValueError("Field Diagnostic command must be absolute")
        allowed = {"link_id", "gpu_uuid", "pci_bdf"}
        placeholders = {
            item
            for argument in command
            for item in re.findall(r"\{([^{}]+)\}", argument)
        }
        unsupported = placeholders - allowed
        if unsupported:
            raise ValueError(
                "unsupported Field Diagnostic placeholders: "
                + ", ".join(sorted(unsupported))
            )
        if not expected_sha256 or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
            raise ValueError("Field Diagnostic executable SHA-256 is required")
        executable = Path(command[0])
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ValueError("Field Diagnostic executable is missing or not executable")
        actual = hashlib.sha256(executable.read_bytes()).hexdigest()
        if not hmac.compare_digest(actual.lower(), expected_sha256.lower()):
            raise ValueError("Field Diagnostic executable SHA-256 mismatch")

    def _run_field_diagnostic(self, command: NodeActionCommand) -> dict[str, Any]:
        if not self.field_diagnostic_enabled:
            raise RuntimeError("Field Diagnostic is disabled")
        diagnostic_command = self.field_diagnostic_command
        diagnostic_sha256 = self.field_diagnostic_sha256
        if command.operation is WorkflowOperation.RUN_FIELD_DIAGNOSTIC:
            diagnostic_command = self.memory_field_diagnostic_command
            diagnostic_sha256 = self.memory_field_diagnostic_sha256
            if not diagnostic_command:
                raise RuntimeError("memory Field Diagnostic is not configured")
        link_id = command.parameters.get("nvlink_link_id")
        if command.operation is WorkflowOperation.RUN_NVLINK74_WORKFLOW and (
            not isinstance(link_id, int) or link_id < 0
        ):
            raise RuntimeError("Field Diagnostic requires an explicit NVLink ID")
        if link_id is not None and (not isinstance(link_id, int) or link_id < 0):
            raise RuntimeError("Field Diagnostic NVLink ID is invalid")
        if link_id is None and any(
            "{link_id}" in argument for argument in diagnostic_command
        ):
            raise RuntimeError(
                "configured Field Diagnostic command requires an "
                "NVLink ID and cannot diagnose a memory fault"
            )
        if len(command.gpu_uuids) != 1:
            raise RuntimeError("Field Diagnostic requires exactly one GPU UUID")
        gpu_uuid = command.gpu_uuids[0]
        pci_bdf = command.parameters.get("pci_bdf")
        if pci_bdf is not None and not isinstance(pci_bdf, str):
            raise RuntimeError("Field Diagnostic PCI BDF is invalid")
        self._verify_no_clients({gpu_uuid})
        replacements = {
            "{link_id}": str(link_id) if link_id is not None else "",
            "{gpu_uuid}": gpu_uuid,
            "{pci_bdf}": pci_bdf or "",
        }
        rendered = []
        for argument in diagnostic_command:
            for placeholder, value in replacements.items():
                argument = argument.replace(placeholder, value)
            rendered.append(argument)
        completed = self._run_checked(
            rendered,
            timeout=self.field_diagnostic_timeout_seconds,
        )
        combined_output = "\n".join(
            value
            for value in (
                completed.stdout or "",
                completed.stderr or "",
            )
            if value
        )
        failed_lines = [
            line.strip()
            for line in combined_output.splitlines()
            if self._field_diagnostic_line_failed(line)
        ]
        if failed_lines:
            raise RuntimeError(
                "Field Diagnostic reported failure despite exit status 0: "
                + " | ".join(failed_lines[-10:])
            )
        return {
            "field_diagnostic": "PASSED",
            "procedure": command.parameters.get("procedure", "NVIDIA_FIELD_DIAGNOSTIC"),
            "gpu_uuid": gpu_uuid,
            "nvlink_link_id": link_id,
            "pci_bdf": pci_bdf,
            "command_sha256": diagnostic_sha256,
            "stdout_tail": (completed.stdout or "")[-4096:],
            "stderr_tail": (completed.stderr or "")[-4096:],
        }

    @staticmethod
    def _field_diagnostic_line_failed(line: str) -> bool:
        normalized = line.strip()
        if not normalized:
            return False
        # Blank out the phrases that *report* a clean count and judge what is
        # left. The old code suppressed the whole line on one benign phrase, so
        # "Error count: 0 ... Overall Result: FAIL" would have passed once the
        # counter spelling was understood.
        remainder = BENIGN_DIAGNOSTIC_COUNT_PATTERN.sub(" ", normalized)
        return bool(FIELD_DIAGNOSTIC_FAILURE_PATTERN.search(remainder))

    def _collect_diagnostic_bundle(self, command: NodeActionCommand) -> dict[str, Any]:
        self._cleanup_diagnostic_archives()
        bundle_key = hashlib.sha256(command.command_id.encode()).hexdigest()[:24]
        work_dir = self.diagnostic_output_dir / bundle_key
        archive = self.diagnostic_output_dir / (f"gpu-diagnostic-{bundle_key}.tar.gz")
        partial_archive = archive.with_suffix(archive.suffix + ".partial")
        self.diagnostic_output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(mode=0o700, exist_ok=True)
        captures = [
            (
                "nvidia-smi-q.txt",
                ["nvidia-smi", "-q"],
                60,
            ),
            (
                "nvidia-smi-nvlink.txt",
                ["nvidia-smi", "nvlink", "--status"],
                60,
            ),
            (
                "nvidia-smi-nvlink-errors.txt",
                # nvidia-smi has no --errors option; the error counters
                # live behind --errorcounters. The old spelling exited 2
                # with "Option --errors is not recognized" on every
                # driver, so NVLink error counters never reached a bundle.
                ["nvidia-smi", "nvlink", "--errorcounters"],
                60,
            ),
            (
                "nvidia-smi-topology.txt",
                ["nvidia-smi", "topo", "--matrix"],
                60,
            ),
            (
                "dcgm-diag.json",
                ["dcgmi", "diag", "-r", "1", "-j"],
                300,
            ),
            (
                "fabric-manager-journal.txt",
                [
                    "journalctl",
                    "-u",
                    "nvidia-fabricmanager",
                    "--since",
                    "-30 minutes",
                    "--no-pager",
                ],
                60,
            ),
            (
                "kernel-nvidia.txt",
                [
                    "journalctl",
                    "-k",
                    "--since",
                    "-30 minutes",
                    "--no-pager",
                ],
                60,
            ),
            (
                "nvidia-bug-report.log.gz",
                ["/usr/bin/nvidia-bug-report.sh"],
                600,
            ),
        ]
        if command.parameters.get("diagnostic_reason") == "EFA_TRAFFIC_HUNG_SUSPECTED":
            captures.append(
                (
                    "rdma-link-show.txt",
                    ["rdma", "link", "show"],
                    60,
                )
            )
        manifest = {
            "command_id": command.command_id,
            "workflow_request_id": command.workflow_request_id,
            "incident_id": command.incident_id,
            "node_id": command.node_id,
            "collected_at": self.now().isoformat(),
            "fault_context": command.parameters,
            "captures": [],
        }
        try:
            for filename, argv, timeout in captures:
                output_path = work_dir / filename
                if argv[0] == "/usr/bin/nvidia-bug-report.sh":
                    # The NVIDIA script appends .gz itself. Passing an
                    # already-compressed filename produces a misleading
                    # .gz.gz file and leaves command output at the expected
                    # path.
                    script_output_path = output_path.with_suffix("")
                    argv = [
                        *argv,
                        "--output-file",
                        str(script_output_path),
                    ]
                try:
                    completed = self.runner(
                        argv,
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                    )
                    output = completed.stdout or ""
                    if completed.stderr:
                        output += "\n[stderr]\n" + completed.stderr
                    if (
                        argv[0] == "/usr/bin/nvidia-bug-report.sh"
                        and not output_path.exists()
                        and script_output_path.exists()
                    ):
                        with (
                            script_output_path.open("rb") as source,
                            gzip.open(output_path, "wb") as target,
                        ):
                            shutil.copyfileobj(source, target)
                        script_output_path.unlink()
                    if not output_path.exists():
                        if output_path.suffix == ".gz":
                            with gzip.open(
                                output_path,
                                "wt",
                                encoding="utf-8",
                                errors="replace",
                            ) as target:
                                target.write(output)
                        else:
                            output_path.write_text(
                                output,
                                encoding="utf-8",
                                errors="replace",
                            )
                    returncode = completed.returncode
                    error = None
                except (
                    OSError,
                    subprocess.TimeoutExpired,
                ) as exc:
                    returncode = None
                    error = f"{type(exc).__name__}: {exc}"
                    if output_path.suffix == ".gz":
                        with gzip.open(
                            output_path,
                            "wt",
                            encoding="utf-8",
                            errors="replace",
                        ) as target:
                            target.write(error)
                    else:
                        output_path.write_text(error, encoding="utf-8")
                capture = {
                    "file": filename,
                    "command": argv,
                    "returncode": returncode,
                }
                if error:
                    capture["error"] = error
                manifest["captures"].append(capture)
            if command.parameters.get("capture_process_state"):
                self._capture_efa_rdma_state(work_dir, command, manifest)
                self._capture_hung_process_state(work_dir, command, manifest)
            (work_dir / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True, indent=2),
                encoding="utf-8",
            )
            partial_archive.unlink(missing_ok=True)
            with tarfile.open(partial_archive, "w:gz") as tar:
                tar.add(work_dir, arcname="diagnostics")
            partial_archive.replace(archive)
        finally:
            partial_archive.unlink(missing_ok=True)
            shutil.rmtree(work_dir, ignore_errors=True)
        digest_state = hashlib.sha256()
        with archive.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest_state.update(chunk)
        digest = digest_state.hexdigest()
        evidence_ref = f"file://{archive}"
        if self.diagnostic_s3_uri:
            from urllib.parse import urlparse

            import boto3

            parsed = urlparse(self.diagnostic_s3_uri)
            if parsed.scheme != "s3" or not parsed.netloc:
                raise RuntimeError("invalid diagnostic S3 URI")
            key = "/".join(
                item
                for item in [
                    parsed.path.strip("/"),
                    command.node_id,
                    archive.name,
                ]
                if item
            )
            boto3.client("s3").upload_file(str(archive), parsed.netloc, key)
            evidence_ref = f"s3://{parsed.netloc}/{key}"
        result = {
            "evidence_ref": evidence_ref,
            "sha256": digest,
            "size_bytes": archive.stat().st_size,
            "manifest_summary": {
                "capture_count": len(manifest["captures"]),
                "failed_capture_count": sum(
                    1
                    for item in manifest["captures"]
                    if item.get("returncode") not in {0, None} or item.get("error")
                ),
                "diagnostic_reason": command.parameters.get("diagnostic_reason"),
                "capture_process_state": bool(
                    command.parameters.get("capture_process_state")
                ),
            },
        }
        self._cleanup_diagnostic_archives()
        return result

    def _cleanup_diagnostic_archives(self, now: datetime | None = None) -> list[str]:
        """Prune every family of diagnostic evidence this agent writes.

        The sweep used to glob only ``gpu-diagnostic-*.tar.gz``, so the JSON a
        quick diagnostic writes for its evidence reference was never removed:
        on a node that flaps XIDs those accumulate one per attempt in the same
        0700 directory until the root filesystem fills, which takes kubelet
        with it. Each family keeps its own count budget so a burst of quick
        diagnostics cannot evict the bundle a support case is waiting on.
        """

        if not self.diagnostic_output_dir.exists():
            return []
        timestamp = now or self.now()
        cutoff = (
            timestamp - timedelta(seconds=self.diagnostic_retention_seconds)
        ).timestamp()
        removed: list[str] = []
        for pattern in DIAGNOSTIC_EVIDENCE_PATTERNS:
            removed.extend(self._prune_diagnostic_family(pattern, cutoff))
        return removed

    def _prune_diagnostic_family(self, pattern: str, cutoff: float) -> list[str]:
        # A file can be gone between the glob and the stat: a concurrent triage
        # sweeps the same 0700 directory, and evidence is pulled off the node
        # with tools that remove the source. It needs no pruning, and it must
        # not raise out of the sweep -- that used to fail the diagnostic step
        # over the archive it had just written, and prune nothing at all.
        candidates: list[tuple[float, str, Path]] = []
        for archive in self.diagnostic_output_dir.glob(pattern):
            try:
                modified = archive.stat().st_mtime
            except OSError:
                continue
            candidates.append((modified, archive.name, archive))
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        removed = []
        for index, (modified, name, archive) in enumerate(candidates):
            if index >= self.diagnostic_max_archives or modified < cutoff:
                archive.unlink(missing_ok=True)
                removed.append(name)
        return removed

    @staticmethod
    def _dcgm_diagnostic_statuses(
        value: Any,
        *,
        path: str = "$",
        depth: int = 0,
        max_depth: int = 64,
    ) -> list[dict[str, str]]:
        if depth > max_depth:
            raise ValueError("DCGM diagnostic JSON exceeds maximum depth")
        statuses = []
        if isinstance(value, dict):
            for key, item in value.items():
                child_path = f"{path}.{key}"
                normalized_key = re.sub(r"[^a-z]", "", str(key).lower())
                if normalized_key in {
                    "status",
                    "result",
                    "testresult",
                    "overallresult",
                }:
                    normalized_value = normalize_dcgm_status(item)
                    if normalized_value is not None:
                        statuses.append(
                            {
                                "path": child_path,
                                "status": normalized_value,
                            }
                        )
                statuses.extend(
                    DiagnosticOperationsMixin._dcgm_diagnostic_statuses(
                        item,
                        path=child_path,
                        depth=depth + 1,
                        max_depth=max_depth,
                    )
                )
        elif isinstance(value, list):
            for index, item in enumerate(value):
                statuses.extend(
                    DiagnosticOperationsMixin._dcgm_diagnostic_statuses(
                        item,
                        path=f"{path}[{index}]",
                        depth=depth + 1,
                        max_depth=max_depth,
                    )
                )
        return statuses

    def _run_dcgm_diagnostic(self, command: NodeActionCommand) -> dict[str, Any]:
        result = self.runner(
            ["dcgmi", "diag", "-r", "1", "-j"],
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = result.stdout or ""
        if result.stderr:
            output += "\n[stderr]\n" + result.stderr
        evidence_key = hashlib.sha256(command.command_id.encode()).hexdigest()[:24]
        output_path = self.diagnostic_output_dir / (
            f"dcgm-quick-diagnostic-{evidence_key}.json"
        )
        self.diagnostic_output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        # Sweep before writing, the way ``_collect_diagnostic_bundle`` does: a
        # node whose only diagnostic operation is the quick one would otherwise
        # never run retention at all, and the sweep can never reach the file
        # this call is about to produce.
        self._cleanup_diagnostic_archives()
        output_path.write_text(output, encoding="utf-8", errors="replace")
        digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
        evidence_ref = f"file://{output_path}"
        if self.diagnostic_s3_uri:
            from urllib.parse import urlparse

            import boto3

            parsed = urlparse(self.diagnostic_s3_uri)
            if parsed.scheme != "s3" or not parsed.netloc:
                raise RuntimeError("invalid diagnostic S3 URI")
            key = "/".join(
                item
                for item in [
                    parsed.path.strip("/"),
                    command.node_id,
                    output_path.name,
                ]
                if item
            )
            boto3.client("s3").upload_file(str(output_path), parsed.netloc, key)
            evidence_ref = f"s3://{parsed.netloc}/{key}"

        parse_error = None
        payload: Any = None
        statuses: list[dict[str, str]] = []
        try:
            payload = json.loads(result.stdout)
            statuses = self._dcgm_diagnostic_statuses(payload)
        except (json.JSONDecodeError, TypeError) as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
        observed = {item["status"] for item in statuses}
        diagnostic_findings = (
            extract_dcgm_diagnostic_findings(payload) if payload is not None else []
        )
        # DCGM grades every failure with dcgmErrorSeverity_t. When the
        # only failures are CONFIG severity ("this error can be
        # configured", e.g. persistence mode disabled) the GPU is not
        # diagnosed as faulty, so the outcome must be WARN rather than
        # FAIL — otherwise a host configuration gap drains the node.
        configuration_only = (
            parse_error is None
            and dcgm_failures_are_configuration_only(diagnostic_findings)
        )
        if configuration_only:
            outcome = "WARN"
        elif result.returncode != 0 or "FAIL" in observed:
            outcome = "FAIL"
        elif "WARN" in observed:
            outcome = "WARN"
        elif "PASS" in observed:
            outcome = "PASS"
        else:
            outcome = "INCONCLUSIVE"
        recommendations = build_dcgm_recommendations(
            diagnostic_findings,
            returncode=result.returncode,
            parse_error=parse_error,
            configuration_only=configuration_only,
        )
        return {
            "diagnostic_outcome": outcome,
            "configuration_only_failures": configuration_only,
            "returncode": result.returncode,
            "status_counts": {
                status: sum(item["status"] == status for item in statuses)
                for status in sorted(observed)
            },
            "failed_checks": [
                item["path"] for item in statuses if item["status"] == "FAIL"
            ],
            "warning_checks": [
                item["path"] for item in statuses if item["status"] == "WARN"
            ],
            "diagnostic_findings": diagnostic_findings,
            "recommended_actions": recommendations,
            "parse_error": parse_error,
            "evidence_ref": evidence_ref,
            "sha256": digest,
            "size_bytes": output_path.stat().st_size,
        }

    def _run_checked(
        self, command: list[str], *, timeout: int
    ) -> subprocess.CompletedProcess:
        try:
            return self.runner(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.CalledProcessError as exc:
            output = (exc.stderr or exc.stdout or "").strip()
            if len(output) > 2000:
                output = output[-2000:]
            detail = f": {output}" if output else ""
            raise RuntimeError(
                f"{command[0]} exited with status {exc.returncode}{detail}"
            ) from exc
