from __future__ import annotations

import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

from gpu_fault.node_agent.protocol import (
    NodeActionCommand,
)
from gpu_fault.node_agent.late_ownership import (
    OwnershipRefused,
    final_ownership_boundary,
    require_physical_ownership,
)
from gpu_fault.node_agent.operations.holders import (
    IDLE_DISPLAY_MODULES,
    display_module_state,
    gpu_client_holder_evidence,
    gpu_client_holder_summary,
    idle_display_modules,
)

MAX_RESET_BUSY_ATTEMPTS = 3


class ResetProgressError(RuntimeError):
    """A reset failure that carries how far the per-GPU loop got.

    The failure itself only names one GPU; an operator also needs the GPUs
    that finished and the ones nothing touched, so those travel to the
    workflow in the result details instead of only reaching the log.
    """

    def __init__(self, message: str, *, action_details: dict[str, Any]) -> None:
        super().__init__(message)
        self.action_details = action_details


class GpuResetBusy(ResetProgressError):
    """``In use by another client`` after every busy retry, naming who held it.

    The holders travel in ``gpu_client_holders`` (every NVIDIA device node, the
    compute apps and the kernel modules in use at the failure) so the operator
    reads the client from the step instead of logging in to the node.
    """

    def __init__(
        self, message: str, *, evidence: dict[str, Any], attempts: int
    ) -> None:
        super().__init__(
            f"{message}; {gpu_client_holder_summary(evidence)}",
            action_details={
                "gpu_client_holders": evidence,
                "gpu_reset_busy_attempts": attempts,
                # Every reset this command spawned returned "In use by another
                # client": a verdict, not an outcome in flight.
                "node_action_outcome_known": True,
            },
        )


class ResetOutcomeUnknown(ResetProgressError):
    """A driver reset may outlive nvidia-smi; only an operator can resolve it."""

    def __init__(self, message: str, *, gpu_uuids: list[str]) -> None:
        super().__init__(
            message,
            action_details={
                "outcome_unknown": True,
                "manual_confirmation_required": True,
                "reset_completed": [],
                "reset_outcome_unknown": list(gpu_uuids),
                "reset_failed": [],
                "reset_not_attempted": [],
            },
        )


class ResetOperationsMixin:
    # Attributes supplied by the composed concrete implementation.
    _gpu_inventory: Callable[..., Any]
    _run_checked: Callable[..., Any]
    _verify_no_clients: Callable[..., Any]
    _compute_clients: Callable[[], list[dict[str, str]]]
    _gpu_device_paths: Callable[[], dict[str, str]]
    proc_root: Path
    device_client_sample_interval_seconds: float
    fabric_manager_restart_enabled: bool
    fabric_reset_enabled: bool
    reset_enabled: bool
    single_gpu_reset_supported: bool
    sleep: Callable[[float], None]

    _RESET_BUSY_ATTEMPTS = MAX_RESET_BUSY_ATTEMPTS
    #: What a refused or busy reset reads back from its GPU afterwards: that the
    #: device is still enumerated and answers, and the driver's own view of
    #: whether a reset is wanted. Evidence for the record (live the refused
    #: reset was recorded as an unknown outcome and every operator exit was
    #: refused until an EC2 reboot); never a failure of its own.
    _GPU_READBACK_FIELDS = (
        "uuid",
        "pstate",
        "retired_pages.pending",
        "reset_status.reset_required",
        "reset_status.drain_and_reset_recommended",
    )

    def _gpu_readback(self, gpu_uuid: str) -> dict[str, Any]:
        fields = list(self._GPU_READBACK_FIELDS)
        try:
            completed = self._run_checked(
                [
                    "nvidia-smi",
                    "--query-gpu=" + ",".join(fields),
                    "--format=csv,noheader",
                    "-i",
                    gpu_uuid,
                ],
                timeout=60,
            )
        except Exception as exc:  # noqa: BLE001 - the readback is evidence only
            return {"error": type(exc).__name__}
        values = [item.strip() for item in str(completed.stdout).strip().split(",")]
        if len(values) != len(fields):
            return {"raw": str(completed.stdout).strip()[:200]}
        return dict(zip(fields, values, strict=True))

    def _release_idle_display_modules(
        self, gpu_uuids: list[str], released: list[str]
    ) -> bool:
        """Unload ``nvidia_drm`` / ``nvidia_modeset`` when they are the only
        thing between a busy reset and a free GPU; appends what it unloaded to
        ``released`` and says whether it unloaded anything.

        Only when no process holds any NVIDIA device node and no compute app
        is running (``gpu_client_holder_evidence``): a holder is a client to
        wait for or report, not something to pull a kernel module from under.
        The modules are loaded again after the reset (``_reload_display_modules``).
        """

        evidence = self._gpu_client_holder_evidence(gpu_uuids)
        if evidence.get("device_holders") or evidence.get("compute_client_count"):
            return False
        unloaded = False
        # Re-read ``/proc/modules`` after every unload: ``nvidia_modeset`` is
        # judged with ``nvidia_drm`` really gone (live the single snapshot
        # released the DRM module alone and left modeset at refcount 1).
        for _ in IDLE_DISPLAY_MODULES:
            candidates = [
                module
                for module in idle_display_modules(self.proc_root)
                if module not in released
            ]
            if not candidates:
                break
            require_physical_ownership()
            self._run_checked(["rmmod", candidates[0]], timeout=30)
            released.append(candidates[0])
            unloaded = True
        return unloaded

    def _reload_display_modules(self, released: list[str]) -> dict[str, Any]:
        """Load the display stack this reset unloaded; evidence, never a failure."""

        if not released:
            return {}
        details: dict[str, Any] = {"released_kernel_modules": list(released)}
        try:
            retained = display_module_state(self.proc_root)
        except Exception:  # noqa: BLE001 - evidence only
            retained = []
        if retained:
            # What the release left loaded (and why: its refcount and users),
            # so a modeset that stayed behind is readable from the record.
            details["display_modules_retained"] = retained
        # ``nvidia_drm`` depends on ``nvidia_modeset``: loading the outermost
        # module restores the stack exactly as the image had it, modeset first.
        outermost = "nvidia_drm" if "nvidia_drm" in released else released[-1]
        try:
            self._run_checked(["modprobe", outermost], timeout=60)
            details["reloaded_kernel_modules"] = list(reversed(released))
        except Exception as exc:  # noqa: BLE001 - the error is the evidence
            details["kernel_module_reload_error"] = f"{type(exc).__name__}: {exc}"
        return details

    def _run_reset_with_busy_retry(
        self,
        command: list[str],
        *,
        gpu_uuids: list[str],
        timeout: int,
        released: list[str] | None = None,
    ) -> int:
        released = [] if released is None else released
        for attempt in range(1, self._RESET_BUSY_ATTEMPTS + 1):
            require_physical_ownership()
            try:
                self._run_checked(command, timeout=timeout)
                return attempt
            except subprocess.TimeoutExpired as exc:
                # subprocess.run SIGKILLs nvidia-smi, but the reset it asked
                # the driver for keeps going: the outcome is unknown. A
                # TimeoutExpired would be classified retryable and the control
                # plane would resubmit, so convert it to a RuntimeError that
                # nothing retries. Recovery is a human decision (re-quiesce,
                # or reboot the node).
                raise ResetOutcomeUnknown(
                    f"gpu reset outcome unknown after {timeout}s; "
                    "refusing to retry automatically",
                    gpu_uuids=gpu_uuids,
                ) from exc
            except (RuntimeError, OwnershipRefused) as exc:
                busy = "In use by another client" in str(exc)
                if busy and attempt == self._RESET_BUSY_ATTEMPTS:
                    if isinstance(exc, OwnershipRefused):
                        raise
                    # The last busy refusal carries who held the device: the
                    # sampled preflight saw nobody on /dev/nvidiaN, so the
                    # client is elsewhere (nvidiactl, uvm, a replaced device
                    # node, a kernel module) and only this moment can show it.
                    raise GpuResetBusy(
                        str(exc),
                        evidence=self._gpu_client_holder_evidence(gpu_uuids),
                        attempts=attempt,
                    ) from exc
                if not busy:
                    raise
                if self._release_idle_display_modules(gpu_uuids, released):
                    # The DRM client was the holder; the next attempt is the
                    # first one that can succeed, so it needs no settle time.
                    self._verify_no_clients(gpu_uuids)
                    continue
                self.sleep(self.device_client_sample_interval_seconds)
                self._verify_no_clients(gpu_uuids)
        raise AssertionError("unreachable GPU reset retry state")

    def _gpu_client_holder_evidence(self, gpu_uuids: list[str]) -> dict[str, Any]:
        return gpu_client_holder_evidence(
            proc_root=self.proc_root,
            gpu_uuids=gpu_uuids,
            compute_clients=self._compute_clients,
            device_paths=self._gpu_device_paths,
        )

    def _reset_gpu_preflight(self, gpu_uuids: list[str]) -> None:
        """Everything a single-GPU reset can refuse on, before it resets.

        The executor runs this before it claims the quiesce window's single
        reset allowance. A config refusal, or a compute-app probe whose
        ``TimeoutExpired`` is retryable, must not spend that allowance: the
        resubmit the control plane sends would otherwise be refused for a
        reset that never ran. ``_reset_gpu`` runs it again immediately before
        nvidia-smi, so the check the reset relies on remains its own.
        """

        if not self.reset_enabled:
            raise RuntimeError("GPU reset is disabled by node configuration")
        if not self.single_gpu_reset_supported:
            raise RuntimeError(
                "single-GPU reset is not supported by this node "
                "topology; use RESET_ALL_GPUS_NVSWITCHES or reboot"
            )
        if not gpu_uuids:
            raise RuntimeError("GPU reset requires explicit GPU UUID targets")
        self._verify_no_clients(gpu_uuids)

    @contextmanager
    def _pre_spawn_guard(self, *, window_claimed: bool) -> Iterator[None]:
        """Make every step between the window claim and the reset final.

        The quiesce window's single reset allowance is already spent here, so a
        retryable failure would have the control plane resubmit into a claim it
        can never pass -- a step that dies for a transient reason, reported to
        the operator as if a reset had been issued. Anything that fails after
        the claim is therefore a non-retryable failure for a human.

        The claim is deliberately *not* released: nothing has read the node's
        state, and reopening the window would allow a second reset on a GPU
        whose first reset may already be running.
        """

        if not window_claimed:
            yield
            return
        try:
            yield
        except Exception as exc:
            raise RuntimeError(
                f"reset window claimed but pre-spawn re-check failed: {exc}; "
                "manual review required. The claim stays closed, so restore "
                "the GPU services and quiesce again before any further reset"
            ) from exc

    def _reset_gpu(
        self,
        gpu_uuids: list[str],
        *,
        window_claimed: bool = False,
    ) -> dict[str, Any]:
        with self._pre_spawn_guard(window_claimed=window_claimed):
            self._reset_gpu_preflight(gpu_uuids)
        reset_attempts = 0
        completed: list[str] = []
        released: list[str] = []
        for index, gpu_uuid in enumerate(gpu_uuids):
            try:
                reset_attempts += self._run_reset_with_busy_retry(
                    [
                        "nvidia-smi",
                        "--gpu-reset",
                        "-i",
                        gpu_uuid,
                    ],
                    gpu_uuids=[gpu_uuid],
                    timeout=120,
                    released=released,
                )
            except (RuntimeError, OwnershipRefused) as exc:
                # Report the loop's progress: the GPUs already reset, the one
                # that failed (with or without a readable outcome) and the
                # ones nothing touched. Only RuntimeError is wrapped, so the
                # retryable classification of anything else is unchanged.
                unknown = isinstance(exc, ResetOutcomeUnknown)
                raise ResetProgressError(
                    f"{exc} (GPU {gpu_uuid}, {index + 1} of {len(gpu_uuids)})",
                    action_details={
                        **(
                            exc.action_details
                            if isinstance(exc, (ResetProgressError, OwnershipRefused))
                            else {}
                        ),
                        **(
                            {"node_action_not_started": not completed}
                            if isinstance(exc, OwnershipRefused)
                            else {}
                        ),
                        # Every nvidia-smi this loop spawned returned (the
                        # unknown case is ``ResetOutcomeUnknown``): a refusal
                        # at the final ownership boundary after a busy retry is
                        # a known verdict -- not applied -- and the GPU is
                        # read back to say so, not an outcome to confirm.
                        **(
                            {}
                            if unknown
                            else {
                                "node_action_outcome_known": True,
                                "reset_verdict": "not-applied",
                                "gpu_readback": {
                                    gpu_uuid: self._gpu_readback(gpu_uuid)
                                },
                            }
                        ),
                        "reset_completed": completed,
                        "reset_outcome_unknown": [gpu_uuid] if unknown else [],
                        "reset_failed": [] if unknown else [gpu_uuid],
                        "reset_not_attempted": gpu_uuids[index + 1 :],
                        **self._reload_display_modules(released),
                    },
                ) from exc
            completed.append(gpu_uuid)
        return {
            "reset_gpu_uuids": gpu_uuids,
            "verified_no_gpu_clients": True,
            "reset_attempts": reset_attempts,
            "reset_successes": len(completed),
            "reset_busy_refusals": reset_attempts - len(completed),
            **self._reload_display_modules(released),
        }

    def _reset_all_preflight(self, gpu_uuids: list[str]) -> list[str]:
        """Verify a full fabric reset is allowed, and return the inventory.

        Same reason as :meth:`_reset_gpu_preflight`: the inventory query and
        the client probe can time out retryably, and neither has touched a
        GPU, so they run before the quiesce window's reset is claimed.
        """

        if not self.fabric_reset_enabled:
            raise RuntimeError(
                "full GPU/NVSwitch reset is disabled by node configuration"
            )
        requested = sorted(set(gpu_uuids))
        if not requested or len(requested) != len(gpu_uuids):
            raise RuntimeError(
                "full GPU/NVSwitch reset requires unique explicit GPU UUIDs"
            )
        local_inventory: list[str] = self._gpu_inventory()
        if requested != local_inventory:
            raise RuntimeError(
                "requested GPU inventory does not match local inventory: "
                f"requested={requested}; local={local_inventory}"
            )
        self._verify_no_clients(local_inventory)
        return local_inventory

    def _reset_all_gpus_nvswitches(
        self,
        gpu_uuids: list[str],
        *,
        window_claimed: bool = False,
    ) -> dict[str, Any]:
        with self._pre_spawn_guard(window_claimed=window_claimed):
            local_inventory = self._reset_all_preflight(gpu_uuids)
        released: list[str] = []
        try:
            reset_attempts = self._run_reset_with_busy_retry(
                ["nvidia-smi", "--gpu-reset"],
                gpu_uuids=local_inventory,
                timeout=180,
                released=released,
            )
        finally:
            reload = self._reload_display_modules(released)
        observed_after = self._gpu_inventory()
        if observed_after != local_inventory:
            raise RuntimeError(
                "GPU inventory changed after full fabric reset: "
                f"before={local_inventory}; after={observed_after}"
            )
        return {
            "reset_scope": "ALL_LOCAL_GPUS_AND_NVSWITCHES",
            "reset_gpu_uuids": local_inventory,
            "inventory_verified_before": True,
            "inventory_verified_after": True,
            "verified_no_gpu_clients": True,
            "command": ["nvidia-smi", "--gpu-reset"],
            "reset_attempts": reset_attempts,
            "reset_successes": 1,
            "reset_busy_refusals": reset_attempts - 1,
            **reload,
        }

    @final_ownership_boundary
    def _restart_fabric_manager(self, command: NodeActionCommand) -> dict[str, Any]:
        if not self.fabric_manager_restart_enabled:
            raise RuntimeError(
                "Fabric Manager restart is disabled by node configuration"
            )
        self._verify_no_clients(
            command.gpu_uuids,
            include_device_clients=False,
        )
        service = "nvidia-fabricmanager"
        before = self._run_checked(
            [
                "systemctl",
                "show",
                "--property=MainPID",
                "--value",
                service,
            ],
            timeout=30,
        ).stdout.strip()
        require_physical_ownership()
        self._run_checked(
            ["systemctl", "restart", service],
            timeout=120,
        )
        self._run_checked(
            ["systemctl", "is-active", "--quiet", service],
            timeout=30,
        )
        after = self._run_checked(
            [
                "systemctl",
                "show",
                "--property=MainPID",
                "--value",
                service,
            ],
            timeout=30,
        ).stdout.strip()
        return {
            "service": service,
            "active": True,
            "previous_main_pid": before or None,
            "current_main_pid": after or None,
        }
