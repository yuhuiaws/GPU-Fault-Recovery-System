from __future__ import annotations

import os
import subprocess
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator
from xml.etree import ElementTree as ET


from gpu_fault.node_agent.protocol import (
    NodeActionCommand,
)


class ClientOperationsMixin:
    # Attributes supplied by the composed concrete implementation.
    device_client_samples: Any
    gpu_device_path_finder: Callable[..., Any]

    _run_checked: Callable[..., Any]
    device_client_finder: Callable[..., Any]
    device_client_sample_interval_seconds: Any
    proc_root: Any
    sleep: Callable[..., Any]

    # Owned by ``_device_path_cache_window``. Per thread, because the action
    # pool runs four commands on one executor instance and an instance counter
    # is a read-modify-write nobody holds a lock for. A plain assignment, not a
    # slot the concrete executor has to build.
    _device_path_state = threading.local()

    def _compute_clients(self) -> list[dict[str, str]]:
        completed = self._run_checked(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name",
                "--format=csv,noheader,nounits",
            ],
            timeout=15,
        )
        clients = []
        for line in completed.stdout.splitlines():
            values = [value.strip() for value in line.split(",", 2)]
            if len(values) == 3 and values[0]:
                clients.append(
                    {
                        "gpu_uuid": values[0],
                        "pid": values[1],
                        "process_name": values[2],
                    }
                )
        return clients

    @contextmanager
    def _device_path_cache_window(self) -> Iterator[None]:
        """Resolve the uuid -> device-node map at most once inside the block.

        The map is a property of the driver's probe order and cannot change
        while the node holds a GPU still, yet ``_persistent_device_clients``
        rebuilt it for every sample: with three samples on an 8-GPU node, one
        busy-retrying reset spent up to 96 ``nvidia-smi`` invocations learning
        the same answer -- and each one is a driver round trip that competes
        with the very reset it is waiting for.

        The window lives in thread-local state and restores whatever it found,
        so the four action-pool workers neither share a table nor need a lock,
        and nothing survives the block: across commands a driver reload really
        can renumber the devices.
        """

        state = self._device_path_state
        outer_entry = getattr(state, "entry", None)
        outer_caching = getattr(state, "caching", False)
        outer_reasons = getattr(state, "reasons", None)
        state.entry = None
        # Cleared with the table: a reason kept from an earlier command would
        # explain this command's unresolvable target with the previous driver
        # report, which is worse than saying nothing.
        state.reasons = None
        state.caching = True
        try:
            yield
        finally:
            state.entry = outer_entry
            state.caching = outer_caching
            state.reasons = outer_reasons

    def _gpu_device_paths(self) -> dict[str, str]:
        state = self._device_path_state
        entry = getattr(state, "entry", None)
        # The table belongs to one executor: the id keeps a window opened by one
        # from answering another's question.
        if entry is not None and entry[0] == id(self):
            cached: dict[str, str] = entry[1]
            return cached
        paths, reasons = self._minor_number_device_map()
        if paths is None:
            paths = self._index_device_paths()
            # The index map resolves by position and leaves nobody out, so a
            # UUID missing from it is missing from the driver's report itself.
            reasons = {}
        # Kept beside the table (same thread, same executor) so a refusal can
        # say *why* a target has no device node instead of only that it has
        # none. Written whether or not a window is caching: the reader is the
        # very next call after this one.
        state.reasons = (id(self), reasons)
        if getattr(state, "caching", False):
            state.entry = (id(self), paths)
        return paths

    def _device_path_reasons(self) -> dict[str, str]:
        """Why this executor's current map left a UUID out, if it knows.

        Empty when the map came from an injected finder or from the index
        fallback: the caller then falls back to a cause it can still establish
        (a UUID no local GPU reports at all) or to a generic one.
        """

        entry = getattr(self._device_path_state, "reasons", None)
        if entry is None or entry[0] != id(self):
            return {}
        reasons: dict[str, str] = entry[1]
        return reasons

    def _minor_number_device_map(
        self,
    ) -> tuple[dict[str, str] | None, dict[str, str]]:
        """uuid -> ``/dev/nvidia<minor>`` from the driver's own minor numbers.

        ``--query-gpu=index`` is ordered by PCI bus id; the device node's minor
        number is the order the driver probed the GPUs in, which is why
        nvidia-smi reports "Minor Number" separately at all.  Assuming they
        match made ``VERIFY_NO_GPU_CLIENTS`` inspect a *different* GPU's device
        node on any node where the two disagree: it passed with a live holder
        on the target GPU, and blocked on a holder of a GPU nobody was
        resetting.

        Answers ``None`` only when the report carries no usable minor number at
        all -- a driver that does not report them -- and the caller then falls
        back to the index map as a whole. A single GPU that answers ``N/A``,
        which is what a card that fell off the bus reports, is left out of the
        map instead: degrading its seven healthy siblings to the index map is
        exactly the mapping that can point a UUID at another GPU's device node.
        An unresolved target fails the verification closed instead.

        The second answer is why each excluded UUID is out, because only this
        parse knows: "the driver reports no minor number" and "two GPUs claim
        one device node" are the same symptom (an unresolvable target) with
        different operator actions behind them, and the refusal is the only
        place either is ever seen.
        """

        try:
            completed = self._run_checked(["nvidia-smi", "-q", "-x"], timeout=15)
        except (OSError, RuntimeError, subprocess.SubprocessError):
            return None, {}
        try:
            root = ET.fromstring(completed.stdout or "")
        except ET.ParseError:
            return None, {}
        paths: dict[str, str] = {}
        reasons: dict[str, str] = {}
        claimed: set[str] = set()
        duplicated: set[str] = set()
        for gpu in root.findall(".//gpu"):
            gpu_uuid = (gpu.findtext("uuid") or "").strip()
            minor = (gpu.findtext("minor_number") or "").strip()
            if not gpu_uuid:
                continue
            if not minor.isdigit():
                reasons[gpu_uuid] = (
                    "the driver reports no minor number for it, so it has no "
                    "device node"
                )
                continue
            device = f"/dev/nvidia{minor}"
            if device in claimed:
                duplicated.add(device)
            claimed.add(device)
            paths[gpu_uuid] = device
        if not paths:
            # No GPU reported a usable minor number: this driver does not give
            # them, so the index map is all there is.
            return None, {}
        # Two UUIDs on one device node is a report that cannot be trusted for
        # either of them, so neither is resolved and both fail closed. The
        # answer stays a (possibly empty) map, never ``None``: minor numbers
        # were reported, so falling back to the index would hand exactly those
        # UUIDs the device node this report already proved is ambiguous.
        for gpu_uuid in [
            uuid for uuid, device in paths.items() if device in duplicated
        ]:
            reasons[gpu_uuid] = (
                f"the driver reports device node {paths[gpu_uuid]} for more "
                "than one GPU"
            )
            del paths[gpu_uuid]
        return paths, reasons

    def _index_device_paths(self) -> dict[str, str]:
        """The pre-minor-number mapping, kept for drivers that omit it."""

        completed = self._run_checked(
            [
                "nvidia-smi",
                "--query-gpu=uuid,index",
                "--format=csv,noheader,nounits",
            ],
            timeout=15,
        )
        paths = {}
        for line in completed.stdout.splitlines():
            values = [value.strip() for value in line.split(",", 1)]
            if len(values) == 2 and values[0] and values[1].isdigit():
                paths[values[0]] = f"/dev/nvidia{values[1]}"
        return paths

    def _quiesce_scope(self, command: NodeActionCommand) -> tuple[set[str], set[str]]:
        device_paths = self.gpu_device_path_finder()
        if command.gpu_uuids:
            missing = sorted(set(command.gpu_uuids) - set(device_paths))
            if missing:
                raise RuntimeError(
                    "quiesce cannot resolve target GPU UUIDs: " + ", ".join(missing)
                )
            target_device_paths = {
                device_paths[gpu_uuid] for gpu_uuid in command.gpu_uuids
            }
        else:
            target_device_paths = set(device_paths.values())

        raw_by_node = command.parameters.get("workload_cgroup_paths_by_node")
        raw_paths = (
            raw_by_node.get(command.node_id, [])
            if isinstance(raw_by_node, dict)
            else command.parameters.get("workload_cgroup_paths", [])
        )
        if raw_paths is None:
            raw_paths = []
        if not isinstance(raw_paths, list) or not all(
            isinstance(path, str) for path in raw_paths
        ):
            raise RuntimeError("workload cgroup paths must be a list of strings")
        workload_cgroup_paths = {
            path.rstrip("/") for path in raw_paths if path and path.rstrip("/")
        }
        return target_device_paths, workload_cgroup_paths

    def _gpu_inventory(self) -> list[str]:
        completed = self._run_checked(
            [
                "nvidia-smi",
                "--query-gpu=uuid",
                "--format=csv,noheader,nounits",
            ],
            timeout=15,
        )
        inventory = sorted(
            {line.strip() for line in completed.stdout.splitlines() if line.strip()}
        )
        if not inventory:
            raise RuntimeError("local GPU inventory is empty")
        return inventory

    @staticmethod
    def _process_name(process: Path) -> str:
        try:
            return (process / "comm").read_text().strip()
        except (OSError, UnicodeError):
            return "unknown"

    def _device_clients(self, gpu_uuids: set[str]) -> list[dict[str, str]]:
        device_paths = self.gpu_device_path_finder()
        targets = {
            uuid: path
            for uuid, path in device_paths.items()
            if not gpu_uuids or uuid in gpu_uuids
        }
        path_to_uuid = {path: uuid for uuid, path in targets.items()}
        clients: dict[tuple[str, str], dict[str, str]] = {}
        try:
            processes = list(self.proc_root.iterdir())
        except OSError:
            processes = []
        for process in processes:
            if not process.name.isdigit():
                continue
            try:
                descriptors = list((process / "fd").iterdir())
            except OSError:
                continue
            for descriptor in descriptors:
                try:
                    target = os.readlink(descriptor)
                except OSError:
                    continue
                gpu_uuid = path_to_uuid.get(target)
                if gpu_uuid is None:
                    continue
                key = (gpu_uuid, process.name)
                clients[key] = {
                    "gpu_uuid": gpu_uuid,
                    "pid": process.name,
                    "process_name": self._process_name(process),
                    "device": target,
                }
        return sorted(
            clients.values(),
            key=lambda item: (
                item["gpu_uuid"],
                int(item["pid"]),
            ),
        )

    def _require_resolvable_targets(self, gpu_uuids: list[str]) -> None:
        """Refuse to scan when a target GPU has no device node of its own.

        ``_quiesce_scope`` already refuses a target it cannot map; the
        verification that gates the reset did not, and dropped such a target
        from the scan instead -- so a GPU whose minor number the driver no
        longer reports passed the only gate the reset has.

        Reads the same finder ``_device_clients`` scans with, so the map that
        answers "resolvable" is the map the scan actually walks.

        An empty ``gpu_uuids`` is not "no targets", it is "every GPU on this
        node" -- that is how ``_device_clients`` and the reset itself read it --
        so the local inventory becomes the target set and has to be just as
        resolvable. Without this, a whole-node verification passed on a node
        with one unmappable GPU, which is precisely the node a whole-node reset
        should not run on.
        """

        inventory = None if gpu_uuids else self._gpu_inventory()
        targets = set(gpu_uuids) if gpu_uuids else set(inventory or ())
        device_paths = self.gpu_device_path_finder()
        missing = sorted(targets - set(device_paths))
        if not missing:
            return
        raise RuntimeError(
            "cannot resolve device node for "
            + ", ".join(
                f"{gpu_uuid} ({cause})"
                for gpu_uuid, cause in self._unresolvable_causes(
                    missing, inventory
                ).items()
            )
        )

    def _unresolvable_causes(
        self,
        missing: list[str],
        inventory: list[str] | None,
    ) -> dict[str, str]:
        """Why each of these UUIDs has no device node, in the evidence's words.

        The three causes need three different operator actions -- a card that
        fell off the bus, a workflow carrying a UUID this node never had (a
        stale plan, or the wrong node), and a driver report two GPUs share a
        device node in -- and the refusal used to name none of them, leaving
        "cannot resolve device node for GPU-..." as the whole diagnosis of a
        blocked destructive step.

        Takes the whole list because the inventory it may need is one probe for
        all of them: asked per UUID, a step naming eight GPUs from a stale plan
        spent eight ``nvidia-smi`` calls (15 s timeout each) building one error
        message, on a node that is already in trouble.
        """

        reasons = self._device_path_reasons()
        causes = {
            gpu_uuid: reasons[gpu_uuid] for gpu_uuid in missing if reasons.get(gpu_uuid)
        }
        if len(causes) == len(missing):
            return {gpu_uuid: causes[gpu_uuid] for gpu_uuid in missing}
        if inventory is None:
            try:
                inventory = self._gpu_inventory()
            except (OSError, RuntimeError, subprocess.SubprocessError):
                # Only ever asked on the way to a refusal, so a probe that also
                # fails costs nothing but the more specific cause.
                inventory = None
        for gpu_uuid in missing:
            if gpu_uuid in causes:
                continue
            causes[gpu_uuid] = (
                "no GPU on this node reports this UUID"
                if inventory is not None and gpu_uuid not in inventory
                else "the driver reported no device node for it"
            )
        return {gpu_uuid: causes[gpu_uuid] for gpu_uuid in missing}

    def _persistent_device_clients(
        self, target: set[str]
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        """Split device holders into persistent and transient ones.

        A holder counts as persistent only when the *same* pid still
        holds the *same* device across every sample.  Without this,
        short lived GPU processes that respawn on a timer deadlock the
        remediation forever: quiesce sweeps the holder, the next
        ``VERIFY_NO_GPU_CLIENTS`` attempt observes a brand new pid, and
        the step keeps failing until the quiesce maintenance window
        expires -- so the GPU is never reset and the node stays
        quarantined.  Measured on a HyperPod node, 96 of 98 attempts
        failed this way, blocked by ``nvidia-persistenced`` (which these
        AMIs ship with no systemd unit, so ``systemctl stop`` cannot
        reach it) and by per-second ``nvidia-smi`` invocations.

        Sampling stays fail-closed: a holder that survives every sample
        still blocks the reset, and a single sample (the default when
        the deadline leaves no room) behaves exactly as before.
        """
        samples: list[dict[tuple[str, str, str], dict[str, str]]] = []
        for index in range(self.device_client_samples):
            if index:
                self.sleep(self.device_client_sample_interval_seconds)
            samples.append(
                {
                    (
                        item["gpu_uuid"],
                        item["pid"],
                        item.get("device", ""),
                    ): item
                    for item in self.device_client_finder(target)
                }
            )
        if not samples:
            return [], []
        shared = set(samples[0])
        for sample in samples[1:]:
            shared &= set(sample)
        persistent = [samples[-1][key] for key in sorted(shared)]
        transient = [
            item
            for sample in samples
            for key, item in sorted(sample.items())
            if key not in shared
        ]
        return persistent, transient

    def _verify_no_clients(
        self,
        gpu_uuids: list[str],
        *,
        include_device_clients: bool = True,
    ) -> dict[str, Any]:
        target = set(gpu_uuids)
        compute_clients = [
            item
            for item in self._compute_clients()
            if not target or item["gpu_uuid"] in target
        ]
        if compute_clients:
            raise RuntimeError(
                "GPU compute clients are still active: "
                + ", ".join(
                    f"{item['gpu_uuid']}:{item['pid']}" for item in compute_clients
                )
            )
        transient_device_clients: list[dict[str, str]] = []
        if include_device_clients:
            with self._device_path_cache_window():
                # A target the device map cannot resolve used to be filtered out
                # of the scan and the verification passed with a live holder on
                # it. Say so instead, the way ``_quiesce_scope`` does: the reset
                # that follows is destructive and this is its only gate.
                self._require_resolvable_targets(gpu_uuids)
                (
                    device_clients,
                    transient_device_clients,
                ) = self._persistent_device_clients(target)
            if device_clients:
                raise RuntimeError(
                    "GPU device clients are still active: "
                    + ", ".join(
                        f"{item['gpu_uuid']}:{item['pid']}:{item['process_name']}"
                        for item in device_clients
                    )
                )
        return {
            "verified_no_gpu_clients": True,
            "gpu_uuids": gpu_uuids,
            "device_clients_checked": include_device_clients,
            "transient_device_clients": transient_device_clients,
        }
