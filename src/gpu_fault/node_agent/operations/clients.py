from __future__ import annotations

import os
import re
import subprocess
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence
from xml.etree import ElementTree as ET

from gpu_fault.node_agent.common import (
    COLD_GPU_INIT_BOUND_SECONDS,
    DEFAULT_NODE_GPU_COUNT,
    nvidia_gpu_count,
)
from gpu_fault.node_agent.gpu_activity import GpuActivityGate
from gpu_fault.node_agent.operations.holders import (
    CGROUP_PATH_LIMIT,
    GPU_CLIENT_HOLDER_LIMIT,
    kubernetes_attribution,
    process_cgroup_paths,
)
from gpu_fault.node_agent.protocol import (
    NodeActionCommand,
)
from gpu_fault.node_agent.query_clients import query_only_device_client

PCI_BUS_ID_PATTERN = re.compile(
    r"^[0-9A-Fa-f]{4,8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]$"
)

#: A read-only ``nvidia-smi`` query (compute apps, ``-q -x``, the index map,
#: the inventory) on a driver that keeps its GPUs initialised.
NVIDIA_SMI_QUERY_TIMEOUT_SECONDS = 15
#: What each GPU of the node adds to that bound: with nothing holding them,
#: each query initialises every GPU of the node first (``common``). The flat
#: 15 s killed the first compute-apps query after a busy reset unloaded the
#: display stack -- eight cold H200 -- and the final ownership boundary refused
#: the reset OWNERSHIP_FINAL_SAFETY_CHANGED (live, gpu-eks-cluster p5en,
#: GF-REGIONAL-DESTR-010). A driver that does not answer within the larger
#: bound still fails the query, and the boundary still refuses.
NVIDIA_SMI_COLD_INIT_SECONDS_PER_GPU = COLD_GPU_INIT_BOUND_SECONDS


def nvidia_smi_query_timeout(gpu_count: int | None) -> int:
    """15 s plus 5 s per GPU: 55 s on an eight-GPU node, cold or warm."""

    return NVIDIA_SMI_QUERY_TIMEOUT_SECONDS + NVIDIA_SMI_COLD_INIT_SECONDS_PER_GPU * (
        gpu_count or DEFAULT_NODE_GPU_COUNT
    )


class GpuClientsActive(RuntimeError):
    """``VERIFY_NO_GPU_CLIENTS`` found a client; ``action_details`` names it.

    The message keeps the ``clients are still active`` wording the executor's
    retry fold matches on; the details carry every holder it saw -- pid,
    process name, device, cgroup paths and the Kubernetes container/pod the
    cgroup names -- so the step record of each attempt says who held the GPU
    (live 2026-10-05, gpu-eks-cluster: seven minutes of retries and a record
    that said only "quiesce maintenance window expired").
    """

    def __init__(self, message: str, *, action_details: dict[str, Any]) -> None:
        super().__init__(message)
        self.action_details = action_details


class ClientOperationsMixin:
    # Attributes supplied by the composed concrete implementation.
    device_client_samples: Any
    gpu_device_path_finder: Callable[..., Any]

    _run_checked: Callable[..., Any]
    device_client_finder: Callable[..., Any]
    device_client_sample_interval_seconds: Any
    gpu_activity: GpuActivityGate
    proc_root: Any
    sleep: Callable[..., Any]

    # Owned by ``_device_path_cache_window``. Per thread, because the action
    # pool runs four commands on one executor instance and an instance counter
    # is a read-modify-write nobody holds a lock for. A plain assignment, not a
    # slot the concrete executor has to build.
    _device_path_state = threading.local()

    def _nvidia_smi_query_timeout(self) -> int:
        """The bound of one read-only query on this node (its driver's GPU list)."""

        return nvidia_smi_query_timeout(nvidia_gpu_count(self.proc_root))

    def _compute_clients(self) -> list[dict[str, str]]:
        completed = self._run_checked(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name",
                "--format=csv,noheader,nounits",
            ],
            timeout=self._nvidia_smi_query_timeout(),
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
        and no table survives the block: across commands a driver reload really
        can renumber the devices. The report's per-GPU attributes are evidence,
        not a table: the block starts without them and leaves its own read's.
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
        state.attributes = None
        state.caching = True
        try:
            yield
        finally:
            state.entry = outer_entry
            state.caching = outer_caching
            state.reasons = outer_reasons
            # Not the attributes: the reset reads the targets' persistence right
            # after the final ownership window, as that window's own report
            # gave it -- none if it read none, never an older report.

    def _gpu_device_paths(self) -> dict[str, str]:
        state = self._device_path_state
        entry = getattr(state, "entry", None)
        # The table belongs to one executor: the id keeps a window opened by one
        # from answering another's question.
        if entry is not None and entry[0] == id(self):
            cached: dict[str, str] = entry[1]
            return cached
        paths, reasons, attributes = self._minor_number_device_map()
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
        state.attributes = (id(self), attributes)
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

    def _gpu_attributes(self) -> dict[str, dict[str, str]]:
        """uuid -> PCI bus id and persistence mode, from the last report parsed.

        Only the executor's own resolver parses that report: with an injected
        finder (tests) there is none, and a table some earlier command left
        behind on this thread must not answer for this one.
        """

        entry = getattr(self._device_path_state, "attributes", None)
        if entry is None or entry[0] != id(self):
            return {}
        if self.gpu_device_path_finder != self._gpu_device_paths:
            return {}
        attributes: dict[str, dict[str, str]] = entry[1]
        return attributes

    def _target_pci_bus_ids(self, gpu_uuids: Sequence[str]) -> list[str] | None:
        """The targets' PCI bus ids, for the quiesce's per-GPU persistence release.

        From the same ``nvidia-smi -q -x`` report the device nodes were just
        resolved from -- never an index or a minor number, which need not match
        the bus order. ``None`` unless every target has one: the release then
        leaves the targets to the device sweep instead of guessing.
        """

        attributes = self._gpu_attributes()
        bus_ids = [attributes.get(uuid, {}).get("pci_bus_id", "") for uuid in gpu_uuids]
        if not bus_ids or not all(bus_ids):
            return None
        return bus_ids

    def _target_persistence_states(self, gpu_uuids: Sequence[str]) -> dict[str, Any]:
        """Each target's persistence mode and a count of the others', by mode.

        Read from the report the device map came from, so it costs nothing:
        the reset records it for each physical call as the final ownership
        window read it. The other GPUs say which regime the window runs in:
        with persistence off on them every ``nvidia-smi`` re-initialises them
        all, and one node of a parallel reset falls behind the other.
        """

        attributes = self._gpu_attributes()
        if not attributes:
            return {}
        targets = list(gpu_uuids) or sorted(attributes)
        others: dict[str, int] = {}
        for uuid in sorted(set(attributes) - set(targets)):
            mode = attributes[uuid].get("persistence_mode", "")
            others[mode] = others.get(mode, 0) + 1
        return {
            "target_persistence": [
                {"gpu_uuid": uuid, **attributes[uuid]}
                for uuid in targets
                if uuid in attributes
            ][:GPU_CLIENT_HOLDER_LIMIT],
            "non_target_persistence_modes": others,
        }

    def _minor_number_device_map(
        self,
    ) -> tuple[dict[str, str] | None, dict[str, str], dict[str, dict[str, str]]]:
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

        The third answer is each GPU's PCI bus id and persistence mode, from
        the same report: what the quiesce releases persistence by, and what the
        reset receipt records about the window.
        """

        try:
            completed = self._run_checked(
                ["nvidia-smi", "-q", "-x"], timeout=self._nvidia_smi_query_timeout()
            )
        except (OSError, RuntimeError, subprocess.SubprocessError):
            return None, {}, {}
        try:
            root = ET.fromstring(completed.stdout or "")
        except ET.ParseError:
            return None, {}, {}
        paths: dict[str, str] = {}
        reasons: dict[str, str] = {}
        attributes: dict[str, dict[str, str]] = {}
        claimed: set[str] = set()
        duplicated: set[str] = set()
        for gpu in root.findall(".//gpu"):
            gpu_uuid = (gpu.findtext("uuid") or "").strip()
            minor = (gpu.findtext("minor_number") or "").strip()
            if not gpu_uuid:
                continue
            bus_id = (gpu.findtext("pci/pci_bus_id") or gpu.get("id") or "").strip()
            if PCI_BUS_ID_PATTERN.fullmatch(bus_id):
                mode = (gpu.findtext("persistence_mode") or "").strip()[:32]
                attributes[gpu_uuid] = {"pci_bus_id": bus_id, "persistence_mode": mode}
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
            return None, {}, attributes
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
        return paths, reasons, attributes

    def _index_device_paths(self) -> dict[str, str]:
        """The pre-minor-number mapping, kept for drivers that omit it."""

        completed = self._run_checked(
            [
                "nvidia-smi",
                "--query-gpu=uuid,index",
                "--format=csv,noheader,nounits",
            ],
            timeout=self._nvidia_smi_query_timeout(),
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
            timeout=self._nvidia_smi_query_timeout(),
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

    def _client_holder_details(
        self, kind: str, clients: list[dict[str, str]]
    ) -> dict[str, Any]:
        """The verification's holders with their cgroup and Kubernetes identity.

        The same attribution the busy-reset evidence uses (``holders``): the
        cgroup paths of ``/proc/<pid>`` and the container id / pod uid a
        ``kubepods`` path names. Bounded, no command lines.
        """

        holders: list[dict[str, Any]] = []
        for item in clients[:GPU_CLIENT_HOLDER_LIMIT]:
            cgroups = process_cgroup_paths(Path(self.proc_root) / str(item["pid"]))
            holders.append(
                {
                    **item,
                    "cgroup_paths": cgroups[:CGROUP_PATH_LIMIT],
                    **kubernetes_attribution(cgroups),
                }
            )
        return {
            "gpu_client_kind": kind,
            "gpu_client_holders": holders,
            "gpu_client_holders_total": len(clients),
        }

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
        except OSError as exc:
            raise OSError("host process inventory is unavailable") from exc
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

    def _refuse_agent_gpu_activities(self) -> None:
        """Fail closed while one of the Agent's own GPU operations is running.

        Its ``dcgmi``/``nvvs``/``nvidia-smi`` children come and go between the
        samples, so the sampled scan below calls them transient and passes in a
        gap between two captures -- and the reset that follows meets the next
        one. The registry (``gpu_activity``) sees the operation for its whole
        run; the reset waits, as for any other client.
        """

        activities = self.gpu_activity.in_flight()
        if activities:
            raise GpuClientsActive(
                "GPU agent clients are still active: "
                + ", ".join(
                    f"{item['operation']}:{item['command_id']}"
                    for item in activities[:8]
                ),
                action_details={
                    "gpu_client_kind": "agent",
                    "gpu_client_holders": activities[:GPU_CLIENT_HOLDER_LIMIT],
                    "gpu_client_holders_total": len(activities),
                },
            )

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
        # A query that outlived every sample (``nvidia-smi`` with persistence
        # mode off initialises the driver on each call) is still not a GPU
        # user; it is reported with the transient holders, classified.
        blocking = [
            item
            for item in persistent
            if not query_only_device_client(self.proc_root, item)
        ]
        transient.extend(
            {**item, "classification": "query-only"}
            for item in persistent
            if item not in blocking
        )
        return blocking, transient

    def _verify_no_clients(
        self,
        gpu_uuids: list[str],
        *,
        include_device_clients: bool = True,
        include_agent_activities: bool = False,
    ) -> dict[str, Any]:
        target = set(gpu_uuids)
        if include_agent_activities:
            self._refuse_agent_gpu_activities()
        compute_clients = [
            item
            for item in self._compute_clients()
            if not target or item["gpu_uuid"] in target
        ]
        if compute_clients:
            raise GpuClientsActive(
                "GPU compute clients are still active: "
                + ", ".join(
                    f"{item['gpu_uuid']}:{item['pid']}" for item in compute_clients
                ),
                action_details=self._client_holder_details("compute", compute_clients),
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
                raise GpuClientsActive(
                    "GPU device clients are still active: "
                    + ", ".join(
                        f"{item['gpu_uuid']}:{item['pid']}:{item['process_name']}"
                        for item in device_clients
                    ),
                    action_details={
                        **self._client_holder_details("device", device_clients),
                        "transient_device_clients": transient_device_clients,
                    },
                )
        return {
            "verified_no_gpu_clients": True,
            "gpu_uuids": gpu_uuids,
            "device_clients_checked": include_device_clients,
            "transient_device_clients": transient_device_clients,
            "query_only_device_clients": [
                item
                for item in transient_device_clients
                if item.get("classification") == "query-only"
            ],
        }
