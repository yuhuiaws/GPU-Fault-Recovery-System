from __future__ import annotations

import logging
from datetime import datetime, timezone

from gpu_fault.models import (
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStepSpec,
    lifetime_exceeded,
    resolved_step_indexes,
)
from gpu_fault.operation_registry import SHARED_DAG_OPERATIONS
from gpu_fault.orchestration.arbitration import (
    RecoveryArbiter,
    fault_scope_covered,
)

LOGGER = logging.getLogger(__name__)


class DagBrancher:
    def __init__(self, arbiter: RecoveryArbiter) -> None:
        self.arbiter = arbiter

    def node_branch_step_indexes(
        self, workflow: WorkflowRequest, node_id: str
    ) -> list[int]:
        branch_ids = {
            step.branch_id
            for step in workflow.official_steps
            if step.branch_id not in {None, "shared", "join"}
            and step.operation not in self.arbiter.WORKLOAD_SCOPED_OPERATIONS
            # Branch identity, not execution scope: a step widened onto
            # another node still belongs to the branch it was created for.
            and node_id in (step.branch_node_ids or step.node_ids)
        }
        if not branch_ids:
            return []
        return [
            index
            for index, step in enumerate(workflow.official_steps)
            if step.branch_id in branch_ids
            and step.operation not in self.arbiter.WORKLOAD_SCOPED_OPERATIONS
        ]

    def step_indexes_recovery_rank(
        self, workflow: WorkflowRequest, indexes: list[int]
    ) -> int:
        return self.arbiter.recovery_rank(
            {workflow.official_steps[index].operation for index in indexes}
        )

    def step_indexes_merge_intent(
        self, workflow: WorkflowRequest, indexes: list[int]
    ) -> frozenset[WorkflowOperation]:
        return frozenset(
            (
                workflow.official_steps[index].operation
                for index in indexes
                if workflow.official_steps[index].operation
                in self.arbiter.MERGE_INTENT_OPERATIONS
            )
        )

    def node_branch_covers_fault_scope(
        self,
        workflow: WorkflowRequest,
        indexes: list[int],
        node_id: str,
        gpu_uuids: set[str],
        *,
        candidate: WorkflowRequest | None = None,
    ) -> bool:
        return fault_scope_covered(
            workflow, node_id, gpu_uuids, indexes=indexes, candidate=candidate
        )

    def branch_has_started(self, workflow: WorkflowRequest, indexes: list[int]) -> bool:
        branch_indexes = set(indexes)
        return bool(
            branch_indexes.intersection(workflow.completed_step_indexes)
            or branch_indexes.intersection(workflow.superseded_step_indexes)
            or any(
                (
                    execution.step_index in branch_indexes
                    for execution in workflow.step_executions
                )
            )
        )

    def dag_join_accepts_new_branch(
        self, workflow: WorkflowRequest, *, now: datetime | None = None
    ) -> bool:
        if lifetime_exceeded(workflow, now):
            # Past its lifetime a workflow takes no new node branch (F-N1);
            # disposition() records such events on the incident instead.
            return False
        restart_index = next(
            (
                index
                for index, step in enumerate(workflow.official_steps)
                if step.operation is WorkflowOperation.RESTART_WORKLOAD
            ),
            None,
        )
        if restart_index is None:
            return False
        return (
            restart_index not in workflow.completed_step_indexes
            and restart_index not in workflow.superseded_step_indexes
            and (
                not any(
                    (
                        item.step_index == restart_index
                        and item.operation is WorkflowOperation.RESTART_WORKLOAD
                        for item in workflow.step_executions
                    )
                )
            )
        )

    def requires_live_pre_stop_capture(self, step: WorkflowStepSpec) -> bool:
        return bool(
            step.operation is WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE
            and step.parameters.get("capture_process_state")
        )

    def is_pre_stop_step(self, step: WorkflowStepSpec) -> bool:
        return step.operation in {
            WorkflowOperation.FREEZE_EVIDENCE,
            WorkflowOperation.MARK_UNSCHEDULABLE,
        } or self.requires_live_pre_stop_capture(step)

    def _after_stop_step(self, step: WorkflowStepSpec) -> WorkflowStepSpec:
        if not self.requires_live_pre_stop_capture(step):
            return step
        return step.model_copy(
            update={
                "parameters": {
                    **step.parameters,
                    "capture_process_state": False,
                    "running_process_evidence_unavailable": True,
                    "running_process_evidence_reason": (
                        "shared STOP_WORKLOADS already completed"
                    ),
                }
            }
        )

    _SHARED_OPERATIONS = SHARED_DAG_OPERATIONS

    def _normalize_existing(
        self, existing: WorkflowRequest
    ) -> tuple[list[WorkflowStepSpec], int | None, bool]:
        steps = []
        for index, step in enumerate(existing.official_steps):
            dependencies = (
                list(step.depends_on_step_indexes)
                if existing.dag_enabled
                else []
                if index == 0
                else [index - 1]
            )
            branch_id = step.branch_id or (
                "shared"
                if step.operation in self._SHARED_OPERATIONS
                else "branch:initial"
            )
            steps.append(
                step.model_copy(
                    update={
                        "depends_on_step_indexes": dependencies,
                        "branch_id": branch_id,
                    }
                )
            )
        stop_index = next(
            (
                index
                for index, step in enumerate(steps)
                if step.operation is WorkflowOperation.STOP_WORKLOADS
            ),
            None,
        )
        stop_is_open = (
            stop_index is not None
            and stop_index not in existing.completed_step_indexes
            and stop_index not in existing.superseded_step_indexes
            and not any(
                execution.step_index == stop_index
                and execution.operation is WorkflowOperation.STOP_WORKLOADS
                for execution in existing.step_executions
            )
        )
        return steps, stop_index, stop_is_open

    def _candidate_branch(
        self,
        candidate: WorkflowRequest,
        predecessor_step_index: int | None,
        revision: int,
    ) -> tuple[
        int | None,
        list[tuple[int, WorkflowStepSpec]],
        list[WorkflowStepSpec],
        list[str],
        str,
        bool,
    ]:
        stop_index = next(
            (
                index
                for index, step in enumerate(candidate.official_steps)
                if step.operation is WorkflowOperation.STOP_WORKLOADS
            ),
            None,
        )
        indexed = [
            (index, step)
            for index, step in enumerate(candidate.official_steps)
            if step.operation not in self._SHARED_OPERATIONS
        ]
        steps = [step for _, step in indexed]
        nodes = sorted({node for step in steps for node in step.node_ids})
        branch_id = "branch:" + ",".join(nodes)
        if predecessor_step_index is not None:
            branch_id += f":successor:{revision + 1}"
        operations = {step.operation for step in candidate.official_steps}
        # A candidate that isolates the node or hands it to support ends the
        # node's part in the job; the branch it replaces owes no release and
        # the job restart is retired only where this node was all of it
        # (F-C5).
        terminal = (
            bool(
                operations
                & {WorkflowOperation.QUARANTINE, WorkflowOperation.ESCALATE_SUPPORT}
            )
            and WorkflowOperation.RESTART_WORKLOAD not in operations
        )
        return (
            stop_index,
            indexed,
            steps,
            nodes,
            branch_id,
            terminal,
        )

    @staticmethod
    def _branch_semantics(
        steps: list[WorkflowStepSpec],
    ) -> list[dict]:
        return [
            step.model_copy(
                update={
                    "branch_id": None,
                    "branch_node_ids": [],
                    "depends_on_step_indexes": [],
                }
            ).model_dump(mode="json")
            for step in steps
        ]

    def _branch_already_present(
        self,
        existing: WorkflowRequest,
        steps: list[WorkflowStepSpec],
        candidate_steps: list[WorkflowStepSpec],
        candidate_nodes: list[str],
    ) -> bool:
        candidate = self._branch_semantics(candidate_steps)
        completed = set(existing.completed_step_indexes)
        superseded = set(existing.superseded_step_indexes)
        branch_ids = {
            step.branch_id
            for index, step in enumerate(steps)
            if step.branch_id not in {None, "shared", "join"}
            and index not in superseded
            and set(step.node_ids) == set(candidate_nodes)
        }
        # A branch whose every live step already completed has run; an
        # identical new fault needs a new branch, not a no-op (F-C3).
        branch_ids = {
            branch_id
            for branch_id in branch_ids
            if any(
                index not in completed
                for index, step in enumerate(steps)
                if step.branch_id == branch_id and index not in superseded
            )
        }
        return any(
            self._branch_semantics(
                [
                    step
                    for index, step in enumerate(steps)
                    if step.branch_id == branch_id
                    and index not in existing.superseded_step_indexes
                ]
            )
            == candidate
            for branch_id in branch_ids
        )

    @staticmethod
    def _terminal_superseded(
        existing: WorkflowRequest,
        steps: list[WorkflowStepSpec],
        branch_nodes: list[str],
        replaced: frozenset[int],
        terminal: bool,
    ) -> set[int]:
        superseded = set(replaced)
        if not terminal:
            return superseded
        nodes = set(branch_nodes)
        for index, step in enumerate(steps):
            if step.operation not in {
                WorkflowOperation.RESTART_WORKLOAD,
                WorkflowOperation.RESTORE_SCHEDULING,
            }:
                continue
            # Only a step the isolated node wholly owns goes with it: a
            # release shared with another node must still uncordon that
            # node, and a job restart the other nodes still need must still
            # run (F-C5). A step that already ran is history either way.
            if not set(step.node_ids) <= nodes:
                continue
            if index in existing.completed_step_indexes or any(
                execution.step_index == index and execution.operation is step.operation
                for execution in existing.step_executions
            ):
                continue
            superseded.add(index)
        return superseded

    @staticmethod
    def _append_chain(
        steps: list[WorkflowStepSpec],
        values: list[WorkflowStepSpec],
        dependency: int | None,
        branch_id: str,
        appended: list[int],
    ) -> int | None:
        previous = dependency
        base = len(steps)
        for branch_step in values:
            index = len(steps)
            # Keep the candidate's own edges (its indexes are local to
            # ``values``; remap them onto the appended positions) alongside the
            # chain edge, instead of discarding them (F-C2, P1-66E).
            local = {
                base + dep
                for dep in branch_step.depends_on_step_indexes
                if 0 <= dep < len(values) and base + dep < index
            }
            if previous is not None:
                local.add(previous)
            steps.append(
                branch_step.model_copy(
                    update={
                        "depends_on_step_indexes": sorted(local),
                        "branch_id": branch_id,
                        # Identity is fixed at creation; later widening of
                        # ``node_ids`` does not move the step to another
                        # node's branch (F-B6 (2)).
                        "branch_node_ids": (
                            list(branch_step.branch_node_ids)
                            or list(branch_step.node_ids)
                        ),
                    }
                )
            )
            appended.append(index)
            previous = index
        return previous

    def _attach_candidate_branch(
        self,
        steps: list[WorkflowStepSpec],
        indexed: list[tuple[int, WorkflowStepSpec]],
        branch_steps: list[WorkflowStepSpec],
        branch_id: str,
        candidate_stop_index: int | None,
        stop_index: int | None,
        stop_is_open: bool,
        predecessor: int | None,
        appended: list[int],
    ) -> int | None:
        if predecessor is not None:
            return self._append_chain(
                steps,
                branch_steps,
                predecessor,
                branch_id,
                appended,
            )
        if stop_is_open:
            prefix = [
                step
                for original_index, step in indexed
                if (
                    candidate_stop_index is None
                    or original_index < candidate_stop_index
                )
                and self.is_pre_stop_step(step)
            ]
            prefix_ids = {id(step) for step in prefix}
            # The suffix chains behind the shared STOP either way, so a step
            # that needs live processes is told they are gone in this branch
            # too, not only when the STOP has already run (F-C4).
            suffix = [
                self._after_stop_step(step)
                for _, step in indexed
                if id(step) not in prefix_ids
            ]
        else:
            prefix = []
            suffix = [self._after_stop_step(step) for step in branch_steps]
        prefix_tail = self._append_chain(steps, prefix, None, branch_id, appended)
        if prefix_tail is not None and stop_is_open and stop_index is not None:
            stop = steps[stop_index]
            steps[stop_index] = stop.model_copy(
                update={
                    "depends_on_step_indexes": sorted(
                        set(stop.depends_on_step_indexes) | {prefix_tail}
                    )
                }
            )
        dependency = stop_index if stop_index is not None else prefix_tail
        suffix_tail = self._append_chain(steps, suffix, dependency, branch_id, appended)
        # Index 0 is a valid tail; ``or`` treated it as absent (P3-65H).
        return suffix_tail if suffix_tail is not None else prefix_tail

    @staticmethod
    def _reconnect_join(
        steps: list[WorkflowStepSpec],
        branch_tail: int | None,
        superseded: set[int],
    ) -> None:
        for index, step in enumerate(steps):
            if step.operation is not WorkflowOperation.RESTART_WORKLOAD:
                continue
            dependencies = set(step.depends_on_step_indexes) - superseded
            steps[index] = step.model_copy(
                update={
                    "depends_on_step_indexes": sorted(
                        dependencies
                        | ({branch_tail} if branch_tail is not None else set())
                    ),
                    "branch_id": "join",
                }
            )

    def append_parallel_job_branch(
        self,
        existing: WorkflowRequest,
        candidate: WorkflowRequest,
        *,
        predecessor_step_index: int | None = None,
        replaced_step_indexes: frozenset[int] = frozenset(),
    ) -> WorkflowRequest:
        steps, stop_index, stop_is_open = self._normalize_existing(existing)
        (
            candidate_stop,
            indexed,
            branch_steps,
            branch_nodes,
            branch_id,
            terminal,
        ) = self._candidate_branch(
            candidate,
            predecessor_step_index,
            existing.dag_revision,
        )
        if (
            predecessor_step_index is None
            and not replaced_step_indexes
            and self._branch_already_present(
                existing,
                steps,
                branch_steps,
                branch_nodes,
            )
        ):
            return existing
        if any(step.branch_id == branch_id for step in steps):
            # A second, different branch on the same node set must not share
            # the first one's identity: every consumer that groups steps by
            # ``branch_id`` would otherwise merge the two (F-C3). The first
            # occurrence keeps the plain spelling.
            branch_id = f"{branch_id}:{existing.dag_revision + 1}"
        superseded = self._terminal_superseded(
            existing,
            steps,
            branch_nodes,
            replaced_step_indexes,
            terminal,
        )
        appended: list[int] = []
        tail = self._attach_candidate_branch(
            steps,
            indexed,
            branch_steps,
            branch_id,
            candidate_stop,
            stop_index,
            stop_is_open,
            predecessor_step_index,
            appended,
        )
        if appended:
            self._reconnect_join(steps, tail, superseded)
        rank = self.arbiter.workflow_recovery_rank
        return existing.model_copy(
            update={
                "dag_enabled": True,
                "dag_revision": existing.dag_revision + 1,
                "superseded_step_indexes": sorted(
                    set(existing.superseded_step_indexes) | superseded
                ),
                "official_action": (
                    candidate.official_action
                    if rank(candidate) > rank(existing)
                    else existing.official_action
                ),
                "official_steps": steps,
                "updated_at": datetime.now(timezone.utc),
            }
        )

    def widen_parallel_job_branch(
        self,
        existing: WorkflowRequest,
        candidate: WorkflowRequest,
        node_id: str,
        gpu_uuids: set[str],
    ) -> WorkflowRequest:
        indexes = set(self.node_branch_step_indexes(existing, node_id))
        # Steps that finished, were superseded or already have a command in
        # flight keep the scope they ran with; only pending steps widen
        # (F-B6). Rewriting history would make the ledger claim a wider
        # reset than the agent performed.
        untouchable = set(resolved_step_indexes(existing)) | {
            execution.step_index for execution in existing.step_executions
        }
        candidate_by_operation = {
            step.operation: step
            for step in candidate.official_steps
            if node_id in step.node_ids
            and step.operation not in self.arbiter.WORKLOAD_SCOPED_OPERATIONS
        }
        steps = []
        for index, step in enumerate(existing.official_steps):
            if index not in indexes or index in untouchable:
                steps.append(step)
                continue
            matching = candidate_by_operation.get(step.operation)
            merged_gpus = sorted(
                set(step.gpu_uuids)
                | set(matching.gpu_uuids if matching else [])
                | gpu_uuids
            )
            parameters = dict(step.parameters)
            raw = parameters.get("gpu_uuids_by_node")
            if isinstance(raw, dict):
                mapping = {
                    str(key): list(values)
                    for key, values in raw.items()
                    if isinstance(values, list)
                }
                widened_gpus = set(mapping.get(node_id, [])) | gpu_uuids
                if widened_gpus:
                    mapping[node_id] = sorted(widened_gpus)
                parameters["gpu_uuids_by_node"] = mapping
            steps.append(
                step.model_copy(
                    update={
                        "gpu_uuids": merged_gpus,
                        "parameters": parameters,
                    }
                )
            )
        return existing.model_copy(
            update={
                "dag_revision": existing.dag_revision + 1,
                "official_steps": steps,
                "updated_at": datetime.now(timezone.utc),
            }
        )

    def replace_parallel_job_branch(
        self,
        existing: WorkflowRequest,
        candidate: WorkflowRequest,
        node_id: str,
    ) -> WorkflowRequest:
        indexes = self.node_branch_step_indexes(existing, node_id)
        if not indexes:
            return existing
        branch_set = set(indexes)
        first_index = min(indexes)
        external_dependencies = [
            dependency
            for dependency in existing.official_steps[
                first_index
            ].depends_on_step_indexes
            if dependency not in branch_set
        ]
        predecessor = external_dependencies[-1] if external_dependencies else None
        tails = {
            index
            for index in indexes
            if not any(
                (
                    index in step.depends_on_step_indexes
                    for step in existing.official_steps
                    if step.branch_id == existing.official_steps[index].branch_id
                )
            )
        }
        return self.append_parallel_job_branch(
            existing,
            candidate,
            predecessor_step_index=predecessor,
            replaced_step_indexes=frozenset(branch_set | tails),
        )

    def append_parallel_job_branch_successor(
        self,
        existing: WorkflowRequest,
        candidate: WorkflowRequest,
        node_id: str,
    ) -> WorkflowRequest:
        indexes = self.node_branch_step_indexes(existing, node_id)
        if not indexes and (not existing.dag_enabled):
            indexes = [
                index
                for index, step in enumerate(existing.official_steps)
                if step.operation
                not in {
                    WorkflowOperation.CHECKPOINT_WORKLOADS,
                    WorkflowOperation.STOP_WORKLOADS,
                    WorkflowOperation.RESTART_WORKLOAD,
                }
                and node_id in step.node_ids
            ]
        if not indexes:
            # The node has no branch to queue behind. Returning ``existing``
            # unchanged dropped the candidate on the floor (P0-55B); the
            # right outcome is a fresh parallel branch for that node.
            LOGGER.info(
                "no branch for node %s in workflow %s; appending %s as a new "
                "parallel branch",
                node_id,
                existing.request_id,
                candidate.request_id,
            )
            return self.append_parallel_job_branch(existing, candidate)
        tail = max(
            (
                index
                for index in indexes
                if not existing.dag_enabled
                or not any(
                    (
                        index in step.depends_on_step_indexes
                        for step in existing.official_steps
                        if step.branch_id == existing.official_steps[index].branch_id
                    )
                )
            )
        )
        return self.append_parallel_job_branch(
            existing, candidate, predecessor_step_index=tail
        )
