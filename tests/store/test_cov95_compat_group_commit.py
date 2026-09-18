from __future__ import annotations

from threading import Event

import pytest

from gpu_fault.store import NotFoundError
from tests.store._cov95_compat_group_commit import (
    commit_group_fixture as commit_group_fixture,
)
from tests.store._cov95_compat_support import (
    compat_store_fixture as compat_store_fixture,
)


class WriterLost(BaseException):
    pass


def test_group_commit_drains_bounded_batches_and_concurrent_arrivals(commit_group):
    group = commit_group
    initial = [group.entry(f"initial-{index}") for index in range(5)]
    late = [group.entry(f"late-{index}") for index in range(3)]
    group.queue.extend(initial)
    submitted = group.entry("submitted")
    first = True

    def flush(batch):
        nonlocal first
        group.persist(batch)
        if first:
            first = False
            with group.condition:
                group.queue.extend(late)

    group.submit(submitted, flush=flush, batch_size=2)
    assert [len(batch) for batch in group.batches] == [2, 2, 2, 2, 1]
    assert group.queue == []
    assert group.host.active is False
    for entry in [*initial, submitted, *late]:
        assert entry["event"].is_set(), (
            "every flushed caller must receive its completion event"
        )
        assert entry["result"].request_id == entry["request"].request_id
        assert group.store.get_processor_request(
            entry["request"].request_id
        ).body() == (entry["request"].body())


def test_waiter_promotes_after_the_leader_loses_its_flush(commit_group):
    group = commit_group
    leader_entered, permit_crash = group.gate(), group.gate()
    leader_entry, follower_entry = group.entry("leader"), group.entry("follower")

    def flush(batch):
        if batch[0] is leader_entry:
            leader_entered.set()
            assert permit_crash.wait(10), (
                "test never released the simulated leader crash"
            )
            raise WriterLost("unit writer disappeared before persistence")
        group.persist(batch)

    leader = group.start(leader_entry, flush=flush)
    assert leader_entered.wait(5), "leader never entered the controlled flush"
    follower = group.start(follower_entry, flush=flush)
    assert follower_entry["event"].waiting.wait(5), "follower never reached its wait"
    permit_crash.set()
    group.join(leader)
    group.join(follower)
    assert set(group.errors) == {"leader"}
    assert isinstance(group.errors["leader"], WriterLost), (
        "leader loss must propagate to its caller"
    )
    assert follower_entry["event"].is_set(), "promoted follower was not completed"
    assert follower_entry["result"].request_id == "follower"
    assert group.queue == []
    assert group.host.active is False
    with pytest.raises(NotFoundError):
        group.store.get_processor_request("leader")
    group.submit(group.entry("after-crash"))
    assert group.store.get_processor_request("after-crash").request_id == "after-crash"


def test_timeout_removes_only_its_own_queued_identity(commit_group):
    group = commit_group
    entry = group.entry("duplicate")
    twin = dict(entry)
    group.queue.append(twin)
    group.host.active = True
    with pytest.raises(TimeoutError, match="batch did not flush"):
        group.submit(entry, timeout=0)
    assert len(group.queue) == 1
    assert group.queue[0] is twin, (
        "a timed-out caller must not remove an equal peer entry"
    )
    assert entry["result"] is None
    with pytest.raises(NotFoundError):
        group.store.get_processor_request("duplicate")
    group.host.active = False
    group.submit(group.entry("later"))
    assert group.queue == []
    assert group.store.get_processor_request("duplicate").request_id == "duplicate"


def test_timeout_does_not_mean_an_inflight_batch_was_rolled_back(commit_group):
    group = commit_group
    first_entered, release_first = group.gate(), group.gate()
    second_entered, release_second = group.gate(), group.gate()
    first, second = group.entry("first"), group.entry("inflight")

    def flush(batch):
        if batch[0] is first:
            first_entered.set()
            assert release_first.wait(10), "first batch was not released"
        else:
            second_entered.set()
            assert release_second.wait(10), "inflight batch was not released"
        group.persist(batch)

    leader = group.start(first, flush=flush)
    assert first_entered.wait(5), "leader did not enter the first flush"
    follower = group.start(second, timeout=1)
    assert second["event"].waiting.wait(5), "follower did not enter its timeout window"
    release_first.set()
    assert second_entered.wait(5), "leader did not take the follower into a batch"
    group.join(follower)
    assert set(group.errors) == {"inflight"}
    assert isinstance(group.errors["inflight"], TimeoutError), (
        "only the waiting caller should time out"
    )
    assert group.queue == []
    with pytest.raises(NotFoundError):
        group.store.get_processor_request("inflight")
    release_second.set()
    group.join(leader)
    assert group.host.active is False
    assert second["event"].is_set(), (
        "the inflight flush may complete after its caller timed out"
    )
    assert group.store.get_processor_request("inflight").request_id == "inflight"


def test_completion_between_event_checks_is_not_reported_as_timeout(commit_group):
    group = commit_group
    sampled, resume = group.gate(), group.gate()

    class SampledEvent(Event):
        observed = False

        def is_set(self):
            observed = super().is_set()
            if not observed and not self.observed:
                self.observed = True
                sampled.set()
                assert resume.wait(10), "concurrent completion was not released"
            return observed

    entry = group.entry("racing", event=SampledEvent())
    group.host.active = True
    caller = group.start(entry, timeout=0)
    assert sampled.wait(5), "waiter did not sample the unfinished event"
    with group.condition:
        batch = list(group.queue)
        group.queue.clear()
    group.persist(batch)
    group.host.active = False
    resume.set()
    group.join(caller)
    assert group.errors == {}
    assert entry["result"].request_id == "racing"
    assert group.store.get_processor_request("racing").request_id == "racing"


def test_per_entry_failure_is_signaled_without_blocking_other_results(commit_group):
    group = commit_group
    failed, successful = group.entry("failed"), group.entry("successful")
    group.queue.append(failed)

    def flush(batch):
        for entry in batch:
            if entry is failed:
                entry["error"] = ValueError("unit item rejected")
                entry["event"].set()
            else:
                group.persist([entry])

    group.submit(successful, flush=flush, batch_size=2, first_wait=0)
    assert failed["event"].is_set(), "failed items must release their own waiters"
    assert isinstance(failed["error"], ValueError), "per-item error must be retained"
    assert failed["result"] is None
    assert successful["result"].request_id == "successful"
    assert group.queue == []
    assert group.host.active is False
    with pytest.raises(NotFoundError):
        group.store.get_processor_request("failed")
