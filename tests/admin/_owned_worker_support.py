"""Deadline budgets for tests that expire a task while an owned worker is blocked.

These tests enter a short ``deadline_scope`` and prove that the supervised
worker chain is reaped when it expires: the isolated HTTP worker inside a
framing read, or the credential helper and the ``credential_process`` it
started that never answer. The deadline is absolute from scope entry by
contract -- the supervisor receives it as a monotonic instant at spawn, so a
slow start can never extend it -- while the chain that has to be *blocked*
when it expires is two to four interpreters started in series (supervisor,
worker, the fixture's ``aws`` shim, its provider). Idle, the credential chain
writes its start record about 0.4 s after scope entry and the HTTP worker
connects after about 0.2 s. Under the release gate's load (32 xdist workers
and 16 Postgres shards, host load ~16) the credential chain took longer than
its 1 s budget: the deadline expired before the provider existed and the test
failed with "the owned credential process never started" (deploy gate,
2026-10-12) although nothing in the product had changed. A sub-second budget
measures the host, not the product. The budget below is an order of magnitude
above the idle start so the chain is blocked when the deadline fires on any
gate host, and every elapsed bound is taken from the deadline rather than from
scope entry, so what the tests prove about prompt reaping is unchanged.
"""

from __future__ import annotations

# Long enough for the slowest owned chain (four interpreters) to be blocked
# when the deadline fires under release-gate load; a test takes this long.
BLOCKED_WORKER_BUDGET_SECONDS = 5.0
# The supervisor's 20 ms poll, SIGTERM delivery to the tree, both completion
# reports and the parent's 100 ms communicate() step; idle this is about 0.15 s.
CLEANUP_SLACK_SECONDS = 2.0
