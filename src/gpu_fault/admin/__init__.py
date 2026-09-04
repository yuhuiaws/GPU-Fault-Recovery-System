"""The administrator tool: bootstrap, cluster membership, release and cleanup.

These modules run on an operator's machine, not in the control plane. They are
the only part of the package allowed to create or destroy AWS infrastructure, so
the boundary is worth being able to see: everything under here is measured by the
``administrator-tools`` coverage floor and is excluded from the runtime shards'
coverage, and nothing outside imports it.

Deliberately empty of re-exports. Importing ``gpu_fault.admin`` must not drag in
forty-odd modules -- and boto-shaped work with them -- just because a caller
wanted one command.
"""

from __future__ import annotations
