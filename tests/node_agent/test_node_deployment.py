"""Compatibility re-export; the tests live in ``test_node_deployment_*``.

``test_node_deployment.py`` outgrew the 1500-line file limit and was split
into six thematic modules (rollout, units, wheel, verify, degraded_gpu,
ledger_drain) with the shared fixtures in ``_deployment_support``. This
module keeps the import path ``test_node_installer_drain_budget.py`` uses
until that import is retargeted at the support module; it defines no tests.
"""

from __future__ import annotations

from tests.node_agent._deployment_support import (
    NODE_SCRIPTS,
    ROOT,
    _ledger_drain_probe,
    _ledger_with_state,
    _write_stub,
)

__all__ = [
    "NODE_SCRIPTS",
    "ROOT",
    "_ledger_drain_probe",
    "_ledger_with_state",
    "_write_stub",
]
