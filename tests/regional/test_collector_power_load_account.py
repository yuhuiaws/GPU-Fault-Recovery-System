"""The restore receipt carries the GPU load unit's own account.

Live (plain EKS, COLLECT-002): the load unit ran for the whole window while the
loaded GPU stayed idle and nothing in the evidence said why. The restore now
records what the unit itself said before it was stopped -- still active or
not, its systemd result and exit status, its journal tail -- read-only.
"""

from __future__ import annotations

from scripts.e2e.regional.probes import collector_node_probe as probe
from tests.regional.test_collector_power_safety import PowerHost
from tests.regional.test_collector_power_safety import power_host as power_host_fixture

power_host = power_host_fixture


def test_the_restore_receipt_carries_the_load_units_own_account(
    power_host: PowerHost,
) -> None:
    """Live (plain EKS): the load unit ran for the whole window while the GPU
    stayed idle and nothing in the evidence said why. The restore now records
    what the unit itself said before it was stopped -- still active or not, its
    systemd result and exit status, its journal tail -- read-only."""

    host = power_host
    probe.throttle_gpu(host.arguments())
    assert host.units()[2] in host.active

    probe.restore_gpu_power_limit(host.arguments())

    receipt = host.emitted[-1]
    assert receipt["cleanup_verified"] is True and receipt["load_stopped"] is True
    assert receipt["load_unit_active_at_restore"] is True
    assert receipt["load_journal_tail"] == host.journal_lines
    assert "load_unit_result" in receipt and "load_exit_status" in receipt
    journal_reads = [call for call in host.calls if call[0] == "journalctl"]
    assert len(journal_reads) == 1
    assert host.calls.index(journal_reads[0]) < host.calls.index(
        ["systemctl", "stop", host.units()[2]]
    ), "the account is read before the stop, while the unit still has it"
