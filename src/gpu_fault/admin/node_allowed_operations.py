"""The Node Agent operations a default node install grants, as the deploy pins them.

``bootstrap_common.compute_agent_config_digest`` derives the Agent config digest the
control plane requires (its required-Agent-config-digest setting) with this list,
and that digest's payload carries the sorted allowed operations, so the list must be
exactly the default ``NODE_ALLOWED_OPERATIONS`` of
``deploy/node/install-gpu-fault-collector.sh``
(``test_deploy_digest_operations_mirror_the_installer_default`` pins it).

The rendered control-plane manifests carry that digest, so
``config/release-identity.yaml`` lists this module as a renderer and ``cpu`` input:
changing the list changes the release id. Without that, a corrected list rendered
new manifests under the same release id, and a paused upgrade could neither resume
(the approved manifest pin no longer matched) nor be superseded
(``--supersede-failed-transaction`` refuses the release that failed). It lives
apart from ``bootstrap_common`` so ordinary deploy-host changes do not move the
release id.
"""

from __future__ import annotations

NODE_ALLOWED_OPERATIONS = (
    "COLLECT_HUNG_TRIAGE,COLLECT_HUNG_SUMMARY,COLLECT_DIAGNOSTIC_BUNDLE,"
    "RUN_DCGM_DIAGNOSTIC,QUIESCE_GPU_SERVICES,"
    "VERIFY_NO_GPU_CLIENTS,TRIGGER_HEALTH_SNAPSHOT,"
    "RESET_GPU,RECOVER_DRAIN_P2P,RESET_ALL_GPUS_NVSWITCHES,"
    "RESTORE_GPU_SERVICES,RESTART_FABRIC_MANAGER,"
    "REMEDIATE_EFA_DRIVER,COLLECTOR_OUTBOX_MAINTENANCE"
)
