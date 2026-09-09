"""The dedicated DCGM exporter's collect period, owned once for every launch path.

The exporter is started two ways -- our HyperPod DaemonSet
(``deploy/dataplane/hyperpod-dcgm-exporter.yaml``, rendered by
``regional_release_rendering``) and the node installer's docker-mode systemd
unit -- and the host collector checks its duty-cycle carry-over window against
the period it is told (the exporter-interval seconds in ``collector.env``).
Three places, one number: the DaemonSet's ``-c`` is rendered from this constant, the
node installer reconciler passes it into every install Job so ``--dcgm-exporter
existing`` can write the collector's seconds, and the installer's docker-mode
default is held equal to it by contract test. A second literal anywhere would
let a node's grading depend on how its exporter happened to start.

This module is a release input (``config/release-identity.yaml``,
``component_inputs.dcgm``): changing the period changes the rendered DaemonSet
and must re-apply it.
"""

from __future__ import annotations

#: DCGM's ``-c`` argument, in milliseconds. Kept at the collector's 15 s scrape
#: period: on DCGM's 30 s default every second sample repeats the previous one,
#: so a throttling episode cannot be graded.
DCGM_EXPORTER_COLLECT_INTERVAL_MS = 15000

#: What the checked-in DaemonSet carries in place of the period. The renderer
#: substitutes it, and a render that leaves it behind fails the release.
DCGM_EXPORTER_COLLECT_INTERVAL_PLACEHOLDER = (
    "REPLACE_WITH_DCGM_EXPORTER_COLLECT_INTERVAL_MS"
)
