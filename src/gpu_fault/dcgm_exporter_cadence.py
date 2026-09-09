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

The period is not free-standing: it is the dcgm collector's own scrape period
(the collector CLI's metrics-interval default, the installer's
``METRICS_INTERVAL``), because the collector's stale carry-over covers
``DUTY_CYCLE_STALE_CARRY_OVER_INTERVALS`` (8) of its own intervals and the
exporter must refresh at or below that bound (``collectors/gpu/dcgm.py``); on
DCGM's 30 s default every second sample repeats the previous one and a
throttling episode cannot be graded. A contract test pins the seconds below to
both defaults and the milliseconds to the reviewed value.

This module is a release input (``config/release-identity.yaml``,
``component_inputs.dcgm``): changing the period changes the rendered DaemonSet
and must re-apply it. Any edit here -- even to this docstring -- changes the
``dcgm`` component digest and re-applies the DaemonSet once, so the module
holds constants only.
"""

from __future__ import annotations

#: The dcgm collector's scrape period in seconds, which the exporter matches.
DCGM_COLLECTOR_INTERVAL_SECONDS = 15

#: DCGM's ``-c`` argument, in milliseconds: one collector interval.
DCGM_EXPORTER_COLLECT_INTERVAL_MS = DCGM_COLLECTOR_INTERVAL_SECONDS * 1000

#: What the checked-in DaemonSet carries in place of the period. The renderer
#: substitutes it, and a render that leaves it behind fails the release.
DCGM_EXPORTER_COLLECT_INTERVAL_PLACEHOLDER = (
    "REPLACE_WITH_DCGM_EXPORTER_COLLECT_INTERVAL_MS"
)
