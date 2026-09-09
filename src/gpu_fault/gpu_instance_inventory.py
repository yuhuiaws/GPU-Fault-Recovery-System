"""The GPU instance types the data plane supports, with their device counts.

One table, three readers: the node installer reconciler sizes each install
Job's expected GPU and EFA counts from it, the regional release renders the
DCGM exporter DaemonSet's node affinity from it, and the legacy single-cluster
deploy pins the same list. It lives on its own because it is a *release
input*: ``config/release-identity.yaml`` lists this module under
``component_inputs.dcgm`` so that a new supported type re-applies the exporter
DaemonSet. While the table sat inside ``node_installer_reconciler.py`` the
whole reconciler was that input, and every unrelated reconciler fix rolled
every exporter Pod on every GPU cluster.
"""

from __future__ import annotations

#: EC2 instance type (without HyperPod's ``ml.`` prefix) -> (GPUs, EFA devices).
GPU_INSTANCE_INVENTORY: dict[str, tuple[int, int]] = {
    "p5.4xlarge": (1, 1),
    "p5.48xlarge": (8, 32),
    "p5e.48xlarge": (8, 32),
    "p5en.48xlarge": (8, 16),
    "p6-b200.48xlarge": (8, 8),
    "p6-b300.48xlarge": (8, 16),
}


def gpu_instance_inventory(instance_type: str) -> tuple[int, int]:
    """The (GPU, EFA device) counts of ``instance_type``.

    Accepts the type with or without the ``ml.`` prefix, because HyperPod
    labels its nodes ``ml.<type>`` while a self-managed node pool carries the
    bare EC2 type. An unknown type is a :class:`ValueError`: the installer
    must not guess a GPU count, and the caller decides whether that is fatal.
    """

    normalized = instance_type.removeprefix("ml.")
    try:
        return GPU_INSTANCE_INVENTORY[normalized]
    except KeyError as error:
        raise ValueError(
            f"unsupported GPU instance type: {instance_type or 'UNKNOWN'}"
        ) from error
