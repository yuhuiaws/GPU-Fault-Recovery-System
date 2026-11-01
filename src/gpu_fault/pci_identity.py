"""Whether two PCI addresses name one device, however each source spelled it.

One GPU reaches the control plane under several spellings: the kernel's NVRM
Xid line ``PCI:0000:b9:00`` (``nvidia_logs``: a 4-digit domain, no function),
the dcgm-exporter ``pci_bus_id`` label and nvidia-smi ``00000000:B9:00.0`` (an
8-digit domain, upper case, the function), the inventory collectors the same
lower-cased. A join of records from two sources compares the device, not the
string (batch 236: provider-event correlation used to intersect the raw
strings, so an H200 kernel XID and DCGM's report of the same GPU never met).

``pci_bdf_matches`` reads ``[domain:]bus:device[.function]`` in any case and
compares every part both sides carry by value; an absent domain or function
matches any. A string that is no such address matches nothing there, and
``same_pci_device`` lets it match only an identical string -- an unreadable
spelling never joins a record that an exact comparison would not have joined.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_PCI_BDF = re.compile(
    r"(?:(?P<domain>[0-9a-f]{4}|[0-9a-f]{8}):)?"
    r"(?P<bus>[0-9a-f]{2}):(?P<device>[0-9a-f]{2})(?:\.(?P<function>[0-7]))?"
)


def pci_bdf_matches(left: str, right: str) -> bool:
    first = _PCI_BDF.fullmatch(left.strip().lower())
    second = _PCI_BDF.fullmatch(right.strip().lower())
    if first is None or second is None:
        return False
    for name in ("domain", "bus", "device", "function"):
        a, b = first.group(name), second.group(name)
        if a is not None and b is not None and int(a, 16) != int(b, 16):
            return False
    return True


def same_pci_device(left: str, right: str) -> bool:
    """Identical strings, or two spellings of one device (module docstring)."""

    return left == right or pci_bdf_matches(left, right)


def pci_identity_overlaps(left: Iterable[str], right: Iterable[str]) -> bool:
    """Whether an address of ``left`` and one of ``right`` name one device."""

    others = list(right)
    return any(same_pci_device(first, second) for first in left for second in others)
