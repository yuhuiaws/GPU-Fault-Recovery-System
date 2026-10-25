"""BMC sensor alerts through ``ipmitool sensor`` (GF-GENERAL-LIVE-015)."""

from __future__ import annotations

import os
import shutil
import subprocess
from datetime import datetime
from typing import Any, Callable, NamedTuple

from gpu_fault.collectors.host.observation import HostObservationError
from gpu_fault.host_health import HostMetricSample

#: ``ipmitool sensor`` prints ``name | reading | units | status | lnr | lcr |
#: lnc | unc | ucr | unr``. The status is the fourth column -- never the units,
#: which read ``cr`` for a current probe -- ``na`` is a sensor with no reading,
#: and a discrete sensor prints its state word (``0x0180``) there.
_IPMI_ALERT_STATUSES = frozenset({"cr", "critical", "nr", "non-recoverable"})
_IPMI_UNAVAILABLE = "na"
_IPMI_THRESHOLD_COLUMNS = ("lnr", "lcr", "lnc", "unc", "ucr", "unr")
#: The device nodes ipmitool's OpenIPMI interface tries, in its order
#: (``lib/plugins/open/open.c``).
IPMI_DEVICE_NODES = ("/dev/ipmi0", "/dev/ipmi/0", "/dev/ipmidev/0")
#: What ipmitool prints when none of them exists: ``Could not open device at
#: %s or %s or %s: %s`` with ``strerror(ENOENT)``. Matched together, lowercase.
_IPMI_NO_DEVICE = ("could not open device", "no such file or directory")
#: How many alerting sensors the count sample names in full; a BMC whose
#: whole power domain trips lists hundreds, and the count carries the rest.
_BMC_NAMED_SENSORS = 32


class _BmcSensorAlert(NamedTuple):
    """One ``cr``/``nr`` row of ``ipmitool sensor``, as printed."""

    name: str
    status: str
    reading: str
    units: str
    thresholds: tuple[tuple[str, str], ...]

    def describe(self) -> str:
        """``<name>=<reading> <units> [<status>; <threshold> <value>, ...]``."""

        text = f"{self.name}={self.reading}"
        if self.units:
            text += f" {self.units}"
        band = ", ".join(f"{column} {value}" for column, value in self.thresholds)
        return f"{text} [{self.status}" + (f"; {band}]" if band else "]")


class HostBmcMixin:
    # Attributes supplied by the composed concrete implementation.
    ipmi_device_nodes: tuple[str, ...]
    runner: Callable[..., Any]

    def _bmc(self, _: datetime) -> list[HostMetricSample]:
        """BMC sensor alerts, read through ``ipmitool sensor``.

        ``bmc_critical_sensor`` is the number of distinct sensors whose status
        column reads ``cr``/``critical``/``nr``/``non-recoverable`` (one
        finding, CRITICAL QUARANTINE, GF-GENERAL-LIVE-015). The count alone
        told the operator "2": the sensors' identity is carried on the sample
        as labels -- ``sensors`` (names, table order, each once) and
        ``sensor_alerts`` (name, reading, units, status and the threshold
        columns ipmitool printed) -- which the grader puts into the finding's
        reason and details, so the incident reasons, the marker and the
        QUARANTINE step's hardware escalation mail name them. A healthy table
        is the plain ``0`` an older control plane already reads.

        What the BMC could not say is UNKNOWN, never a zero: a non-zero exit,
        a hung or unrunnable ipmitool, a table without a four-column row and
        a table whose every status is ``na`` (a BMC that answered with no
        reading at all) are a ``_bmc`` collection error with no sample. BMC
        the node cannot provide is NOT_SUPPORTED -- ``bmc_status_unsupported``
        on every tick and no error, like SMART's: ``TOOL_MISSING`` without
        ipmitool, ``NO_DEVICE`` when ipmitool reports that no IPMI device node
        exists and none does (EC2). That used to be a collection error every
        tick, a permanent capability gap holding every host validation of the
        node forever. The query is still made every tick, so the gap is read
        from the tool's answer and a BMC that appears is seen at once.
        """

        if not shutil.which("ipmitool"):
            return [self._bmc_unsupported("TOOL_MISSING")]
        try:
            completed = self.runner(
                ["ipmitool", "sensor"],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            raise HostObservationError(
                f"BMC sensor query failed: {type(exc).__name__}: {exc}", []
            ) from exc
        if completed.returncode != 0:
            output = f"{completed.stderr or ''} {completed.stdout or ''}".strip()
            if self._ipmi_device_missing(output):
                return [self._bmc_unsupported("NO_DEVICE")]
            raise HostObservationError(
                f"BMC sensor query failed: exit status {completed.returncode}: "
                f"{output[:200]}",
                [],
            )
        observed = 0
        unavailable = 0
        alerts: dict[str, _BmcSensorAlert] = {}
        for line in (completed.stdout or "").splitlines():
            columns = [item.strip() for item in line.split("|")]
            if len(columns) < 4:
                continue
            observed += 1
            status = columns[3].lower()
            if status == _IPMI_UNAVAILABLE:
                unavailable += 1
                continue
            if status not in _IPMI_ALERT_STATUSES:
                continue
            thresholds = tuple(
                (column, value)
                for column, value in zip(_IPMI_THRESHOLD_COLUMNS, columns[4:10])
                if value and value.lower() != _IPMI_UNAVAILABLE
            )
            alerts.setdefault(
                columns[0],
                _BmcSensorAlert(columns[0], status, columns[1], columns[2], thresholds),
            )
        if not observed:
            raise HostObservationError(
                "BMC sensor query returned no valid sensor rows", []
            )
        if unavailable == observed:
            raise HostObservationError(
                "BMC sensor query returned no sensor reading: the status of all "
                f"{observed} rows is na",
                [],
            )
        return [self._bmc_critical_sensors(tuple(alerts.values()))]

    def _ipmi_device_missing(self, output: str) -> bool:
        """ipmitool said no IPMI device node could be opened, and none exists.

        Both facts are required: the text alone could be a wrapper or an odd
        build, a missing node alone says nothing about why the tool failed,
        and a node that exists while the tool cannot open it is a BMC the
        collector fails to read, not a host without one. Anything short of
        both stays a collection error.
        """

        text = output.lower()
        return all(marker in text for marker in _IPMI_NO_DEVICE) and not any(
            os.path.exists(node) for node in self.ipmi_device_nodes
        )

    @staticmethod
    def _bmc_critical_sensors(alerts: tuple[_BmcSensorAlert, ...]) -> HostMetricSample:
        """The count of alerting sensors, naming them when there are any."""

        labels: dict[str, str] = {}
        if alerts:
            named = alerts[:_BMC_NAMED_SENSORS]
            unnamed = len(alerts) - len(named)
            names = [alert.name for alert in named]
            details = [alert.describe() for alert in named]
            if unnamed:
                names.append(f"+{unnamed} more")
                details.append(f"+{unnamed} more")
            labels = {"sensors": ", ".join(names), "sensor_alerts": "; ".join(details)}
        return HostMetricSample(
            name="bmc_critical_sensor",
            value=len(alerts),
            unit="sensors",
            labels=labels,
        )

    @staticmethod
    def _bmc_unsupported(reason: str) -> HostMetricSample:
        """The marker for BMC this node cannot provide.

        ``TOOL_MISSING`` is a node without ipmitool, ``NO_DEVICE`` one whose
        ipmitool finds no IPMI device node. Not a rule input and not a
        collection error: a capability gap is neither a healthy sensor table
        nor an unknown one.
        """

        return HostMetricSample(
            name="bmc_status_unsupported", value=1, labels={"reason": reason}
        )
