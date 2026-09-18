"""HA-001 role checks against the current leaderless health and metric contract."""

from __future__ import annotations

import math
from typing import Any


def validate_role_health(
    snapshot: dict[str, Any],
    replicas: dict[str, int],
    *,
    ingress_app: str,
    worker_app: str,
) -> list[str]:
    errors = []
    for app, role, processor_role in (
        (ingress_app, "ingress", "inactive"),
        (worker_app, "worker", "active-consumer"),
    ):
        pods = snapshot.get(app, [])
        if not isinstance(pods, list):
            errors.append(f"{role} role snapshot is malformed")
            continue
        expected = replicas.get(app)
        if type(expected) is not int or expected < 1 or len(pods) != expected:
            errors.append(f"{role} role snapshot does not contain {expected} Pods")
        names: set[str] = set()
        for item in pods:
            if not isinstance(item, dict):
                errors.append(f"{role} Pod observation is malformed")
                continue
            name = item.get("name")
            if not isinstance(name, str) or not name or name in names:
                errors.append(f"{role} role snapshot repeats or omits Pod identity")
                continue
            names.add(name)
            health = item.get("health")
            if not isinstance(health, dict):
                errors.append(f"{name} healthz is missing or malformed")
                continue
            if health.get("service_role") != role:
                errors.append(f"{name} is not {role}")
            if health.get("processor_role") != processor_role:
                errors.append(f"{name} processor is not {processor_role}")
            consumer = item.get("processor_active_consumer")
            valid_count = (
                isinstance(consumer, (int, float))
                and not isinstance(consumer, bool)
                and math.isfinite(consumer)
                and consumer >= (0 if role == "ingress" else 1)
                and consumer % 1 == 0
            )
            if role == "ingress":
                if not valid_count or consumer != 0:
                    errors.append(f"{name} active-consumer metric is not zero")
            # The gauge is summed across uvicorn processes, not Pod replicas.
            elif not valid_count:
                errors.append(
                    f"{name} active-consumer metric is {consumer!r}, "
                    "expected >= 1 (one per uvicorn worker process)"
                )
            if health.get("processor_mode") != "active-active":
                errors.append(f"{name} is not in active-active processor mode")
            if health.get("leadership") is not None:
                errors.append(f"{name} reports leadership; expected null")
            # /healthz emits an empty string when no leadership lease is held.
            epoch = health.get("processor_epoch")
            if epoch is None:
                errors.append(
                    f"{name} healthz omits processor_epoch; "
                    "cannot prove leaderless active-active"
                )
            elif epoch != "":
                errors.append(
                    f"{name} holds processor leadership epoch {epoch!r}; "
                    "expected none (leaderless active-active)"
                )
    return errors
