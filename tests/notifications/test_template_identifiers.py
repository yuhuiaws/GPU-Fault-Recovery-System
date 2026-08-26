from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTIFICATIONS = ROOT / "src/gpu_fault/notifications"
CONCRETE_NODE = re.compile(
    r"\b(?:hyperpod-i-[0-9a-f]{17}|node-[a-z](?:\b|[-0-9])|worker-[0-9]+)\b",
    re.IGNORECASE,
)


def test_production_notification_templates_do_not_embed_concrete_node_names() -> None:
    findings = []
    for path in sorted(NOTIFICATIONS.glob("*.py")):
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if CONCRETE_NODE.search(line):
                findings.append(f"{path.name}:{line_number}: {line.strip()}")

    assert findings == [], (
        "production notification templates embed concrete node names:\n"
        + "\n".join(findings)
    )
