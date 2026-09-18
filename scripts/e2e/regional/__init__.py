"""Regional acceptance drivers and audit probes."""

from pathlib import Path
import sys

# Standalone drivers may be invoked outside the checkout. Keep their imported
# implementation paired with this source tree, not an unrelated installed wheel.
_SOURCE = str(Path(__file__).resolve().parents[3] / "src")
if _SOURCE not in sys.path:
    sys.path.insert(0, _SOURCE)
