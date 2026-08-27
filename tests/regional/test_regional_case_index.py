from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_regional_case_index_matches_its_sources() -> None:
    # 索引是生成物，手写一百多行必然漂移。事实源是
    # testcases/regional-execution-order.yaml（顺序）与
    # testcases/fault-scenarios.yaml（字段），改动任一方都必须重新生成。
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/build-regional-case-index.py"), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
