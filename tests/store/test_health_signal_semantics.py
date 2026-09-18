from __future__ import annotations

from contextlib import closing

import pytest

from tests.store._health_signal_semantics import (
    LegacyMemoryStore,
    LegacySqliteStore,
    SemanticSignalContract,
)


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        yield LegacyMemoryStore()
    else:
        with closing(LegacySqliteStore(str(tmp_path / "semantic.db"))) as instance:
            yield instance


class TestHealthSignalSemantics(SemanticSignalContract):
    pass
