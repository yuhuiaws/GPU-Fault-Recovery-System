"""The release engine batches same-kind reads and caches the history document.

Every ``kubectl`` call from the deploy host costs about 1.2 s (exec-plugin auth
plus the API round trip), so a release's duration is its kubectl call count. A
CONTROL_PLANE_ONLY ``config`` release issued ~250 calls, of which three shapes
were pure repetition:

* the 18 role ConfigMaps of the ``previous`` snapshot were read one ``get``
  each (~22 s) -- now one ``kubectl get configmap <a> <b> ... --ignore-not-found``,
  with a missing name still failing the capture and naming every absent one;
* a batched read inside a ``read_snapshot`` files each item under its per-name
  key too, so a later ``config_map_data`` for one of them is a cache hit;
* every checkpoint re-read ``gpu-fault-release-history`` (an existence probe
  plus the read) before appending -- now the process reads it once, lazily, and
  appends in memory; a failed apply leaves the cache untouched so the next
  checkpoint reads again rather than trusting a line the cluster never saw.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault_release import regional_release_history as HISTORY
from gpu_fault_release import regional_release_state as STATE
from gpu_fault_release.regional_release_config import ReleaseError

NAMESPACE = "gpu-fault-system"
ROLE_MAPS = (
    "gpu-fault-api-ha-config-core",
    "gpu-fault-control-worker-config-core",
    "gpu-fault-telemetry-spool-worker-config-core",
)


def _config_map(name: str, data: dict[str, str]) -> dict[str, Any]:
    return {"kind": "ConfigMap", "metadata": {"name": name}, "data": data}


def _list(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {"kind": "List", "apiVersion": "v1", "items": items}


class _Runner:
    """Serves ``kubectl get configmap <names...>`` from a dict, as kubectl would."""

    dry_run = False

    def __init__(self, config_maps: dict[str, dict[str, str]]) -> None:
        self.config_maps = config_maps
        self.calls: list[list[str]] = []

    def run(self, arguments: list[str], **_kwargs: Any) -> str:
        self.calls.append(list(arguments))
        names = [
            word
            for word in arguments[arguments.index("configmap") + 1 :]
            if not word.startswith("-") and word != "json"
        ]
        items = [
            _config_map(name, self.config_maps[name])
            for name in names
            if name in self.config_maps
        ]
        if len(names) == 1 and items:
            return json.dumps(items[0])
        return json.dumps(_list(items)) if items else ""


class _Release:
    """A release double whose ``_get_json`` is the engine's own cached reader."""

    config = SimpleNamespace(namespace=NAMESPACE)

    def __init__(self, runner: _Runner) -> None:
        self.runner = runner

    @staticmethod
    def _cpu(*args: str) -> list[str]:
        return ["kubectl", "--kubeconfig", "/secure/cpu.kubeconfig", *args]

    def _get_json(self, args: list[str]) -> dict[str, Any]:
        return STATE.get_json(self, args)


def test_role_config_maps_are_read_in_one_kubectl_call() -> None:
    runner = _Runner({name: {"GPU_FAULT_ROLE": name} for name in ROLE_MAPS})
    release = _Release(runner)

    data = STATE.config_maps_data(release, list(reversed(ROLE_MAPS)))

    assert len(runner.calls) == 1, (
        f"one kubectl call for every name, got {runner.calls}"
    )
    call = runner.calls[0]
    assert call[call.index("get") :] == [
        "get",
        "configmap",
        *ROLE_MAPS,
        STATE.IGNORE_NOT_FOUND,
        "-o",
        "json",
    ], f"names are requested sorted, in one get, tolerating absence: {call}"
    assert data == {name: {"GPU_FAULT_ROLE": name} for name in ROLE_MAPS}, (
        "every ConfigMap's data must come back keyed by its name"
    )


def test_a_missing_role_config_map_still_fails_the_capture_naming_it() -> None:
    runner = _Runner({ROLE_MAPS[0]: {"A": "1"}})
    release = _Release(runner)

    with pytest.raises(ReleaseError) as excinfo:
        STATE.config_maps_data(release, ROLE_MAPS)

    message = str(excinfo.value)
    assert ROLE_MAPS[1] in message and ROLE_MAPS[2] in message, (
        f"every absent ConfigMap must be named: {message}"
    )
    assert ROLE_MAPS[0] not in message, f"a found ConfigMap is not missing: {message}"
    assert len(runner.calls) == 1, "absence is learned from the one batched answer"


def test_no_names_means_no_kubectl_call() -> None:
    runner = _Runner({})

    assert STATE.config_maps_data(_Release(runner), []) == {}, (
        "nothing requested, nothing returned"
    )
    assert runner.calls == [], "an empty batch must not reach kubectl"


def test_a_batched_read_inside_a_snapshot_serves_later_per_name_reads() -> None:
    runner = _Runner({name: {"GPU_FAULT_ROLE": name} for name in ROLE_MAPS})
    release = _Release(runner)

    with STATE.read_snapshot(release):
        batched = STATE.config_maps_data(release, ROLE_MAPS)
        single = STATE.config_map_data(release, ROLE_MAPS[1])
        again = STATE.config_maps_data(release, ROLE_MAPS)

    assert single == batched[ROLE_MAPS[1]], "the per-name read must see the same data"
    assert again == batched, "the same batch asked twice is the same answer"
    assert len(runner.calls) == 1, (
        f"the batch fills the per-name cache; no further kubectl call: {runner.calls}"
    )


def test_cpu_role_config_maps_uses_the_batched_reader() -> None:
    batches: list[list[str]] = []
    deployments = {
        deployment: {
            "spec": {
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "name": "app",
                                "envFrom": [
                                    {
                                        "configMapRef": {
                                            "name": f"{deployment}-config-core"
                                        }
                                    },
                                    {"configMapRef": {"name": "gpu-fault-unrelated"}},
                                ],
                            }
                        ]
                    }
                }
            }
        }
        for deployment in STATE.inventory.CPU_RUNTIME_DEPLOYMENTS
    }
    release = SimpleNamespace(
        config=SimpleNamespace(namespace=NAMESPACE),
        _config_maps_data=lambda names: (
            batches.append(list(names)) or {name: {"K": "v"} for name in names}
        ),
    )

    snapshot = STATE.cpu_role_config_maps(release, deployments)

    assert len(batches) == 1, f"all role ConfigMaps in one batch, got {batches}"
    assert batches[0] == sorted(
        f"{deployment}-config-core"
        for deployment in STATE.inventory.CPU_RUNTIME_DEPLOYMENTS
    ), "only gpu-fault-*-config-* references are captured, sorted"
    assert set(snapshot) == set(batches[0]), "the snapshot holds what was read"


# --------------------------------------------------------------------------
# Release history: read once per process
# --------------------------------------------------------------------------


class _HistoryRunner:
    dry_run = False

    def __init__(self, *, fail_applies: int = 0) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.fail_applies = fail_applies

    def run(self, arguments: list[str], **kwargs: Any) -> str:
        if arguments[-3:] == ["apply", "-f", "-"] and self.fail_applies:
            self.fail_applies -= 1
            raise RuntimeError("apiserver unavailable")
        # Only what the cluster accepted is recorded, as the cluster would.
        self.calls.append((list(arguments), kwargs))
        return ""

    def probe(self, _arguments: list[str], **_kwargs: Any) -> bool:
        raise AssertionError("the history read must not pay a separate probe")


def _history_release(runner: _HistoryRunner, existing: list[dict]) -> SimpleNamespace:
    reads: list[list[str]] = []

    def get_json(arguments: list[str]) -> dict[str, Any]:
        reads.append(list(arguments))
        return {
            "data": {
                HISTORY.HISTORY_KEY: "\n".join(json.dumps(item) for item in existing)
            }
        }

    release = SimpleNamespace(
        runner=runner,
        release_id="rel-42",
        config=SimpleNamespace(namespace=NAMESPACE),
        state={"phase": "cpu-staged", "execution_plan": None},
        _cpu=lambda *arguments: list(arguments),
        _get_json=get_json,
    )
    release.reads = reads
    return release


def _applied_histories(runner: _HistoryRunner) -> list[list[dict]]:
    documents = [
        json.loads(kwargs["input_text"])
        for arguments, kwargs in runner.calls
        if arguments[-3:] == ["apply", "-f", "-"]
    ]
    return [
        [
            json.loads(line)
            for line in document["data"][HISTORY.HISTORY_KEY].splitlines()
            if line
        ]
        for document in documents
    ]


@pytest.fixture(autouse=True)
def _no_aws_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        HISTORY, "resolve_operator_identity", lambda *, fallback: fallback
    )
    monkeypatch.delenv(HISTORY.HISTORY_DIR_ENV, raising=False)


def test_history_is_read_once_across_checkpoints_and_appended_in_memory() -> None:
    runner = _HistoryRunner()
    release = _history_release(runner, [{"release_id": "rel-41", "phase": "complete"}])

    HISTORY.record_release_history(release, phase="preflight", state_text="{}")
    HISTORY.record_release_history(release, phase="cpu-staged", state_text="{}")

    assert len(release.reads) == 1, (
        f"the history ConfigMap is read once per process, got {release.reads}"
    )
    assert release.reads[0][-1] == "--ignore-not-found", (
        "the one read tolerates a site without history instead of probing first"
    )
    assert len(runner.calls) == 2, (
        f"one apply per checkpoint, nothing else: {runner.calls}"
    )
    first, second = _applied_histories(runner)
    assert [item["phase"] for item in first] == ["complete", "preflight"], (
        "the first checkpoint appends to what the cluster held"
    )
    assert [item["phase"] for item in second] == [
        "complete",
        "preflight",
        "cpu-staged",
    ], "the second checkpoint appends to the in-memory copy, not a re-read"


def test_history_cache_holds_the_truncated_document() -> None:
    runner = _HistoryRunner()
    existing = [
        {"release_id": f"rel-{index}", "phase": "complete"}
        for index in range(HISTORY.HISTORY_MAX_ENTRIES)
    ]
    release = _history_release(runner, existing)

    HISTORY.record_release_history(release, phase="preflight", state_text="{}")
    HISTORY.record_release_history(release, phase="cpu-staged", state_text="{}")

    first, second = _applied_histories(runner)
    assert len(first) == len(second) == HISTORY.HISTORY_MAX_ENTRIES, (
        "the bound holds on every checkpoint"
    )
    assert second[0]["release_id"] == "rel-2", "the two oldest entries were dropped"
    assert [item["phase"] for item in second[-2:]] == ["preflight", "cpu-staged"], (
        "both new entries survive at the end"
    )


def test_a_failed_apply_does_not_poison_the_cache(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _HistoryRunner(fail_applies=1)
    release = _history_release(runner, [])

    HISTORY.record_release_history(release, phase="preflight", state_text="{}")
    assert "was not recorded" in capsys.readouterr().err, "the failure is announced"
    HISTORY.record_release_history(release, phase="cpu-staged", state_text="{}")

    assert len(release.reads) == 2, (
        "after a failed apply the next checkpoint reads the cluster again"
    )
    (applied,) = _applied_histories(runner)
    assert [item["phase"] for item in applied] == ["cpu-staged"], (
        "the line the cluster never accepted is not replayed from memory"
    )


def test_dry_run_never_reads_the_history() -> None:
    runner = _HistoryRunner()
    runner.dry_run = True
    release = _history_release(runner, [])

    HISTORY.record_release_history(release, phase="plan", state_text="{}")

    assert release.reads == [] and runner.calls == [], (
        "a dry run neither reads nor writes the history"
    )
