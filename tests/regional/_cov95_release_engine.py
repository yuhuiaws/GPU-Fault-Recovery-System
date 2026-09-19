from __future__ import annotations

import base64
import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from gpu_fault_release import rollout
from tests.regional._cov95_release_support import OLD_IMAGE, RecordingRunner
from tests.regional._release_orchestrator_support import config_file


class EngineRelease(rollout.RegionalRelease):
    """Real phase and CPU restore logic with recorded transport effects."""

    def __init__(self, root: Path) -> None:
        self.effects: list[tuple[str, Any]] = []
        self.saves: list[dict[str, Any]] = []
        self.maps: list[dict[str, Any]] = []
        self.rbac_manifests: list[str] = []
        self.environments: list[dict[str, str]] = []
        self.old_artifact = b"example previous component"
        self.registry_restored = False
        self.registry_staged = False
        self.aurora_drift = False
        self.idle = True
        self.previous: dict[str, Any] = {
            "release_id": "previous",
            "runtime_image": OLD_IMAGE,
            "node_installer_image": OLD_IMAGE,
            "aurora_refresh": None,
            "cpu_wheel": "previous-wheel",
            "runtime_profile_version": "previous-profile",
            "metadata": {
                "required-agent-artifact-sha256": "a" * 64,
                "required-agent-config-digest": "b" * 64,
                "required-regional-executor-artifact-sha256": "c" * 64,
            },
            "secret_backups": {},
            "cpu_role_config_maps": None,
        }
        config = replace(rollout.ReleaseConfig.load(config_file(root)), clusters=())
        super().__init__(config, RecordingRunner(self.command))

    def command(self, arguments: list[str], kwargs: dict[str, Any]) -> str:
        if "get" in arguments and "configmap" in arguments:
            name = arguments[arguments.index("configmap") + 1]
            if name == "previous-wheel":
                data = (
                    {
                        self.config.wheel.name: base64.b64encode(
                            self.old_artifact
                        ).decode()
                    }
                    if self.old_artifact
                    else {}
                )
                return json.dumps({"binaryData": data})
        if "apply" in arguments and "input_text" in kwargs:
            if "kind: RoleBinding" in kwargs["input_text"]:
                # The release-metadata read grant every CPU apply re-applies.
                self.rbac_manifests.append(kwargs["input_text"])
                self.effects.append(("release-metadata-rbac", None))
                return ""
            value = json.loads(kwargs["input_text"])
            if value["kind"] != "ConfigMap":
                raise AssertionError("unexpected restore object kind")
            self.maps.append(value)
            self.effects.append(("configmap", value["metadata"]["name"]))
            return ""
        if arguments[0] == "bash" and arguments[1].endswith(
            (
                "/render-control-plane-role-split.sh",
                "/apply-control-plane-role-split.sh",
            )
        ):
            self.environments.append(dict(kwargs["env"]))
            self.effects.append(("roles", Path(arguments[1]).name))
            return ""
        raise AssertionError(f"unexpected modeled command: {arguments[:4]}")

    def _ensure_contexts(self) -> None:
        self.effects.append(("contexts", None))

    def _require_cpu_secrets(self, **kwargs: Any) -> None:
        self.effects.append(("cpu-secrets", kwargs))

    def _remote_commands_are_idle(self) -> bool:
        return self.idle

    def _require_no_inflight_installs(self, **kwargs: Any) -> dict[str, Any]:
        self.effects.append(("inflight", kwargs))
        return {"blocked": 0}

    def _refresh_aurora_credentials(self, **kwargs: Any) -> None:
        self.effects.append(("refresh", kwargs))

    def _aurora_refresh_drift(self) -> bool:
        return self.aurora_drift

    def _apply_aurora_refresh(self) -> None:
        self.effects.append(("apply-refresher", None))

    def _capture_previous(self, **_kwargs: Any) -> dict[str, Any]:
        return copy.deepcopy(self.previous)

    def _backup_release_secrets(self) -> dict[str, Any]:
        self.effects.append(("backup", None))
        return {}

    def _load_state(self) -> dict[str, Any]:
        return copy.deepcopy(self.state)

    def _save_state(self, phase: str, **updates: Any) -> None:
        self.state.update(phase=phase, **updates)
        self.saves.append(copy.deepcopy(self.state))

    def _validate_resume_checkpoint(self, **_kwargs: Any) -> None:
        self.effects.append(("resume-checkpoint", None))

    def _restore_registry_backup(self) -> bool:
        self.effects.append(("restore-registry", None))
        return self.registry_restored

    def _publish_restored_registry(self) -> None:
        self.effects.append(("publish-restored", None))

    def _restore_secret(self, *_args: Any, **kwargs: Any) -> None:
        self.effects.append(("restore-secret", kwargs))

    def _validate_rollback(self, _previous: Any, **kwargs: Any) -> None:
        self.effects.append(("verify-rollback", kwargs))

    def _delete_release_secret_backups(self, _previous: Any) -> None:
        self.effects.append(("delete-backups", None))

    def _fleet_command(self, operation: str, _payload: Any) -> dict[str, Any]:
        self.effects.append(("fleet", operation))
        return {"terminalized": []}

    def _upload_release(self, _diff: Any) -> None:
        self.effects.append(("upload", None))

    def _stage_registry(self) -> bool:
        self.effects.append(("stage-registry", None))
        return self.registry_staged

    def _publish_staged_registry(self) -> None:
        self.effects.append(("publish-staged", None))

    def _commit_registry_update(self) -> None:
        self.effects.append(("commit-registry", None))

    def _ensure_profile_transition_safe(self, version: str) -> None:
        self.effects.append(("profile-safe", version))

    def _capture_active_agent_node_sets(self) -> dict[str, Any]:
        return {}

    def _wait_candidate_cpu_agent_heartbeats(self, *_args: Any, **kwargs: Any) -> None:
        self.effects.append(("heartbeat", kwargs))

    def _validate_release_quick(self, _plan: Any) -> None:
        self.effects.append(("verify", None))

    def _ensure_schema(self) -> None:
        self.effects.append(("schema", None))

    def _apply_control_plane_observability(self) -> None:
        self.effects.append(("observability", None))

    def _apply_dataplane_expected_rules(self) -> None:
        self.effects.append(("expected-rules", None))
