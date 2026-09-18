from __future__ import annotations

import hashlib
import json

import pytest

from scripts.verify_release_images import verify_images
from tests.test_release_image_container_cleanup import FakeDocker


@pytest.mark.parametrize("invalid_plugins", [False, True])
def test_executor_image_validates_installed_plugins_before_acceptance(
    invalid_plugins: bool,
) -> None:
    inventory = json.dumps({"files": {"dependency.whl": {"sha256": "a" * 64}}})
    inventory_digest = hashlib.sha256(inventory.encode()).hexdigest()
    descriptor = {
        "schema_version": 3,
        "images": {
            "control_plane": {"tag": "control-image", "dockerfile_sha256": "d" * 64},
            "executor": {"tag": "executor-image", "dockerfile_sha256": "d" * 64},
            "node_dependencies": {
                "tag": "node-image",
                "wheelhouse_sha256": inventory_digest,
            },
        },
        "components": {
            "control_plane": {"module_digest": "c" * 64},
            "executor": {"module_digest": "e" * 64},
        },
    }

    def command_result(reference: str, arguments: list[str]) -> tuple[int, str]:
        output = ""
        code = 0
        if arguments[-1] == "validate-plugins":
            assert reference == "executor-image"
            assert arguments[-6:] == [
                "/opt/gpu-fault/executor/bin/python",
                "-I",
                "-B",
                "-m",
                "gpu_fault.collectors_cli",
                "validate-plugins",
            ]
            code = 1 if invalid_plugins else 0
        elif "module_digest; print(module_digest())" in arguments[-1]:
            output = "c" * 64 if reference == "control-image" else "e" * 64
        elif arguments[-2:] == ["/bin/cat", "/opt/gpu-fault/wheelhouse/inventory.json"]:
            output = inventory
        elif "sha256sum" in arguments[-1]:
            output = f"{'a' * 64}  dependency.whl\n{inventory_digest}  inventory.json\n"
        return code, output

    run = FakeDocker(command_result)

    if invalid_plugins:
        with pytest.raises(ValueError, match="executor image content check failed"):
            verify_images(descriptor, runner=run)
        assert not any(item["reference"] == "node-image" for item in run.creations), (
            "a failed Executor plugin gate must stop before Node image validation"
        )
    else:
        verify_images(descriptor, runner=run)
    assert sum(item["command"][-1] == "validate-plugins" for item in run.creations) == 1
    assert run.containers == {}, (
        "image content verification left owned containers behind"
    )
