from gpu_fault.regional_compatibility import RegionalExecutorCompatibilityPolicy


def test_executor_compatibility_checks_protocol_and_artifact() -> None:
    required = "a" * 64
    compatible = "b" * 64
    policy = RegionalExecutorCompatibilityPolicy.from_mapping(
        {
            "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_PROTOCOL_VERSION": "2",
            "GPU_FAULT_COMPATIBLE_REGIONAL_EXECUTOR_PROTOCOL_VERSIONS": "1",
            "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_ARTIFACT_SHA256": required,
            "GPU_FAULT_COMPATIBLE_REGIONAL_EXECUTOR_ARTIFACT_SHA256S": compatible,
            "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_COMPATIBILITY_DIGEST": (required),
            "GPU_FAULT_COMPATIBLE_REGIONAL_EXECUTOR_COMPATIBILITY_DIGESTS": (
                compatible
            ),
        }
    )

    assert policy.rejection_reason(2, required, required) is None
    assert policy.rejection_reason(1, compatible, compatible) is None
    assert "protocol version mismatch" in (
        policy.rejection_reason(3, required, required) or ""
    )
    assert "artifact mismatch" in (policy.rejection_reason(2, "c" * 64, required) or "")
    assert "compatibility digest mismatch" in (
        policy.rejection_reason(2, required, "c" * 64) or ""
    )


def test_legacy_unpinned_executor_artifact_remains_compatible() -> None:
    policy = RegionalExecutorCompatibilityPolicy.from_mapping(
        {"GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_PROTOCOL_VERSION": "2"}
    )

    assert policy.rejection_reason(2, None) is None


def test_empty_executor_compatibility_digest_falls_back_to_artifact() -> None:
    artifact = "a" * 64
    policy = RegionalExecutorCompatibilityPolicy.from_mapping(
        {
            "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_ARTIFACT_SHA256": artifact,
            "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_COMPATIBILITY_DIGEST": "",
        }
    )

    assert policy.required_compatibility_digest == artifact
