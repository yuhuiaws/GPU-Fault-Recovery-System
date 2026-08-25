def pending_fault_scope_keys(items) -> set[str]:
    return {
        scope_key
        for item in items
        if item.status.value == "PENDING" and item.is_correlated_fault()
        for scope_key in item.correlation_scope_keys
    }


def incomplete_observation_scope_keys(items) -> set[str]:
    return {
        scope_key
        for item in items
        if item.path == "/v1/workload-observations"
        and item.status.value in {"PENDING", "LEASED"}
        for scope_key in item.correlation_scope_keys
    }
