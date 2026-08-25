# E2E and probe manifests

These manifests are test inputs, not production deployment assets. Apply only
the file named by an E2E procedure, against the explicitly selected test
cluster and namespace.

They intentionally live outside `deploy/` so directory-level production
operations cannot include canary, smoke, capacity or connectivity probes.

Manifests that need one specific GPU node select
`gpu-fault.io/e2e-target=true`; they never contain a concrete node name.
Label and clean up the target explicitly:

```bash
kubectl label node "${TARGET_NODE}" gpu-fault.io/e2e-target=true
# apply the procedure's rendered manifest
kubectl label node "${TARGET_NODE}" gpu-fault.io/e2e-target-
```

Manifests mounting the control-plane wheel contain
`REPLACE_WITH_WHEEL_CONFIGMAP`. Render them into `/tmp`; never edit or apply the
source file directly:

```bash
python3 scripts/e2e/render_manifest.py \
  scripts/e2e/manifests/<manifest>.yaml \
  --wheel-configmap "${WHEEL_CONFIG_MAP}" \
  -o /tmp/<manifest>.yaml
kubectl apply -f /tmp/<manifest>.yaml
```

Driver/manifest pairs:

| Manifest | Driver mounted into `/test` |
|---|---|
| `hyperpod-dcgm-metrics-e2e.yaml` | `../run_hyperpod_dcgm_metrics_e2e.py` |
| `hyperpod-efa-traffic-e2e.yaml` | `../run_hyperpod_efa_traffic_e2e.py` |
| `hyperpod-three-source-fault-e2e.yaml` | `../run_hyperpod_three_source_fault_e2e.py` |

The procedure must create the driver ConfigMap from the file in the same row;
it must also include
`--from-file=isolated_api.py=scripts/e2e/isolated_api.py`. Changing the
manifest, driver or shared launcher requires updating and testing the trio
together.
