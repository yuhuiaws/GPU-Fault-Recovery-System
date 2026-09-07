# Fault-injection manifests

These manifests can write synthetic XID/SXID records to `/dev/kmsg`, quiesce
GPU services or otherwise alter test nodes. They are destructive test inputs,
not deployment assets.

Use only from an approved fault-injection case with an explicit Kubernetes
context, isolated node and rollback procedure. Never directory-apply this
tree.

The manifests never contain concrete node names. Before applying a single-node
case, label the approved target:

```bash
kubectl label node "${TARGET_NODE}" gpu-fault.io/e2e-target-a=true
```

Two-node cases use `gpu-fault.io/e2e-target-a=true` and
`gpu-fault.io/e2e-target-b=true`; the labels must resolve to different nodes.
Remove both labels during cleanup:

```bash
kubectl label node "${TARGET_NODE_A}" gpu-fault.io/e2e-target-a-
kubectl label node "${TARGET_NODE_B}" gpu-fault.io/e2e-target-b-
```

## Contents

Every manifest here is named by exactly one case or procedure; a manifest
nobody names gets removed (the 2026-09-07 review dropped ten such files).

| Manifest | Named by |
| --- | --- |
| `kmsg-xid45-xid14.yaml` | `GF-LIVE-KMSG-XID45-XID14-20260724` (`manifest:` in `testcases/fault-scenarios.yaml`) |
| `kmsg-xid54.yaml` | `GF-REGIONAL-COLLECT-009` CHECK_MECHANICALS loop; `tests/regional/test_regional_acceptance_fixtures.py` pins its presence |
| `kmsg-xid63-xid48.yaml` | `GF-REGIONAL-COLLECT-008` companion-branch procedure in `docs/区域模式端到端验收测试用例.md` |

The regional collector runners (`run_collector_acceptance.py`,
`run_collector_destructive.py`) write `/dev/kmsg` through their own probe Pods
and do not apply these manifests; they document the by-hand form of the same
injection.
