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
