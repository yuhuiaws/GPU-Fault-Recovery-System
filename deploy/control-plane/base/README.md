# Control-plane renderer inputs

This directory is not deployable.

`control-plane-deployment.yaml` is shared by the historical single-cluster
installer and the regional CPU control plane. The regional path overlays
`../regional/regional-control-plane-patch.yaml`, then splits the intermediate
Deployment into ingress, control-worker and telemetry-spool-worker roles.

`role-split-input.kustomization.yaml` deliberately uses a non-standard
filename. The render wrapper copies it to a temporary directory as
`kustomization.yaml`; do not rename it or run `kubectl apply -k` here.
