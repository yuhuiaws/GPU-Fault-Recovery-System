"""The control-plane liveness probe: its own port, a sustained-stall threshold.

The regional patch is the source of the probe timing and the renderer gives
each role its liveness listener port (service port + 10); the generated
manifests must carry both. Readiness and startup are deliberately unchanged:
a Pod that cannot accept a request promptly is what readiness should say.

Numbers (scripts/perf/ingress_event_loop_burst.py, 50 clusters x 85 cold-
connect workers against four uvicorn processes): the service socket's accept
queue held 3,600-3,900 of 4,250 connections for the whole burst, a probe on
the service port waited 3.6 s (0.1 s in-app), the longest event-loop stall
was a 0.6 s gen-2 collection. 10 s x 3 is an order of magnitude above a
busy process and only a loop that has not run for 30 s is restarted.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PATCH = ROOT / "deploy/control-plane/regional/regional-control-plane-patch.yaml"
BASE = ROOT / "deploy/control-plane/base/control-plane-deployment.yaml"
GENERATED = ROOT / "deploy/control-plane/regional/generated"
ROLES = {
    "gpu-fault-api-ha-ingress.yaml": ("api", 8080, 8090),
    "gpu-fault-control-worker.yaml": ("control-worker", 8081, 8091),
    "gpu-fault-telemetry-spool-worker.yaml": ("telemetry-spool-worker", 8082, 8092),
}
LIVENESS_TIMING = {
    "initialDelaySeconds": 10,
    "periodSeconds": 10,
    "timeoutSeconds": 10,
    "failureThreshold": 3,
}


def _api_container(path: Path) -> dict:
    documents = [
        document
        for document in yaml.safe_load_all(path.read_text(encoding="utf-8"))
        if isinstance(document, dict) and document.get("kind") == "Deployment"
    ]
    (deployment,) = documents
    containers = deployment["spec"]["template"]["spec"]["containers"]
    return next(item for item in containers if item["name"] in {"api", *_NAMES})


_NAMES = {name for name, _, _ in ROLES.values()}


def test_the_patch_probes_liveness_on_its_own_port_with_a_sustained_threshold():
    container = _api_container(PATCH)
    probe = container["livenessProbe"]
    assert probe["httpGet"] == {"path": "/livez", "port": "livez"}
    assert {key: probe[key] for key in LIVENESS_TIMING} == LIVENESS_TIMING
    assert {"name": "livez", "containerPort": 8090} in container["ports"]
    assert "readinessProbe" not in container, "readiness stays as the base has it"


def test_the_base_readiness_probe_is_unchanged():
    container = _api_container(BASE)
    readiness = container["readinessProbe"]
    assert readiness["httpGet"] == {"path": "/healthz", "port": "http"}
    assert readiness["periodSeconds"] == 5
    assert readiness["timeoutSeconds"] == 5
    assert "failureThreshold" not in readiness, "the Kubernetes default of 3"
    assert container["startupProbe"]["tcpSocket"] == {"port": "http"}


def test_every_generated_role_probes_its_liveness_listener_port():
    for filename, (name, service_port, liveness_port) in ROLES.items():
        container = _api_container(GENERATED / filename)
        assert container["name"] == name
        probe = container["livenessProbe"]
        assert probe["httpGet"] == {"path": "/livez", "port": liveness_port}, filename
        assert {key: probe[key] for key in LIVENESS_TIMING} == LIVENESS_TIMING
        ports = {item["name"]: item["containerPort"] for item in container["ports"]}
        assert ports == {"http": service_port, "livez": liveness_port}, filename
        assert container["readinessProbe"]["httpGet"]["port"] == service_port
        assert container["readinessProbe"]["timeoutSeconds"] == 5
        assert f"--port {service_port} " in container["args"][0]


def test_every_generated_role_tells_its_processes_the_port_and_the_gc_policy():
    for filename, (_, _, liveness_port) in ROLES.items():
        prefix = filename.removesuffix(".yaml").removesuffix("-ingress")
        core = yaml.safe_load(
            (GENERATED / f"{prefix}-config-core.yaml").read_text(encoding="utf-8")
        )["data"]
        assert core["GPU_FAULT_LIVENESS_PORT"] == str(liveness_port), filename
        assert core["GPU_FAULT_GC_FREEZE_AFTER_STARTUP"] == "true", filename
        assert "GPU_FAULT_GC_GEN2_THRESHOLD" not in core, (
            "CPython's gen-2 threshold stays: raising it lengthened every pause"
        )
