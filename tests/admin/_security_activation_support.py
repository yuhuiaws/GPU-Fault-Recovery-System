"""Operational I/O double; the activation state machine itself remains real."""

from __future__ import annotations

import copy
import hashlib

from gpu_fault.admin.node_key_custody_activation_state import (
    owned_wave_data,
    require_owned_wave,
)


class ActivationDouble:
    def __init__(self, *, keys=None, authorization=None):
        self.events = []
        self.failure = None
        self.keys = keys
        self.snapshot = None
        self.current_reads = 0
        self.authorization = authorization
        self.fenced = False
        self.wave = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": "unit-wave",
                "namespace": "gpu-fault-system",
                "uid": "unit-wave-uid",
                "resourceVersion": "1",
            },
            "data": {
                "allowed-nodes": "*",
                "max-unavailable": "1",
                "generation": "steady",
            },
        }

    def capture(self):
        self.events.append("capture")
        self.snapshot = {"binding": "fixture-runtime"}
        self.snapshot["wave"] = {
            "name": self.wave["metadata"]["name"],
            "uid": self.wave["metadata"]["uid"],
            "data": copy.deepcopy(self.wave["data"]),
        }
        if self.keys is not None:
            self.snapshot["siblings"] = {
                key: hashlib.sha256(value.encode()).hexdigest()
                for key, value in self.keys().items()
                if key != "node-a"
            }
        return copy.deepcopy(self.snapshot)

    def bind_completed(self, completed):
        self.completed = completed

    def keys_provisioned(self, snapshot):
        return {"signed_completion_verified": True}

    def verify(self, snapshot):
        assert snapshot == self.snapshot, (
            "activation must keep the prepared runtime binding"
        )
        if self.keys is not None:
            assert snapshot["siblings"] == {
                key: hashlib.sha256(value.encode()).hexdigest()
                for key, value in self.keys().items()
                if key != "node-a"
            }, "activation changed a sibling key"

    def operation(self, name):
        self.events.append(name)
        if self.failure == name:
            raise RuntimeError("simulated lost activation acknowledgement")
        return {"operation": name, "independent_witness": False}

    def guard(self, snapshot):
        return self.operation("guard")

    def fence(self, snapshot):
        evidence = self.operation("fence")
        self.fenced = True
        if self.authorization is not None:
            self.wave["data"] = owned_wave_data(snapshot["wave"], self.authorization)
        return evidence

    def require_owned_wave(self, snapshot):
        assert self.fenced, "activation lost its owned wave"
        if self.authorization is not None:
            require_owned_wave(self.wave, snapshot["wave"], self.authorization)

    def refresh_executor(self, snapshot):
        return self.operation("executor")

    def install(self, snapshot):
        return self.operation("install")

    def refresh_cpu(self, snapshot):
        return self.operation("cpu")

    def observe(self, snapshot):
        return self.operation("observe")

    def unfence(self, snapshot):
        evidence = self.operation("unfence")
        self.wave["data"] = copy.deepcopy(snapshot["wave"]["data"])
        self.fenced = False
        return evidence

    def current(self, snapshot):
        self.current_reads += 1
        return {"independent_witness": False}
