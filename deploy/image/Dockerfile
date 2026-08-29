ARG RUNTIME_BASE_IMAGE=registry.access.redhat.com/ubi9/python-312-minimal@sha256:ddf83888de3388bce2fc9c3d66f917bf683790d4db9f956644a3e1b9b6fd15e7
FROM ${RUNTIME_BASE_IMAGE}

USER 0
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements/runtime.lock /tmp/gpu-fault-runtime.lock
RUN python -m pip install --no-cache-dir --require-hashes \
    -r /tmp/gpu-fault-runtime.lock \
    && rm -f /tmp/gpu-fault-runtime.lock
COPY wheels /tmp/gpu-fault-wheels
RUN python -m venv --system-site-packages /opt/gpu-fault/control-plane \
    && python -m venv --system-site-packages /opt/gpu-fault/executor \
    && for component in control-plane executor; do \
      for library in lib64 lib; do \
        site_dir="/opt/gpu-fault/${component}/${library}/python3.12/site-packages"; \
        install -d "${site_dir}"; \
        printf '%s\n' \
          /opt/app-root/lib64/python3.12/site-packages \
          /opt/app-root/lib/python3.12/site-packages \
          > "${site_dir}/gpu-fault-runtime-dependencies.pth"; \
      done; \
    done \
    && /opt/gpu-fault/control-plane/bin/python -m pip install \
      --no-cache-dir --no-deps \
      /tmp/gpu-fault-wheels/gpu_fault_control_plane-*.whl \
    && /opt/gpu-fault/executor/bin/python -m pip install \
      --no-cache-dir --no-deps \
      /tmp/gpu-fault-wheels/gpu_fault_cluster_executor-*.whl \
    && if python -c 'import gpu_fault' >/dev/null 2>&1; then \
      echo "system Python unexpectedly contains gpu_fault" >&2; exit 1; \
    fi \
    && /opt/gpu-fault/control-plane/bin/python -c \
      'import importlib.metadata as m, pydantic; assert set(m.packages_distributions()["gpu_fault"]) == {"gpu-fault-control-plane"}' \
    && /opt/gpu-fault/executor/bin/python -c \
      'import importlib.metadata as m, pydantic; assert set(m.packages_distributions()["gpu_fault"]) == {"gpu-fault-cluster-executor"}' \
    && rm -rf /tmp/gpu-fault-wheels

USER 1001
ENTRYPOINT []
CMD ["/opt/gpu-fault/control-plane/bin/gpu-fault-api"]
