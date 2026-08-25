"""Local development entry point for the application factory."""

from __future__ import annotations

import os

__all__ = ["run"]


def run() -> None:
    import uvicorn

    uvicorn.run(
        "gpu_fault.app:create_app",
        factory=True,
        host=os.getenv("GPU_FAULT_API_HOST", "127.0.0.1"),
        port=int(os.getenv("GPU_FAULT_API_PORT", "8080")),
        reload=False,
    )


if __name__ == "__main__":
    run()
