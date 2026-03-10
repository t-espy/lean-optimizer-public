import json
import os
import select
import subprocess
import threading
import time
from pathlib import Path
from uuid import uuid4

import docker
from loguru import logger

from optimizer.config import config_builder
from optimizer.runner.evaluation import Evaluation, EvaluationStatus


class Worker:
    """Wraps a single long-lived LEAN engine container with a persistent harness.

    The harness is a dotnet process that stays alive inside the container,
    accepting backtest requests via stdin JSON lines and returning results
    on stdout.  This eliminates per-eval dotnet startup overhead (~2.1s).

    Usage:
        with Worker(artifact_dir, results_root, harness_dir) as worker:
            worker.run_backtest(evaluation)
    """

    def __init__(
        self,
        artifact_dir: Path,
        results_root: Path,
        harness_dir: Path,
        container_prefix: str = "lean_optimizer",
    ):
        self._artifact_dir = artifact_dir.resolve()
        self._results_root = results_root.resolve()
        self._harness_dir = harness_dir.resolve()
        self._results_container_path = os.environ.get("RESULTS_CONTAINER_PATH", "/Results")
        self._artifacts_container_path = os.environ.get("ARTIFACTS_CONTAINER_PATH", "/Artifacts")
        self._lean_data_host = os.environ.get("LEAN_DATA_HOST_PATH", "/path/to/lean/Data")
        self._lean_data_container = os.environ.get("LEAN_DATA_CONTAINER_PATH", "/Lean/Data")
        self._launcher_workdir = os.environ.get(
            "LEAN_LAUNCHER_WORKDIR", "/Lean/Launcher/bin/Debug"
        )
        self._image = os.environ.get("LEAN_IMAGE", "lean-cli/engine:dgx-arm64")
        self._client = docker.from_env()
        self._container = None
        self._harness_proc = None
        self._stderr_thread = None
        self._name = f"{container_prefix}_{uuid4().hex[:8]}"

    def start(self) -> None:
        logger.info(f"[worker] Starting container {self._name}")
        network = os.environ.get("LEAN_DOCKER_NETWORK", "lean_cli")
        self._container = self._client.containers.run(
            image=self._image,
            name=self._name,
            entrypoint=["sleep", "infinity"],
            volumes={
                self._lean_data_host: {"bind": self._lean_data_container, "mode": "ro"},
                str(self._artifact_dir): {"bind": self._artifacts_container_path, "mode": "ro"},
                str(self._results_root): {"bind": self._results_container_path, "mode": "rw"},
                str(self._harness_dir): {"bind": "/Harness", "mode": "ro"},
            },
            network=network,
            detach=True,
        )
        logger.info(f"[worker] Container {self._name} started ({self._container.short_id})")

        # Launch the persistent harness process inside the container
        self._harness_proc = subprocess.Popen(
            [
                "docker", "exec", "-i", self._container.id,
                "dotnet", "/Harness/LeanHarness.dll",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
        )
        # Drain stderr in background to prevent pipe buffer deadlock.
        # LEAN writes extensive trace/debug output to Console.Out, which the
        # harness redirects to stderr.  If nobody reads stderr, the 64KB kernel
        # pipe buffer fills and the harness blocks, never writing its JSON
        # response to stdout.
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True
        )
        self._stderr_thread.start()
        logger.info(f"[worker] Harness launched in {self._name}")

    def stop(self) -> None:
        if self._harness_proc is not None:
            logger.debug(f"[worker] Closing harness stdin for {self._name}")
            try:
                self._harness_proc.stdin.close()
                self._harness_proc.wait(timeout=10)
            except Exception:
                try:
                    self._harness_proc.kill()
                except Exception:
                    pass
            self._harness_proc = None

        if self._container is None:
            return
        logger.info(f"[worker] Stopping container {self._name}")
        try:
            self._container.stop(timeout=5)
        except Exception:
            pass
        try:
            self._container.remove(force=True)
        except Exception:
            pass
        self._container = None
        logger.info(f"[worker] Container {self._name} removed")

    def _drain_stderr(self) -> None:
        """Continuously read stderr from the harness process to prevent pipe deadlock."""
        try:
            for line in self._harness_proc.stderr:
                logger.debug(f"[harness:{self._name[-8:]}] {line.rstrip()}")
        except (ValueError, OSError):
            pass  # pipe closed during shutdown

    def run_backtest(self, evaluation: Evaluation) -> Evaluation:
        """Execute one backtest via the persistent harness; return updated Evaluation."""
        if self._harness_proc is None or self._harness_proc.poll() is not None:
            raise RuntimeError("Harness not running; call start() or use context manager")

        evaluation.status = EvaluationStatus.RUNNING
        (self._results_root / evaluation.id).mkdir(parents=True, exist_ok=True)

        config_builder.build(
            evaluation=evaluation,
            results_root=self._results_root,
            artifacts_container_path=self._artifacts_container_path,
            results_container_path=self._results_container_path,
            artifact_dir=self._artifact_dir,
        )

        config_container_path = f"{self._results_container_path}/{evaluation.id}/config.json"
        request = json.dumps({"id": evaluation.id, "config_path": config_container_path})

        logger.info(f"[worker] [{evaluation.id[:8]}] → harness")
        t0 = time.monotonic()

        try:
            self._harness_proc.stdin.write(request + "\n")
            self._harness_proc.stdin.flush()

            # Wait for response with timeout to prevent indefinite blocking
            timeout = 120  # seconds
            ready, _, _ = select.select(
                [self._harness_proc.stdout], [], [], timeout
            )
            if not ready:
                raise RuntimeError(
                    f"Harness timeout: no response after {timeout}s"
                )

            response_line = self._harness_proc.stdout.readline()
            evaluation.runtime_seconds = time.monotonic() - t0

            if not response_line:
                raise RuntimeError("Harness closed stdout unexpectedly")

            response = json.loads(response_line)
        except Exception as exc:
            evaluation.runtime_seconds = time.monotonic() - t0
            evaluation.status = EvaluationStatus.FAILED
            evaluation.error_message = f"Harness communication error: {exc}"
            logger.error(f"[worker] [{evaluation.id[:8]}] {evaluation.error_message}")
            return evaluation

        if response.get("status") == "ok":
            evaluation.status = EvaluationStatus.SUCCESS
            evaluation.result_path = self._results_root / evaluation.id
        else:
            evaluation.status = EvaluationStatus.FAILED
            evaluation.error_message = response.get("message", "unknown harness error")
            logger.error(
                f"[worker] [{evaluation.id[:8]}] backtest failed: "
                f"{evaluation.error_message}"
            )

        return evaluation

    @property
    def worker_id(self) -> str:
        return self._name

    def is_alive(self) -> bool:
        """Return True if both the container and harness process are running."""
        if self._container is None:
            return False
        try:
            self._container.reload()
            if self._container.status != "running":
                return False
        except Exception:
            return False
        if self._harness_proc is None or self._harness_proc.poll() is not None:
            return False
        return True

    def __enter__(self) -> "Worker":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()
