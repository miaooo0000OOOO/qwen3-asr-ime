from __future__ import annotations

import logging
import shutil
import subprocess
import time

import requests

logger = logging.getLogger(__name__)


class ASRProcessManager:
    def __init__(self, model: str, device: str = "auto", quantization: str = "auto"):
        self.model = model
        self.device = device
        self.quantization = quantization
        self._process: subprocess.Popen | None = None

    def _detect_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            if torch.cuda.is_available():
                mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if mem >= 6:
                    return "cuda"
        except Exception:
            pass
        return "cpu"

    def _build_command(self) -> list[str]:
        device = self._detect_device()
        if shutil.which("vllm"):
            cmd = [
                "python",
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                self.model,
                "--port",
                "8000",
            ]
            if device == "cpu":
                cmd.extend(["--device", "cpu"])
            return cmd
        if shutil.which("llama-server"):
            return ["llama-server", "-m", self.model, "--port", "8000"]
        raise RuntimeError("No supported ASR server backend found (vllm or llama-server)")

    def start(self) -> None:
        if self.is_running():
            logger.info("ASR service already running")
            return
        cmd = self._build_command()
        logger.info("Starting ASR service: %s", " ".join(cmd))
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._wait_for_ready()

    def _wait_for_ready(self, timeout: float = 120.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp = requests.get("http://127.0.0.1:8000/health", timeout=1)
                if resp.status_code == 200:
                    logger.info("ASR service ready")
                    return
            except requests.RequestException:
                pass
            time.sleep(1)
        raise TimeoutError("ASR service did not become ready")

    def stop(self) -> None:
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

    @staticmethod
    def is_running() -> bool:
        try:
            resp = requests.get("http://127.0.0.1:8000/health", timeout=1)
            return resp.status_code == 200
        except requests.RequestException:
            return False
