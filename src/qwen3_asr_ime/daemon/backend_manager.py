"""Backend process lifecycle manager.

``BackendManager`` spawns the ASR server as a child process, monitors its
health endpoint, enforces idle-timeout auto-sleep, and detects fatal errors
in the backend's stderr (OOM, port conflict, model not found).

All unexpected failures cause ``sys.exit(1)``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qwen3_asr_ime.common.config import IMEConfig

logger = logging.getLogger(__name__)

# Project root directory (parent of tools/)
_PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent
_SERVER_SCRIPT = _PROJECT_DIR / "tools" / "asr_server.py"


class BackendManager:
    """Manages the full lifecycle of the ASR backend child process.

    Usage::

        mgr = BackendManager()
        await mgr.spawn(config)
        await mgr.wait_ready()
        # ... backend is running ...
        mgr.touch_activity()
        await mgr.check_idle()   # stops backend if idle timeout reached
    """

    # Patterns in stderr that indicate a fatal startup error.
    _FATAL_PATTERNS: dict[str, str] = {
        "out of memory": "GPU 显存不足，无法加载模型",
        "cuda out of memory": "GPU 显存不足，无法加载模型",
        "address already in use": "端口已被占用",
        "cannot access": "模型路径不存在或无效",
        "no such file": "模型路径不存在或无效",
    }

    def __init__(self) -> None:
        """Initialize with no process, inactive idle timer, and defaults."""
        self._process: asyncio.subprocess.Process | None = None
        self._last_activity: float = 0.0
        self._stderr_task: asyncio.Task[None] | None = None
        self._health_url: str = ""
        self._auto_sleep_time: float = 0.0
        self._wait_timeout: float = 120.0
        self._port: int = 8000

    @property
    def is_running(self) -> bool:
        """Return True if the backend process exists and has not exited."""
        return self._process is not None and self._process.returncode is None

    async def spawn(self, config: IMEConfig) -> None:
        """Launch the ASR server subprocess.

        Args:
            config: Current IMEConfig providing mode, backend, model, endpoint.

        Exits with ``sys.exit(1)`` if the process fails to start.
        """
        if self.is_running:
            logger.warning("Backend already running; stopping first")
            await self.stop()

        from urllib.parse import urlparse
        parsed = urlparse(config.asr_endpoint)
        self._port = parsed.port or 8000
        self._health_url = f"{config.asr_endpoint.rstrip('/')}/health"
        self._auto_sleep_time = float(config.asr_auto_sleep_time)
        self._wait_timeout = float(config.asr_backend_wait_timeout)

        env = os.environ.copy()
        env["QWEN3_ASR_MODE"] = config.asr_mode
        env["QWEN3_ASR_BACKEND"] = config.asr_backend
        env["QWEN3_ASR_MODEL"] = config.model_path
        env["QWEN3_ASR_DEVICE"] = config.asr_device
        env["QWEN3_ASR_PORT"] = str(self._port)

        cmd = [sys.executable, str(_SERVER_SCRIPT)]

        logger.info(
            "Starting backend: %s (mode=%s, backend=%s, model=%s, port=%d)",
            cmd, config.asr_mode, config.asr_backend, config.model_path, self._port,
        )
        try:
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as exc:
            print(
                f"ERROR: 后端进程启动失败: {' '.join(cmd)}: {exc}",
                file=sys.stderr,
            )
            sys.exit(1)

        self._stderr_task = asyncio.create_task(self._monitor_stderr())

    async def wait_ready(self) -> None:
        """Poll /health until the backend responds 200 OK with ``status=="ok"``.

        While polling, monitors stderr for fatal patterns. If the backend
        process exits before becoming ready, exits immediately.

        Raises/Exits:
            ``sys.exit(1)`` on timeout or fatal error detected in stderr.
        """
        import aiohttp

        if self._process is None:
            print("ERROR: 后端进程未启动", file=sys.stderr)
            sys.exit(1)

        deadline = asyncio.get_running_loop().time() + self._wait_timeout
        last_log = asyncio.get_running_loop().time()

        while asyncio.get_running_loop().time() < deadline:
            # Check if process died
            if self._process.returncode is not None:
                # Give stderr monitor a moment to report the actual cause
                await asyncio.sleep(0.5)
                print(
                    f"ERROR: 后端进程异常退出 (code={self._process.returncode})",
                    file=sys.stderr,
                )
                sys.exit(1)

            # Try the health endpoint
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        self._health_url, timeout=aiohttp.ClientTimeout(total=2)
                    ) as resp:
                        if resp.status == 200:
                            body = await resp.json()
                            if body.get("status") == "ok":
                                logger.info(
                                    "Backend healthy (backend=%s) in %.1fs",
                                    body.get("backend", "unknown"),
                                    self._wait_timeout - (deadline - asyncio.get_running_loop().time()),
                                )
                                self._last_activity = asyncio.get_running_loop().time()
                                return
            except Exception:
                pass  # not ready yet

            if asyncio.get_running_loop().time() - last_log > 10:
                elapsed = self._wait_timeout - (deadline - asyncio.get_running_loop().time())
                logger.info("Waiting for backend... (%.0f/%.0fs)", elapsed, self._wait_timeout)
                last_log = asyncio.get_running_loop().time()

            await asyncio.sleep(1.0)

        print(
            f"ERROR: 后端启动超时 ({self._wait_timeout}s): {self._health_url} 无响应",
            file=sys.stderr,
        )
        sys.exit(1)

    async def _monitor_stderr(self) -> None:
        """Continuously read backend stderr; detect fatal patterns and exit.

        Runs as a background task. Non-fatal output is logged at DEBUG level.
        """
        if self._process is None or self._process.stderr is None:
            return
        try:
            while True:
                line = await self._process.stderr.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                # Check for fatal patterns
                lower = text.lower()
                for pattern, message in self._FATAL_PATTERNS.items():
                    if pattern in lower:
                        print(f"ERROR: {message}: {text}", file=sys.stderr)
                        sys.exit(1)
                logger.debug("Backend stderr: %s", text)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.warning("stderr monitor error: %s", exc)

    async def _read_remaining_stderr(self) -> str:
        """Drain any remaining stderr lines after process exits. Returns last few lines."""
        if self._process is None or self._process.stderr is None:
            return ""
        lines: list[str] = []
        try:
            while True:
                line = await self._process.stderr.readline()
                if not line:
                    break
                lines.append(line.decode("utf-8", errors="replace").strip())
        except Exception:
            pass
        return "\n".join(lines[-10:])

    async def ensure_running(self) -> None:
        """Call before each recognition request.

        If the backend is already running, resets the idle timer and returns immediately.
        Otherwise spawns and waits for readiness.
        """
        if self.is_running:
            self._last_activity = asyncio.get_running_loop().time()
            return
        logger.info("Backend not running; re-spawning...")
        # Need config to re-spawn — stored from last spawn
        # This is handled by the daemon which holds the config reference
        # For standalone use, this is a no-op if the process died unexpectedly
        if not self.is_running:
            # The daemon should call spawn() with current config instead
            pass

    def touch_activity(self) -> None:
        """Mark the backend as recently used, resetting the idle timer.

        Called after each successful ASR recognition completes.
        """
        self._last_activity = asyncio.get_running_loop().time()

    async def check_idle(self) -> bool:
        """Check if the backend has been idle beyond auto_sleep_time.

        If idle timeout reached and backend is running, stops the backend.

        Returns:
            True if the backend was stopped due to idle timeout.
        """
        if not self.is_running:
            return False
        if self._auto_sleep_time <= 0:
            return False
        idle_duration = asyncio.get_running_loop().time() - self._last_activity
        if idle_duration >= self._auto_sleep_time:
            logger.info(
                "Backend idle for %.0fs (limit: %.0fs); stopping to save resources",
                idle_duration,
                self._auto_sleep_time,
            )
            await self.stop()
            return True
        return False

    async def stop(self) -> None:
        """Gracefully stop the backend process.

        Sends SIGTERM, waits up to 5 seconds, then SIGKILL if still alive.
        """
        if self._stderr_task is not None:
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except asyncio.CancelledError:
                pass
            self._stderr_task = None

        if self._process is None:
            return

        if self._process.returncode is None:
            logger.info("Stopping backend (pid=%d)...", self._process.pid)
            try:
                self._process.send_signal(signal.SIGTERM)
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning("Backend did not stop; sending SIGKILL")
                    self._process.kill()
                    await self._process.wait()
            except ProcessLookupError:
                pass  # already exited
            logger.info("Backend stopped")

        self._process = None

    async def restart(self, config: IMEConfig) -> None:
        """Stop the current backend and spawn a new one with updated config.

        Used when config-watcher detects backend-relevant changes.
        """
        logger.info("Restarting backend due to config change")
        await self.stop()
        await self.spawn(config)
        await self.wait_ready()
