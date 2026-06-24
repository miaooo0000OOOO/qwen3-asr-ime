"""Tests for BackendManager."""
import asyncio
from unittest.mock import MagicMock

import pytest

from qwen3_asr_ime.common.config import IMEConfig


class TestBackendManager:
    """Unit tests for BackendManager without a real backend process."""

    @pytest.fixture
    def config(self):
        return IMEConfig.defaults()

    @pytest.fixture
    def manager(self):
        from qwen3_asr_ime.daemon.backend_manager import BackendManager
        return BackendManager()

    def test_initial_state(self, manager):
        """Manager starts with no process and not running."""
        assert not manager.is_running

    @pytest.mark.asyncio
    async def test_touch_activity_resets_timer(self, manager):
        """touch_activity updates last_activity timestamp."""
        old = manager._last_activity
        await asyncio.sleep(0.01)
        manager.touch_activity()
        assert manager._last_activity > old

    @pytest.mark.asyncio
    async def test_check_idle_not_running(self, manager):
        """check_idle returns False when no process is running."""
        result = await manager.check_idle()
        assert result is False

    @pytest.mark.asyncio
    async def test_check_idle_zero_sleep_time(self, manager):
        """check_idle returns False when auto_sleep_time is 0."""
        manager._auto_sleep_time = 0
        # Mock is_running
        manager._process = MagicMock()
        manager._process.returncode = None
        result = await manager.check_idle()
        manager._process = None
        assert result is False
