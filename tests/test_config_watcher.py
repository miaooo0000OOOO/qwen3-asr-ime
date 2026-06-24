"""Tests for ConfigWatcher."""
import asyncio
import tempfile
import time
from pathlib import Path

import pytest

from qwen3_asr_ime.common.config import ConfigWatcher, IMEConfig


def test_create_default_config():
    """ConfigWatcher creates a default config file when none exists."""
    with tempfile.TemporaryDirectory() as td:
        config_path = Path(td) / "config.yaml"
        watcher = ConfigWatcher(path=config_path)
        assert config_path.exists()
        assert watcher.config.asr_mode == "offline"
        assert watcher.config.asr_model == "1.7B"


def test_load_existing_config():
    """ConfigWatcher loads an existing valid config."""
    with tempfile.TemporaryDirectory() as td:
        config_path = Path(td) / "config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("""
asr:
  mode: "streaming"
  model: "0.6B"
  backend: "vllm"
""")
        watcher = ConfigWatcher(path=config_path)
        assert watcher.config.asr_mode == "streaming"
        assert watcher.config.asr_model == "0.6B"
        assert watcher.config.asr_backend == "vllm"


def test_invalid_mode_backend_combo_exits():
    """Config with invalid mode+backend combo exits immediately."""
    with tempfile.TemporaryDirectory() as td:
        config_path = Path(td) / "config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("""
asr:
  mode: "streaming"
  backend: "transformers"
""")
        with pytest.raises(SystemExit):
            ConfigWatcher(path=config_path)


def test_reload_detects_mtime_change():
    """ConfigWatcher re-reads config when mtime changes."""
    with tempfile.TemporaryDirectory() as td:
        config_path = Path(td) / "config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("""
asr:
  mode: "offline"
  auto_sleep_time: 60
""")
        watcher = ConfigWatcher(path=config_path)
        assert watcher.config.asr_auto_sleep_time == 60

        # Update the file
        time.sleep(0.01)  # ensure mtime changes
        config_path.write_text("""
asr:
  mode: "offline"
  auto_sleep_time: 120
""")

        # Manually trigger reload via the internal method
        watcher._mtime = 0  # force reload
        watcher._reload()
        assert watcher.config.asr_auto_sleep_time == 120


def test_reload_failure_keeps_old_config():
    """When reload fails, the old config is retained."""
    with tempfile.TemporaryDirectory() as td:
        config_path = Path(td) / "config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("""
asr:
  mode: "offline"
  auto_sleep_time: 60
""")
        watcher = ConfigWatcher(path=config_path)
        old_sleep = watcher.config.asr_auto_sleep_time

        # Corrupt the file
        time.sleep(0.01)
        config_path.write_text("this is not valid: yaml: :::")

        watcher._mtime = 0
        watcher._reload()
        # Config should be unchanged
        assert watcher.config.asr_auto_sleep_time == old_sleep
