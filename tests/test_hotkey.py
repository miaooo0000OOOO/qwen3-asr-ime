"""Tests for the global hotkey listener."""

from __future__ import annotations

from unittest.mock import MagicMock

from pynput import keyboard

from qwen3_asr_ime.daemon.hotkey import HotkeyEvent, PynputHotkeyListener


def test_ctrl_press_release_triggers_events() -> None:
    """Pressing and releasing Ctrl fires press/release events."""
    callback = MagicMock()
    listener = PynputHotkeyListener("CTRL", callback)

    listener._on_press(keyboard.Key.ctrl_l)
    callback.assert_called_once_with(HotkeyEvent("press"))

    callback.reset_mock()
    listener._on_release(keyboard.Key.ctrl_l)
    callback.assert_called_once_with(HotkeyEvent("release"))


def test_ctrl_plus_other_key_triggers_interrupt() -> None:
    """Pressing another key while Ctrl is held fires an interrupt event."""
    callback = MagicMock()
    listener = PynputHotkeyListener("CTRL", callback)

    listener._on_press(keyboard.Key.ctrl_l)
    callback.assert_called_once_with(HotkeyEvent("press"))

    callback.reset_mock()
    listener._on_press(keyboard.KeyCode.from_char("c"))
    callback.assert_called_once_with(HotkeyEvent("interrupt"))

    # Further non-target presses should not re-trigger interrupt.
    callback.reset_mock()
    listener._on_press(keyboard.KeyCode.from_char("v"))
    callback.assert_not_called()

    # Releasing Ctrl resets state.
    callback.reset_mock()
    listener._on_release(keyboard.Key.ctrl_l)
    callback.assert_called_once_with(HotkeyEvent("release"))

    # A fresh Ctrl press works normally after release.
    callback.reset_mock()
    listener._on_press(keyboard.Key.ctrl_l)
    callback.assert_called_once_with(HotkeyEvent("press"))


def test_dual_ctrl_does_not_interrupt() -> None:
    """Pressing the other Ctrl key while one Ctrl is held is not an interrupt."""
    callback = MagicMock()
    listener = PynputHotkeyListener("CTRL", callback)

    listener._on_press(keyboard.Key.ctrl_l)
    callback.assert_called_once_with(HotkeyEvent("press"))

    callback.reset_mock()
    listener._on_press(keyboard.Key.ctrl_r)
    callback.assert_not_called()

    # Releasing one Ctrl while the other is still held should not release.
    listener._on_release(keyboard.Key.ctrl_l)
    callback.assert_not_called()

    # Releasing the remaining Ctrl fires release.
    listener._on_release(keyboard.Key.ctrl_r)
    callback.assert_called_once_with(HotkeyEvent("release"))
