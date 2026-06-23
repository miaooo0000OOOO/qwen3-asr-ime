"""Global hotkey listener for the voice input daemon.

Provides a pynput-based listener that supports chord combos (e.g.
``"<Super>+<Shift>+R"``) and flex modifier matching (e.g. ``"CTRL"`` matches
both left and right Ctrl keys).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from pynput import keyboard

logger = logging.getLogger(__name__)

# Modifier families: each canonical name maps to the left/right pynput keys.
_MODIFIER_KEYS: dict[str, tuple[keyboard.Key, keyboard.Key]] = {
    "CTRL": (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r),
    "SHIFT": (keyboard.Key.shift_l, keyboard.Key.shift_r),
    "ALT": (keyboard.Key.alt_l, keyboard.Key.alt_r),
    "SUPER": (keyboard.Key.cmd_l, keyboard.Key.cmd_r),
}

# Non-modifier special keys addressable by canonical names in combo strings.
_SPECIAL_KEYS: dict[str, keyboard.Key] = {
    "SPACE": keyboard.Key.space,
    "ENTER": keyboard.Key.enter,
    "TAB": keyboard.Key.tab,
    "ESC": keyboard.Key.esc,
    "BACKSPACE": keyboard.Key.backspace,
    "LEFT": keyboard.Key.left,
    "RIGHT": keyboard.Key.right,
    "UP": keyboard.Key.up,
    "DOWN": keyboard.Key.down,
}


def _normalized_key(key: keyboard.Key | keyboard.KeyCode | None) -> keyboard.Key | keyboard.KeyCode | None:
    """Return a normalized form of a pynput key for set comparison.

    Character keys are normalized to lower-case so that ``"R"`` in a combo
    matches a physical ``r`` press regardless of Shift/CapsLock state.
    """
    if key is None:
        return None
    if isinstance(key, keyboard.KeyCode) and key.char is not None:
        return keyboard.KeyCode.from_char(key.char.lower())
    return key


@dataclass(frozen=True, slots=True)
class HotkeyEvent:
    """A hotkey press or release event."""

    action: Literal["press", "release"]


class PynputHotkeyListener:
    """Global hotkey listener using ``pynput``.

    Works with X11 by monitoring global key events through the X11 XTEST
    extension (or the platform-native equivalent).

    Supports chord combos and flex key families. For example:

    - ``"CTRL"`` matches left Ctrl *or* right Ctrl.
    - ``"<Super>+<Shift>+R"`` requires all three keys to be pressed together.
    """

    def __init__(self, key_combo: str, on_event: Callable[[HotkeyEvent], None]):
        """Initialize the listener.

        Args:
            key_combo: Human-readable combo such as ``"CTRL"`` or
                ``"<Super>+<Shift>+R"``.
            on_event: Callback invoked with ``HotkeyEvent("press")`` or
                ``HotkeyEvent("release")`` when the combo triggers.
        """
        self.key_combo = key_combo
        self.on_event = on_event
        self._triggered = False
        self._pressed: set[keyboard.Key | keyboard.KeyCode] = set()
        self._target_keys = self._parse_combo(key_combo)
        # Flex mode: for single-modifier combos like "CTRL", any target key
        # press triggers; for true chords, all target keys must be down.
        self._any_mode = len(self._target_keys) <= 2
        self._listener: keyboard.Listener | None = None

    @classmethod
    def _parse_combo(cls, combo: str) -> set[keyboard.Key | keyboard.KeyCode]:
        """Parse a human-readable key combo into a set of pynput keys.

        Args:
            combo: Key combo string. Angle brackets are stripped and ``+``
                separates chord members.

        Returns:
            Set of pynput keys that must be pressed to trigger the combo.
        """
        keys: set[keyboard.Key | keyboard.KeyCode] = set()
        for part in combo.upper().replace("<", "").replace(">", "").split("+"):
            part = part.strip()
            if not part:
                continue
            if part in _MODIFIER_KEYS:
                keys.update(_MODIFIER_KEYS[part])
            elif part in _SPECIAL_KEYS:
                keys.add(_SPECIAL_KEYS[part])
            elif len(part) == 1:
                keys.add(keyboard.KeyCode.from_char(part.lower()))
            else:
                # Try function keys (F1..F12) and other pynput Key members.
                attr = part.lower()
                if hasattr(keyboard.Key, attr):
                    keys.add(getattr(keyboard.Key, attr))
        return keys

    def start(self) -> None:
        """Start listening for global keyboard events."""
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()
        logger.info("Started pynput hotkey listener for '%s'", self.key_combo)

    def stop(self) -> None:
        """Stop listening and release resources."""
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def _on_press(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        """pynput callback: handle a global key press."""
        norm = _normalized_key(key)
        if norm is None:
            return
        from_key = norm in self._target_keys
        if from_key:
            self._pressed.add(norm)
        if self._any_mode:
            if from_key and not self._triggered:
                self._triggered = True
                self.on_event(HotkeyEvent("press"))
        else:
            if self._pressed == self._target_keys and not self._triggered:
                self._triggered = True
                self.on_event(HotkeyEvent("press"))

    def _on_release(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        """pynput callback: handle a global key release."""
        norm = _normalized_key(key)
        if norm is None:
            return
        from_key = norm in self._target_keys
        if norm in self._pressed:
            self._pressed.discard(norm)
        if self._any_mode:
            if from_key and self._triggered and not self._pressed & self._target_keys:
                self._triggered = False
                self.on_event(HotkeyEvent("release"))
        else:
            if from_key and self._triggered:
                self._triggered = False
                self.on_event(HotkeyEvent("release"))


# Supported values for the ``device`` argument of ``create_hotkey_listener``.
# Only ``"pynput"`` is implemented; ``"auto"`` is an alias for compatibility.
HotkeyDevice = Literal["auto", "pynput"]


def create_hotkey_listener(
    device: HotkeyDevice, key_combo: str, on_event: Callable[[HotkeyEvent], None]
) -> PynputHotkeyListener:
    """Create a global hotkey listener.

    Args:
        device: Backend selector. ``"auto"`` and ``"pynput"`` both create a
            ``PynputHotkeyListener``.
        key_combo: Hotkey combo such as ``"CTRL"`` or ``"<Super>+<Shift>+R"``.
        on_event: Callback invoked on press/release events.

    Returns:
        A started hotkey listener instance.

    Raises:
        ValueError: If ``device`` is not ``"auto"`` or ``"pynput"``.
    """
    if device not in ("auto", "pynput"):
        raise ValueError(f"Unsupported hotkey device: {device}")
    return PynputHotkeyListener(key_combo, on_event)
