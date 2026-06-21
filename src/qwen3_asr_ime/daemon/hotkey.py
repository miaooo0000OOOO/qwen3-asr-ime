from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import evdev


logger = logging.getLogger(__name__)
# Linux input event codes (evdev.KEY_*) — hardcoded for testability & speed
_EC_CODES: dict[str, int] = {
    "KEY_LEFTCTRL": 29,
    "KEY_RIGHTCTRL": 97,
    "KEY_LEFTMETA": 125,
    "KEY_RIGHTMETA": 126,
    "KEY_LEFTSHIFT": 42,
    "KEY_RIGHTSHIFT": 54,
    "KEY_LEFTALT": 56,
    "KEY_RIGHTALT": 100,
    "KEY_LEFT": 105,
    "KEY_RIGHT": 106,
    "KEY_UP": 103,
    "KEY_DOWN": 108,
    "KEY_ESC": 1,
    "KEY_TAB": 15,
    "KEY_SPACE": 57,
    "KEY_ENTER": 28,
    "KEY_BACKSPACE": 14,
}


@dataclass(frozen=True, slots=True)
class HotkeyEvent:
    action: Literal["press", "release"]


class EvdevHotkeyListener:
    """Hotkey listener that supports chord combos and flex key families.

    - Chord mode (default): all keys in target_codes must be pressed simultaneously.
    - Flex mode (``_any_mode``): any key in target_codes triggers press (all must
      be released to trigger release). Used for "CTRL" meaning left OR right Ctrl.
    """

    def __init__(
        self,
        key_combo: str,
        on_event: Callable[[HotkeyEvent], None],
    ):
        self.key_combo = key_combo
        self.on_event = on_event
        self._triggered = False
        self._pressed: set[int] = set()
        self._target_codes = self._parse_combo(key_combo)
        self._any_mode = len(self._target_codes) <= 2 and "CTRL" in key_combo.upper()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._stop = threading.Event()

    @staticmethod
    def _parse_combo(combo: str) -> set[int]:
        # Linux input event codes (evdev.KEY_*)
        # Avoid runtime ec.ecodes lookup for testability.
        name_map: dict[str, int] = {
            "SUPER": 125,  # KEY_LEFTMETA
            "SHIFT": 42,  # KEY_LEFTSHIFT
            "CTRL": 0,
            "ALT": 56,  # KEY_LEFTALT
        }
        codes: set[int] = set()
        for part in combo.upper().replace("<", "").replace(">", "").split("+"):
            part = part.strip()
            if part == "CTRL":
                codes.add(29)  # KEY_LEFTCTRL
                codes.add(97)  # KEY_RIGHTCTRL
            elif part in name_map:
                codes.add(name_map[part])
            else:
                ec_code = _EC_CODES.get(f"KEY_{part}")
                if ec_code:
                    codes.add(ec_code)
        return codes

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        try:
            devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
        except OSError:
            logger.error(
                "Cannot access input devices. Are you in the 'input' group? Try: sudo usermod -aG input $USER"
            )
            return
        if not devices:
            logger.error("No input devices found. Hotkey listener will not work.")
            return
        logger.info("Listening on %d input devices (passive, no grab)", len(devices))
        try:
            while not self._stop.is_set():
                for dev in devices:
                    try:
                        for event in dev.read():
                            if event.type == 1:  # EV_KEY
                                self._handle(event)
                    except BlockingIOError:
                        continue
        except Exception:
            logger.exception("Hotkey listener error")

    def _handle(self, event) -> None:
        code = event.code
        from_code = code in self._target_codes

        if event.value == 1:  # key down
            if from_code:
                self._pressed.add(code)
            if self._any_mode:
                if from_code and not self._triggered:
                    self._triggered = True
                    self.on_event(HotkeyEvent("press"))
            else:
                self._pressed.add(code)
                if self._pressed == self._target_codes and not self._triggered:
                    self._triggered = True
                    self.on_event(HotkeyEvent("press"))

        elif event.value == 0:  # key up
            if code in self._pressed:
                self._pressed.discard(code)
            if self._any_mode:
                if from_code and self._triggered and not self._pressed & self._target_codes:
                    self._triggered = False
                    self.on_event(HotkeyEvent("release"))
            else:
                if code in self._target_codes and self._triggered:
                    self._triggered = False
                    self.on_event(HotkeyEvent("release"))


class PynputHotkeyListener:
    """Hotkey listener using ``pynput`` (no special permissions needed).

    Works with X11/Wayland by monitoring global key events.
    """

    def __init__(self, key_combo: str, on_event: Callable[[HotkeyEvent], None]):
        self.on_event = on_event
        self._target_codes = EvdevHotkeyListener._parse_combo(key_combo)
        self._pressed: set[int] = set()
        self._triggered = False
        self._listener = None
        self._any_mode = len(self._target_codes) <= 2

    def start(self):
        from pynput import keyboard

        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()

    def stop(self):
        if self._listener:
            self._listener.stop()

    def _key_code(self, key) -> int | None:
        """Convert pynput key to evdev key code using _EC_CODES."""
        try:
            name = key.name.upper()
        except AttributeError:
            name = str(key).upper()
        return _EC_CODES.get(f"KEY_{name}")

    def _on_press(self, key):
        code = self._key_code(key)
        if code is None:
            return
        from_code = code in self._target_codes
        if from_code:
            self._pressed.add(code)
        if self._any_mode:
            if from_code and not self._triggered:
                self._triggered = True
                self.on_event(HotkeyEvent("press"))
        else:
            if self._pressed == self._target_codes and not self._triggered:
                self._triggered = True
                self.on_event(HotkeyEvent("press"))

    def _on_release(self, key):
        code = self._key_code(key)
        if code is None:
            return
        from_code = code in self._target_codes
        if code in self._pressed:
            self._pressed.discard(code)
        if self._any_mode:
            if from_code and self._triggered and not self._pressed & self._target_codes:
                self._triggered = False
                self.on_event(HotkeyEvent("release"))
        else:
            if code in self._target_codes and self._triggered:
                self._triggered = False
                self.on_event(HotkeyEvent("release"))


def create_hotkey_listener(device: str, key_combo: str, on_event):
    if device == "auto":
        try:
            devs = evdev.list_devices()
            if devs:
                logger.info("Using evdev hotkey listener (input devices accessible)")
                return EvdevHotkeyListener(key_combo, on_event)
        except Exception:
            pass
        logger.info("evdev not available, using pynput fallback")
        return PynputHotkeyListener(key_combo, on_event)
    if device == "evdev":
        return EvdevHotkeyListener(key_combo, on_event)
    if device == "pynput":
        return PynputHotkeyListener(key_combo, on_event)
    raise ValueError(f"Unsupported hotkey device: {device}")
