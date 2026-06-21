#!/usr/bin/env python3
"""Minimal hotkey detection test — isomorphic with hotkey.py logic.

Usage:
    python tools/test_hotkey.py                      # default CTRL, auto device
    python tools/test_hotkey.py <Super>+<Shift>+R    # custom combo
    python tools/test_hotkey.py --device evdev        # force evdev
    python tools/test_hotkey.py --device pynput       # force pynput
    python tools/test_hotkey.py --device auto         # try evdev, then pynput
"""

from __future__ import annotations

import argparse
import sys
import threading
from dataclasses import dataclass
from typing import Literal


# ── Same hardcoded event-code table as hotkey.py ──────────────────────
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


# ── Same HotkeyEvent dataclass ────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class HotkeyEvent:
    action: Literal["press", "release"]


# ── EvdevHotkeyListener — isomorphic with hotkey.py ──────────────────
class EvdevHotkeyListener:
    """Hotkey listener via evdev (Linux input subsystem)."""

    def __init__(self, key_combo: str, on_event):
        import evdev as _evdev

        self._evdev = _evdev
        self.on_event = on_event
        self._triggered = False
        self._pressed: set[int] = set()
        self._target_codes = self._parse_combo(key_combo)
        self._any_mode = len(self._target_codes) <= 2 and "CTRL" in key_combo.upper()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._stop = threading.Event()

    @staticmethod
    def _parse_combo(combo: str) -> set[int]:
        name_map: dict[str, int] = {
            "SUPER": 125,   # KEY_LEFTMETA
            "SHIFT": 42,    # KEY_LEFTSHIFT
            "CTRL": 0,
            "ALT": 56,      # KEY_LEFTALT
        }
        codes: set[int] = set()
        for part in combo.upper().replace("<", "").replace(">", "").split("+"):
            part = part.strip()
            if part == "CTRL":
                codes.add(29)   # KEY_LEFTCTRL
                codes.add(97)   # KEY_RIGHTCTRL
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
            devices = [self._evdev.InputDevice(path) for path in self._evdev.list_devices()]
        except OSError:
            print("Cannot access input devices.", file=sys.stderr)
            return
        if not devices:
            print("No input devices found.", file=sys.stderr)
            return
        try:
            while not self._stop.is_set():
                for dev in devices:
                    try:
                        for event in dev.read():
                            if event.type == 1:  # EV_KEY
                                self._handle(event)
                    except BlockingIOError:
                        continue

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


# ── PynputHotkeyListener — isomorphic with hotkey.py ────────────────
class PynputHotkeyListener:
    """Hotkey listener using ``pynput`` (no special permissions needed)."""

    def __init__(self, key_combo: str, on_event):
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


# ── Fallback logic (isomorphic with create_hotkey_listener) ─────────
def _create_listener(device: str, key_combo: str, on_event):
    if device == "auto":
        try:
            import evdev
            devs = evdev.list_devices()
            if devs:
                print("evdev: input devices accessible, using evdev listener")
                return EvdevHotkeyListener(key_combo, on_event)
        except Exception:
            pass
        print("evdev not available, falling back to pynput")
        return PynputHotkeyListener(key_combo, on_event)
    if device == "evdev":
        return EvdevHotkeyListener(key_combo, on_event)
    if device == "pynput":
        return PynputHotkeyListener(key_combo, on_event)
    raise ValueError(f"Unsupported hotkey device: {device}")


# ── Entry point ──────────────────────────────────────────────────────
def _combo_label(combo: str) -> str:
    """Strip angle brackets for display (<Super> → Super)."""
    return combo.replace("<", "").replace(">", "")


def main():
    parser = argparse.ArgumentParser(description="Minimal hotkey test")
    parser.add_argument(
        "combo", nargs="?", default="CTRL",
        help='Hotkey combo, e.g. CTRL or <Super>+<Shift>+R (default: CTRL)',
    )
    parser.add_argument(
        "--device", choices=["evdev", "pynput", "auto"],
        default="auto", help="Detection backend (default: auto)",
    )
    args = parser.parse_args()
    label = _combo_label(args.combo)

    def _on_event(ev: HotkeyEvent):
        if ev.action == "press":
            print(f"⬇ {label} 按下")
        else:
            print(f"⬆ {label} 松开")

    listener = _create_listener(args.device, args.combo, _on_event)

    print(f"Listening for: {label}  (device: {args.device})")
    print("Press Ctrl+C to exit.")
    listener.start()

    try:
        if hasattr(listener, '_thread'):
            # evdev — join the daemon thread until it stops (or Ctrl+C hits)
            while listener._thread.is_alive():
                listener._thread.join(timeout=1)
        else:
            # pynput — just sleep
            import time
            while True:
                time.sleep(3600)
    except KeyboardInterrupt:
        print("\nExiting...")
        listener.stop()


if __name__ == "__main__":
    main()
