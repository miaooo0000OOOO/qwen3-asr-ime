from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import evdev


@dataclass(frozen=True, slots=True)
class HotkeyEvent:
    action: Literal["press", "release"]


class EvdevHotkeyListener:
    def __init__(
        self,
        key_combo: str,
        on_event: Callable[[HotkeyEvent], None],
    ):
        self.key_combo = key_combo
        self.on_event = on_event
        self._pressed: set[int] = set()
        self._target_codes = self._parse_combo(key_combo)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._stop = threading.Event()

    @staticmethod
    def _parse_combo(combo: str) -> set[int]:
        import evdev.ecodes as ec
        name_map = {
            "SUPER": ec.ecodes["KEY_LEFTMETA"],
            "SHIFT": ec.ecodes["KEY_LEFTSHIFT"],
            "CTRL": ec.ecodes["KEY_LEFTCTRL"],
            "ALT": ec.ecodes["KEY_LEFTALT"],
        }
        codes: set[int] = set()
        for part in combo.upper().replace("<", "").replace(">", "").split("+"):
            part = part.strip()
            if part in name_map:
                codes.add(name_map[part])
            else:
                codes.add(ec.ecodes.get(f"KEY_{part}", 0))
        return codes

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
        for dev in devices:
            dev.grab()
        try:
            while not self._stop.is_set():
                for dev in devices:
                    try:
                        for event in dev.read():
                            if event.type == evdev.ecodes.EV_KEY:
                                self._handle(event)
                    except BlockingIOError:
                        continue
        finally:
            for dev in devices:
                dev.ungrab()

    def _handle(self, event: evdev.InputEvent) -> None:
        code = event.code
        if event.value == 1:
            self._pressed.add(code)
            if self._pressed == self._target_codes:
                self.on_event(HotkeyEvent("press"))
        elif event.value == 0:
            if code in self._pressed:
                self._pressed.remove(code)
                if code in self._target_codes:
                    self.on_event(HotkeyEvent("release"))


def create_hotkey_listener(device: str, key_combo: str, on_event):
    if device == "evdev":
        return EvdevHotkeyListener(key_combo, on_event)
    raise ValueError(f"Unsupported hotkey device: {device}")
