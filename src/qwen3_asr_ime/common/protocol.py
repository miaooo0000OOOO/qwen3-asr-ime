"""JSON-line message protocol for daemon IPC clients."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class RecognizedText:
    """Message broadcast when ASR produces new or final text.

    Attributes:
        type: Message discriminator, always ``"recognized"``.
        text: Recognized text content.
        confidence: Optional confidence score.
        error: Optional error message when recognition fails.
    """

    type: Literal["recognized"] = "recognized"
    text: str = ""
    confidence: float | None = None
    error: str | None = None

    def to_json(self) -> str:
        """Serialize the message to a JSON string."""
        return json.dumps(
            {
                "type": self.type,
                "text": self.text,
                "confidence": self.confidence,
                "error": self.error,
            }
        )

    @classmethod
    def from_dict(cls, data: dict) -> "RecognizedText":
        """Deserialize from a JSON-decoded dictionary."""
        msg_type = data.get("type", "recognized")
        if msg_type != "recognized":
            raise ValueError(f"RecognizedText type must be 'recognized', got {msg_type!r}")
        return cls(
            type=msg_type,
            text=data.get("text", ""),
            confidence=data.get("confidence"),
            error=data.get("error"),
        )


@dataclass(frozen=True, slots=True)
class StateUpdate:
    """Message broadcast when the daemon state changes.

    Attributes:
        type: Message discriminator, always ``"state"``.
        state: One of ``"idle"``, ``"recording"``, ``"recognizing"``, ``"error"``.
        message: Optional human-readable status message.
    """

    type: Literal["state"] = "state"
    state: Literal["idle", "recording", "recognizing", "error"] = "idle"
    message: str | None = None

    def to_json(self) -> str:
        """Serialize the message to a JSON string."""
        return json.dumps(
            {
                "type": self.type,
                "state": self.state,
                "message": self.message,
            }
        )

    @classmethod
    def from_dict(cls, data: dict) -> "StateUpdate":
        """Deserialize from a JSON-decoded dictionary."""
        msg_type = data.get("type", "state")
        if msg_type != "state":
            raise ValueError(f"StateUpdate type must be 'state', got {msg_type!r}")
        state = data.get("state", "idle")
        allowed_states = ("idle", "recording", "recognizing", "error")
        if state not in allowed_states:
            raise ValueError(f"Invalid state: {state!r}; must be one of {allowed_states}")
        return cls(
            type=msg_type,
            state=state,
            message=data.get("message"),
        )


@dataclass(frozen=True, slots=True)
class HotkeyCommand:
    """Incoming hotkey command from GNOME Shell extension over IPC."""

    type: Literal["hotkey"] = "hotkey"
    action: Literal["press", "release"] = "press"

    @classmethod
    def from_dict(cls, data: dict) -> "HotkeyCommand":
        """Deserialize from a JSON-decoded dictionary."""
        msg_type = data.get("type", "hotkey")
        if msg_type != "hotkey":
            raise ValueError(f"HotkeyCommand type must be 'hotkey', got {msg_type!r}")
        action = data.get("action", "press")
        if action not in ("press", "release"):
            raise ValueError(f"Invalid action: {action!r}; must be 'press' or 'release'")
        return cls(type=msg_type, action=action)


def parse_message(line: str) -> RecognizedText | StateUpdate | HotkeyCommand:
    """Parse a JSON-line IPC message into the appropriate dataclass.

    Args:
        line: A JSON-encoded string.

    Returns:
        One of ``RecognizedText``, ``StateUpdate``, or ``HotkeyCommand``.

    Raises:
        ValueError: If the line is not a JSON object or has an unknown type.
    """
    data = json.loads(line)
    if not isinstance(data, dict):
        raise ValueError("Message must be a JSON object")
    msg_type = data.get("type")
    if msg_type == "recognized":
        return RecognizedText.from_dict(data)
    if msg_type == "state":
        return StateUpdate.from_dict(data)
    if msg_type == "hotkey":
        return HotkeyCommand.from_dict(data)
    raise ValueError(f"Unknown message type: {msg_type}")
