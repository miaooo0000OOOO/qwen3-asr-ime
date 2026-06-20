from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class RecognizedText:
    type: Literal["recognized"] = "recognized"
    text: str = ""
    confidence: float | None = None
    error: str | None = None

    def to_json(self) -> str:
        return json.dumps({
            "type": self.type,
            "text": self.text,
            "confidence": self.confidence,
            "error": self.error,
        })

    @classmethod
    def from_dict(cls, data: dict) -> "RecognizedText":
        return cls(
            type=data.get("type", "recognized"),
            text=data.get("text", ""),
            confidence=data.get("confidence"),
            error=data.get("error"),
        )


@dataclass(frozen=True, slots=True)
class StateUpdate:
    type: Literal["state"] = "state"
    state: Literal["idle", "recording", "recognizing", "error"] = "idle"
    message: str | None = None

    def to_json(self) -> str:
        return json.dumps({
            "type": self.type,
            "state": self.state,
            "message": self.message,
        })

    @classmethod
    def from_dict(cls, data: dict) -> "StateUpdate":
        return cls(
            type=data.get("type", "state"),
            state=data.get("state", "idle"),
            message=data.get("message"),
        )


def parse_message(line: str) -> RecognizedText | StateUpdate:
    data = json.loads(line)
    msg_type = data.get("type")
    if msg_type == "recognized":
        return RecognizedText.from_dict(data)
    if msg_type == "state":
        return StateUpdate.from_dict(data)
    raise ValueError(f"Unknown message type: {msg_type}")
