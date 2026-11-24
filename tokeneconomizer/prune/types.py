"""Evidence types and strict JSON shared by the read-only review paths."""
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


class InputError(ValueError):
    pass


def _object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise InputError("duplicate JSON object key")
        obj[key] = value
    return obj


def _constant(_value):
    raise InputError("non-finite JSON number")


def _number(value):
    number = float(value)
    if not math.isfinite(number):
        raise InputError("JSON number exceeds finite float range")
    return number


def strict_json(text):
    return json.loads(text, object_pairs_hook=_object, parse_constant=_constant,
                      parse_float=_number)


def timestamp(value, location):
    if value is None:
        return None
    if not isinstance(value, str):
        raise InputError(f"{location}: timestamp must be ISO-8601 text")
    try:
        datetime.fromisoformat(value.removesuffix("Z") + ("+00:00" if value.endswith("Z") else ""))
    except ValueError as exc:
        raise InputError(f"{location}: invalid ISO-8601 timestamp") from exc
    return value


def epoch(value):
    if value is None:
        return None
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return stamp.timestamp() if stamp.tzinfo is not None else None


def optional_text(value, location):
    if value is not None and (not isinstance(value, str) or not value):
        raise InputError(f"{location}: expected nonempty text or null")
    return value


def count(value, location):
    if value is not None and (type(value) is not int or value < 0):
        raise InputError(f"{location}: token count must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class ToolCall:
    call_id: str | None
    tool: str
    arguments: dict[str, Any]
    result: str | None
    success: bool | None
    line: int
    result_line: int | None
    turn: int | None
    timestamp: str | None = None
    completed_at: str | None = None
    agent_id: str | None = None
    model: str | None = None
    facts: dict[str, Any] = field(default_factory=dict)
    evidence_source: str | None = None
    message_id: str | None = None
    billed_model: str | None = None


@dataclass(frozen=True)
class Event:
    line: int
    role: str
    text: str
    timestamp: str | None = None
    agent_id: str | None = None
    facts: dict[str, Any] = field(default_factory=dict)
    evidence_source: str | None = None
    message_id: str | None = None


@dataclass(frozen=True)
class Usage:
    unit_id: str
    line: int
    model: str | None
    scope: str
    input_tokens: int | None
    cache_read_input_tokens: int | None
    cache_creation_input_tokens: int | None
    output_tokens: int | None
    cache_creation_1h_input_tokens: int | None = None
    total_input_tokens: int | None = None
    reasoning_output_tokens: int | None = None


@dataclass(frozen=True)
class Transcript:
    source: str
    host: str
    records: int
    calls: tuple[ToolCall, ...]
    limitations: tuple[str, ...]
    sha256: str = ""
    session_id: str | None = None
    agent_id: str | None = None
    parent_agent_id: str | None = None
    usage: tuple[Usage, ...] = ()
    events: tuple[Event, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    evidence_source: str | None = None
