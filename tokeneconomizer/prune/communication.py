"""Verbatim context forwarding and explicitly addressed no-progress exchanges."""
from collections import deque
from datetime import datetime

from .agents import SPAWN_TOOLS
from .checks import finding
from .types import InputError

MESSAGE_TOOLS = {"write_agent", "send_input"}


def _brief(call):
    key = "message" if call.tool.rsplit(".", 1)[-1] in MESSAGE_TOOLS else "prompt"
    value = call.arguments.get(key)
    return value if isinstance(value, str) and value else None


def context_forwarding(trace):
    findings, groups, eligible = [], {}, 0
    for call in trace.calls:
        facts = call.facts.get("forwarding") or {}
        brief = _brief(call)
        if (call.tool.rsplit(".", 1)[-1] not in SPAWN_TOOLS | MESSAGE_TOOLS
                or call.success is not True or not brief
                or facts.get("sufficient") is not True
                or facts.get("preserves_requirements") is not True or not facts.get("reason")):
            continue
        kind = facts.get("kind")
        if kind == "conversation":
            pieces = {event.line: event.text for event in trace.events
                      if event.text and event.line < call.line}
        elif kind == "tool_output":
            pieces = {previous.result_line: previous.result for previous in trace.calls
                      if previous.success is True and previous.result is not None
                      and previous.result_line is not None and previous.result_line < call.line}
        else:
            continue
        source_lines = facts.get("source_lines") or []
        needed = set(facts.get("needed_source_lines") or [])
        if not source_lines or not needed.issubset(source_lines) or not set(source_lines).issubset(pieces):
            raise InputError(f"{trace.source}:{call.line}: invalid forwarded source references")
        if kind == "conversation" and set(source_lines) != set(pieces):
            continue
        cursor, copied, necessary = 0, 0, 0
        for line in sorted(source_lines):
            text = pieces[line]
            position = brief.find(text, cursor)
            if position < 0:
                break
            cursor = position + len(text)
            size = len(text.encode("utf-8"))
            copied += size
            if line in needed:
                necessary += size
        else:
            eligible += 1
            if copied <= necessary:
                continue
            key = (kind, tuple(sorted(source_lines)), tuple(sorted(needed)))
            group = groups.get(key)
            if group is None:
                group = finding(
                    trace, [call], "context_amplification",
                    "Forwarded context beyond the declared worker need",
                    "Send a bounded task-specific brief and required safeguards, with references "
                    "to full evidence. Keep full context when the worker genuinely needs it.",
                    "The brief contains the referenced text verbatim and in order; supplied "
                    "task evidence declares the smaller selection sufficient and requirement-preserving.",
                    context_kind=kind, avoidable_context_bytes=copied - necessary)
                groups[key] = group
                findings.append(group)
            else:
                group["calls"] += 1
                group["last_line"] = call.result_line or call.line
                group["avoidable_context_bytes"] += copied - necessary
    return findings, eligible


def _message(call, trace):
    tool = call.tool.rsplit(".", 1)[-1]
    facts = call.facts.get("communication") or {}
    if (tool not in MESSAGE_TOOLS or call.success is not True or call.turn is None
            or call.result_line is None or facts.get("required") is not False
            or facts.get("new_information") is not False
            or not facts.get("conversation_id") or not facts.get("progress_fingerprint")):
        return None
    receiver = call.arguments.get("agent_id" if tool == "write_agent" else "id")
    native_sender = call.agent_id or trace.agent_id
    sender = facts.get("sender") or native_sender
    if facts.get("sender") and native_sender and facts["sender"] != native_sender:
        raise InputError(f"{trace.source}:{call.line}: message sender conflicts with native identity")
    if facts.get("receiver") not in (None, receiver):
        raise InputError(f"{trace.source}:{call.line}: message receiver conflicts with arguments")
    text = _brief(call)
    if not isinstance(receiver, str) or not receiver or not sender or not text or sender == receiver:
        return None
    return sender, receiver, text, facts["conversation_id"], facts["progress_fingerprint"]


def _epoch(stamp):
    if stamp is None:
        return None
    value = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    return value.timestamp() if value.tzinfo is not None else None


def ping_pong(traces):
    global_order = len(traces) > 1 and all(
        not trace.limitations
        and all(_epoch(call.timestamp) is not None and _epoch(call.completed_at) is not None
                and _epoch(call.completed_at) >= _epoch(call.timestamp)
                for call in trace.calls)
        and all(_epoch(event.timestamp) is not None for event in trace.events
                if event.role in ("user", "boundary"))
        for trace in traces)
    streams = []
    if global_order:
        timeline = [(_epoch(call.timestamp), trace, call) for trace in traces for call in trace.calls]
        timeline.extend((_epoch(event.timestamp), trace, None) for trace in traces
                        for event in trace.events if event.role in ("user", "boundary"))
        timeline.sort(key=lambda item: item[0])
        streams.append(timeline)
    else:
        streams.extend([(None, trace, call) for call in trace.calls] for trace in traces)
    findings, eligible = [], 0
    for stream in streams:
        previous, group = deque(maxlen=2), None
        for stamp, trace, call in stream:
            message = _message(call, trace) if call is not None else None
            if message is None:
                previous.clear()
                group = None
                continue
            eligible += 1
            if previous:
                _old_stamp, _old_trace, old_call, _old_message = previous[-1]
                ordered = (_epoch(old_call.completed_at) < stamp if global_order else
                           old_call.result_line < call.line and old_call.turn == call.turn)
                if not ordered:
                    previous.clear()
                    group = None
            if len(previous) == 2:
                first = previous[0]
                second = previous[1]
                a, b = first[3], second[3]
                repeats = (message == a and b[0:2] == (a[1], a[0]) and b[3:] == a[3:])
                if repeats:
                    if group is None:
                        root_trace = first[1]
                        group = finding(
                            root_trace, [first[2]], "agent_ping_pong",
                            "Agent exchanges repeated without recorded progress",
                            "Consolidate the request and stopping condition; wait for substantive "
                            "new evidence. Preserve necessary coordination and required handoffs.",
                            "At least three alternating addressed messages repeat a prior message; "
                            "supplied evidence declares unchanged progress and no information or requirement.",
                            message_evidence=[])
                        group["calls"] = 2
                        group["message_evidence"] = [
                            {"source": entry[1].source, "line": entry[2].line,
                             "evidence_source": entry[2].evidence_source or entry[1].evidence_source}
                            for entry in previous]
                        findings.append(group)
                    group["calls"] += 1
                    if trace.source == group["source"]:
                        group["last_line"] = call.result_line
                    if len(group["message_evidence"]) < 6:
                        group["message_evidence"].append({
                            "source": trace.source, "line": call.line,
                            "evidence_source": call.evidence_source or trace.evidence_source})
                    group["additional_messages"] = max(0, group["calls"] - 6)
                else:
                    group = None
            previous.append((stamp, trace, call, message))
    return findings, eligible, global_order
