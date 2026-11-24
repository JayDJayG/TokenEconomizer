"""Strict, source-backed transcript review without session discovery or side calls."""
import json
import hashlib
from pathlib import Path
from typing import Any, Iterable

from ..trace import claude_calls, codex_exec_calls, copilot_event_calls
from ..trace.codex import rollout_item_to_exec
from .evidence import coverage, enrich
from .checks import (duplicate_investigation, extend_finding, finding as make_finding,
                     output_overfetch, rank_findings, redundant_verification)
from .agents import agent_graph, duplicate_delegation
from .communication import context_forwarding, ping_pong
from .workflows import fragmented_calls, repeated_planning, reusable_procedures, scope_drift
from .types import InputError, ToolCall, Transcript, strict_json as _json

HOSTS = ("claude", "codex", "copilot")
Record = tuple[int, dict[str, Any], str]


def _mapping(value, source, line, field):
    if not isinstance(value, dict):
        raise InputError(f"{source}:{line}: {field} must be a JSON object")
    return value


def _arguments(value, source, line):
    if isinstance(value, str):
        try:
            value = _json(value)
        except ValueError as exc:
            raise InputError(f"{source}:{line}: invalid tool arguments ({exc})") from exc
    return _mapping(value, source, line, "tool arguments")


def _id(value, source, line):
    if not isinstance(value, str) or not value:
        raise InputError(f"{source}:{line}: tool call ID must be a nonempty string")
    return value


def _turn(turn):
    return 1 if turn is None else turn + 1


def _text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, list) and all(
            isinstance(block, dict) and isinstance(block.get("text"), str)
            for block in value):
        return "\n".join(block["text"] for block in value)
    return None


def _claude(rows, source):
    starts, ends, turn, unsupported = {}, {}, None, set()
    for line, event, _raw in rows:
        kind = event.get("type", "assistant")
        if kind == "system" and event.get("subtype") != "init":
            turn = _turn(turn) if turn is not None else None
        if kind not in ("assistant", "user"):
            continue
        message = _mapping(event.get("message", {}), source, line, "message")
        content = message.get("content", "")
        if content is None:
            continue
        if isinstance(content, str):
            if kind == "user":
                turn = _turn(turn)
            continue
        if not isinstance(content, list):
            raise InputError(f"{source}:{line}: message content must be text or a list")
        blocks = [_mapping(block, source, line, "content block") for block in content]
        if kind == "user" and any(block.get("type") != "tool_result" for block in blocks):
            turn = _turn(turn)
        for block in blocks:
            block_kind = block.get("type", "")
            if (isinstance(block_kind, str) and block_kind not in ("tool_use", "tool_result")
                    and block_kind.endswith(("_tool_use", "_tool_result"))):
                unsupported.add(block_kind)
                turn = _turn(turn) if turn is not None else None
            if block.get("type") == "tool_use":
                cid = _id(block.get("id"), source, line)
                _arguments(block.get("input", {}), source, line)
                if not isinstance(block.get("name"), str) or not block["name"]:
                    raise InputError(f"{source}:{line}: tool name must be a nonempty string")
                starts.setdefault(cid, (line, turn))
            elif block.get("type") == "tool_result":
                cid = _id(block.get("tool_use_id"), source, line)
                if "is_error" in block and not isinstance(block["is_error"], bool):
                    raise InputError(f"{source}:{line}: is_error must be a boolean")
                ends[cid] = (line, _text(block.get("content")),
                             not block.get("is_error", False))
    native = claude_calls([raw for _line, _event, raw in rows])
    if len(native) != len(starts):
        raise InputError(f"{source}: tool calls cannot be aligned with source records")
    calls = []
    for (cid, (line, scope)), (tool, args, _result, _ok, _cwd) in zip(starts.items(), native):
        end, result, success = ends.get(cid, (None, None, None))
        if end is not None and end < line:
            raise InputError(f"{source}:{end}: result precedes its tool start")
        calls.append(ToolCall(cid, tool, _arguments(args, source, line),
                              result, success, line, end, scope))
    orphaned = len(ends.keys() - starts.keys())
    return calls, orphaned, unsupported


def _copilot(rows, source):
    starts, ends, turn, unsupported = {}, {}, None, set()
    for line, event, raw in rows:
        kind = event.get("type", "")
        if kind == "user.message":
            _mapping(event.get("data", {}), source, line, "data")
        if kind == "user.message" or "notification" in kind:
            turn = _turn(turn)
        if kind not in ("tool.execution_start", "tool.execution_complete"):
            if kind.startswith("tool.execution_"):
                unsupported.add(kind)
                turn = _turn(turn) if turn is not None else None
            continue
        data = _mapping(event.get("data"), source, line, "data")
        cid = _id(data.get("toolCallId"), source, line)
        if kind == "tool.execution_start":
            _arguments(data.get("arguments", {}), source, line)
            if not isinstance(data.get("toolName"), str) or not data["toolName"]:
                raise InputError(f"{source}:{line}: tool name must be a nonempty string")
            starts.setdefault(cid, (line, turn, raw))
        else:
            if "success" in data and not isinstance(data["success"], bool):
                raise InputError(f"{source}:{line}: success must be a boolean")
            ends[cid] = (line, raw, data)
    calls = []
    for cid, (line, scope, raw) in starts.items():
        if cid not in ends:
            data = _json(raw)["data"]
            calls.append(ToolCall(cid, data["toolName"],
                                  _arguments(data.get("arguments", {}), source, line),
                                  None, None, line, None, scope))
            continue
        end, result_raw, data = ends[cid]
        if end < line:
            raise InputError(f"{source}:{end}: result precedes its tool start")
        native = copilot_event_calls([raw, result_raw])
        if len(native) != 1:
            raise InputError(f"{source}:{end}: tool result cannot be normalized")
        tool, args, _result, success = native[0]
        if data.get("error"):
            success = False
        result = data.get("result")
        if isinstance(result, dict):
            result = result.get("content")
        calls.append(ToolCall(cid, tool, _arguments(args, source, line),
                              _text(result), success, line, end, scope))
    return calls, len(ends.keys() - starts.keys()), unsupported


def _codex(rows, source):
    calls, pending, seen, turn, orphaned = [], {}, set(), None, 0
    unsupported = set()
    task_starts = any(event.get("type") == "event_msg"
                      and isinstance(event.get("payload"), dict)
                      and event["payload"].get("type") == "task_started"
                      for _line, event, _raw in rows)
    for line, event, _raw in rows:
        kind = event.get("type")
        payload = event.get("payload", {})
        if kind in ("event_msg", "response_item", "turn_context"):
            payload = _mapping(payload, source, line, "payload")
        if (kind in ("turn.started", "compacted") or kind == "event_msg"
                and payload.get("type") in ("task_started", "user_message")
                or kind == "turn_context" and not task_starts
                or kind == "response_item" and payload.get("type") == "message"
                and payload.get("role") == "user"):
            turn = _turn(turn)
        if kind == "response_item" and payload.get("type") in (
                "custom_tool_call", "custom_tool_call_output"):
            unsupported.add(payload["type"])
            turn = _turn(turn) if turn is not None else None
        if kind == "response_item" and payload.get("type") == "function_call":
            cid = _id(payload.get("call_id"), source, line)
            args = _arguments(payload.get("arguments", {}), source, line)
            name = payload.get("name")
            if not isinstance(name, str) or not name:
                raise InputError(f"{source}:{line}: tool name must be a nonempty string")
            pending.setdefault(cid, (line, turn, name, args))
        elif kind == "response_item" and payload.get("type") == "function_call_output":
            cid = _id(payload.get("call_id"), source, line)
            if cid in seen:
                continue
            seen.add(cid)
            start = pending.pop(cid, None)
            if start is None:
                orphaned += 1
            else:
                begin, scope, name, args = start
                calls.append(ToolCall(cid, name, args, _text(payload.get("output")),
                                      None, begin, line, scope))
        else:
            item = None
            if kind == "item.completed":
                item = _mapping(event.get("item"), source, line, "item")
            elif kind == "event_msg" and payload.get("type") == "item_completed":
                item = rollout_item_to_exec(
                    _mapping(payload.get("item"), source, line, "item"))
            if item is None:
                continue
            cid = item.get("id")
            if cid is not None:
                _id(cid, source, line)
                if cid in seen:
                    continue
                seen.add(cid)
            if item.get("type") == "command_execution":
                if not isinstance(item.get("command", ""), str):
                    raise InputError(f"{source}:{line}: command must be text")
                if not isinstance(item.get("aggregated_output", ""), str):
                    raise InputError(f"{source}:{line}: command output must be text")
            if item.get("type") == "file_change":
                changes = item.get("changes", [])
                if not isinstance(changes, list) or any(
                        not isinstance(change, dict) for change in changes):
                    raise InputError(f"{source}:{line}: changes must be a list of objects")
            native = codex_exec_calls([json.dumps({"type": "item.completed", "item": item})])
            if not native and item.get("type") not in ("agent_message", "reasoning", "todo_list"):
                unsupported.add(str(item.get("type") or "unknown item"))
                turn = _turn(turn) if turn is not None else None
            for tool, args, result, success in native:
                calls.append(ToolCall(cid, tool, args, result, success, line, line, turn))
    for cid, (line, scope, name, args) in pending.items():
        calls.append(ToolCall(cid, name, args, None, None, line, None, scope))
    return sorted(calls, key=lambda call: call.line), orphaned, unsupported


def _detect_host(rows, source, requested):
    detected = set()
    for line, event, _raw in rows:
        kind = event.get("type", "")
        if not isinstance(kind, str):
            raise InputError(f"{source}:{line}: event type must be text")
        if kind.startswith(("tool.execution_", "session.", "assistant.", "user.")):
            detected.add("copilot")
        elif kind in ("session_meta", "turn_context", "response_item", "event_msg",
                      "thread.started", "turn.started", "turn.completed", "turn.failed",
                      "item.started", "item.completed"):
            detected.add("codex")
        elif isinstance(event.get("message"), dict):
            detected.add("claude")
    if len(detected) != 1:
        raise InputError(f"{source}: unsupported or mixed transcript format")
    host = detected.pop()
    if requested is not None and requested != host:
        raise InputError(f"{source}: detected {host}, not requested host {requested}")
    return host


def load_trace(path: str | Path, host: str | None = None) -> Transcript:
    if host is not None and host not in HOSTS:
        raise InputError(f"unsupported host: {host}")
    source = str(Path(path).resolve())
    rows: list[Record] = []
    digest = hashlib.sha256()
    try:
        with open(source, encoding="utf-8", newline="") as stream:
            for line, raw in enumerate(stream, 1):
                digest.update(raw.encode("utf-8"))
                if not raw.strip():
                    continue
                normalized = raw.removeprefix("\ufeff") if line == 1 else raw
                try:
                    event = _json(normalized)
                except ValueError as exc:
                    raise InputError(f"{source}:{line}: invalid JSON ({exc})") from exc
                rows.append((line, _mapping(event, source, line, "record"), normalized))
    except UnicodeError as exc:
        raise InputError(f"{source}: transcript must be UTF-8") from exc
    if not rows:
        raise InputError(f"{source}: transcript contains no JSON records")
    detected = _detect_host(rows, source, host)
    calls, orphaned, unsupported = {"claude": _claude, "copilot": _copilot,
                                    "codex": _codex}[detected](rows, source)
    limits = []
    if unsupported:
        limits.append("Tool event kinds not normalized: " + ", ".join(sorted(unsupported))
                      + ". They break polling comparisons.")
    if orphaned:
        limits.append(f"{orphaned} result(s) lack a matching tool start and were not assessed.")
    incomplete = sum(call.result_line is None for call in calls)
    if incomplete:
        limits.append(f"{incomplete} call(s) lack a recorded result; outcome remains unknown.")
    return enrich(rows, calls, detected, source, digest.hexdigest(), limits)


def _poll(call):
    if call.success is False or call.result is None:
        return None
    tool = call.tool.rsplit(".", 1)[-1]
    if tool not in ("read_agent", "TaskOutput"):
        return None
    target_key, wait_key = ("agent_id", "wait") if tool == "read_agent" else ("task_id", "block")
    target = call.arguments.get(target_key)
    if not isinstance(target, str) or not target or call.arguments.get(wait_key) is not False:
        return None
    try:
        result = _json(call.result)
    except ValueError:
        return None
    if (not isinstance(result, dict) or result.get("error") or result.get("is_error")
            or result.get("success") is False):
        return None
    state = result.get("task") if tool == "TaskOutput" else result
    if not isinstance(state, dict):
        return None
    status = state.get("status")
    if status not in ("running", "pending", "queued", "completed"):
        return None
    if target_key in state and state[target_key] != target:
        return None
    signature = json.dumps(call.arguments, sort_keys=True) + "\n" + call.result
    return target, status, signature


def review(traces: Iterable[Transcript],
           notifications_supported: bool = False) -> dict[str, Any]:
    if not isinstance(notifications_supported, bool):
        raise InputError("notification support must be an explicit boolean")
    traces = tuple(traces)
    if not traces:
        raise InputError("at least one transcript is required")
    findings, eligible, busy, duplicate_eligible, output_eligible = [], 0, 0, 0, 0
    verification_eligible = 0
    delegation_eligible = 0
    context_eligible = 0
    workflow_counts = {"fragmented_calls": 0, "scope_drift": 0,
                       "repeated_visible_planning": 0, "reusable_procedure": 0}
    sources = set()
    for trace in traces:
        if trace.source in sources:
            raise InputError(f"duplicate input transcript: {trace.source}")
        sources.add(trace.source)
        duplicates, count = duplicate_investigation(trace)
        findings.extend(duplicates)
        duplicate_eligible += count
        overfetch, count = output_overfetch(trace)
        findings.extend(overfetch)
        output_eligible += count
        verification, count = redundant_verification(trace)
        findings.extend(verification)
        verification_eligible += count
        delegation, count = duplicate_delegation(trace)
        findings.extend(delegation)
        delegation_eligible += count
        context, count = context_forwarding(trace)
        findings.extend(context)
        context_eligible += count
        for pattern, detector in (("fragmented_calls", fragmented_calls),
                                  ("scope_drift", scope_drift),
                                  ("repeated_visible_planning", repeated_planning),
                                  ("reusable_procedure", reusable_procedures)):
            detected, count = detector(trace)
            findings.extend(detected)
            workflow_counts[pattern] += count
        previous, group = None, None
        for call in trace.calls:
            poll = _poll(call)
            if poll is None or call.turn is None or call.result_line is None:
                previous, group = None, None
                continue
            eligible += 1
            target, status, signature = poll
            busy += status != "completed"
            supported = status == "completed" or notifications_supported
            key = (call.tool, call.turn, target, status, signature)
            if (supported and previous is not None and previous[0] == key
                    and previous[1].result_line < call.line):
                if group is None:
                    completed = status == "completed"
                    group = make_finding(
                        trace, [previous[1], call],
                        "completed_agent_rereads" if completed else "busy_polling",
                        "Repeated completed-agent retrieval" if completed else
                        "Repeated non-blocking polling returned no new information",
                        (
                            "Reuse the retrieved result; read again only after a new request "
                            "or changed result." if completed else
                            "Use the confirmed completion notifications, then retrieve once; "
                            "keep necessary blocking waits or bounded fallback waiting."),
                        ("recorded completed state and identical results"
                                           if completed else
                                           "identical non-blocking results and user-confirmed "
                                           "notification support"),
                        agent_id=target)
                    findings.append(group)
                else:
                    extend_finding(group, call)
            else:
                group = None
            previous = (key, call)
    graph_findings, graph, graph_assessable = agent_graph(traces)
    findings.extend(graph_findings)
    exchanges, communication_eligible, cross_source_order = ping_pong(traces)
    findings.extend(exchanges)
    findings, coalesced = rank_findings(findings)
    checks = [
        {"pattern": "completed_agent_rereads",
         "status": "assessed" if eligible else "not_assessable",
         "reason": "Requires structured status, an explicit turn, and non-blocking call data."},
        {"pattern": "busy_polling",
         "status": "assessed" if busy and notifications_supported else "not_assessable",
         "reason": ("Notification support was explicitly supplied by the reviewer."
                    if notifications_supported else
                    "Notification support is unknown; running-state repeats are not findings.")},
        {"pattern": "duplicate_reads_searches",
         "status": "assessed" if duplicate_eligible else "not_assessable",
         "reason": "Requires non-required investigation, equivalent requests/results, and "
                   "complete supplied call-start input-state evidence."},
        {"pattern": "output_overfetch",
         "status": "assessed" if output_eligible else "not_assessable",
         "reason": "Requires a successful text result and supplied task evidence that a "
                   "specific line selection is sufficient; size alone is never a finding."},
        {"pattern": "redundant_verification",
         "status": "assessed" if verification_eligible else "not_assessable",
         "reason": "Requires successful deterministic checks, full historical source/test/"
                   "configuration/dependency/environment state, and no required rerun or concern."},
        {"pattern": "agent_graph",
         "status": "assessed" if graph_assessable else "not_assessable",
         "reason": "Only explicit task-scoped parent links are used. Budgets are optional; "
                   "the observed graph is a lower bound, not complete compliance evidence."},
        {"pattern": "duplicate_delegation",
         "status": "assessed" if delegation_eligible else "not_assessable",
         "reason": "Requires equivalent briefs/inputs and explicit completed-result evidence "
                   "of no new information or independent value."},
        {"pattern": "context_amplification",
         "status": "assessed" if context_eligible else "not_assessable",
         "reason": "Requires a verified verbatim copy and explicit smaller-brief sufficiency "
                   "that preserves requirements; context size alone is not evidence."},
        {"pattern": "agent_ping_pong",
         "status": "assessed" if communication_eligible else "not_assessable",
         "reason": "Requires addressed actors, unchanged progress and non-required messages. "
                   + ("Cross-source ordering used complete timezone-aware timestamps."
                      if cross_source_order else
                      "Cross-source ordering is unavailable; only within-source chains are assessed.")},
    ]
    for pattern, reason in (
            ("model_use", "Model/task evidence requires further checks."),
            ("cost_accounting", "No billed-cost or savings calculation is implemented yet.")):
        checks.append({"pattern": pattern, "status": "not_implemented", "reason": reason})
    for pattern, reason in (
            ("fragmented_calls", "Requires independent calls, distinct model round trips, and confirmed batching."),
            ("scope_drift", "Requires complete lexical task scope and explicit non-required/no-information evidence."),
            ("repeated_visible_planning", "Only visible messages with full input/progress and intent evidence are assessed."),
            ("reusable_procedure", "Only complete successful, explicitly reusable procedure runs are candidates.")):
        checks.append({"pattern": pattern,
                       "status": "assessed" if workflow_counts[pattern] else "not_assessable",
                       "reason": reason})
    return {
        "schema_version": 1,
        "mode": "advisory",
        "sources": [{"path": trace.source, "host": trace.host, "records": trace.records,
                     "sha256": trace.sha256, "session_id": trace.session_id,
                     "tool_calls": len(trace.calls), "evidence": coverage(trace),
                     "limitations": list(trace.limitations)}
                    for trace in traces],
        "findings": findings[:3],
        "additional_findings": max(0, len(findings) - 3),
        "coalesced_findings": coalesced,
        "checks": checks,
        "agent_graph": graph,
        "safeguards": [
            "Preserve necessary investigation, required checks, failure details, and approvals.",
            "No execution controls, model calls, settings changes, or session discovery.",
            "Missing evidence is not a passed check; no cost savings are inferred.",
        ],
    }
