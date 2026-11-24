"""Native metadata and optional, digest-bound annotations; never discover inputs."""
from dataclasses import replace
from pathlib import Path

from .types import (Event, InputError, Transcript, Usage, count, optional_text,
                    strict_json, timestamp)

TOKEN_KEYS = ("input_tokens", "cache_read_input_tokens",
              "cache_creation_input_tokens", "output_tokens")


def _usage(raw, host, model, unit, line, source, scope):
    if not isinstance(raw, dict):
        raise InputError(f"{source}:{line}: usage must be an object")
    names = {
        "claude": TOKEN_KEYS,
        "copilot": ("inputTokens", "cacheReadTokens", "cacheWriteTokens", "outputTokens"),
        "codex": ("input_tokens", "cached_input_tokens",
                  "cache_write_input_tokens", "output_tokens"),
    }[host]
    values = [count(raw.get(key), f"{source}:{line}:{key}") for key in names]
    if host != "claude" and all(value is not None for value in values[:3]):
        fresh = values[0] - values[1] - values[2]
        if fresh < 0:
            raise InputError(f"{source}:{line}: cache counts exceed total input")
        values[0] = fresh
    elif host != "claude":
        values[0] = None
    split = raw.get("cache_creation")
    if split is not None and not isinstance(split, dict):
        raise InputError(f"{source}:{line}: cache_creation must be an object")
    hour = count((split or {}).get("ephemeral_1h_input_tokens"),
                 f"{source}:{line}:ephemeral_1h_input_tokens")
    if hour is not None and (values[2] is None or hour > values[2]):
        raise InputError(f"{source}:{line}: 1-hour cache writes exceed total cache writes")
    return Usage(unit, line, model, scope, *values, hour)


def enrich(rows, calls, host, source, digest, limitations):
    native, usage, events = {}, {}, []
    session_id = agent_id = model = None
    limits = list(limitations)
    task_starts = any(event.get("type") == "event_msg"
                      and isinstance(event.get("payload"), dict)
                      and event["payload"].get("type") == "task_started"
                      for _line, event, _raw in rows)
    for line, event, _raw in rows:
        location = f"{source}:{line}"
        stamp = timestamp(event.get("timestamp"), location)
        kind = event.get("type")
        message, data, payload = (event.get(key) or {} for key in ("message", "data", "payload"))
        if host == "claude":
            session_id = optional_text(event.get("sessionId", session_id), location)
            agent_id = optional_text(event.get("agentId", agent_id), location)
            if kind not in ("assistant", "user"):
                if kind == "system" and event.get("subtype") != "init":
                    events.append(Event(line, "boundary", "", stamp, agent_id))
                native[line] = (stamp, agent_id, model)
                continue
            current = optional_text(message.get("model"), location)
            if current and current != "<synthetic>":
                model = current
            blocks = message.get("content", "")
            text = blocks if isinstance(blocks, str) else "\n".join(
                b["text"] for b in blocks or []
                if b.get("type") == "text" and isinstance(b.get("text"), str))
            role = kind
            if kind == "user" and not isinstance(blocks, str) and not any(
                    block.get("type") != "tool_result" for block in blocks or []):
                role = ""
            if message.get("usage") is not None and kind == "assistant" and current != "<synthetic>":
                mid = optional_text(message.get("id"), location)
                if mid is None:
                    limits.append(f"Usage at line {line} lacks a message ID; not counted.")
                else:
                    usage[mid] = _usage(message["usage"], host, current, mid, line, source,
                                        "message")
        elif host == "copilot":
            text, role = "", ""
            if kind in ("session.start", "user.message", "assistant.message", "session.shutdown"):
                data = _object(data, location)
            if kind == "session.start":
                session_id = optional_text(data.get("sessionId"), location)
                model = optional_text(data.get("selectedModel"), location)
            if kind in ("user.message", "assistant.message"):
                role = kind.split(".", 1)[0]
                text = data.get("content", "")
                current = optional_text(data.get("model"), location)
                if current:
                    model = current
            if "notification" in kind:
                role = "boundary"
            if kind == "session.shutdown":
                metrics = data.get("modelMetrics", {})
                if not isinstance(metrics, dict):
                    raise InputError(f"{location}: modelMetrics must be an object")
                for mid, metric in metrics.items():
                    optional_text(mid, location)
                    if not isinstance(metric, dict):
                        raise InputError(f"{location}: model metric must be an object")
                    if metric.get("usage") is not None:
                        usage[mid] = _usage(metric["usage"], host, mid, mid, line, source,
                                            "session")
        else:
            text, role = "", ""
            if kind in ("session_meta", "turn_context", "response_item", "event_msg"):
                payload = _object(payload, location)
            if kind == "thread.started":
                session_id = optional_text(event.get("thread_id"), location)
            if kind == "session_meta":
                session_id = optional_text(payload.get("id", payload.get("session_id")), location)
            if kind == "turn_context":
                model = optional_text(payload.get("model", model), location)
            if kind == "response_item" and payload.get("type") == "message":
                role = payload.get("role", "")
                blocks = payload.get("content", [])
                if isinstance(blocks, list):
                    text = "\n".join(b["text"] for b in blocks if isinstance(b, dict)
                                     and isinstance(b.get("text"), str)
                                     and b.get("type") in ("input_text", "output_text"))
            if kind == "item.completed" and event.get("item", {}).get("type") == "agent_message":
                role, text = "assistant", event["item"].get("text", "")
            if (kind in ("turn.started", "compacted") or kind == "event_msg"
                    and payload.get("type") in ("task_started", "user_message")
                    or kind == "turn_context" and not task_starts):
                role = "boundary"
            if kind == "event_msg" and payload.get("type") == "token_count":
                info = payload.get("info") or {}
                if not isinstance(info, dict):
                    raise InputError(f"{location}: token-count info must be an object")
                total = info.get("total_token_usage")
                if total is not None:
                    usage["total"] = _usage(total, host, model, "total", line, source, "session")
        if isinstance(text, str) and (text and role == "assistant" or role in ("user", "boundary")):
            events.append(Event(line, role, text, stamp, agent_id))
        native[line] = (stamp, agent_id, model)
    enriched = []
    for call in calls:
        start = native[call.line]
        end = native.get(call.result_line, (None, None, None))
        enriched.append(replace(call, timestamp=start[0], completed_at=end[0],
                                agent_id=start[1], model=start[2]))
    return Transcript(source, host, len(rows), tuple(enriched), tuple(limits), digest,
                      session_id, agent_id, usage=tuple(usage.values()), events=tuple(events))


def _object(value, location):
    if not isinstance(value, dict):
        raise InputError(f"{location}: expected an object")
    return value


def validate_facts(facts, location):
    facts = _object(facts, location)
    for name in ("purpose", "scope"):
        if name in facts:
            optional_text(facts[name], f"{location}:{name}")
    if "required" in facts and type(facts["required"]) is not bool:
        raise InputError(f"{location}: required must be a boolean")
    forwarding = facts.get("forwarding")
    if forwarding is not None:
        forwarding = _object(forwarding, f"{location}:forwarding")
        if forwarding.get("kind") not in ("conversation", "tool_output"):
            raise InputError(f"{location}: forwarding kind must be conversation or tool_output")
        for key in ("sufficient", "preserves_requirements"):
            if type(forwarding.get(key)) is not bool:
                raise InputError(f"{location}: forwarding {key} must be a boolean")
        optional_text(forwarding.get("reason"), f"{location}:forwarding reason")
        for key in ("source_lines", "needed_source_lines"):
            values = forwarding.get(key)
            if (not isinstance(values, list) or any(type(n) is not int or n < 1 for n in values)
                    or len(set(values)) != len(values)):
                raise InputError(f"{location}: forwarding {key} must be unique positive indexes")
    communication = facts.get("communication")
    if communication is not None:
        communication = _object(communication, f"{location}:communication")
        for key in ("sender", "receiver", "conversation_id", "progress_fingerprint"):
            if key in communication:
                optional_text(communication[key], f"{location}:{key}")
        for key in ("required", "new_information"):
            if key in communication and type(communication[key]) is not bool:
                raise InputError(f"{location}: communication {key} must be a boolean")
    spawning = facts.get("spawn")
    if spawning is not None:
        spawning = _object(spawning, f"{location}:spawn")
        for key in ("agent_id", "parent_agent_id", "objective_fingerprint", "result_fingerprint"):
            if key in spawning:
                optional_text(spawning[key], f"{location}:{key}")
        for key in ("independent", "completed", "new_information"):
            if key in spawning and type(spawning[key]) is not bool:
                raise InputError(f"{location}: spawn {key} must be a boolean")
    verification = facts.get("verification")
    if verification is not None:
        verification = _object(verification, f"{location}:verification")
        for key in ("deterministic", "unresolved_concerns"):
            if key in verification and type(verification[key]) is not bool:
                raise InputError(f"{location}: verification {key} must be a boolean")
    need = facts.get("output_need")
    if need is not None:
        need = _object(need, f"{location}:output_need")
        if type(need.get("sufficient")) is not bool:
            raise InputError(f"{location}: output sufficiency must be a boolean")
        optional_text(need.get("reason"), f"{location}:output reason")
        selected = need.get("selected_lines")
        if (not isinstance(selected, list) or not selected
                or any(type(n) is not int or n < 1 for n in selected)
                or len(set(selected)) != len(selected)):
            raise InputError(f"{location}: selected_lines must be unique positive integers")
    state = facts.get("input_state")
    if state is not None:
        state = _object(state, f"{location}:input_state")
        optional_text(state.get("fingerprint"), f"{location}:fingerprint")
        optional_text(state.get("scope"), f"{location}:scope")
        timestamp(state.get("captured_at"), f"{location}:captured_at")
        if "at" in state and state["at"] != "call_start":
            raise InputError(f"{location}: input-state at must be call_start")
        if "complete" in state and type(state["complete"]) is not bool:
            raise InputError(f"{location}: complete must be a boolean")
        covers = state.get("covers", [])
        if not isinstance(covers, list) or any(not isinstance(k, str) or not k for k in covers):
            raise InputError(f"{location}: covers must be a list of nonempty strings")
    return facts


def apply_evidence(traces, path):
    """Attach only annotations for already supplied sources with matching physical digests."""
    traces = tuple(traces)
    source = str(Path(path).resolve())
    try:
        with open(source, encoding="utf-8-sig") as stream:
            evidence = strict_json(stream.read())
    except UnicodeError as exc:
        raise InputError(f"{source}: evidence must be UTF-8") from exc
    except ValueError as exc:
        raise InputError(f"{source}: invalid evidence JSON ({exc})") from exc
    evidence = _object(evidence, source)
    if evidence.get("schema_version") != 1 or type(evidence.get("schema_version")) is not int:
        raise InputError(f"{source}: evidence schema_version must be 1")
    entries = evidence.get("sources")
    if not isinstance(entries, list):
        raise InputError(f"{source}: evidence sources must be a list")
    by_path = {trace.source: trace for trace in traces}
    seen = set()
    for entry in entries:
        entry = _object(entry, source)
        name = optional_text(entry.get("path"), source)
        if name is None:
            raise InputError(f"{source}: each evidence source requires a path")
        resolved = str((Path(source).parent / name).resolve())
        if resolved not in by_path or resolved in seen:
            raise InputError(f"{source}: evidence source is not a unique supplied transcript: {name}")
        seen.add(resolved)
        trace = by_path[resolved]
        if entry.get("sha256") != trace.sha256:
            raise InputError(f"{source}: transcript digest mismatch: {name}")
        session = _object(entry.get("session", {}), source)
        if "task_id" in session:
            optional_text(session["task_id"], f"{source}:task_id")
        if "is_root" in session and type(session["is_root"]) is not bool:
            raise InputError(f"{source}: is_root must be a boolean")
        budget = session.get("budgets")
        if budget is not None:
            budget = _object(budget, f"{source}:budgets")
            for key, value in budget.items():
                if key not in ("max_depth", "max_fan_out", "max_spawns"):
                    raise InputError(f"{source}: unknown agent budget {key}")
                if type(value) is not int or value < 0:
                    raise InputError(f"{source}: agent budgets must be nonnegative integers")
        for name in ("agent_id", "parent_agent_id"):
            if name in session:
                optional_text(session[name], f"{source}:{name}")
                native = getattr(trace, name)
                if native is not None and session[name] != native:
                    raise InputError(f"{source}: annotation conflicts with native {name}")
        if "usage_includes_children" in session and type(session["usage_includes_children"]) is not bool:
            raise InputError(f"{source}: usage_includes_children must be a boolean")
        annotated, marked = list(trace.calls), set()
        entries_calls = entry.get("calls", [])
        if not isinstance(entries_calls, list):
            raise InputError(f"{source}: calls must be a list")
        for item in entries_calls:
            item = _object(item, source)
            matches = [i for i, call in enumerate(annotated)
                       if type(item.get("line")) is int and call.line == item["line"]
                       and call.tool == item.get("tool") and call.call_id == item.get("call_id")]
            if len(matches) != 1 or matches[0] in marked:
                raise InputError(f"{source}: annotation must identify one unique call")
            index = matches[0]
            marked.add(index)
            annotated[index] = replace(annotated[index],
                                       facts=validate_facts(item.get("facts"), source),
                                       evidence_source=source)
        by_path[resolved] = replace(trace, calls=tuple(annotated), metadata=session,
                                    agent_id=session.get("agent_id", trace.agent_id),
                                    parent_agent_id=session.get("parent_agent_id",
                                                                trace.parent_agent_id),
                                    evidence_source=source)
    return tuple(by_path[trace.source] for trace in traces)


def coverage(trace):
    return {
        "timestamps": sum(call.timestamp is not None for call in trace.calls),
        "agent_identity": trace.agent_id is not None,
        "agent_ancestry": trace.parent_agent_id is not None,
        "input_states": sum(call.facts.get("input_state") is not None for call in trace.calls),
        "model_identity": sum(call.model is not None for call in trace.calls),
        "usage_units": len(trace.usage),
        "visible_messages": len(trace.events),
        "annotations": bool(trace.metadata or any(call.facts for call in trace.calls)),
    }
