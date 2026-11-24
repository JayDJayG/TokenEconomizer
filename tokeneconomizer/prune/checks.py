"""Small deterministic checks; unknown purpose or input state never proves waste."""
import json
from datetime import datetime

READ_SEARCH_TOOLS = {"Read", "Grep", "Glob", "view", "rg", "glob", "read_file", "search"}
VERIFY_COVERS = {"source", "tests", "configuration", "dependencies", "environment"}


def finding(trace, calls, pattern, title, replacement, basis, **details):
    return {
        "pattern": pattern, "title": title, "source": trace.source,
        "first_line": min(call.line for call in calls),
        "last_line": max(call.result_line or call.line for call in calls),
        "calls": len(calls), "replacement": replacement, "evidence_basis": basis,
        "safeguards": [
            "Preserve required work, new failures, changed inputs, and user-requested reruns.",
            "This is a recommendation, not a skipped call or an execution control.",
        ],
        **details,
    }


def _has_float(value):
    if isinstance(value, float):
        return True
    if isinstance(value, dict):
        return any(_has_float(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_float(item) for item in value)
    return False


def input_key(call, covers=()):
    state = call.facts.get("input_state") or {}
    if (call.facts.get("required") is not False or state.get("complete") is not True
            or state.get("at") != "call_start" or not state.get("fingerprint")
            or not state.get("scope") or not state.get("captured_at")
            or not set(covers).issubset(state.get("covers", []))):
        return None
    captured = datetime.fromisoformat(state["captured_at"].replace("Z", "+00:00"))
    if call.timestamp:
        started = datetime.fromisoformat(call.timestamp.replace("Z", "+00:00"))
        if (captured.tzinfo is None or started.tzinfo is None or captured > started):
            return None
    return state["fingerprint"], state["scope"], tuple(sorted(state.get("covers", [])))


def duplicate_investigation(trace):
    findings, eligible, previous, group = [], 0, None, None
    for call in trace.calls:
        tool = call.tool.rsplit(".", 1)[-1]
        state = input_key(call)
        if (tool not in READ_SEARCH_TOOLS or call.facts.get("purpose") != "investigation"
                or state is None or call.success is not True or call.result is None
                or call.turn is None or call.result_line is None or _has_float(call.arguments)):
            previous, group = None, None
            continue
        eligible += 1
        key = (call.tool, call.turn, json.dumps(call.arguments, sort_keys=True), call.result, state)
        if previous and previous[0] == key and previous[1].result_line < call.line:
            if group is None:
                group = finding(
                    trace, [previous[1], call], "duplicate_reads_searches",
                    "Repeated investigation returned identical evidence",
                    "Reuse the existing read/search result; repeat after relevant input or "
                    "purpose changes, or when explicitly required.",
                    "Equivalent requests and identical results against supplied complete "
                    "call-start input snapshots; both calls declared non-required.",
                    tool=call.tool)
                findings.append(group)
            else:
                group["calls"] += 1
                group["last_line"] = call.result_line
        else:
            group = None
        previous = (key, call)
    return findings, eligible
