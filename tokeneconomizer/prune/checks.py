"""Small deterministic checks; unknown purpose or input state never proves waste."""
import json
from datetime import datetime
from .types import InputError

READ_SEARCH_TOOLS = {"Read", "Grep", "Glob", "view", "rg", "glob", "read_file", "search"}
VERIFY_TOOLS = {"Bash", "shell", "powershell", "run_tests", "test", "execute_command", "terminal"}
VERIFY_COVERS = {"source", "tests", "configuration", "dependencies", "environment"}


def finding(trace, calls, pattern, title, replacement, basis, **details):
    return {
        "pattern": pattern, "title": title, "source": trace.source,
        "first_line": min((call.line for call in calls), default=None),
        "last_line": max((call.result_line or call.line for call in calls), default=None),
        "calls": len(calls), "replacement": replacement, "evidence_basis": basis,
        "safeguards": [
            "Preserve required work, new failures, changed inputs, and user-requested reruns.",
            "This is a recommendation, not a skipped call or an execution control.",
        ],
        "evidence_sources": sorted({source for source in
                                   [trace.evidence_source, *(call.evidence_source for call in calls)]
                                   if source}),
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


def _request_key(call, state):
    if (state is None or call.success is not True or call.result is None
            or call.turn is None or call.result_line is None or _has_float(call.arguments)):
        return None
    return (call.tool, call.turn, json.dumps(call.arguments, sort_keys=True), call.result, state)


def _repeated(trace, key_for, pattern, title, replacement, basis):
    findings, eligible, previous, group = [], 0, None, None
    for call in trace.calls:
        key = key_for(call)
        if key is None:
            previous, group = None, None
            continue
        eligible += 1
        if previous and previous[0] == key and previous[1].result_line < call.line:
            if group is None:
                group = finding(trace, [previous[1], call], pattern, title, replacement,
                                basis, tool=call.tool)
                findings.append(group)
            else:
                group["calls"] += 1
                group["last_line"] = call.result_line
        else:
            group = None
        previous = (key, call)
    return findings, eligible


def duplicate_investigation(trace):
    def key(call):
        if (call.tool.rsplit(".", 1)[-1] not in READ_SEARCH_TOOLS
                or call.facts.get("purpose") != "investigation"):
            return None
        return _request_key(call, input_key(call))
    return _repeated(
        trace, key, "duplicate_reads_searches",
        "Repeated investigation returned identical evidence",
        "Reuse the existing read/search result; repeat after relevant input or "
        "purpose changes, or when explicitly required.",
        "Equivalent requests and identical results against supplied complete "
        "call-start input snapshots; both calls declared non-required.")


def redundant_verification(trace):
    def key(call):
        evidence = call.facts.get("verification") or {}
        if (call.tool.rsplit(".", 1)[-1] not in VERIFY_TOOLS
                or call.facts.get("purpose") != "verification"
                or evidence.get("deterministic") is not True
                or evidence.get("unresolved_concerns") is not False):
            return None
        return _request_key(call, input_key(call, VERIFY_COVERS))
    return _repeated(
        trace, key, "redundant_verification", "Repeated verification of unchanged inputs",
        "Reuse the passing check for its exact recorded input state. Rerun after a relevant "
        "source, test, configuration, dependency, or environment change, a new concern, "
        "failure, or explicit requirement.",
        "Identical successful deterministic checks, complete supplied call-start snapshots "
        "of all verification inputs, and explicit absence of rerun requirements or concerns.")


def output_overfetch(trace):
    findings, eligible = [], 0
    for call in trace.calls:
        need = call.facts.get("output_need") or {}
        if (call.success is not True or call.result is None
                or need.get("sufficient") is not True or not need.get("reason")
                or call.facts.get("purpose") is None):
            continue
        selected = need.get("selected_lines")
        if not isinstance(selected, list) or not selected:
            continue
        lines = call.result.splitlines(keepends=True)
        if any(type(n) is not int or n < 1 or n > len(lines) for n in selected):
            raise InputError(f"{trace.source}:{call.line}: selected output line is out of range")
        eligible += 1
        original = len(call.result.encode("utf-8"))
        reduced = len("".join(lines[n - 1] for n in sorted(set(selected))).encode("utf-8"))
        if original <= reduced:
            continue
        findings.append(finding(
            trace, [call], "output_overfetch", "Tool output exceeded the declared evidence need",
            "Filter or request only the declared needed lines at the source. Keep the full "
            "artifact available by reference and preserve failures and required full context.",
            "Supplied task evidence declares the selected result lines sufficient; byte "
            "counts compare the actual result with that selection, not estimated tokens.",
            output_bytes=original, needed_output_bytes=reduced,
            avoidable_output_bytes=original - reduced))
    return findings, eligible
