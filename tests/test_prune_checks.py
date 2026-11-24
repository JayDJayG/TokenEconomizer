from dataclasses import replace

import pytest

from tokeneconomizer.prune import load_trace, review
from tests.test_prune_review import write_trace


def calls_trace(tmp_path, tool="Read", arguments=None, results=("module", "module")):
    rows = [{"type": "user.message", "data": {"content": "Investigate"}}]
    for i, result in enumerate(results):
        rows.extend([
            {"type": "tool.execution_start", "data": {
                "toolCallId": f"c{i}", "toolName": tool,
                "arguments": arguments or {"file_path": "module.py"}}},
            {"type": "tool.execution_complete", "data": {
                "toolCallId": f"c{i}", "success": True, "result": {"content": result}}},
        ])
    trace = load_trace(write_trace(tmp_path, rows))
    facts = {"purpose": "investigation", "required": False, "input_state": {
        "fingerprint": "recorded-state", "scope": "relevant inputs", "complete": True,
        "at": "call_start", "captured_at": "2026-11-24T12:00:00Z", "covers": ["source"]}}
    return replace(trace, calls=tuple(replace(call, facts={
        **facts, "input_state": dict(facts["input_state"])}) for call in trace.calls))


@pytest.mark.parametrize("tool", ["Read", "Grep", "functions.view", "rg"])
def test_duplicate_investigation_needs_input_and_purpose_evidence(tmp_path, tool):
    trace = calls_trace(tmp_path, tool=tool)
    report = review([trace])
    assert report["findings"][0]["pattern"] == "duplicate_reads_searches"
    assert report["findings"][0]["calls"] == 2
    assert report["findings"][0]["safeguards"]
    bare = replace(trace, calls=tuple(replace(call, facts={}) for call in trace.calls))
    assert review([bare])["findings"] == []
    assert next(c for c in review([bare])["checks"]
                if c["pattern"] == "duplicate_reads_searches")["status"] == "not_assessable"


@pytest.mark.parametrize("change", [
    lambda c: replace(c, facts={**c.facts, "required": True}),
    lambda c: replace(c, facts={**c.facts, "purpose": "verification"}),
    lambda c: replace(c, facts={**c.facts, "input_state": {
        **c.facts["input_state"], "fingerprint": "changed"}}),
    lambda c: replace(c, facts={**c.facts, "input_state": {
        **c.facts["input_state"], "complete": False}}),
    lambda c: replace(c, facts={**c.facts, "input_state": {
        **c.facts["input_state"], "at": "now"}}),
    lambda c: replace(c, arguments={"file_path": "other.py"}),
    lambda c: replace(c, arguments={"limit": 1.0}),
    lambda c: replace(c, success=False),
    lambda c: replace(c, result="new evidence"),
    lambda c: replace(c, turn=2),
    lambda c: replace(c, result_line=None),
    lambda c: replace(c, line=2),
])
def test_required_changed_failed_and_ambiguous_reads_are_preserved(tmp_path, change):
    trace = calls_trace(tmp_path)
    trace = replace(trace, calls=(trace.calls[0], change(trace.calls[1])))
    assert review([trace])["findings"] == []


def test_duplicate_runs_group_and_break_at_intervening_work(tmp_path):
    trace = calls_trace(tmp_path, results=("module",) * 4)
    assert review([trace])["findings"][0]["calls"] == 4
    edit = replace(trace.calls[1], tool="edit", facts={})
    interrupted = replace(trace, calls=(trace.calls[0], edit, trace.calls[2]))
    assert review([interrupted])["findings"] == []


def test_future_snapshot_cannot_prove_historical_inputs(tmp_path):
    trace = calls_trace(tmp_path)
    trace = replace(trace, calls=tuple(replace(
        call, timestamp="2026-11-24T11:00:00Z") for call in trace.calls))
    assert review([trace])["findings"] == []
