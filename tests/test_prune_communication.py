from dataclasses import replace

import pytest

from tokeneconomizer.prune import load_trace, review
from tokeneconomizer.prune.types import Event, InputError
from tests.test_prune_checks import calls_trace
from tests.test_prune_review import write_trace


def test_whole_visible_conversation_forwarding_needs_sufficiency_and_actual_copy(tmp_path):
    trace = calls_trace(tmp_path, tool="spawn_agent", arguments={
        "prompt": "Question\nLong prior discussion\nInspect module"}, results=("started",))
    call = replace(trace.calls[0], line=4, result_line=5)
    trace = replace(trace, events=(Event(1, "user", "Question"),
                                  Event(2, "assistant", "Long prior discussion")),
                    calls=(replace(call, facts={"forwarding": {
                        "kind": "conversation", "source_lines": [1, 2],
                        "needed_source_lines": [1], "sufficient": True,
                        "preserves_requirements": True, "reason": "Only the question is relevant"}}),))
    item = review([trace])["findings"][0]
    assert item["pattern"] == "context_amplification"
    assert item["avoidable_context_bytes"] == len("Long prior discussion")
    call = replace(trace.calls[0], arguments={"prompt": "A concise new brief"})
    assert review([replace(trace, calls=(call,))])["findings"] == []


def test_raw_tool_output_forwarding_preserves_required_full_context(tmp_path):
    trace = calls_trace(tmp_path, results=("Raw detailed tool output",))
    first = trace.calls[0]
    worker = replace(first, call_id="worker", tool="spawn_agent", line=4, result_line=5,
                     arguments={"prompt": "Raw detailed tool output\nTask"}, result="started",
                     facts={"forwarding": {"kind": "tool_output", "source_lines": [3],
                         "needed_source_lines": [], "sufficient": True,
                         "preserves_requirements": True, "reason": "A reference suffices"}})
    trace = replace(trace, calls=(first, worker))
    assert review([trace])["findings"][0]["context_kind"] == "tool_output"
    worker = replace(worker, facts={"forwarding": {
        **worker.facts["forwarding"], "needed_source_lines": [3]}})
    assert review([replace(trace, calls=(first, worker))])["findings"] == []


def messages_trace(tmp_path):
    rows = [{"type": "user.message", "timestamp": "2026-11-24T12:00:00Z", "data": {}}]
    for i, (receiver, message) in enumerate((("b", "Any update?"), ("a", "No change"), ("b", "Any update?"))):
        rows.extend([
            {"type": "tool.execution_start", "timestamp": f"2026-11-24T12:00:0{i * 2 + 1}Z",
             "data": {"toolCallId": f"c{i}", "toolName": "write_agent",
                      "arguments": {"agent_id": receiver, "message": message}}},
            {"type": "tool.execution_complete", "timestamp": f"2026-11-24T12:00:0{i * 2 + 2}Z",
             "data": {"toolCallId": f"c{i}", "success": True, "result": {"content": "sent"}}},
        ])
    trace = load_trace(write_trace(tmp_path, rows))
    calls = tuple(replace(call, facts={"communication": {
        "sender": "a" if i % 2 == 0 else "b", "required": False, "new_information": False,
        "conversation_id": "conversation", "progress_fingerprint": "unchanged"}})
        for i, call in enumerate(trace.calls))
    return replace(trace, calls=calls)


def test_three_alternating_no_progress_messages_are_a_ping_pong_cut(tmp_path):
    item = review([messages_trace(tmp_path)])["findings"][0]
    assert item["pattern"] == "agent_ping_pong" and item["calls"] == 3
    assert len(item["message_evidence"]) == 3


@pytest.mark.parametrize("change", [
    lambda c: replace(c, facts={"communication": {**c.facts["communication"], "required": True}}),
    lambda c: replace(c, facts={"communication": {**c.facts["communication"], "new_information": True}}),
    lambda c: replace(c, facts={"communication": {**c.facts["communication"], "progress_fingerprint": "new"}}),
    lambda c: replace(c, tool="edit"),
    lambda c: replace(c, turn=2),
    lambda c: replace(c, success=False),
])
def test_useful_messages_required_coordination_and_other_work_break_chains(tmp_path, change):
    trace = messages_trace(tmp_path)
    assert review([replace(trace, calls=(trace.calls[0], change(trace.calls[1]), trace.calls[2]))])["findings"] == []


def test_cross_source_ping_pong_needs_complete_unambiguous_timestamps(tmp_path):
    trace = messages_trace(tmp_path)
    a = replace(trace, agent_id="a", calls=(trace.calls[0], trace.calls[2]))
    b = replace(trace, source="explicit-worker", agent_id="b", calls=(trace.calls[1],))
    assert review([a, b])["findings"][0]["pattern"] == "agent_ping_pong"
    b = replace(b, calls=(replace(b.calls[0], completed_at=None),))
    assert review([a, b])["findings"] == []


def test_conflicting_sender_metadata_fails(tmp_path):
    trace = replace(messages_trace(tmp_path), agent_id="other")
    with pytest.raises(InputError, match="sender conflicts"):
        review([trace])


def test_cross_source_new_prompt_breaks_a_no_progress_chain(tmp_path):
    trace = messages_trace(tmp_path)
    a = replace(trace, agent_id="a", calls=(trace.calls[0], trace.calls[2]),
                events=trace.events + (Event(6, "user", "", "2026-11-24T12:00:04.500Z"),))
    b = replace(trace, source="worker", agent_id="b", calls=(trace.calls[1],))
    assert review([a, b])["findings"] == []


@pytest.mark.parametrize("gap", ["normalization", "clock"])
def test_cross_source_gaps_prevent_unproven_ordering(tmp_path, gap):
    trace = messages_trace(tmp_path)
    a = replace(trace, agent_id="a", calls=(trace.calls[0], trace.calls[2]))
    b = replace(trace, source="worker", agent_id="b", calls=(trace.calls[1],))
    if gap == "normalization":
        b = replace(b, limitations=("Unnormalized tool activity",))
    else:
        b = replace(b, calls=(replace(b.calls[0], completed_at="2026-11-24T11:59:00Z"),))
    assert review([a, b])["findings"] == []
