import json
from dataclasses import replace

from tokeneconomizer.prune import load_trace, review
from tokeneconomizer.prune.types import Event
from tokeneconomizer.prune.evidence import apply_evidence
from tests.test_prune_evidence import annotate
from tests.test_prune_checks import calls_trace
from tests.test_prune_review import write_trace


def test_independent_calls_need_distinct_round_trips_and_host_support(tmp_path):
    trace = calls_trace(tmp_path)
    calls = tuple(replace(call, arguments={"file_path": f"{i}.py"}, facts={"batch": {
        "group": "module review", "independent": True, "supported": True,
        "round_trip_id": f"request-{i}"}}) for i, call in enumerate(trace.calls))
    trace = replace(trace, calls=calls)
    assert review([trace])["findings"][0]["pattern"] == "fragmented_calls"
    batched = tuple(replace(call, message_id="already-one-request") for call in calls)
    assert review([replace(trace, calls=batched)])["findings"] == []
    dependent = replace(calls[1], facts={"batch": {**calls[1].facts["batch"], "independent": False}})
    assert review([replace(trace, calls=(calls[0], dependent))])["findings"] == []


def test_scope_observation_requires_complete_policy_and_no_useful_contribution(tmp_path):
    trace = calls_trace(tmp_path, results=("data",))
    call = replace(trace.calls[0], arguments={"file_path": "unrelated\\module.py"},
                   facts={"required": False, "new_information": False, "scope_policy_id": "task"})
    policy = {"id": "task", "complete": True, "flavour": "windows",
              "allowed_paths": ["src"], "exceptions": []}
    trace = replace(trace, calls=(call,), metadata={"scope_policy": policy})
    assert review([trace])["findings"][0]["pattern"] == "scope_drift"
    assert review([replace(trace, metadata={"scope_policy": {
        **policy, "exceptions": ["unrelated"]}})])["findings"] == []
    useful = replace(call, facts={**call.facts, "new_information": True})
    assert review([replace(trace, calls=(useful,))])["findings"] == []
    ambiguous = replace(call, arguments={"file_path": "..\\module.py"})
    assert review([replace(trace, calls=(ambiguous,))])["findings"] == []


def planning_events():
    facts = {"planning": {"scope": "task", "input_fingerprint": "same-inputs",
             "progress_fingerprint": "same-progress", "inputs_complete": True,
             "required": False, "new_information": False, "at": "event",
             "captured_at": "2026-11-24T12:00:00Z"}}
    return (Event(2, "assistant", "Inspect, implement, test", facts=facts),
            Event(4, "assistant", "Inspect, implement, test", facts=facts))


def test_only_visible_no_progress_planning_without_work_is_a_cut(tmp_path):
    trace = replace(calls_trace(tmp_path), calls=(), events=planning_events())
    item = review([trace])["findings"][0]
    assert item["pattern"] == "repeated_visible_planning" and item["planning_messages"] == 2
    changed = replace(trace.events[1], facts={"planning": {
        **trace.events[1].facts["planning"], "progress_fingerprint": "new-evidence"}})
    assert review([replace(trace, events=(trace.events[0], changed))])["findings"] == []
    user = Event(3, "user", "Clarify the plan")
    assert review([replace(trace, events=(trace.events[0], user, trace.events[1]))])["findings"] == []


def test_claude_snapshots_and_hidden_thinking_are_not_new_visible_plans(tmp_path):
    row = {"type": "assistant", "message": {"id": "m", "content": [
        {"type": "text", "text": "Visible plan"}, {"type": "thinking", "thinking": "not inspected"}]}}
    trace = load_trace(write_trace(tmp_path, [row, row]))
    assert len(trace.events) == 1
    assert trace.events[0].text == "Visible plan"
    assert review([trace])["findings"] == []


def test_successful_reusable_procedures_are_candidates_not_automatic_skills(tmp_path):
    trace = calls_trace(tmp_path, results=("ok",) * 4)
    calls = tuple(replace(call, tool=("Read" if i % 2 == 0 else "run_tests"), facts={
        "workflow": {"procedure_id": "review-and-test", "run_id": f"run-{i // 2}",
                     "step": i % 2 + 1, "step_key": "read" if i % 2 == 0 else "test",
                     "reusable": True, "final_step": i % 2 == 1}})
        for i, call in enumerate(trace.calls))
    trace = replace(trace, calls=calls)
    item = review([trace])["findings"][0]
    assert item["pattern"] == "reusable_procedure"
    assert item["classification"] == "skill_candidate" and item["executions"] == 2
    failed = replace(calls[3], success=False)
    assert review([replace(trace, calls=(*calls[:3], failed))])["findings"] == []


def test_overlapping_cuts_are_coalesced_without_hiding_related_replacements(tmp_path):
    trace = calls_trace(tmp_path, results=("needed\nextra\n",) * 2)
    calls = tuple(replace(call, facts={**call.facts, "output_need": {
        "sufficient": True, "reason": "One line suffices", "selected_lines": [1]}})
        for call in trace.calls)
    report = review([replace(trace, calls=calls)])
    assert len(report["findings"]) == 1
    assert report["findings"][0]["pattern"] == "duplicate_reads_searches"
    assert report["findings"][0]["related_patterns"] == ["output_overfetch"]
    assert report["coalesced_findings"] == 2


def test_visible_event_annotations_are_digest_bound_and_report_provenance(tmp_path):
    rows = [{"type": "user", "message": {"content": "Implement"}},
            {"type": "assistant", "message": {"id": "one", "content": "Inspect, implement, test"}},
            {"type": "assistant", "message": {"id": "two", "content": "Inspect, implement, test"}}]
    trace = load_trace(write_trace(tmp_path, rows))
    evidence = annotate(tmp_path, trace)
    data = json.loads(evidence.read_text(encoding="utf-8"))
    data["sources"][0]["events"] = [{"line": line, "role": "assistant",
                                   "facts": planning_events()[0].facts} for line in (2, 3)]
    evidence.write_text(json.dumps(data), encoding="utf-8")
    report = review(apply_evidence([trace], evidence))
    assert report["findings"][0]["pattern"] == "repeated_visible_planning"
    assert report["findings"][0]["evidence_sources"] == [str(evidence.resolve())]


def test_unparsed_work_and_intervening_tool_calls_prevent_planning_cuts(tmp_path):
    base = calls_trace(tmp_path)
    trace = replace(base, calls=(), events=planning_events(),
                    limitations=("Tool event kinds not normalized: external_tool_use",))
    assert review([trace])["findings"] == []
    intervening = replace(base.calls[0], line=3)
    trace = replace(trace, limitations=(), calls=(intervening,))
    assert review([trace])["findings"] == []
