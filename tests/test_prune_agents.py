import json
from dataclasses import replace

import pytest

from tokeneconomizer.prune import review
from tokeneconomizer.prune.types import InputError
from tests.test_prune_checks import calls_trace


def parent_trace(tmp_path, children=("a", "b"), budget=None):
    trace = calls_trace(tmp_path, tool="spawn_agent", arguments={"prompt": "Inspect module"},
                        results=tuple(json.dumps({"agent_id": child}) for child in children))
    return replace(trace, agent_id="root", metadata={
        "task_id": "task", "is_root": True, "budgets": budget or {}})


def test_budget_observations_use_explicit_parent_links_and_no_universal_cap(tmp_path):
    trace = parent_trace(tmp_path, budget={"max_fan_out": 1, "max_spawns": 1})
    report = review([trace])
    assert report["agent_graph"]["roots"][0]["max_fan_out"] == 2
    item = report["findings"][0]
    assert item["pattern"] == "agent_budget_exceeded"
    assert item["classification"] == "budget_observation"
    assert {v["metric"] for v in item["violations"]} == {"max_fan_out", "max_spawns"}
    assert review([replace(trace, metadata={"task_id": "task", "is_root": True})])["findings"] == []


def test_nested_depth_and_unrelated_task_scopes(tmp_path):
    root = parent_trace(tmp_path, children=("a",), budget={"max_depth": 1})
    child = replace(root, source="explicit-child", agent_id="a", parent_agent_id="root",
                    calls=(replace(root.calls[0], result=json.dumps({"agent_id": "b"})),),
                    metadata={"task_id": "task"})
    other = replace(child, source="other-task", metadata={"task_id": "independent"})
    report = review([root, child, other])
    assert report["agent_graph"]["roots"][0]["max_depth"] == 2
    assert report["agent_graph"]["roots"][0]["max_spawns"] == 2
    assert report["findings"][0]["violations"][0]["observed"] == 2


def test_unknown_relationships_are_not_invented(tmp_path):
    trace = parent_trace(tmp_path)
    trace = replace(trace, metadata={}, agent_id=None)
    report = review([trace])
    assert report["findings"] == [] and report["agent_graph"]["nodes"] == []


@pytest.mark.parametrize("children", [("root",), ("a",)])
def test_cycles_fail_explicitly(tmp_path, children):
    trace = parent_trace(tmp_path, children=children)
    if children == ("a",):
        trace = replace(trace, parent_agent_id="a", metadata={"task_id": "task"})
    with pytest.raises(InputError, match="cycle"):
        review([trace])


def test_conflicting_parents_fail_instead_of_joining_agents(tmp_path):
    root = parent_trace(tmp_path, children=("shared",))
    other = replace(root, source="other-parent", agent_id="other")
    with pytest.raises(InputError, match="conflicting parents"):
        review([root, other])


def test_duplicate_delegation_needs_explicit_no_independent_value(tmp_path):
    trace = parent_trace(tmp_path)
    calls = tuple(replace(call, facts={**call.facts, "spawn": {
        "independent": False, "completed": True, "new_information": False,
        "objective_fingerprint": "same-brief", "result_fingerprint": "same-result"}})
        for call in trace.calls)
    trace = replace(trace, calls=calls)
    assert review([trace])["findings"][0]["pattern"] == "duplicate_delegation"
    independent = replace(calls[1], facts={**calls[1].facts, "spawn": {
        **calls[1].facts["spawn"], "independent": True}})
    assert review([replace(trace, calls=(calls[0], independent))])["findings"] == []


def test_metadata_only_graph_observations_keep_unknown_source_spans(tmp_path):
    root = replace(parent_trace(tmp_path), calls=(), metadata={
        "task_id": "task", "is_root": True, "budgets": {"max_spawns": 0}})
    child = replace(root, source="child", agent_id="a", parent_agent_id="root",
                    metadata={"task_id": "task"})
    report = review([root, child])
    assert report["findings"][0]["first_line"] is None
    assert report["findings"][0]["calls"] == 0


def test_a_declared_root_cannot_also_be_another_agents_child(tmp_path):
    root = parent_trace(tmp_path, children=("other-root",))
    other = replace(root, source="second-root", calls=(), agent_id="other-root")
    with pytest.raises(InputError, match="recorded parent"):
        review([root, other])
