"""Observed agent graphs and budget diagnostics, never execution controls."""
import json
from collections import defaultdict, deque

from .checks import _has_float, _repeated, finding, input_key
from .types import InputError, strict_json

SPAWN_TOOLS = {"spawn_agent", "task", "Task", "Agent"}


def spawn(call, trace):
    if call.tool.rsplit(".", 1)[-1] not in SPAWN_TOOLS or call.success is not True:
        return None
    facts = call.facts.get("spawn") or {}
    recorded = {}
    if call.result:
        try:
            recorded = strict_json(call.result)
        except ValueError:
            recorded = {}
    if not isinstance(recorded, dict):
        recorded = {}
    child = facts.get("agent_id") or recorded.get("agent_id")
    parent = call.agent_id or trace.agent_id
    if facts.get("agent_id") and recorded.get("agent_id") not in (None, facts["agent_id"]):
        raise InputError(f"{trace.source}:{call.line}: conflicting spawned agent IDs")
    if facts.get("parent_agent_id") not in (None, parent):
        raise InputError(f"{trace.source}:{call.line}: conflicting spawn parent")
    if not isinstance(child, str) or not child or parent is None:
        return None
    return parent, child


def duplicate_delegation(trace):
    def key(call):
        facts = call.facts.get("spawn") or {}
        relation = spawn(call, trace)
        state = input_key(call)
        if (relation is None or state is None or call.turn is None
                or call.result_line is None or _has_float(call.arguments)
                or facts.get("independent") is not False
                or facts.get("completed") is not True
                or facts.get("new_information") is not False
                or not facts.get("objective_fingerprint") or not facts.get("result_fingerprint")):
            return None
        return (relation[0], call.tool, call.turn, json.dumps(call.arguments, sort_keys=True),
                state, facts["objective_fingerprint"], facts["result_fingerprint"])
    return _repeated(
        trace, key, "duplicate_delegation", "Delegated the same investigation more than once",
        "Reuse or partition the worker's completed result; retain genuinely independent "
        "investigations and any explicit confirmation requirement.",
        "Equivalent briefs and complete input snapshots; supplied completion/result evidence "
        "explicitly records no new information or independent value.")


def agent_graph(traces):
    parents, children, roots, anchors = {}, defaultdict(set), {}, defaultdict(list)

    def add(scope, agent, parent):
        key = (scope, agent)
        if key in parents and parents[key] is not None and parent is not None and parents[key] != parent:
            raise InputError(f"conflicting parents for agent {agent!r} in task {scope!r}")
        if parent is not None or key not in parents:
            parents[key] = parent
        if parent is not None:
            parents.setdefault((scope, parent), None)
            children[(scope, parent)].add(agent)

    for trace in traces:
        scope = trace.metadata.get("task_id")
        if not scope or not trace.agent_id:
            continue
        add(scope, trace.agent_id, trace.parent_agent_id)
        if trace.metadata.get("is_root") is True:
            if trace.parent_agent_id is not None:
                raise InputError(f"{trace.source}: a root agent cannot have a parent")
            roots[(scope, trace.agent_id)] = trace
        for call in trace.calls:
            relation = spawn(call, trace)
            if relation:
                parent, child = relation
                add(scope, child, parent)
                anchors[(scope, parent)].append((trace, call))
    checked = set()
    for key in parents:
        path, current = set(), key
        while current not in checked:
            if current in path:
                raise InputError(f"agent graph contains a cycle in task {key[0]!r}")
            path.add(current)
            parent = parents.get(current)
            if parent is None:
                break
            current = (current[0], parent)
        checked.update(path)
    for key in roots:
        if parents[key] is not None:
            raise InputError(f"root agent {key[1]!r} has a recorded parent")
    findings, metrics = [], []
    for (scope, root), trace in roots.items():
        queue, depth, members = deque([(root, 0)]), 0, set()
        fan_out = 0
        calls = []
        while queue:
            node, level = queue.popleft()
            members.add(node)
            depth = max(depth, level)
            offspring = children[(scope, node)]
            fan_out = max(fan_out, len(offspring))
            queue.extend((child, level + 1) for child in sorted(offspring))
            calls.extend(call for source, call in anchors[(scope, node)] if source.source == trace.source)
        values = {"max_depth": depth, "max_fan_out": fan_out, "max_spawns": len(members) - 1}
        metrics.append({"task_id": scope, "root_agent_id": root, **values,
                        "coverage": "observed_lower_bound"})
        budget = trace.metadata.get("budgets") or {}
        exceeded = [{"metric": key, "observed": value, "budget": budget[key]}
                    for key, value in values.items() if key in budget and value > budget[key]]
        if exceeded:
            item = finding(
                trace, calls, "agent_budget_exceeded", "Observed agent structure exceeded supplied budgets",
                "Review the supplied depth/fan-out/spawn budgets and any justified exception. "
                "Recommend host-level controls or narrower delegation; do not stop agents.",
                "Explicit task-scoped parent relationships exceed reviewer-supplied limits; "
                "counts are observed lower bounds, not complete compliance guarantees.",
                classification="budget_observation", violations=exceeded, task_id=scope)
            item["safeguards"].append(
                "Preserve justified parallel work; a budget exceedance alone does not prove waste.")
            findings.append(item)
    nodes = [{"task_id": scope, "agent_id": agent, "parent_agent_id": parents[(scope, agent)]}
             for scope, agent in sorted(parents)]
    graph = {"nodes": nodes[:20], "additional_nodes": max(0, len(nodes) - 20),
             "roots": metrics[:10], "additional_roots": max(0, len(metrics) - 10),
             "coverage": "observed_lower_bound"}
    return findings, graph, any(parent is not None for parent in parents.values())
