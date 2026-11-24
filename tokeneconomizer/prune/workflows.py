"""Intent-backed workflow observations and optional procedure suggestions."""
from collections import defaultdict
from pathlib import PurePosixPath, PureWindowsPath

from .checks import READ_SEARCH_TOOLS, extend_finding, finding
from .types import epoch


def fragmented_calls(trace):
    findings, eligible, previous, group, rounds = [], 0, None, None, set()
    for call in trace.calls:
        facts = call.facts.get("batch") or {}
        round_trip = call.message_id or facts.get("round_trip_id")
        if (call.tool.rsplit(".", 1)[-1] not in READ_SEARCH_TOOLS or call.success is not True
                or call.result_line is None or call.turn is None or not facts.get("group")
                or facts.get("independent") is not True or facts.get("supported") is not True
                or not round_trip):
            previous, group, rounds = None, None, set()
            continue
        eligible += 1
        key = (call.tool, call.turn, facts["group"])
        if (previous and previous[0] == key and previous[1].result_line < call.line
                and round_trip not in rounds):
            if group is None:
                group = finding(
                    trace, [previous[1], call], "fragmented_calls",
                    "Independent investigation used separate model round trips",
                    "Batch the independent requests using the confirmed host capability; "
                    "preserve dependencies, separate results, and all required work.",
                    "Distinct recorded request/message IDs and serial results; supplied "
                    "evidence marks the calls independent and batching supported.")
                findings.append(group)
            else:
                extend_finding(group, call)
            rounds.add(round_trip)
        else:
            group, rounds = None, {round_trip}
        previous = (key, call)
    return findings, eligible


def scope_drift(trace):
    policy = trace.metadata.get("scope_policy") or {}
    findings, eligible = [], 0
    if policy.get("complete") is not True or not policy.get("id") or not policy.get("allowed_paths"):
        return findings, eligible
    path_type = PureWindowsPath if policy.get("flavour") == "windows" else PurePosixPath
    allowed = [path_type(path) for path in policy["allowed_paths"] + policy.get("exceptions", [])]
    for call in trace.calls:
        resource = call.arguments.get("file_path", call.arguments.get("path"))
        if (call.tool.rsplit(".", 1)[-1] not in {"Read", "view", "read_file"}
                or call.success is not True or not isinstance(resource, str) or not resource
                or call.facts.get("scope_policy_id") != policy["id"]
                or call.facts.get("required") is not False
                or call.facts.get("new_information") is not False):
            continue
        path = path_type(resource)
        if ".." in path.parts or any(part.is_absolute() != path.is_absolute() for part in allowed):
            continue
        eligible += 1
        if any(path.is_relative_to(prefix) for prefix in allowed):
            continue
        findings.append(finding(
            trace, [call], "scope_drift", "Investigation left the supplied task scope",
            "Return to the recorded task scope, or obtain/record a justified exception. "
            "Preserve necessary dependency, failure, security, and correctness investigation.",
            "A known read path falls outside the complete supplied scope and exceptions; "
            "the call is explicitly non-required and contributed no new information.",
            classification="scope_observation", scope_policy_id=policy["id"]))
    return findings, eligible


def repeated_planning(trace):
    if any("Tool event kinds not normalized" in limit for limit in trace.limitations):
        return [], 0
    timeline = [(event.line, event) for event in trace.events]
    timeline.extend((call.line, None) for call in trace.calls)
    timeline.sort(key=lambda entry: entry[0])
    findings, eligible, previous, group = [], 0, None, None
    for _line, event in timeline:
        facts = event.facts.get("planning", {}) if event is not None else {}
        if (event is None or event.role != "assistant" or facts.get("required") is not False
                or facts.get("new_information") is not False or facts.get("inputs_complete") is not True
                or not facts.get("input_fingerprint") or not facts.get("progress_fingerprint")
                or not facts.get("scope") or not event.text or facts.get("at") != "event"
                or epoch(facts.get("captured_at")) is None):
            previous, group = None, None
            continue
        if event.timestamp and (epoch(event.timestamp) is None
                                or epoch(facts["captured_at"]) > epoch(event.timestamp)):
            previous, group = None, None
            continue
        eligible += 1
        key = (event.text, facts["scope"], facts["input_fingerprint"], facts["progress_fingerprint"])
        if previous and previous[0] == key:
            if group is None:
                group = finding(
                    trace, [], "repeated_visible_planning", "Repeated visible planning added no progress",
                    "Reuse the visible plan and proceed to evidence or implementation; revisit "
                    "it when inputs, requirements, or unresolved concerns change.",
                    "Identical visible planning and supplied complete input/progress evidence "
                    "record no new information or requirement; no tool work intervened.")
                group.update(first_line=previous[1].line, last_line=event.line, calls=0,
                             planning_messages=2)
                group["evidence_sources"] = sorted({source for source in
                    (event.evidence_source, previous[1].evidence_source, trace.evidence_source) if source})
                findings.append(group)
            else:
                group["planning_messages"] += 1
                group["last_line"] = event.line
        else:
            group = None
        previous = (key, event)
    return findings, eligible


def reusable_procedures(trace):
    if any("Tool event kinds not normalized" in limit for limit in trace.limitations):
        return [], 0
    runs, complete, eligible = defaultdict(list), defaultdict(list), 0
    for call in trace.calls:
        facts = call.facts.get("workflow") or {}
        if facts.get("procedure_id") and facts.get("run_id"):
            runs[(facts["procedure_id"], facts["run_id"])].append(call)
    for (procedure, _run), calls in runs.items():
        if (len(calls) < 2 or any(call.success is not True for call in calls)
                or any(call.facts["workflow"].get("reusable") is not True for call in calls)
                or [call.facts["workflow"].get("step") for call in calls] != list(range(1, len(calls) + 1))
                or calls[-1].facts["workflow"].get("final_step") is not True
                or any(call.facts["workflow"].get("final_step") is True for call in calls[:-1])
                or any(not call.facts["workflow"].get("step_key") for call in calls)):
            continue
        eligible += 1
        signature = tuple((call.tool, call.facts["workflow"]["step_key"]) for call in calls)
        complete[(procedure, signature)].append(calls)
    findings = []
    for (procedure, _signature), executions in complete.items():
        if len(executions) < 2:
            continue
        calls = [call for execution in executions for call in execution]
        findings.append(finding(
            trace, calls, "reusable_procedure", "Repeated successful procedure is a skill candidate",
            "Consider a reviewed, parameterized optional skill. Keep per-task judgment, "
            "required checks and approvals; do not automatically persist or execute it.",
            "At least two complete successful runs share explicitly reusable procedure steps; "
            "this is a candidate, not proof that the original work was wasteful.",
            classification="skill_candidate", procedure_id=procedure, executions=len(executions)))
    return findings, eligible
