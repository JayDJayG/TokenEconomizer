"""Offline labeled trace calibration. Explicit manifests are inputs, not report discovery."""
from pathlib import Path

from .evidence import apply_evidence
from .review import HOSTS, load_trace, review
from .types import InputError, optional_text, strict_json

PATTERN_CHECKS = {
    "completed_agent_rereads": "completed_agent_rereads", "busy_polling": "busy_polling",
    "duplicate_reads_searches": "duplicate_reads_searches", "output_overfetch": "output_overfetch",
    "redundant_verification": "redundant_verification", "agent_budget_exceeded": "agent_graph",
    "duplicate_delegation": "duplicate_delegation", "context_amplification": "context_amplification",
    "agent_ping_pong": "agent_ping_pong", "fragmented_calls": "fragmented_calls",
    "scope_drift": "scope_drift", "repeated_visible_planning": "repeated_visible_planning",
    "reusable_procedure": "reusable_procedure", "model_use": "model_use",
}


def read_manifest(path, kind):
    source = Path(path).resolve()
    try:
        with open(source, encoding="utf-8-sig") as stream:
            manifest = strict_json(stream.read())
    except UnicodeError as exc:
        raise InputError(f"{source}: manifest must be UTF-8") from exc
    except ValueError as exc:
        raise InputError(f"{source}: invalid manifest JSON ({exc})") from exc
    if (not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != 1 or manifest.get("kind") != kind):
        raise InputError(f"{source}: requires schema_version 1 and kind {kind}")
    return source, manifest


def case_traces(case, directory):
    if not isinstance(case, dict):
        raise InputError("manifest case must be an object")
    paths = case.get("transcripts")
    if not isinstance(paths, list) or not paths or any(not isinstance(p, str) or not p for p in paths):
        raise InputError("each case needs explicitly named transcripts")
    host = case.get("input_host")
    if host is not None and host not in HOSTS:
        raise InputError("unsupported case input_host")
    supported = case.get("notifications_supported", False)
    if type(supported) is not bool:
        raise InputError("case notification support must be a boolean")
    traces = tuple(load_trace(directory / path, host) for path in paths)
    evidence = case.get("evidence")
    if evidence is not None:
        optional_text(evidence, "case evidence")
        traces = apply_evidence(traces, directory / evidence)
    return traces, supported


def calibrate(path):
    source, manifest = read_manifest(path, "calibration")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise InputError("calibration requires a nonempty cases list")
    results, names, passed = [], set(), True
    metrics = {"true_positives": 0, "false_positives": 0, "true_negatives": 0,
               "false_negatives": 0, "abstentions": 0}
    for case in cases:
        if not isinstance(case, dict):
            raise InputError("calibration case must be an object")
        name = optional_text(case.get("name"), "case name")
        labels = case.get("labels")
        if name is None or name in names or not isinstance(labels, dict) or not labels:
            raise InputError("calibration requires unique names and explicit labels")
        names.add(name)
        if any(key not in PATTERN_CHECKS or value not in ("present", "absent", "unassessable")
               for key, value in labels.items()):
            raise InputError("unknown calibration pattern or label")
        traces, notifications = case_traces(case, source.parent)
        report = review(traces, notifications_supported=notifications)
        checks = {check["pattern"]: check["status"] for check in report["checks"]}
        outcomes = []
        for pattern, expected in labels.items():
            detected = report["pattern_counts"].get(pattern, 0) > 0
            assessable = checks[PATTERN_CHECKS[pattern]] == "assessed"
            match = (not assessable and not detected if expected == "unassessable"
                     else assessable and (detected if expected == "present" else not detected))
            passed = passed and match
            if not assessable:
                metrics["abstentions"] += 1
            elif expected != "unassessable":
                key = ("true_positives" if detected and expected == "present" else
                       "false_positives" if detected else
                       "false_negatives" if expected == "present" else "true_negatives")
                metrics[key] += 1
            outcomes.append({"pattern": pattern, "expected": expected, "detected": detected,
                             "assessable": assessable, "match": match})
        results.append({"name": name, "outcomes": outcomes})
    return {"schema_version": 1, "mode": "advisory_calibration", "passed": passed,
            "cases": results[:20], "additional_cases": max(0, len(results) - 20),
            "metrics": metrics, "source": str(source),
            "limits": ["Labels are supplied ground truth, not independently verified.",
                       "Abstentions are not true negatives or evidence of efficiency.",
                       "Reviewer coverage is conservative; matches do not prove complete compliance.",
                       "Offline replay measures detection, not cost savings."]}
