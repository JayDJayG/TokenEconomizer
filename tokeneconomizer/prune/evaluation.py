"""Offline calibration and controlled comparisons over explicitly named recordings."""
from decimal import Decimal, DecimalException, localcontext, Overflow, Underflow
from pathlib import Path
from time import perf_counter, process_time

from .accounting import _amount, _money_sum
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


def _elapsed(wall, cpu):
    return {"wall_seconds": round(perf_counter() - wall, 6),
            "cpu_seconds": round(process_time() - cpu, 6), "review_model_calls": 0,
            "paid_review_tokens": 0}


def review_case(case, directory):
    wall, cpu = perf_counter(), process_time()
    traces, supported = case_traces(case, directory)
    report = review(traces, notifications_supported=supported)
    return report, traces, _elapsed(wall, cpu)


def _truth(values):
    values = tuple(values)
    return False if False in values else None if None in values else True


def _boolean(value, name):
    if value is not None and type(value) is not bool:
        raise InputError(f"{name} must be a boolean or null")
    return value


def _names(value, name):
    if (not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value)
            or len(set(value)) != len(value)):
        raise InputError(f"{name} must be an explicit list of unique nonempty names")
    return value


def _outcome(case, checks, approvals, requirements_complete):
    outcome = case.get("outcome", {})
    if not isinstance(outcome, dict):
        raise InputError("run outcome must be an object")
    task = _boolean(outcome.get("task_success"), "task_success")
    complete = _boolean(outcome.get("complete"), "outcome completeness")
    recorded = []
    for key in ("checks", "approvals"):
        values = outcome.get(key, {})
        if not isinstance(values, dict):
            raise InputError(f"outcome {key} must be an object")
        for name, value in values.items():
            optional_text(name, f"outcome {key} name")
            _boolean(value, f"outcome {key}")
        recorded.append(values)
    quality = _truth([task, True if complete is True and requirements_complete is True else None,
                      *(recorded[0].get(name) for name in checks),
                      *(recorded[1].get(name) for name in approvals)])
    details = {
        "task_success": task, "requirements_satisfied": quality,
        "failed_checks": [name for name, value in recorded[0].items() if value is False],
        "missing_required_checks": [name for name in checks if recorded[0].get(name) is None],
        "failed_approvals": [name for name, value in recorded[1].items() if value is False],
        "missing_required_approvals": [name for name in approvals if recorded[1].get(name) is None],
    }
    details["additional_outcome_items"] = sum(max(0, len(value) - 10) for value in details.values()
                                             if isinstance(value, list))
    return {key: value[:10] if isinstance(value, list) else value for key, value in details.items()}


def _controls(baseline, candidate, basis):
    fields = ["task_fingerprint", "input_fingerprint", "requirements_fingerprint",
              "environment_fingerprint"]
    if basis == "estimated_usd":
        fields.append("pricing_fingerprint")
    supplied = []
    for case in (baseline, candidate):
        values = case.get("controls", {})
        if not isinstance(values, dict):
            raise InputError("run controls must be an object")
        supplied.append({key: optional_text(values.get(key), key) for key in fields})
    missing = [key for key in fields if any(values[key] is None for values in supplied)]
    changed = [key for key in fields if key not in missing and supplied[0][key] != supplied[1][key]]
    return {"match": False if changed else None if missing else True,
            "missing": missing, "mismatched": changed}


def _register_recordings(traces, run_id, seen):
    for trace in traces:
        identities = [("capture", trace.sha256)]
        for usage in trace.usage:
            if usage.scope == "message":
                identities.append(("message", trace.host, usage.unit_id))
            elif trace.session_id:
                identities.append(("session", trace.host, trace.session_id))
        bill = trace.metadata.get("billing")
        if bill:
            identities.append(("receipt", bill["receipt_id"]))
        for identity in identities:
            if identity in seen and seen[identity] != run_id:
                raise InputError("comparison reuses a recording or billing scope across runs")
            seen[identity] = run_id


def _pricing_consistent(traces):
    signatures = {}
    for trace in traces:
        models = {usage.model for usage in trace.usage}
        if trace.metadata.get("ledger_overhead"):
            models.add(trace.metadata.get("ledger_overhead_model"))
        for model in models:
            quote = trace.metadata.get("pricing", {}).get(model)
            if quote:
                weights = quote.get("weights")
                signature = (_amount(quote["input_usd_per_million"], "price"),
                             tuple(sorted(weights.items())) if weights is not None else None)
                signatures.setdefault(model, set()).add(signature)
    return all(len(values) == 1 for values in signatures.values())


def _sum_usd(values):
    values = tuple(values)
    if any(value is None for value in values):
        return None
    return _money_sum(Decimal(value) for value in values)


def compare(path):
    wall, cpu = perf_counter(), process_time()
    source, manifest = read_manifest(path, "comparison")
    basis = manifest.get("cost_basis")
    if basis not in ("estimated_usd", "reported_billed_usd"):
        raise InputError("comparison needs estimated_usd or reported_billed_usd cost_basis")
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise InputError("comparison requires a nonempty pairs list")
    results, names, run_ids, recordings = [], set(), set(), {}
    amounts = {"baseline": [], "candidate": []}
    qualities, controls, price_checks = [], [], []
    for pair in pairs:
        if not isinstance(pair, dict):
            raise InputError("comparison pair must be an object")
        name = optional_text(pair.get("name"), "pair name")
        if name is None or name in names:
            raise InputError("comparison requires unique pair names")
        names.add(name)
        checks = _names(pair.get("required_checks"), "required_checks")
        approvals = _names(pair.get("required_approvals"), "required_approvals")
        complete = _boolean(pair.get("requirements_complete"), "requirements_complete")
        runs, traces = {}, []
        for side in ("baseline", "candidate"):
            case = pair.get(side)
            if not isinstance(case, dict):
                raise InputError(f"comparison needs an explicit {side} run")
            run_id = optional_text(case.get("run_id"), "run_id")
            if run_id is None or run_id in run_ids:
                raise InputError("comparison requires unique run IDs")
            run_ids.add(run_id)
            outcome = _outcome(case, checks, approvals, complete)
            report, inputs, overhead = review_case(case, source.parent)
            _register_recordings(inputs, run_id, recordings)
            traces.extend(inputs)
            costs = report["costs"]
            amount = costs["total_estimated_usd" if basis == "estimated_usd" else "reported_billed_usd"]
            amounts[side].append(amount)
            qualities.append(outcome["requirements_satisfied"])
            runs[side] = {
                "run_id": run_id, "selected_usd": amount, "outcome": outcome,
                "sources": [trace.source for trace in inputs[:4]],
                "additional_sources": max(0, len(inputs) - 4),
                "cost_notes": costs["notes"][:3],
                "additional_cost_notes": costs["additional_notes"] + max(0, len(costs["notes"]) - 3),
                "scope_complete": costs["scope_complete"],
                "overhead_complete": costs["overhead_complete"], "local_review_overhead": overhead,
            }
        control = _controls(pair["baseline"], pair["candidate"], basis)
        prices = basis != "estimated_usd" or _pricing_consistent(traces)
        controls.append(control["match"])
        price_checks.append(prices)
        results.append({"name": name, **runs, "controls": control, "pricing_consistent": prices})
    try:
        with localcontext() as context:
            context.traps[Overflow] = context.traps[Underflow] = True
            baseline, candidate = (_sum_usd(amounts[side]) for side in ("baseline", "candidate"))
            difference = (_money_sum((baseline, candidate.copy_negate()))
                          if baseline is not None and candidate is not None else None)
            percent = difference * 100 / baseline if difference is not None and baseline > 0 else None
    except DecimalException as exc:
        raise InputError("comparison costs exceed the supported decimal range") from exc
    comparable = (all(value is True for value in controls) and all(price_checks)
                  and difference is not None)
    quality = _truth(qualities)
    useful = (False if quality is False else difference > 0 if comparable and quality is True else None)
    report = {
        "schema_version": 1, "mode": "advisory_comparison", "source": str(source),
        "cost_basis": basis, "comparable": comparable, "quality_preserved": quality,
        "useful_reduction": useful, "pairs": results[:20], "additional_pairs": max(0, len(results) - 20),
        "totals": {"baseline_usd": str(baseline) if baseline is not None else None,
                   "candidate_usd": str(candidate) if candidate is not None else None,
                   "observed_reduction_usd": str(difference) if difference is not None else None,
                   "observed_reduction_percent": str(percent) if percent is not None else None},
        "failed_runs": qualities.count(False), "unknown_outcomes": qualities.count(None),
        "all_supplied_runs_included": True,
        "comparability": {"matched_controls": controls.count(True),
                          "mismatched_controls": controls.count(False),
                          "unknown_controls": controls.count(None),
                          "inconsistent_pricing_pairs": price_checks.count(False),
                          "unpriced_baseline_runs": amounts["baseline"].count(None),
                          "unpriced_candidate_runs": amounts["candidate"].count(None)},
        "limits": ["Controls, outcomes, scope completeness and billing are caller assertions.",
                   "Failed and incomplete runs are retained; cheaper failures are not useful reductions.",
                   "Unknown cost, overlap or overhead is unavailable, not zero; cost bases are never mixed.",
                   "Recorded controlled differences are not causal savings guarantees.",
                   "All supplied runs are retained; omitted failed runs cannot be independently detected.",
                   "Local timers cover manifest processing, input loading and review, not rendering or "
                   "original task time. Paid use of reports needs separate supplied usage."],
    }
    report["local_review_overhead"] = _elapsed(wall, cpu)
    return report
