"""Cache-aware estimates over reported scopes; never substitute estimates for billing."""
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
import math

from .. import ledger, pricing
from .checks import extend_finding, finding
from .types import InputError, optional_text, timestamp


def _amount(value, location):
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise InputError(f"{location}: amount must be a finite nonnegative number")
    try:
        amount = Decimal(str(value))
    except InvalidOperation as exc:
        raise InputError(f"{location}: invalid decimal amount") from exc
    if not amount.is_finite() or amount < 0:
        raise InputError(f"{location}: amount must be finite and nonnegative")
    return amount


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def validate_accounting(session, location):
    for key in ("accounting_complete", "overhead_complete", "ledger_overhead_included"):
        if key in session and type(session[key]) is not bool:
            raise InputError(f"{location}: {key} must be a boolean")
    if "usage_role" in session and session["usage_role"] not in ("execution", "worker", "review", "overhead"):
        raise InputError(f"{location}: invalid usage_role")
    optional_text(session.get("ledger_overhead_model"), f"{location}:ledger overhead model")
    overhead = session.get("ledger_overhead", {})
    if not isinstance(overhead, dict):
        raise InputError(f"{location}: ledger_overhead must be an object")
    for key, value in overhead.items():
        if key not in ledger.OVERHEAD_KEYS or not _finite(value):
            raise InputError(f"{location}: invalid ledger overhead field or count")
    quotes = session.get("pricing", {})
    if not isinstance(quotes, dict):
        raise InputError(f"{location}: pricing must be an object")
    for model, quote in quotes.items():
        optional_text(model, location)
        if not isinstance(quote, dict):
            raise InputError(f"{location}: price quote must be an object")
        _amount(quote.get("input_usd_per_million"), location)
        if not optional_text(quote.get("source"), location) or not timestamp(quote.get("as_of"), location):
            raise InputError(f"{location}: price quote requires source and as_of")
        weights = quote.get("weights")
        if weights is not None:
            if (not isinstance(weights, dict) or not {"input", "cache_read", "cache_write", "output"}.issubset(weights)
                    or set(weights) - {"input", "cache_read", "cache_write", "output", "cache_write_1h"}
                    or weights.get("input") != 1):
                raise InputError(f"{location}: quote weights need normalized input and all cache/output rates")
            for value in weights.values():
                if not _finite(value):
                    raise InputError(f"{location}: price weights must be finite nonnegative numbers")
    policy = session.get("model_policy")
    if policy is not None:
        if not isinstance(policy, dict) or not optional_text(policy.get("id"), location):
            raise InputError(f"{location}: model_policy requires an ID")
        preferred = policy.get("preferred_models")
        if not isinstance(preferred, dict):
            raise InputError(f"{location}: preferred_models must be an object")
        for task_class, model in preferred.items():
            optional_text(task_class, location)
            optional_text(model, location)
    billing = session.get("billing")
    if billing is not None:
        if not isinstance(billing, dict) or not optional_text(billing.get("receipt_id"), location):
            raise InputError(f"{location}: billing requires a receipt ID")
        _amount(billing.get("usd"), location)
        for key in ("includes_children", "scope_complete"):
            if type(billing.get(key)) is not bool:
                raise InputError(f"{location}: billing needs explicit {key}")


def model_use(trace):
    policy = trace.metadata.get("model_policy") or {}
    findings, eligible = [], 0
    groups = {}
    for call in trace.calls:
        evidence = call.facts.get("model_use") or {}
        model = evidence.get("billed_model") or call.billed_model
        if evidence.get("billed_model") and call.billed_model and evidence["billed_model"] != call.billed_model:
            raise InputError(f"{trace.source}:{call.line}: billed model conflicts with native message")
        preferred = policy.get("preferred_models", {}).get(evidence.get("task_class"))
        if (not model or not preferred or evidence.get("policy_id") != policy.get("id")
                or evidence.get("substitution_approved") is not True
                or evidence.get("quality_preserved") is not True
                or evidence.get("context_sufficient") is not True):
            continue
        eligible += 1
        if model == preferred:
            continue
        key = (model, preferred, evidence["task_class"])
        if key in groups:
            extend_finding(groups[key], call)
            continue
        item = finding(
            trace, [call], "model_use", "Model choice differed from the supplied task-routing policy",
            "Recommend the approved model for this task class through the host. Validate "
            "quality and cost on controlled runs; do not switch the model automatically.",
            "Billed identity and task class are recorded, with an explicit approved substitution "
            "and preserved quality/context requirements; volume or model names alone are not evidence.",
            classification="model_observation", model=model, preferred_model=preferred,
            task_class=evidence["task_class"])
        groups[key] = item
        findings.append(item)
    return findings, eligible


def _mask(traces, flag, has_data):
    by_actor = {(trace.metadata.get("task_id"), trace.agent_id): trace for trace in traces
                if trace.metadata.get("task_id") and trace.agent_id}
    included, problems = set(), []
    for trace in traces:
        scope, parent = trace.metadata.get("task_id"), trace.parent_agent_id
        if parent and not scope and any(other.agent_id == parent for other in traces):
            problems.append("Related totals lack a task scope; aggregation unavailable.")
        visited = set()
        unknown = False
        while scope and parent and (scope, parent) in by_actor:
            if parent in visited:
                raise InputError("cyclic accounting ancestry")
            visited.add(parent)
            ancestor = by_actor[(scope, parent)]
            if has_data(ancestor):
                value = flag(ancestor)
                if value is True:
                    included.add(trace.source)
                    unknown = False
                    break
                if value is None and has_data(trace):
                    unknown = True
            parent = ancestor.parent_agent_id
        if unknown:
            problems.append("Parent/child totals have unknown inclusion; aggregation unavailable.")
    return included, problems


def _usd(effective, quote):
    return (Decimal(str(effective)) * _amount(quote["input_usd_per_million"], "price")
            / Decimal(1_000_000)) if quote else None


def cost_report(traces):
    for trace in traces:
        validate_accounting(trace.metadata, trace.source)
    included, blockers = _mask(traces, lambda t: t.metadata.get("usage_includes_children"), lambda t: bool(t.usage))
    notes = list(blockers)
    units, seen, effective, dollars, all_priced, all_effective = [], {}, [], [], True, True
    for trace in traces:
        if not trace.usage and trace.source not in included:
            notes.append("A supplied transcript lacks usage; missing spend is not zero.")
            blockers.append(notes[-1])
        for usage in trace.usage:
            owner = ((trace.metadata.get("task_id"), trace.agent_id) if trace.agent_id
                     else (trace.host, trace.session_id or trace.sha256 or trace.source))
            identity = (owner, usage.scope, usage.unit_id)
            data = asdict(usage)
            counts = tuple(data[key] for key in ("model", "input_tokens", "cache_read_input_tokens",
                                                "cache_creation_input_tokens", "output_tokens",
                                                "cache_creation_1h_input_tokens"))
            duplicate = identity in seen
            if duplicate and seen[identity] != counts:
                notes.append("Conflicting snapshots of one usage scope prevent aggregation.")
                blockers.append(notes[-1])
            seen[identity] = counts
            unit = {"source": trace.source, "role": trace.metadata.get("usage_role", "recorded_session"),
                    **data, "included_in_parent": trace.source in included, "duplicate_scope": duplicate,
                    "effective_input_token_equivalents": None, "estimated_usd": None}
            complete = all(data[key] is not None for key in
                           ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens"))
            quote = trace.metadata.get("pricing", {}).get(usage.model)
            if complete and usage.model:
                try:
                    row = dict(data, model=usage.model)
                    value = (pricing.effective_tokens(row, weights=quote["weights"])
                             if quote and quote.get("weights") else ledger.effective_exec(row))
                    if not math.isfinite(value):
                        raise ValueError("Effective-cost estimate exceeds finite numeric range.")
                except (ValueError, OverflowError) as exc:
                    unit["pricing_limit"] = str(exc)[:240]
                else:
                    unit["effective_input_token_equivalents"] = value
                    unit["price_basis"] = "supplied_weights" if quote and quote.get("weights") else "upstream_snapshot"
                    if quote:
                        unit["quote_source"] = quote["source"]
                        unit["quote_as_of"] = quote["as_of"]
                    if usage.cache_creation_input_tokens and usage.cache_creation_1h_input_tokens is None:
                        unit["pricing_assumption"] = "Unspecified cache-write lifetime uses the selected rate's default."
                    amount = _usd(value, quote)
                    unit["estimated_usd"] = str(amount) if amount is not None else None
                    if not duplicate and trace.source not in included:
                        effective.append((usage.model, value))
                        if amount is not None:
                            dollars.append(amount)
            else:
                unit["pricing_limit"] = "Missing billed model or complete cache/token breakdown."
            if not duplicate and trace.source not in included:
                if unit.get("pricing_limit"):
                    notes.append("A reported usage unit is unpriced: " + unit["pricing_limit"])
                elif unit["estimated_usd"] is None:
                    notes.append("An absolute USD price quote is missing; token equivalents are not dollars.")
            if not duplicate and trace.source not in included and unit["estimated_usd"] is None:
                all_priced = False
            if not duplicate and trace.source not in included and unit["effective_input_token_equivalents"] is None:
                all_effective = False
            units.append(unit)
    overhead_values, overhead_dollars, overhead_known, overhead_records = [], [], True, []
    overhead_coverage = all(trace.metadata.get("overhead_complete") is True for trace in traces)
    for trace in traces:
        metadata = trace.metadata
        if metadata.get("overhead_complete") is not True:
            overhead_known = False
        overhead = ledger.overhead(metadata.get("ledger_overhead", {}))
        if not _finite(overhead):
            raise InputError(f"{trace.source}: summed ledger overhead exceeds finite numeric range")
        if metadata.get("ledger_overhead"):
            overhead_records.append({"source": trace.source, "model": metadata.get("ledger_overhead_model"),
                                     "fields": metadata["ledger_overhead"]})
        if overhead:
            if metadata.get("ledger_overhead_included") is None:
                notes.append("Ledger overhead inclusion is unknown; aggregation unavailable.")
                blockers.append(notes[-1])
                overhead_known = False
            elif metadata["ledger_overhead_included"] is False:
                model = metadata.get("ledger_overhead_model")
                overhead_values.append((model, overhead))
                amount = _usd(overhead, metadata.get("pricing", {}).get(model))
                if amount is None:
                    overhead_known = False
                else:
                    overhead_dollars.append(amount)
    billing_included, billing_notes = _mask(
        traces, lambda t: t.metadata.get("billing", {}).get("includes_children"),
        lambda t: bool(t.metadata.get("billing")))
    receipts, billing_complete = {}, True
    for trace in traces:
        bill = trace.metadata.get("billing")
        if trace.source in billing_included:
            continue
        if not bill or bill.get("scope_complete") is not True:
            billing_complete = False
            continue
        amount = _amount(bill["usd"], "billing")
        identity = bill["receipt_id"]
        if identity in receipts and receipts[identity] != amount:
            raise InputError("conflicting amounts for one billing receipt")
        receipts[identity] = amount
    scope_complete = all(trace.metadata.get("accounting_complete") is True for trace in traces)
    aggregation_safe = not blockers
    usd_execution = sum(dollars, Decimal(0)) if units and all_priced and aggregation_safe else None
    total = (usd_execution + sum(overhead_dollars, Decimal(0))
             if usd_execution is not None and overhead_known and scope_complete else None)
    priced_models = {model for model, _value in effective + overhead_values}
    equivalents = (sum(value for _model, value in effective + overhead_values)
                   if len(priced_models) == 1 and units and all_effective and aggregation_safe and overhead_coverage else None)
    if equivalents is not None and not _finite(equivalents):
        equivalents = None
        notes.append("Aggregated token equivalents exceed finite numeric range.")
    billed = (sum(receipts.values(), Decimal(0))
              if receipts and billing_complete and not billing_notes and scope_complete else None)
    if not scope_complete:
        notes.append("Complete task/worker/review scope was not explicitly attested.")
    if not overhead_known:
        notes.append("Complete overhead and its pricing were not supplied.")
    unique_notes = list(dict.fromkeys(notes + billing_notes))
    return {
        "usage_units": units[:20], "additional_usage_units": max(0, len(units) - 20),
        "execution_estimated_usd": str(usd_execution) if usd_execution is not None else None,
        "total_estimated_usd": str(total) if total is not None else None,
        "reported_billed_usd": str(billed) if billed is not None else None,
        "effective_input_token_equivalents": equivalents,
        "ledger_overhead_equivalents": sum(value for _model, value in overhead_values) if overhead_known else None,
        "reported_ledger_overhead": overhead_records[:20],
        "scope_complete": scope_complete, "overhead_complete": overhead_coverage,
        "overhead_priced": overhead_known,
        "aggregate_available": total is not None,
        "notes": unique_notes[:10], "additional_notes": max(0, len(unique_notes) - 10),
        "basis": "Measured token fields; explicit quotes/upstream ratios are estimates. "
                 "Supplied billing receipts are unverified. No savings inferred.",
        "review_model_calls": 0,
    }
