from dataclasses import replace
from decimal import Decimal

import pytest

from tokeneconomizer.prune import load_trace, review
from tokeneconomizer.prune.accounting import validate_accounting
from tokeneconomizer.prune.types import InputError
from tests.test_prune_checks import calls_trace
from tests.test_prune_review import write_trace


def priced_trace(tmp_path, name="priced.jsonl", message_id="usage"):
    row = {"type": "assistant", "message": {
        "id": message_id, "model": "claude-sonnet-5", "content": "Done",
        "usage": {"input_tokens": 100, "cache_read_input_tokens": 900,
                  "cache_creation_input_tokens": 0, "output_tokens": 10}}}
    trace = load_trace(write_trace(tmp_path, [row], name=name))
    return replace(trace, metadata={
        "accounting_complete": True, "overhead_complete": True,
        "ledger_overhead_included": False,
        "pricing": {"claude-sonnet-5": {"input_usd_per_million": 2,
                    "source": "synthetic test quote", "as_of": "2026-11-24"}}})


def test_cache_aware_estimates_include_explicit_ledger_overhead(tmp_path):
    trace = priced_trace(tmp_path)
    trace = replace(trace, metadata={**trace.metadata, "ledger_overhead": {"optimizer_tokens": 20},
                                     "ledger_overhead_model": "claude-sonnet-5"})
    costs = review([trace])["costs"]
    assert costs["usage_units"][0]["effective_input_token_equivalents"] == 240
    assert costs["effective_input_token_equivalents"] == 260
    assert Decimal(costs["total_estimated_usd"]) == Decimal("0.00052")
    assert costs["reported_billed_usd"] is None and costs["review_model_calls"] == 0


def test_missing_usage_breakdown_prices_or_overhead_never_become_zero(tmp_path):
    trace = priced_trace(tmp_path)
    unpriced = replace(trace, usage=(replace(trace.usage[0], model="unknown-model"),))
    assert review([unpriced])["costs"]["total_estimated_usd"] is None
    missing = replace(trace, usage=(replace(trace.usage[0], cache_read_input_tokens=None),))
    assert review([missing])["costs"]["effective_input_token_equivalents"] is None
    assert review([replace(trace, metadata={})])["costs"]["total_estimated_usd"] is None


@pytest.mark.parametrize("inclusive,expected", [(True, "0.00048"), (False, "0.00096"), (None, None)])
def test_child_usage_is_not_double_counted_and_unknown_overlap_is_unpriced(tmp_path, inclusive, expected):
    root = replace(priced_trace(tmp_path), agent_id="root", metadata={
        **priced_trace(tmp_path).metadata, "task_id": "task", "is_root": True,
        "usage_includes_children": inclusive})
    child = replace(priced_trace(tmp_path, name="child.jsonl", message_id="child"),
                    agent_id="child", parent_agent_id="root", metadata={
                        **root.metadata, "is_root": False, "usage_includes_children": False})
    costs = review([root, child])["costs"]
    value = costs["total_estimated_usd"]
    assert (Decimal(value) if value is not None else None) == (Decimal(expected) if expected else None)


def test_duplicate_receipts_are_distinct_from_estimates_and_not_added_twice(tmp_path):
    trace = priced_trace(tmp_path)
    billing = {"receipt_id": "synthetic:receipt", "usd": "0.0015",
               "includes_children": False, "scope_complete": True}
    trace = replace(trace, metadata={**trace.metadata, "billing": billing})
    copy = replace(trace, source="explicit-copy")
    costs = review([trace, copy])["costs"]
    assert costs["reported_billed_usd"] == "0.0015"
    assert Decimal(costs["total_estimated_usd"]) == Decimal("0.00048")
    assert costs["usage_units"][1]["duplicate_scope"]


def test_mixed_codex_cumulative_usage_does_not_adopt_last_model(tmp_path):
    rows = [{"type": "turn_context", "payload": {"model": model}} for model in ("gpt-6.1-sol", "gpt-5-mini")]
    rows.append({"type": "event_msg", "payload": {"type": "token_count", "info": {
        "total_token_usage": {"input_tokens": 100, "cached_input_tokens": 10,
                             "cache_write_input_tokens": 0, "output_tokens": 5}}}})
    trace = load_trace(write_trace(tmp_path, rows))
    assert trace.usage[0].model is None
    assert review([trace])["costs"]["usage_units"][0]["effective_input_token_equivalents"] is None


def test_model_diagnostics_need_billed_identity_and_approved_quality(tmp_path):
    trace = calls_trace(tmp_path, results=("ok",))
    evidence = {"policy_id": "route", "task_class": "lookup", "billed_model": "large-model",
                "substitution_approved": True, "quality_preserved": True, "context_sufficient": True}
    call = replace(trace.calls[0], facts={"model_use": evidence})
    trace = replace(trace, calls=(call,), metadata={"model_policy": {
        "id": "route", "preferred_models": {"lookup": "approved-model"}}})
    assert review([trace])["findings"][0]["pattern"] == "model_use"
    unsafe = replace(call, facts={"model_use": {**evidence, "quality_preserved": False}})
    assert review([replace(trace, calls=(unsafe,))])["findings"] == []
    selected_only = replace(call, model="large-model", facts={"model_use": {
        key: value for key, value in evidence.items() if key != "billed_model"}})
    assert review([replace(trace, calls=(selected_only,))])["findings"] == []


@pytest.mark.parametrize("session", [
    {"pricing": {"model": {"input_usd_per_million": True}}},
    {"ledger_overhead": {"optimizer_tokens": True}},
    {"ledger_overhead": {"optimizer_tokens": -1}},
    {"billing": {"receipt_id": "receipt", "usd": "NaN"}},
    {"overhead_complete": "true"},
])
def test_invalid_accounting_evidence_fails_explicitly(session):
    with pytest.raises(InputError):
        validate_accounting(session, "evidence")


def test_partial_copilot_usage_retains_measured_grand_input(tmp_path):
    trace = load_trace(write_trace(tmp_path, [{"type": "session.shutdown", "data": {
        "modelMetrics": {"gpt-6.1-sol": {"usage": {"inputTokens": 100, "outputTokens": 5}}}}}]))
    unit = review([trace])["costs"]["usage_units"][0]
    assert unit["total_input_tokens"] == 100
    assert unit["input_tokens"] is None and unit["estimated_usd"] is None


def test_unpriced_scope_cannot_disappear_from_token_equivalent_totals(tmp_path):
    trace = priced_trace(tmp_path)
    trace = replace(trace, usage=(*trace.usage, replace(trace.usage[0], unit_id="other", model="unknown")))
    costs = review([trace])["costs"]
    assert costs["effective_input_token_equivalents"] is None
    assert costs["total_estimated_usd"] is None and costs["notes"]


def test_parent_totals_cover_a_child_even_when_the_child_lacks_usage(tmp_path):
    root = priced_trace(tmp_path)
    root = replace(root, agent_id="root", metadata={**root.metadata, "task_id": "task",
                    "is_root": True, "usage_includes_children": True})
    child = replace(root, source="child", agent_id="child", parent_agent_id="root",
                    usage=(), metadata={**root.metadata, "is_root": False})
    assert Decimal(review([root, child])["costs"]["total_estimated_usd"]) == Decimal("0.00048")


def test_finite_overhead_fields_cannot_overflow_into_a_successful_cost_report(tmp_path):
    trace = priced_trace(tmp_path)
    trace = replace(trace, metadata={**trace.metadata, "ledger_overhead": {
        "optimizer_tokens": 1e308, "spiral_judge_tokens": 1e308}})
    with pytest.raises(InputError, match="summed ledger overhead"):
        review([trace])


def test_repeated_model_policy_mismatches_are_one_bounded_cut(tmp_path):
    trace = calls_trace(tmp_path)
    facts = {"model_use": {"policy_id": "route", "task_class": "lookup",
             "billed_model": "large", "substitution_approved": True,
             "quality_preserved": True, "context_sufficient": True}}
    trace = replace(trace, calls=tuple(replace(call, facts=facts) for call in trace.calls),
                    metadata={"model_policy": {"id": "route", "preferred_models": {"lookup": "approved"}}})
    findings = review([trace])["findings"]
    assert len(findings) == 1 and findings[0]["calls"] == 2
