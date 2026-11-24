import json
from decimal import Decimal
from pathlib import Path

import pytest

from tokeneconomizer.prune.cli import main
from tokeneconomizer.prune.evaluation import compare
from tokeneconomizer.prune.types import InputError
from tests.test_prune_evidence import annotate
from tests.test_prune_review import write_trace
from tokeneconomizer.prune import load_trace


def recorded_run(tmp_path, run_id, tokens, *, role="execution", parent=None, bill=None):
    rows = [{"type": "assistant", "sessionId": run_id, "agentId": run_id, "message": {
        "id": f"{run_id}-message", "model": "claude-sonnet-5", "content": "Recorded completion",
        "usage": {"input_tokens": tokens, "cache_read_input_tokens": 0,
                  "cache_creation_input_tokens": 0, "output_tokens": 0}}}]
    path = write_trace(tmp_path, rows, name=f"{run_id}.jsonl")
    trace = load_trace(path)
    metadata = {
        "task_id": parent or run_id, "is_root": parent is None,
        "usage_role": role, "usage_includes_children": False,
        "accounting_complete": True, "overhead_complete": True,
        "ledger_overhead_included": False,
        "pricing": {"claude-sonnet-5": {"input_usd_per_million": 1,
                    "source": "synthetic quote", "as_of": "2026-11-24",
                    "weights": {"input": 1, "cache_read": .1, "cache_write": 1.25, "output": 5}}},
    }
    if parent:
        metadata["parent_agent_id"] = parent
    if bill is not None:
        metadata["billing"] = {"receipt_id": f"{run_id}-receipt", "usd": bill,
                               "includes_children": False, "scope_complete": True}
    evidence = annotate(tmp_path, trace, session=metadata)
    destination = tmp_path / f"{run_id}-evidence.json"
    evidence.replace(destination)
    return {"run_id": run_id, "transcripts": [path.name], "evidence": destination.name,
            "controls": {key: "recorded-control" for key in (
                "task_fingerprint", "input_fingerprint", "requirements_fingerprint",
                "environment_fingerprint", "pricing_fingerprint")},
            "outcome": {"task_success": True, "complete": True,
                        "checks": {"tests": True}, "approvals": {"review": True}}}


def comparison(tmp_path, baseline=None, candidate=None, *, basis="estimated_usd", pairs=None):
    if pairs is None:
        pairs = [{"name": "scenario", "required_checks": ["tests"], "required_approvals": ["review"],
                  "requirements_complete": True,
                  "baseline": baseline or recorded_run(tmp_path, "baseline", 100),
                  "candidate": candidate or recorded_run(tmp_path, "candidate", 50)}]
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps({"schema_version": 1, "kind": "comparison",
                               "cost_basis": basis, "pairs": pairs}), encoding="utf-8")
    return path


def update_evidence(tmp_path, case, change):
    path = tmp_path / case["evidence"]
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data["sources"][0]["session"])
    path.write_text(json.dumps(data), encoding="utf-8")


def test_controlled_comparison_uses_usd_and_measures_local_review(tmp_path, monkeypatch):
    from tokeneconomizer.prune import evaluation

    path = comparison(tmp_path)
    wall, cpu = iter([0, 1, 3, 4, 6, 8]), iter([0, 2, 3, 4, 5, 6])
    monkeypatch.setattr(evaluation, "perf_counter", lambda: next(wall))
    monkeypatch.setattr(evaluation, "process_time", lambda: next(cpu))
    report = compare(path)
    assert report["comparable"] and report["quality_preserved"] and report["useful_reduction"]
    assert Decimal(report["totals"]["observed_reduction_usd"]) == Decimal("0.00005")
    assert Decimal(report["totals"]["observed_reduction_percent"]) == 50
    assert report["pairs"][0]["baseline"]["local_review_overhead"]["wall_seconds"] == 2
    assert report["local_review_overhead"]["wall_seconds"] == 8
    assert report["local_review_overhead"]["cpu_seconds"] == 6


@pytest.mark.parametrize("failure", ["task", "tests", "approval"])
def test_cheaper_failed_runs_are_retained_not_useful_savings(tmp_path, failure, capsys):
    candidate = recorded_run(tmp_path, "candidate", 50)
    outcome = candidate["outcome"]
    if failure == "task":
        outcome["task_success"] = False
    elif failure == "tests":
        outcome["checks"]["tests"] = False
    else:
        outcome["approvals"]["review"] = False
    path = comparison(tmp_path, candidate=candidate)
    assert main(["compare", str(path), "--format", "json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["comparable"] and report["quality_preserved"] is False
    assert report["failed_runs"] == 1 and report["all_supplied_runs_included"]
    assert report["useful_reduction"] is False
    assert Decimal(report["totals"]["candidate_usd"]) == Decimal("0.00005")


@pytest.mark.parametrize("field", ["task_fingerprint", "input_fingerprint",
                                    "requirements_fingerprint", "environment_fingerprint", "pricing_fingerprint"])
def test_changed_controls_prevent_controlled_savings_claims(tmp_path, field):
    candidate = recorded_run(tmp_path, "candidate", 50)
    candidate["controls"][field] = "different"
    report = compare(comparison(tmp_path, candidate=candidate))
    assert not report["comparable"] and report["useful_reduction"] is None
    assert report["pairs"][0]["controls"]["mismatched"] == [field]


def test_missing_outcomes_or_costs_are_unknown_not_success_or_zero(tmp_path):
    candidate = recorded_run(tmp_path, "candidate", 50)
    candidate["outcome"]["checks"].clear()
    update_evidence(tmp_path, candidate, lambda session: session.update(overhead_complete=False))
    report = compare(comparison(tmp_path, candidate=candidate))
    assert not report["comparable"] and report["quality_preserved"] is None
    assert report["unknown_outcomes"] == 1 and report["useful_reduction"] is None
    assert report["totals"]["candidate_usd"] is None


def test_reported_billing_is_an_explicit_never_mixed_basis(tmp_path):
    baseline = recorded_run(tmp_path, "baseline", 100, bill="2.40")
    candidate = recorded_run(tmp_path, "candidate", 50, bill="1.20")
    for case in (baseline, candidate):
        update_evidence(tmp_path, case, lambda session: session.pop("pricing"))
    report = compare(comparison(tmp_path, baseline, candidate, basis="reported_billed_usd"))
    assert report["comparable"] and Decimal(report["totals"]["observed_reduction_usd"]) == Decimal("1.20")
    update_evidence(tmp_path, candidate, lambda session: session.pop("billing"))
    report = compare(comparison(tmp_path, baseline, candidate, basis="reported_billed_usd"))
    assert not report["comparable"] and report["totals"]["candidate_usd"] is None


def test_worker_review_and_legacy_overhead_can_reverse_apparent_savings(tmp_path):
    baseline = recorded_run(tmp_path, "baseline", 100)
    candidate = recorded_run(tmp_path, "candidate", 50)
    worker = recorded_run(tmp_path, "worker", 100, role="worker", parent="candidate")
    paid_review = recorded_run(tmp_path, "reviewer", 10, role="review", parent="candidate")
    candidate["transcripts"] += worker["transcripts"] + paid_review["transcripts"]
    evidence_path = tmp_path / candidate["evidence"]
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    for case in (worker, paid_review):
        evidence["sources"] += json.loads((tmp_path / case["evidence"]).read_text(encoding="utf-8"))["sources"]
    evidence["sources"][0]["session"].update(
        ledger_overhead={"optimizer_tokens": 5}, ledger_overhead_model="claude-sonnet-5")
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    report = compare(comparison(tmp_path, baseline, candidate))
    assert report["comparable"] and report["useful_reduction"] is False
    assert Decimal(report["totals"]["candidate_usd"]) == Decimal("0.000165")
    assert Decimal(report["totals"]["observed_reduction_usd"]) == Decimal("-0.000065")


def test_inconsistent_recorded_prices_are_not_hidden_by_a_matching_price_fingerprint(tmp_path):
    candidate = recorded_run(tmp_path, "candidate", 50)
    update_evidence(tmp_path, candidate, lambda session: session["pricing"]["claude-sonnet-5"].update(
        input_usd_per_million=".5"))
    report = compare(comparison(tmp_path, candidate=candidate))
    assert not report["comparable"] and not report["pairs"][0]["pricing_consistent"]
    assert report["useful_reduction"] is None


def test_mixed_models_are_compared_in_quoted_usd_not_token_equivalents(tmp_path):
    baseline = recorded_run(tmp_path, "baseline", 100)
    candidate = recorded_run(tmp_path, "candidate", 200)
    path = tmp_path / candidate["transcripts"][0]
    row = json.loads(path.read_text(encoding="utf-8"))
    row["message"]["model"] = "fixture-small-model"
    write_trace(tmp_path, [row], name=path.name)
    evidence = tmp_path / candidate["evidence"]
    data = json.loads(evidence.read_text(encoding="utf-8"))
    entry = data["sources"][0]
    entry["sha256"] = load_trace(path).sha256
    quote = entry["session"]["pricing"].pop("claude-sonnet-5")
    quote["input_usd_per_million"] = ".1"
    entry["session"]["pricing"]["fixture-small-model"] = quote
    evidence.write_text(json.dumps(data), encoding="utf-8")
    report = compare(comparison(tmp_path, baseline, candidate))
    assert report["comparable"] and report["useful_reduction"]
    assert Decimal(report["totals"]["candidate_usd"]) == Decimal(".00002")


def test_reused_captures_and_billing_scopes_are_not_independent_runs(tmp_path):
    baseline = recorded_run(tmp_path, "baseline", 100)
    candidate = {**baseline, "run_id": "different"}
    with pytest.raises(InputError, match="reuses"):
        compare(comparison(tmp_path, baseline, candidate))


def test_different_captures_cannot_reuse_a_billing_receipt(tmp_path):
    baseline = recorded_run(tmp_path, "baseline", 100, bill="2")
    candidate = recorded_run(tmp_path, "candidate", 50, bill="1")
    update_evidence(tmp_path, candidate, lambda session: session["billing"].update(receipt_id="baseline-receipt"))
    with pytest.raises(InputError, match="reuses"):
        compare(comparison(tmp_path, baseline, candidate, basis="reported_billed_usd"))


def test_different_snapshots_of_one_paid_message_are_not_independent_runs(tmp_path):
    baseline = recorded_run(tmp_path, "baseline", 100)
    candidate = recorded_run(tmp_path, "candidate", 50)
    path = tmp_path / candidate["transcripts"][0]
    row = json.loads(path.read_text(encoding="utf-8"))
    row["message"]["id"] = "baseline-message"
    write_trace(tmp_path, [row], name=path.name)
    evidence = tmp_path / candidate["evidence"]
    data = json.loads(evidence.read_text(encoding="utf-8"))
    data["sources"][0]["sha256"] = load_trace(path).sha256
    evidence.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(InputError, match="reuses"):
        compare(comparison(tmp_path, baseline, candidate))


@pytest.mark.parametrize("value", [False, 0, [], "true"])
def test_invalid_outcome_types_fail_explicitly(tmp_path, value):
    candidate = recorded_run(tmp_path, "candidate", 50)
    candidate["outcome"]["task_success"] = value
    if value is False:
        assert compare(comparison(tmp_path, candidate=candidate))["quality_preserved"] is False
    else:
        with pytest.raises(InputError, match="boolean"):
            compare(comparison(tmp_path, candidate=candidate))


def test_exact_receipt_totals_and_differences_survive_default_decimal_precision(tmp_path):
    baseline = recorded_run(tmp_path, "baseline", 100, bill="2.000000000000000000000000000002")
    candidate = recorded_run(tmp_path, "candidate", 50, bill="2.000000000000000000000000000001")
    report = compare(comparison(tmp_path, baseline, candidate, basis="reported_billed_usd"))
    assert report["useful_reduction"]
    assert Decimal(report["totals"]["observed_reduction_usd"]) == Decimal("1e-30")


def test_zero_baseline_has_no_invented_percentage(tmp_path):
    baseline = recorded_run(tmp_path, "baseline", 0)
    candidate = recorded_run(tmp_path, "candidate", 50)
    report = compare(comparison(tmp_path, baseline, candidate))
    assert report["comparable"] and report["totals"]["observed_reduction_percent"] is None
    assert report["useful_reduction"] is False


def test_all_pairs_affect_totals_and_quality_even_beyond_the_detail_cap(tmp_path):
    pairs = []
    for i in range(21):
        baseline = recorded_run(tmp_path, f"baseline-{i}", 100)
        candidate = recorded_run(tmp_path, f"candidate-{i}", 50)
        if i == 20:
            candidate["outcome"]["task_success"] = False
        pairs.append({"name": str(i), "baseline": baseline, "candidate": candidate,
                      "required_checks": ["tests"], "required_approvals": [],
                      "requirements_complete": True})
    report = compare(comparison(tmp_path, pairs=pairs))
    assert len(report["pairs"]) == 20 and report["additional_pairs"] == 1
    assert Decimal(report["totals"]["candidate_usd"]) == Decimal("0.00105")
    assert report["failed_runs"] == 1 and report["useful_reduction"] is False


def test_incomplete_requirements_and_missing_controls_prevent_a_quality_claim(tmp_path):
    path = comparison(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["pairs"][0]["requirements_complete"] = False
    data["pairs"][0]["candidate"]["controls"].pop("input_fingerprint")
    path.write_text(json.dumps(data), encoding="utf-8")
    report = compare(path)
    assert not report["comparable"] and report["quality_preserved"] is None
    assert report["comparability"]["unknown_controls"] == 1
    assert report["unknown_outcomes"] == 2 and report["useful_reduction"] is None


@pytest.mark.parametrize("amount", ["1e999999999", "1e-999999999"])
def test_out_of_range_money_fails_explicitly_instead_of_overflow_or_free_cost(tmp_path, amount, capsys):
    candidate = recorded_run(tmp_path, "candidate", 50, bill=amount)
    path = comparison(tmp_path, candidate=candidate, basis="reported_billed_usd")
    assert main(["compare", str(path), "--format", "json"]) == 2
    output = capsys.readouterr()
    assert not output.out and "decimal" in output.err


@pytest.mark.parametrize("rate", ["1e999999999", "1e-999999999"])
def test_unrepresentable_price_quotes_are_not_zero_estimates(tmp_path, rate):
    candidate = recorded_run(tmp_path, "candidate", 50)
    update_evidence(tmp_path, candidate, lambda session: session["pricing"]["claude-sonnet-5"].update(
        input_usd_per_million=rate))
    with pytest.raises(InputError, match="decimal"):
        compare(comparison(tmp_path, candidate=candidate))


def test_comparison_does_not_overwrite_manifests_and_explains_unavailable_costs(tmp_path, capsys):
    candidate = recorded_run(tmp_path, "candidate", 50)
    update_evidence(tmp_path, candidate, lambda session: session.update(overhead_complete=False))
    path = comparison(tmp_path, candidate=candidate)
    original = path.read_bytes()
    assert main(["compare", str(path), "--output", str(path)]) == 2
    assert path.read_bytes() == original
    capsys.readouterr()
    assert main(["compare", str(path)]) == 1
    output = capsys.readouterr().out
    assert "Complete overhead and its pricing were not supplied." in output


def test_invalid_manifests_and_review_only_flags_have_no_success_report(tmp_path, capsys):
    path = comparison(tmp_path)
    assert main(["compare", str(path), "--input-host", "claude", "--format", "json"]) == 2
    assert not capsys.readouterr().out
    data = json.loads(path.read_text(encoding="utf-8"))
    data["cost_basis"] = "tokens"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert main(["compare", str(path), "--format", "json"]) == 2
    output = capsys.readouterr()
    assert not output.out and "cost_basis" in output.err


@pytest.mark.parametrize("plugin", [False, True])
def test_comparison_dispatch_is_read_only_and_never_initializes_live_state(tmp_path, monkeypatch, capsys, plugin):
    import builtins
    import socket
    import subprocess
    from tokeneconomizer.plugin import cli, paths

    path = comparison(tmp_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    allowed = {path.resolve()}
    for case in (manifest["pairs"][0]["baseline"], manifest["pairs"][0]["candidate"]):
        allowed.add((tmp_path / case["evidence"]).resolve())
        allowed.update((tmp_path / item).resolve() for item in case["transcripts"])
    original = builtins.open

    def read_only(file, mode="r", *args, **kwargs):
        assert Path(file).resolve() in allowed and mode == "r"
        return original(file, mode, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("comparison attempted live state, process execution, or networking")

    monkeypatch.setattr(builtins, "open", read_only)
    monkeypatch.setattr(cli, "_ctx", forbidden)
    monkeypatch.setattr(cli, "_recent_host", forbidden)
    monkeypatch.setattr(paths, "secure_root", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    arguments = ["compare", str(path), "--format", "json"]
    assert (cli.main(["prune", *arguments]) if plugin else main(arguments)) == 0
    assert json.loads(capsys.readouterr().out)["mode"] == "advisory_comparison"
