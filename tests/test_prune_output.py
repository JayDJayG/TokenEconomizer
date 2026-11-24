from dataclasses import replace

import pytest

from tokeneconomizer.prune import review
from tokeneconomizer.prune.evidence import validate_facts
from tokeneconomizer.prune.types import InputError
from tests.test_prune_checks import calls_trace


def output_trace(tmp_path, result="needed\nextra\nmore\n"):
    trace = calls_trace(tmp_path, results=(result,))
    call = trace.calls[0]
    return replace(trace, calls=(replace(call, facts={
        "purpose": "investigation", "output_need": {
            "sufficient": True, "reason": "Only first line answers the question",
            "selected_lines": [1]}}),))


def test_overfetch_uses_intent_and_actual_bytes_not_length_alone(tmp_path):
    trace = output_trace(tmp_path, result="required\n\u00e9vidence\n")
    finding = review([trace])["findings"][0]
    assert finding["pattern"] == "output_overfetch"
    assert finding["output_bytes"] == len("required\n\u00e9vidence\n".encode("utf-8"))
    assert finding["needed_output_bytes"] == len("required\n".encode("utf-8"))
    assert "full artifact" in finding["replacement"]
    bare = replace(trace, calls=(replace(trace.calls[0], facts={}),))
    assert review([bare])["findings"] == []


@pytest.mark.parametrize("change", [
    lambda c: replace(c, success=False),
    lambda c: replace(c, success=None),
    lambda c: replace(c, result=None),
    lambda c: replace(c, facts={**c.facts, "output_need": {
        **c.facts["output_need"], "sufficient": False}}),
    lambda c: replace(c, facts={**c.facts, "output_need": {
        **c.facts["output_need"], "selected_lines": [1, 2, 3]}}),
])
def test_failures_missing_evidence_and_required_full_output_are_not_cut(tmp_path, change):
    trace = output_trace(tmp_path)
    assert review([replace(trace, calls=(change(trace.calls[0]),))])["findings"] == []


@pytest.mark.parametrize("selected", [[0], [True], [1, 1], [], "1"])
def test_invalid_line_selection_is_an_explicit_error(selected):
    with pytest.raises(InputError, match="selected_lines"):
        validate_facts({"output_need": {"sufficient": True, "selected_lines": selected}}, "evidence")


def test_out_of_range_result_reference_fails(tmp_path):
    trace = output_trace(tmp_path)
    call = trace.calls[0]
    call = replace(call, facts={**call.facts, "output_need": {
        **call.facts["output_need"], "selected_lines": [99]}})
    with pytest.raises(InputError, match="out of range"):
        review([replace(trace, calls=(call,))])
