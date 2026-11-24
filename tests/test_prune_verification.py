from dataclasses import replace

import pytest

from tokeneconomizer.prune import review
from tokeneconomizer.prune.checks import VERIFY_COVERS
from tokeneconomizer.prune.evidence import validate_facts
from tokeneconomizer.prune.types import InputError
from tests.test_prune_checks import calls_trace


def verification_trace(tmp_path):
    trace = calls_trace(tmp_path, tool="functions.powershell",
                        arguments={"command": "python -m pytest tests"},
                        results=("3 passed", "3 passed"))
    calls = tuple(replace(call, facts={
        **call.facts, "purpose": "verification",
        "input_state": {**call.facts["input_state"], "covers": sorted(VERIFY_COVERS)},
        "verification": {"deterministic": True, "unresolved_concerns": False},
    }) for call in trace.calls)
    return replace(trace, calls=calls)


def test_redundant_verification_has_full_historical_evidence(tmp_path):
    trace = verification_trace(tmp_path)
    finding = review([trace])["findings"][0]
    assert finding["pattern"] == "redundant_verification"
    assert finding["calls"] == 2
    assert "environment" in finding["replacement"]


@pytest.mark.parametrize("component", sorted(VERIFY_COVERS))
def test_missing_input_component_prevents_a_verification_cut(tmp_path, component):
    trace = verification_trace(tmp_path)
    calls = tuple(replace(call, facts={**call.facts, "input_state": {
        **call.facts["input_state"], "covers": sorted(VERIFY_COVERS - {component})}})
        for call in trace.calls)
    assert review([replace(trace, calls=calls)])["findings"] == []


@pytest.mark.parametrize("change", [
    lambda c: replace(c, facts={**c.facts, "required": True}),
    lambda c: replace(c, facts={**c.facts, "input_state": {
        **c.facts["input_state"], "fingerprint": "changed-config"}}),
    lambda c: replace(c, facts={**c.facts, "verification": {
        "deterministic": True, "unresolved_concerns": True}}),
    lambda c: replace(c, facts={**c.facts, "verification": {"deterministic": False}}),
    lambda c: replace(c, arguments={"command": "python -m pytest tests -k edge"}),
    lambda c: replace(c, success=False),
    lambda c: replace(c, result="3 passed, new warning"),
])
def test_changes_requirements_and_new_concerns_preserve_reruns(tmp_path, change):
    trace = verification_trace(tmp_path)
    assert review([replace(trace, calls=(trace.calls[0], change(trace.calls[1])))])["findings"] == []


def test_verification_booleans_do_not_coerce_strings():
    with pytest.raises(InputError, match="must be a boolean"):
        validate_facts({"verification": {"unresolved_concerns": "false"}}, "evidence")
