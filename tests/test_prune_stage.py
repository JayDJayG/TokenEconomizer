import builtins
import copy
from pathlib import Path
import socket
import subprocess

import pytest

from tokeneconomizer.prune import load_trace
from tokeneconomizer.prune.stage import PruneReviewStage
from tokeneconomizer.prune.types import InputError
from tests.test_prune_evidence import annotate
from tests.test_prune_review import copilot_polls, write_trace


def forbidden(*args, **kwargs):
    pytest.fail("Prune stage attempted undeclared I/O or execution")


def test_disabled_stage_does_not_inspect_inputs_or_touch_state(tmp_path, monkeypatch):
    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    stage = PruneReviewStage()
    record = {"resolved": True}
    assert stage.after_task(object(), object(), object(), record) == 0
    assert record == {"resolved": True}


@pytest.mark.parametrize("enabled", [False, True])
def test_launch_hooks_remain_noops(enabled):
    stage = PruneReviewStage(enabled=enabled)
    assert stage.before_task(object(), "original prefix") == "original prefix"
    assert stage.build_tools(object()) is None
    assert stage.run_kwargs(object()) == {}


def test_opt_in_reads_only_named_inputs_and_attaches_a_bounded_report(tmp_path, monkeypatch):
    path = write_trace(tmp_path, copilot_polls())
    evidence = annotate(tmp_path, load_trace(path))
    row = {"prune_inputs": {"transcripts": [str(path)], "evidence": str(evidence)}}
    before = copy.deepcopy(row)
    record = {"resolved": True}
    original = builtins.open
    opened = []

    def read_only(file, mode="r", *args, **kwargs):
        assert Path(file).resolve() in (path.resolve(), evidence.resolve()) and mode == "r"
        opened.append(Path(file).resolve())
        return original(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", read_only)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    assert PruneReviewStage(enabled=True).after_task(object(), row, object(), record) == 0
    output = record["prune_review"]
    assert output["status"] == "reviewed" and len(output["report"]["findings"]) == 1
    assert output["local_review_overhead"]["review_model_calls"] == 0
    assert opened == [path.resolve(), evidence.resolve()] and row == before
    assert record["resolved"] is True


@pytest.mark.parametrize("inputs", [
    None, {}, {"transcripts": []}, {"transcripts": ["events.jsonl"]},
    {"transcripts": [False]}, {"transcripts": [None]},
])
def test_missing_or_relative_inputs_fail_explicitly_without_directory_discovery(tmp_path, monkeypatch, capsys, inputs):
    monkeypatch.setattr(builtins, "open", forbidden)
    record = {"prune_review": {"status": "reviewed", "report": "stale"}}
    stage = PruneReviewStage(enabled=True)
    assert stage.after_task({}, {"prune_inputs": inputs}, tmp_path, record) == 0
    assert record["prune_review"]["status"] == "error"
    assert "report" not in record["prune_review"] and capsys.readouterr().err


def test_input_failure_preserves_task_outcome_and_reports_error_overhead(tmp_path, capsys):
    record = {"resolved": False}
    row = {"prune_inputs": {"transcripts": [str(tmp_path / "missing.jsonl")]}}
    assert PruneReviewStage(enabled=True).after_task({}, row, tmp_path, record) == 0
    error = record["prune_review"]
    assert error["status"] == "error" and error["local_review_overhead"]["paid_review_tokens"] == 0
    assert record["resolved"] is False and "missing.jsonl" in capsys.readouterr().err


def test_programming_errors_are_not_silently_swallowed(tmp_path, monkeypatch):
    from tokeneconomizer.prune import stage

    def broken(*args, **kwargs):
        raise RuntimeError("unexpected implementation failure")

    monkeypatch.setattr(stage, "review_case", broken)
    row = {"prune_inputs": {"transcripts": [str(tmp_path / "named.jsonl")]}}
    with pytest.raises(RuntimeError, match="implementation failure"):
        PruneReviewStage(enabled=True).after_task({}, row, tmp_path, {})


def test_enablement_and_output_record_types_are_not_coerced():
    with pytest.raises(InputError, match="boolean"):
        PruneReviewStage(enabled="false")
    with pytest.raises(InputError, match="mutable task record"):
        PruneReviewStage(enabled=True).after_task({}, {}, object(), None)


def test_after_task_bounds_details_without_skipping_supplied_sources(tmp_path):
    paths = [str(write_trace(tmp_path, copilot_polls(), name=f"{i}.jsonl")) for i in range(25)]
    record = {}
    row = {"prune_inputs": {"transcripts": paths}}
    assert PruneReviewStage(enabled=True).after_task({}, row, object(), record) == 0
    report = record["prune_review"]["report"]
    assert len(report["findings"]) == 3 and report["additional_findings"] == 22
    assert len(report["sources"]) == 20 and report["additional_sources"] == 5
    assert report["pattern_counts"]["completed_agent_rereads"] == 25


def test_relative_evidence_is_rejected_before_reading_transcripts(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(builtins, "open", forbidden)
    row = {"prune_inputs": {"transcripts": [str(tmp_path / "named.jsonl")],
                            "evidence": "relative.json"}}
    record = {}
    assert PruneReviewStage(enabled=True).after_task({}, row, object(), record) == 0
    assert record["prune_review"]["status"] == "error" and "absolute" in capsys.readouterr().err
