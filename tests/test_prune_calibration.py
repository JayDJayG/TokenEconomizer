import json
from pathlib import Path

import pytest

from tokeneconomizer.prune import load_trace, review
from tokeneconomizer.prune.cli import main
from tokeneconomizer.prune.evaluation import calibrate
from tokeneconomizer.prune.types import InputError
from tests.test_prune_review import copilot_polls, write_trace


def manifest(tmp_path, cases):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"schema_version": 1, "kind": "calibration", "cases": cases}), encoding="utf-8")
    return path


def test_host_schema_vectors_and_abstentions_are_labeled_honestly(tmp_path):
    data = json.loads((Path(__file__).parent / "fixtures" / "prune_samples.json").read_text(encoding="utf-8"))
    cases = []
    for name, rows in data["samples"].items():
        trace = write_trace(tmp_path, rows, name=f"{name}.jsonl")
        pattern = "busy_polling" if name == "copilot_running" else "completed_agent_rereads"
        expected = "present" if name.endswith("_completed") or name == "copilot_running" else "unassessable"
        cases.append({"name": name, "transcripts": [trace.name],
                      "notifications_supported": name == "copilot_running",
                      "labels": {pattern: expected}})
    report = calibrate(manifest(tmp_path, cases))
    assert report["passed"] and report["metrics"]["true_positives"] == 3
    assert report["metrics"]["abstentions"] == 2


def test_calibration_mismatches_do_not_return_success_exit_codes(tmp_path, capsys):
    trace = write_trace(tmp_path, copilot_polls())
    path = manifest(tmp_path, [{"name": "incorrect-label", "transcripts": [trace.name],
                               "labels": {"completed_agent_rereads": "absent"}}])
    assert main(["calibrate", str(path), "--format", "json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["metrics"]["false_positives"] == 1 and not report["passed"]


def test_unparsed_rollout_activity_is_a_barrier_not_an_absence_of_work(tmp_path):
    samples = json.loads((Path(__file__).parent / "fixtures" / "prune_samples.json").read_text(encoding="utf-8"))
    rows = samples["samples"]["codex_completed"]
    assert review([load_trace(write_trace(tmp_path, rows))])["findings"]
    rows.insert(3, {"type": "event_msg", "payload": {"type": "item_completed",
                   "item": {"type": "UnknownToolExecution", "id": "unknown"}}})
    trace = load_trace(write_trace(tmp_path, rows))
    assert "UnknownToolExecution" in trace.limitations[0]
    assert review([trace])["findings"] == []


@pytest.mark.parametrize("info", [False, 0, []])
def test_invalid_token_info_is_rejected_but_native_null_is_allowed(tmp_path, info):
    rows = [{"type": "event_msg", "payload": {"type": "token_count", "info": info}}]
    with pytest.raises(InputError, match="token-count info"):
        load_trace(write_trace(tmp_path, rows))
    rows[0]["payload"]["info"] = None
    assert load_trace(write_trace(tmp_path, rows)).usage == ()


@pytest.mark.parametrize("data", [False, 0, [], None])
def test_falsey_invalid_native_objects_do_not_become_empty_defaults(tmp_path, data):
    with pytest.raises(InputError):
        load_trace(write_trace(tmp_path, [{"type": "session.start", "data": data}]))


def test_codex_unknown_exit_stays_unknown_and_invalid_exit_is_rejected(tmp_path):
    rows = [{"type": "turn.started"}, {"type": "item.completed", "item": {
        "type": "command_execution", "id": "command", "command": "test",
        "aggregated_output": "unknown outcome"}}]
    assert load_trace(write_trace(tmp_path, rows)).calls[0].success is None
    rows[1]["item"]["exit_code"] = False
    with pytest.raises(InputError, match="exit_code"):
        load_trace(write_trace(tmp_path, rows))


def test_unknown_labels_and_non_manifest_reports_are_rejected(tmp_path):
    path = manifest(tmp_path, [{"name": "case", "transcripts": ["unused.jsonl"],
                               "labels": {"made_up": "present"}}])
    with pytest.raises(InputError, match="unknown calibration"):
        calibrate(path)
    path.write_text(json.dumps({"schema_version": 1, "mode": "advisory", "sources": []}))
    with pytest.raises(InputError, match="kind calibration"):
        calibrate(path)


def test_all_pattern_counts_survive_the_three_cut_limit(tmp_path):
    traces = [load_trace(write_trace(tmp_path, copilot_polls(), name=f"{i}.jsonl")) for i in range(5)]
    report = review(traces)
    assert len(report["findings"]) == 3
    assert report["pattern_counts"]["completed_agent_rereads"] == 5


def test_float_arguments_are_not_guessed_equal_after_json_precision_loss(tmp_path):
    rows = copilot_polls(timeout=1.0)
    assert review([load_trace(write_trace(tmp_path, rows))])["findings"] == []


def test_calibration_does_not_spawn_or_call_models(tmp_path, monkeypatch):
    import socket
    import subprocess

    trace = write_trace(tmp_path, copilot_polls())
    path = manifest(tmp_path, [{"name": "case", "transcripts": [trace.name],
                               "labels": {"completed_agent_rereads": "present"}}])

    def forbidden(*args, **kwargs):
        pytest.fail("calibration attempted process execution or networking")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    assert calibrate(path)["passed"]


@pytest.mark.parametrize("expected,key,passed", [
    ("absent", "true_negatives", True),
    ("present", "false_negatives", False),
])
def test_progress_is_assessed_without_flagging_justified_retrieval(tmp_path, expected, key, passed):
    trace = write_trace(tmp_path, copilot_polls(("running", "completed")))
    path = manifest(tmp_path, [{"name": "progress", "transcripts": [trace.name],
                               "labels": {"completed_agent_rereads": expected}}])
    report = calibrate(path)
    assert report["passed"] is passed and report["metrics"][key] == 1


def test_missing_evidence_is_not_a_matched_absence_or_true_negative(tmp_path):
    trace = write_trace(tmp_path, copilot_polls(wait=True))
    path = manifest(tmp_path, [{"name": "blocking", "transcripts": [trace.name],
                               "labels": {"completed_agent_rereads": "absent"}}])
    report = calibrate(path)
    assert not report["passed"] and report["metrics"]["abstentions"] == 1
    assert report["metrics"]["true_negatives"] == 0


def test_parallel_retrieval_and_hidden_case_mismatches_are_preserved(tmp_path):
    rows = copilot_polls()
    rows[2], rows[3] = rows[3], rows[2]
    trace = write_trace(tmp_path, rows)
    cases = [{"name": f"parallel-{i}", "transcripts": [trace.name],
              "labels": {"completed_agent_rereads": "absent" if i < 24 else "present"}}
             for i in range(25)]
    report = calibrate(manifest(tmp_path, cases))
    assert len(report["cases"]) == 20 and report["additional_cases"] == 5
    assert report["metrics"]["true_negatives"] == 24
    assert report["metrics"]["false_negatives"] == 1 and not report["passed"]


@pytest.mark.parametrize("plugin", [False, True])
def test_calibration_dispatch_reads_only_explicit_inputs_without_live_state(tmp_path, monkeypatch, capsys, plugin):
    import builtins
    from tokeneconomizer.plugin import cli, paths

    trace = write_trace(tmp_path, copilot_polls())
    path = manifest(tmp_path, [{"name": "case", "transcripts": [trace.name],
                               "labels": {"completed_agent_rereads": "present"}}])
    original = builtins.open
    opened = []

    def read_only(file, mode="r", *args, **kwargs):
        assert Path(file).resolve() in (path.resolve(), trace.resolve()) and mode == "r"
        opened.append(Path(file).resolve())
        return original(file, mode, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("calibration touched live plugin state")

    monkeypatch.setattr(builtins, "open", read_only)
    monkeypatch.setattr(cli, "_ctx", forbidden)
    monkeypatch.setattr(cli, "_recent_host", forbidden)
    monkeypatch.setattr(paths, "secure_root", forbidden)
    arguments = ["calibrate", str(path), "--format", "json"]
    assert (cli.main(["prune", *arguments]) if plugin else main(arguments)) == 0
    assert json.loads(capsys.readouterr().out)["mode"] == "advisory_calibration"
    assert opened == [path.resolve(), trace.resolve()]
