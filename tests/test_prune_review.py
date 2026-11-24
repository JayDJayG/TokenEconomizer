"""Source-backed review, conservative polling, and the no-live-state boundary."""
import builtins
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tokeneconomizer.prune import load_trace, review
from tokeneconomizer.prune.cli import main
from tokeneconomizer.prune.review import InputError


def write_trace(tmp_path, rows, name="events.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def copilot_polls(statuses=("completed", "completed"), **args):
    rows = [{"type": "user.message", "data": {"content": "Review the worker result"}}]
    for i, status in enumerate(statuses):
        rows.extend([
            {"type": "tool.execution_start", "data": {
                "toolCallId": f"c{i}", "toolName": "read_agent",
                "arguments": {"agent_id": "worker", "wait": False, **args}}},
            {"type": "tool.execution_complete", "data": {
                "toolCallId": f"c{i}", "success": True,
                "result": {"content": json.dumps({"agent_id": "worker", "status": status})}}},
        ])
    return rows


def test_copilot_completed_rereads_are_grouped_with_source_evidence(tmp_path):
    path = write_trace(tmp_path, copilot_polls(("completed",) * 4))
    trace = load_trace(path)
    report = review([trace])
    assert trace.host == "copilot"
    assert len(trace.calls) == 4
    assert trace.calls[0].line == 2 and trace.calls[0].result_line == 3
    assert len(report["findings"]) == 1
    assert report["findings"][0]["calls"] == 4
    assert report["findings"][0]["first_line"] == 2
    assert report["findings"][0]["last_line"] == 9
    assert report["findings"][0]["source"] == str(path.resolve())


def test_running_polling_needs_explicit_notification_capability(tmp_path):
    trace = load_trace(write_trace(tmp_path, copilot_polls(("running",) * 3)))
    assert review([trace])["findings"] == []
    report = review([trace], notifications_supported=True)
    assert report["findings"][0]["pattern"] == "busy_polling"
    assert "user-confirmed" in report["findings"][0]["evidence_basis"]
    with pytest.raises(InputError, match="explicit boolean"):
        review([trace], notifications_supported="false")


@pytest.mark.parametrize("statuses,args", [
    (("running", "completed"), {}),
    (("completed", "completed"), {"wait": True}),
    (("failed", "failed"), {}),
])
def test_progress_blocking_waits_and_failures_are_not_waste(tmp_path, statuses, args):
    trace = load_trace(write_trace(tmp_path, copilot_polls(statuses, **args)))
    assert review([trace], notifications_supported=True)["findings"] == []


def test_new_prompt_and_intervening_work_break_a_polling_chain(tmp_path):
    rows = copilot_polls()
    rows.insert(3, {"type": "user.message", "data": {"content": "Check it again"}})
    assert review([load_trace(write_trace(tmp_path, rows))])["findings"] == []
    rows = copilot_polls()
    rows[3:3] = [
        {"type": "tool.execution_start", "data": {
            "toolCallId": "edit", "toolName": "edit", "arguments": {}}},
        {"type": "tool.execution_complete", "data": {
            "toolCallId": "edit", "success": True, "result": {"content": "changed"}}},
    ]
    assert review([load_trace(write_trace(tmp_path, rows))])["findings"] == []


def test_partial_parallel_and_changed_results_are_not_duplicate_polls(tmp_path):
    rows = copilot_polls()
    rows[2], rows[3] = rows[3], rows[2]
    assert review([load_trace(write_trace(tmp_path, rows))])["findings"] == []
    rows = copilot_polls()
    rows[-1]["data"]["result"]["content"] = json.dumps(
        {"agent_id": "worker", "status": "completed", "new_result": "updated"})
    assert review([load_trace(write_trace(tmp_path, rows))])["findings"] == []
    rows = copilot_polls()[1:]
    assert review([load_trace(write_trace(tmp_path, rows))])["findings"] == []


def test_tool_errors_and_different_raw_results_are_not_identical_polling(tmp_path):
    rows = copilot_polls()
    rows[-1]["data"]["error"] = {"message": "result retrieval failed"}
    assert review([load_trace(write_trace(tmp_path, rows))])["findings"] == []
    rows = copilot_polls()
    rows[2]["data"]["result"]["content"] = (
        '{"agent_id":"worker","status":"completed","progress":1.00000000000000001}')
    rows[-1]["data"]["result"]["content"] = (
        '{"agent_id":"worker","status":"completed","progress":1.00000000000000002}')
    assert review([load_trace(write_trace(tmp_path, rows))])["findings"] == []


def test_claude_request_blobs_do_not_duplicate_calls_and_missing_results_stay_unknown(tmp_path):
    use = {"type": "tool_use", "id": "t1", "name": "Read",
           "input": {"file_path": "module.py"}}
    rows = [
        {"type": "user", "message": {"content": "Read the module"}},
        {"type": "assistant", "message": {"content": [use]}},
        {"type": "api-request-blob", "message": {"content": [use]}},
        {"type": "assistant", "message": {"content": [use]}},
    ]
    trace = load_trace(write_trace(tmp_path, rows))
    assert trace.host == "claude"
    assert len(trace.calls) == 1
    assert trace.calls[0].success is None and trace.calls[0].result is None
    assert trace.limitations
    assert review([trace])["findings"] == []


def test_claude_taskoutput_results_can_be_reviewed(tmp_path):
    rows = [{"type": "user", "message": {"content": "Read the result"}}]
    for i in range(2):
        rows.extend([
            {"type": "assistant", "message": {"content": [{
                "type": "tool_use", "id": f"t{i}", "name": "TaskOutput",
                "input": {"task_id": "worker", "block": False}}]}},
            {"type": "user", "message": {"content": [{
                "type": "tool_result", "tool_use_id": f"t{i}",
                "content": json.dumps({"task": {"task_id": "worker", "status": "completed"}})}]}},
        ])
    trace = load_trace(write_trace(tmp_path, rows))
    assert trace.host == "claude"
    assert review([trace])["findings"][0]["pattern"] == "completed_agent_rereads"


def test_json_argument_strings_are_normalized_before_review(tmp_path):
    rows = copilot_polls()
    for event in rows:
        if event["type"] == "tool.execution_start":
            data = event["data"]
            data["arguments"] = json.dumps(data["arguments"])
    trace = load_trace(write_trace(tmp_path, rows))
    assert isinstance(trace.calls[0].arguments, dict)
    assert len(review([trace])["findings"]) == 1


def test_codex_exec_and_rollout_reuse_readers_without_executing_commands(tmp_path):
    rows = [{"type": "turn.started"}, {"type": "item.completed", "item": {
        "type": "command_execution", "id": "i1", "command": "python -m pytest",
        "aggregated_output": "1 passed", "exit_code": 0}}]
    trace = load_trace(write_trace(tmp_path, rows))
    assert trace.host == "codex"
    assert trace.calls[0].tool == "shell"
    assert trace.calls[0].arguments == {"command": "python -m pytest"}
    rows = [{"type": "turn_context", "payload": {"model": "unknown-model"}},
            {"type": "event_msg", "payload": {"type": "item_completed", "item": {
                "type": "CommandExecution", "id": "i1", "command": ["python", "-m", "pytest"],
                "aggregated_output": "1 passed", "exit_code": 0}}}]
    assert load_trace(write_trace(tmp_path, rows)).calls[0].tool == "shell"


def test_codex_function_calls_keep_source_provenance_and_unknown_success(tmp_path):
    rows = [{"type": "turn_context", "payload": {}}]
    for i in range(2):
        rows.extend([
            {"type": "response_item", "payload": {
                "type": "function_call", "name": "read_agent", "call_id": f"c{i}",
                "arguments": json.dumps({"agent_id": "worker", "wait": False})}},
            {"type": "response_item", "payload": {
                "type": "function_call_output", "call_id": f"c{i}",
                "output": json.dumps({"agent_id": "worker", "status": "completed"})}},
        ])
    trace = load_trace(write_trace(tmp_path, rows))
    assert trace.calls[0].success is None
    assert trace.calls[0].line == 2 and trace.calls[0].result_line == 3
    assert len(review([trace])["findings"]) == 1


def test_unparsed_codex_tool_activity_breaks_comparisons_and_is_reported(tmp_path):
    rows = [{"type": "turn_context", "payload": {}}]
    for i in range(2):
        if i:
            rows.append({"type": "response_item", "payload": {
                "type": "custom_tool_call", "name": "exec", "input": "worker follow-up"}})
        rows.extend([
            {"type": "response_item", "payload": {
                "type": "function_call", "name": "read_agent", "call_id": f"c{i}",
                "arguments": json.dumps({"agent_id": "worker", "wait": False})}},
            {"type": "response_item", "payload": {
                "type": "function_call_output", "call_id": f"c{i}",
                "output": json.dumps({"agent_id": "worker", "status": "completed"})}},
        ])
    trace = load_trace(write_trace(tmp_path, rows))
    assert "custom_tool_call" in trace.limitations[0]
    assert review([trace])["findings"] == []


def test_utf8_bom_is_supported_and_invalid_encoding_is_an_explicit_error(tmp_path):
    path = write_trace(tmp_path, copilot_polls())
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    assert load_trace(path).host == "copilot"
    path.write_bytes(b"\xff\xfe")
    with pytest.raises(InputError, match="UTF-8"):
        load_trace(path)


@pytest.mark.parametrize("text", [
    "", "not JSON\n", "[]\n", '{"type":"other"}\n',
    '{"type":"user.message","type":"user.message"}\n',
    '{"type":"user.message","data":{"number":NaN}}\n',
    '{"type":"user.message","data":{"number":1e400}}\n',
    '{"type":"tool.execution_start","data":[]}\n',
    '{"type":"user.message","data":[]}\n',
])
def test_bad_inputs_fail_explicitly(tmp_path, text):
    path = tmp_path / "bad.jsonl"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(InputError):
        load_trace(path)


def test_host_mismatch_orphan_results_and_duplicate_inputs(tmp_path):
    path = write_trace(tmp_path, copilot_polls())
    with pytest.raises(InputError, match="detected copilot"):
        load_trace(path, "claude")
    trace = load_trace(path)
    with pytest.raises(InputError, match="duplicate input"):
        review([trace, trace])
    orphan = write_trace(tmp_path, [copilot_polls()[2]], name="orphan.jsonl")
    assert load_trace(orphan).limitations


def test_review_accepts_iterators_but_not_missing_evidence(tmp_path):
    trace = load_trace(write_trace(tmp_path, copilot_polls()))
    assert review(iter([trace]))["sources"][0]["path"] == trace.source
    with pytest.raises(InputError, match="at least one"):
        review([])


def test_reports_do_not_invent_state_or_savings_and_are_limited_to_three(tmp_path):
    paths = [write_trace(tmp_path, copilot_polls(), name=f"{i}.jsonl") for i in range(5)]
    report = review([load_trace(path) for path in paths])
    assert len(report["findings"]) == 3 and report["additional_findings"] == 2
    assert "cost" not in report["findings"][0]
    checks = {check["pattern"]: check["status"] for check in report["checks"]}
    assert checks["redundant_verification"] == "not_assessable"
    assert checks["cost_accounting"] == "not_implemented"


def test_cli_read_only_path_reads_only_the_named_file(tmp_path, monkeypatch, capsys):
    path = write_trace(tmp_path, copilot_polls())
    opened = []
    original = builtins.open

    def read_only(file, mode="r", *args, **kwargs):
        assert Path(file).resolve() == path.resolve()
        assert mode == "r"
        opened.append(file)
        return original(file, mode, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("review attempted live discovery, networking, or process execution")

    monkeypatch.setattr(builtins, "open", read_only)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    import socket
    monkeypatch.setattr(socket, "socket", forbidden)
    assert main(["review", str(path), "--format", "json"]) == 0
    assert len(opened) == 1
    assert json.loads(capsys.readouterr().out)["mode"] == "advisory"


def test_te_dispatch_never_initializes_or_secures_live_state(tmp_path, monkeypatch, capsys):
    from tokeneconomizer.plugin import cli, paths

    def forbidden(*args, **kwargs):
        pytest.fail("review touched the live plugin context")

    monkeypatch.setattr(cli, "_ctx", forbidden)
    monkeypatch.setattr(cli, "_recent_host", forbidden)
    monkeypatch.setattr(paths, "secure_root", forbidden)
    path = write_trace(tmp_path, copilot_polls())
    assert cli.main(["--host", "pi", "prune", "review", str(path), "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["findings"]


def test_explicit_output_never_overwrites_input_or_existing_reports(tmp_path, capsys):
    path = write_trace(tmp_path, copilot_polls())
    original = path.read_bytes()
    assert main(["review", str(path), "--output", str(path)]) == 2
    assert path.read_bytes() == original
    assert capsys.readouterr().err
    output = tmp_path / "report.json"
    assert main(["review", str(path), "--format", "json", "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["findings"]
    assert main(["review", str(path), "--output", str(output)]) == 2


def test_cli_error_has_no_success_report_and_markdown_explains_limits(tmp_path, capsys):
    missing = tmp_path / "missing.jsonl"
    assert main(["review", str(missing), "--format", "json"]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "prune:" in captured.err
    path = write_trace(tmp_path, [{"type": "user.message", "data": {}}])
    assert main(["review", str(path)]) == 0
    text = capsys.readouterr().out
    assert "No supported high-confidence cut" in text
    assert "not_implemented" in text and "Missing evidence is not a passed check" in text


@pytest.mark.parametrize("module,prefix", [
    ("tokeneconomizer.prune", []),
    ("tokeneconomizer.plugin.cli", ["--host", "pi", "prune"]),
])
def test_module_entrypoints_review_a_file_without_creating_live_state(tmp_path, module, prefix):
    path = write_trace(tmp_path, copilot_polls())
    state = tmp_path / "must-not-exist"
    env = dict(os.environ, TOKENECONOMIZER_HOME=str(state), PYTHONDONTWRITEBYTECODE="1")
    run = subprocess.run(
        [sys.executable, "-m", module, *prefix, "review", str(path), "--format", "json"],
        cwd=Path(__file__).resolve().parents[1], env=env,
        capture_output=True, text=True, timeout=30,
    )
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)["findings"][0]["pattern"] == "completed_agent_rereads"
    assert not state.exists()
