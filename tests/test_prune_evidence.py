import hashlib
import json

import pytest

from tokeneconomizer.prune import load_trace, review
from tokeneconomizer.prune.evidence import apply_evidence
from tokeneconomizer.prune.types import InputError
from tests.test_prune_review import copilot_polls, write_trace


def annotate(tmp_path, trace, facts=None, session=None):
    path = tmp_path / "evidence.json"
    source = {"path": trace.source, "sha256": trace.sha256, "session": session or {}}
    if facts is not None:
        call = trace.calls[0]
        source["calls"] = [{"line": call.line, "tool": call.tool,
                            "call_id": call.call_id, "facts": facts}]
    path.write_text(json.dumps({"schema_version": 1, "sources": [source]}), encoding="utf-8")
    return path


def test_metadata_and_exact_byte_digest(tmp_path):
    rows = copilot_polls()
    rows.insert(0, {"type": "session.start", "data": {
        "sessionId": "session", "selectedModel": "gpt-6.1-sol"}})
    rows[2]["timestamp"] = "2026-11-24T12:00:00Z"
    rows[3]["timestamp"] = "2026-11-24T12:00:01Z"
    path = write_trace(tmp_path, rows)
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes().replace(b"\n", b"\r\n"))
    trace = load_trace(path)
    assert trace.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert trace.session_id == "session"
    assert trace.calls[0].timestamp == rows[2]["timestamp"]
    assert trace.calls[0].completed_at == rows[3]["timestamp"]
    assert trace.calls[0].model == "gpt-6.1-sol"
    assert review([trace])["sources"][0]["evidence"]["timestamps"] == 1


def test_claude_usage_is_message_deduplicated_and_does_not_collect_subagents(tmp_path):
    row = {"type": "assistant", "agentId": "worker", "message": {
        "id": "m1", "model": "claude-sonnet-5", "content": [{"type": "text", "text": "Done"}],
        "usage": {"input_tokens": 10, "cache_read_input_tokens": 20,
                  "cache_creation_input_tokens": 4, "output_tokens": 2,
                  "cache_creation": {"ephemeral_1h_input_tokens": 1}}}}
    trace = load_trace(write_trace(tmp_path, [row, row]))
    assert len(trace.usage) == 1 and trace.usage[0].input_tokens == 10
    assert trace.usage[0].cache_creation_1h_input_tokens == 1
    assert trace.agent_id == "worker" and trace.parent_agent_id is None
    row["message"].pop("id")
    trace = load_trace(write_trace(tmp_path, [row]))
    assert trace.usage == () and "message ID" in trace.limitations[0]


def test_copilot_usage_keeps_models_and_unknown_cache_fields(tmp_path):
    row = {"type": "session.shutdown", "data": {"modelMetrics": {
        "gpt-6.1-sol": {"usage": {"inputTokens": 100, "cacheReadTokens": 60,
                                "cacheWriteTokens": 10, "outputTokens": 8}},
        "other": {"usage": {"inputTokens": 50, "outputTokens": 5}}}}}
    trace = load_trace(write_trace(tmp_path, [row, row]))
    assert len(trace.usage) == 2
    assert trace.usage[0].input_tokens == 30
    assert trace.usage[1].input_tokens is None


def test_codex_cumulative_usage_is_not_summed(tmp_path):
    def usage(n):
        return {"type": "event_msg", "payload": {"type": "token_count", "info": {
            "total_token_usage": {"input_tokens": n, "cached_input_tokens": 4,
                                 "cache_write_input_tokens": 0, "output_tokens": 3}}}}
    trace = load_trace(write_trace(tmp_path, [
        {"type": "turn_context", "payload": {"model": "gpt-6.1-sol"}}, usage(10), usage(20)]))
    assert len(trace.usage) == 1 and trace.usage[0].input_tokens == 16
    assert trace.usage[0].model == "gpt-6.1-sol"


def test_annotations_bind_to_named_sources_and_unique_calls(tmp_path):
    trace = load_trace(write_trace(tmp_path, copilot_polls()))
    facts = {"purpose": "investigation", "required": False, "input_state": {
        "fingerprint": "historical", "scope": "module", "complete": True,
        "captured_at": "2026-11-24T12:00:00Z", "covers": ["source"]}}
    path = annotate(tmp_path, trace, facts, {"agent_id": "worker", "parent_agent_id": "root"})
    result = apply_evidence([trace], path)[0]
    assert result.calls[0].facts == facts
    assert result.agent_id == "worker" and result.parent_agent_id == "root"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["sources"][0]["sha256"] = "stale"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(InputError, match="digest mismatch"):
        apply_evidence([trace], path)
    with pytest.raises(InputError, match="unique supplied"):
        apply_evidence([], path)


@pytest.mark.parametrize("facts", [
    {"required": "false"}, {"purpose": 1},
    {"input_state": {"complete": "true"}}, {"input_state": {"covers": "source"}},
    {"input_state": {"captured_at": "yesterday"}},
])
def test_invalid_annotation_facts_are_explicit_errors(tmp_path, facts):
    trace = load_trace(write_trace(tmp_path, copilot_polls()))
    with pytest.raises(InputError):
        apply_evidence([trace], annotate(tmp_path, trace, facts))


@pytest.mark.parametrize("mutation", [
    lambda data: data["sources"][0]["calls"].append(data["sources"][0]["calls"][0]),
    lambda data: data["sources"][0]["calls"][0].update(line=999),
    lambda data: data["sources"].append(data["sources"][0]),
    lambda data: data.update(schema_version=True),
])
def test_ambiguous_or_malformed_annotations_fail(tmp_path, mutation):
    trace = load_trace(write_trace(tmp_path, copilot_polls()))
    path = annotate(tmp_path, trace, {})
    data = json.loads(path.read_text(encoding="utf-8"))
    mutation(data)
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(InputError):
        apply_evidence([trace], path)


def test_invalid_native_metadata_does_not_succeed_silently(tmp_path):
    rows = copilot_polls()
    rows[1]["timestamp"] = "not a date"
    with pytest.raises(InputError, match="timestamp"):
        load_trace(write_trace(tmp_path, rows))
    rows = [{"type": "session.shutdown", "data": {"modelMetrics": {
        "model": {"usage": {"inputTokens": 1, "cacheReadTokens": 2,
                            "cacheWriteTokens": 0, "outputTokens": 3}}}}}]
    with pytest.raises(InputError, match="cache counts"):
        load_trace(write_trace(tmp_path, rows))


def test_cli_annotations_read_only_explicit_inputs(tmp_path, monkeypatch, capsys):
    import builtins
    from pathlib import Path
    from tokeneconomizer.prune.cli import main

    path = write_trace(tmp_path, copilot_polls())
    evidence = annotate(tmp_path, load_trace(path), {"purpose": "coordination"})
    allowed = {path.resolve(), evidence.resolve()}
    opened = []
    original = builtins.open

    def scoped(file, mode="r", *args, **kwargs):
        assert Path(file).resolve() in allowed and mode == "r"
        opened.append(Path(file).resolve())
        return original(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", scoped)
    assert main(["review", str(path), "--evidence", str(evidence), "--format", "json"]) == 0
    assert set(opened) == allowed
    assert json.loads(capsys.readouterr().out)["sources"][0]["evidence"]["annotations"]
