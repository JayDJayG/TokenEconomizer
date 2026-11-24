# Prune diagnostics (fork extension)

This fork adds optional, explicit-input diagnostics in `tokeneconomizer.prune`.
The command does not discover sessions, initialize live plugin state, call a
model, spawn agents, change settings, or enforce budgets. No plugin installation
or hooks are needed to run it from this checkout:

```powershell
python -m tokeneconomizer.prune review 'C:\sessions\events.jsonl'
python -m tokeneconomizer.prune review 'C:\sessions\events.jsonl' --format json
```

The installed console commands are `te-prune review ...` and
`te prune review ...`. The latter dispatches before live host discovery and state
handling. Use `--input-host claude`, `--input-host copilot`, or `--input-host codex`
to require a format; otherwise each named input is detected independently.
The existing `te --host` and `--cwd` options select live context for other
commands and do not constrain a Prune review. Relative input paths resolve from
the command's actual working directory.

## Implemented first slice

Strict UTF-8 JSONL loading supports Claude conversation transcripts, Copilot
event logs, and Codex exec streams or uncompressed rollouts. It reuses upstream
tool readers and supplements call IDs, physical source lines, explicit turns,
and unknown or incomplete outcomes. Native timestamps, model identity, session
identity, visible messages, and token/cache usage are retained where present.
Claude usage is deduplicated by message ID, Codex cumulative usage keeps the
last snapshot, and Copilot shutdown usage stays separated by model. Missing
cache fields stay unknown rather than silently becoming zero. It does not read
linked subagent files or infer ancestry from message UUIDs or timing.
Unsupported tool-event kinds are reported and break polling comparisons rather
than being treated as proof that no intervening work occurred.

The initial detector finds consecutive, non-blocking reads of the same completed
agent result. It requires structured JSON status results, an explicit turn,
identical request/result data, and non-overlapping calls. New user prompts,
intervening tool work, changed results, failures, or blocking waits break a match.
Recognized polling tools are `read_agent` (including a dotted namespace) and
Claude's `TaskOutput`; arbitrary textual status output is not guessed.

Running-state polling is assessed only when the reviewer explicitly confirms
completion notifications for these tasks:

```powershell
python -m tokeneconomizer.prune review 'C:\sessions\events.jsonl' --notifications-supported
```

That flag records supplied evidence; it neither enables notifications nor changes
waiting behavior. Without it, repeated running states are not findings.

## Optional evidence

`--evidence FILE` reads one explicitly named JSON annotation manifest. It can
provide facts the host did not record; it is not a collector or proof that those
facts are true. Every source must already be supplied on the command line and
its `sha256` must match the transcript's exact UTF-8 bytes, including BOM and
line endings. Relative paths resolve beside the manifest. Stale manifests,
unsupplied paths, conflicting native identities, and ambiguous calls fail.

```json
{
  "schema_version": 1,
  "sources": [{
    "path": "events.jsonl",
    "sha256": "<digest from the review report>",
    "session": {
      "agent_id": "coordinator",
      "usage_includes_children": false
    },
    "calls": [{
      "line": 2,
      "tool": "Read",
      "call_id": "c1",
      "facts": {
        "purpose": "investigation",
        "required": false,
        "input_state": {
          "fingerprint": "<recorded state digest>",
          "scope": "module inputs",
          "captured_at": "2026-11-24T12:00:00Z",
          "complete": true,
          "covers": ["source"]
        }
      }
    }]
  }]
}
```

Input-state evidence must have been captured for the historical call, not
computed from today's checkout. `complete` asserts that the stated scope
covers every relevant input; verification also needs source, tests,
configuration, dependencies, and environment. Reports expose coverage counts
without presenting absent metadata as a passed check. Annotations do not grant
permission to read any additional files.

## Duplicate investigation

Read/search checks recognize `Read`, `Grep`, `Glob`, `view`, `rg`, `glob`,
`read_file`, and `search`, including dotted namespaces. Consecutive calls must
have the same turn, exact equivalent arguments and identical successful
results. Each call needs `purpose: "investigation"`, `required: false`, and a
complete `input_state` with matching fingerprint, scope, and coverage.
`input_state.at` must be `"call_start"` and `captured_at` must name the historical
snapshot. A future-dated snapshot is not usable. Float-valued requests are not
compared because JSON decoding can lose precision.

Required reads, absent purpose/state, new prompts, intervening work, changed
results, failures, and overlapping calls are preserved. The evidence is
supplied, not recomputed by Prune.

## Output overfetch

A successful text result can be reviewed when its call annotation declares a
purpose and `output_need: {"sufficient": true, "reason": "task needs these lines",
"selected_lines": [1, 3]}`. Line indexes refer to the recorded result, not source
file line numbers. Prune compares actual UTF-8 byte counts with that selection;
it does not estimate tokens or infer sufficiency from length. Invalid indexes
fail explicitly. Without the supplied intent evidence, even a huge result
remains unassessable. Failed/unknown calls are preserved, and a recommendation
must retain the full artifact by reference rather than destroy failure evidence.

## Redundant verification

Verification calls need `purpose: "verification"`, `required: false`, and
`verification: {"deterministic": true, "unresolved_concerns": false}`.
Both call-start snapshots must completely cover `source`, `tests`,
`configuration`, `dependencies`, and `environment`. Only consecutive,
non-overlapping, identical successful checks with matching requests, results,
state, and turn are findings. Recognized tools include `Bash`, `shell`,
`powershell`, `run_tests`, `test`, `execute_command`, and `terminal`.

Changed selectors, outputs, inputs, failures, explicit reruns, missing coverage,
and unresolved concerns prevent a cut. Prune never executes the check itself.

## Agent structure and duplicate delegation

Related inputs need a common `session.task_id` and explicit `agent_id`/
`parent_agent_id`; `session.is_root: true` identifies a known task root.
Successful `spawn_agent`, `task`, `Task`, or `Agent` calls can add child links
from structured result IDs or call `spawn` annotations. Conflicting identities,
parents, and cycles fail explicitly. Unlinked sessions are not joined by timing.

Optional root `session.budgets` accepts nonnegative `max_depth`, `max_fan_out`
(per parent), and `max_spawns`. Exceedances are budget observations, not claims
that justified parallel work was wasteful. No universal cap is imposed or
enforced. Metrics are observed lower bounds; absent links cannot prove compliance.
Reports bound displayed nodes to 20 while computing metrics over the full graph.

Duplicate delegation additionally needs equivalent briefs and complete call-start
input states, plus `spawn` facts: `independent: false`, `completed: true`,
`new_information: false`, and matching `objective_fingerprint` and
`result_fingerprint`. Different worker IDs alone do not prove duplicated work.
Supplied evidence paths are retained as provenance, not new access permissions.

## Limits and output

Reports include at most three cuts, source references, replacements, safeguards,
and coverage limits. Unimplemented checks are explicitly labeled, not passed:
workflow/model-use checks, and billed-cost accounting remain future work. There
are no savings estimates or claims that the whole session was efficient.

By default the report goes to stdout. `--output FILE` creates only the requested
file and refuses to overwrite an existing report or input. Malformed, empty,
unsupported, mixed-format, or unreadable inputs fail with an error and exit 2;
a valid advisory report exits 0 even if it contains findings.

Validate the new behavior with `python -m pytest tests\test_prune_review.py`.
