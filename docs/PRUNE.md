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
Claude's `TaskOutput`; arbitrary textual status output and precision-ambiguous
float-valued requests are not guessed.

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

## Context and communication

Forwarding checks require actual verbatim text in a successful worker brief.
A `forwarding` annotation declares `kind: "conversation"` or `"tool_output"`,
`source_lines`, `needed_source_lines`, `sufficient: true`,
`preserves_requirements: true`, and a task-specific `reason`. Conversation
references must cover all prior recorded visible messages; tool-output
references point to prior successful result lines. All copies must appear
in order and without overlapping text spans. Smaller-brief sufficiency is
supplied evidence, never inferred from length. Byte amplification is observed,
not a token or dollar estimate. Repeated copies are grouped.

Ping-pong checks recognize `write_agent` and `send_input`. A `communication`
annotation supplies `conversation_id`, `progress_fingerprint`,
`new_information: false`, and `required: false`; `sender` can fill missing
identity and must not contradict native identity. Receivers must match actual
arguments. Three or more alternating messages must repeat an earlier message
without changed progress. Other work, new prompts, overlap, or new information
breaks the chain. Cross-file chains require complete timezone-aware start/end
and user-event timestamps with no known normalization gaps; otherwise only
within-file chains are assessed.

## Workflow diagnostics

`batch` facts (`group`, `independent: true`, `supported: true`) identify calls
that can share a request. Distinct native Claude message IDs, or supplied
`round_trip_id` values, must prove separate model round trips. Already batched,
overlapping, dependent, or unknown-capability calls are preserved.

An optional `session.scope_policy` declares `id`, `flavour` (`windows` or
`posix`), `complete: true`, `allowed_paths`, and optional `exceptions`.
Known read paths can be compared lexically only when their call names that
`scope_policy_id`, is `required: false`, and records `new_information: false`.
Mixed absolute/relative paths and `..` remain unassessable; no filesystem or
symlink discovery occurs. Scope observations preserve justified investigation.

`events` annotations identify visible `line` and `role`, with `facts.planning`
containing `scope`, `input_fingerprint`, `progress_fingerprint`,
`inputs_complete: true`, `required: false`, and `new_information: false`.
The historical snapshot also needs `at: "event"` and timezone-aware `captured_at`.
Only identical visible plans without intervening work or new prompts are
findings. Hidden thinking/reasoning blocks are never inspected. Repeated Claude
message snapshots are not new planning events.

Call `workflow` facts can nominate a `procedure_id`, distinct `run_id`,
consecutive positive `step`, stable `step_key`, `reusable: true`, and
`final_step: true` on the last step. Two complete successful matching procedures
produce an optional skill candidate, not a waste verdict. No skill is written,
installed, or executed.

Cuts on the same recorded work are coalesced with their related replacements.
Waste cuts rank ahead of scope/budget observations and optional skill candidates;
ranking uses observed calls/messages and bytes, never invented monetary savings.

## Model and cost reporting

Model diagnostics require `session.model_policy` with an `id` and a
`preferred_models` task-class map. Call `model_use` evidence needs the matching
`policy_id`, `task_class`, `substitution_approved: true`,
`quality_preserved: true`, and `context_sufficient: true`. Billed identity comes
from a Claude message or explicit `billed_model` evidence, never a host's selected
model alone. This reports a policy mismatch, not a claim that a named model is
universally excessive or that a cheaper model preserves every task's quality.

Cost reports keep measured token/cache fields separate from priced estimates and
unverified supplied billing. No character-to-token approximation, default model,
legacy token-total fallback, or zero-for-missing-usage enters the calculation.
Claude message IDs, cumulative Codex totals and per-model Copilot totals avoid
native duplicate counting. Mixed-model Codex cumulative usage stays unattributed.
Explicit task/agent ancestry and `usage_includes_children` prevent summing child
usage already included in parent totals; unknown overlap blocks aggregation.

Optional `session.pricing` maps each billed model to a quote:
`{"input_usd_per_million": 2, "source": "provider quote",
"as_of": "2026-11-24"}`. It uses upstream `ledger`/`pricing` ratios unless an
explicit normalized `weights` table supplies input/cache-read/cache-write/output
rates. These are price snapshots, not live tariffs. Unspecified cache-write
lifetime uses the selected rate's default and is labeled as an assumption.
Unknown models or partial cache breakdowns stay unpriced. USD values are decimal
strings. Token equivalents are aggregated only for a single billed model.

`ledger_overhead` accepts upstream mechanism-overhead fields in their recorded
effective-token units. Its `ledger_overhead_model` and quote allow USD estimation;
`ledger_overhead_included` declares whether that overhead is already in the
selected aggregate usage. Other worker/reviewer traces can be supplied with
`usage_role: "worker"` or `"review"`. Missing overhead is not zero:
`overhead_complete: true` must explicitly attest coverage. `accounting_complete:
true` attests full task/worker/review scope; it is a caller assertion, not something
Prune can independently prove from a partial transcript.

Optional `billing` contains a scope-unique `receipt_id`, decimal `usd`,
`includes_children`, and `scope_complete`. Duplicate receipts are not added
twice; their authenticity is not verified. No historical review infers savings.
The reviewer makes zero model calls; paid consumption of the resulting report
must be recorded separately if used in a controlled comparison.

## Labeled trace calibration

`te-prune calibrate MANIFEST --format json` checks supplied scenario labels.
A schema-1 manifest has `kind: "calibration"` and `cases`, each with a unique
`name`, explicitly named `transcripts`, optional `evidence`/`input_host`/
`notifications_supported`, and a `labels` map from pattern names to `present`,
`absent`, or `unassessable`. Relative inputs resolve beside the manifest.
Only these named inputs are opened; ordinary reports cannot act as manifests.

Counts cover every detected pattern before the three-cut display limit.
Abstentions are separate from true negatives; missing evidence is not efficiency.
An abstention cannot match `present` or `absent`. Case details are capped at 20,
but all cases contribute to metrics and the overall result. Mismatch exits 1,
malformed inputs exit 2, matched labels exit 0. This replays
history to measure detection, not savings or real-world task success.
The checked-in schema vectors are explicitly synthetic and sanitized, based on
upstream reader/test formats. They are not captured user sessions. Genuine
session calibration still requires explicitly supplied, sanitized real traces.

## Offline cost and quality comparisons

`te-prune compare MANIFEST --format json` reviews previously recorded baseline
and candidate runs. It does not execute tasks, tests, agents, or models. A
schema-1 manifest requires `kind: "comparison"`, one explicit `cost_basis`
(`estimated_usd` or `reported_billed_usd`), and nonempty `pairs`:

```json
{
  "schema_version": 1,
  "kind": "comparison",
  "cost_basis": "estimated_usd",
  "pairs": [{
    "name": "scenario",
    "required_checks": ["tests"],
    "required_approvals": [],
    "requirements_complete": true,
    "baseline": {
      "run_id": "baseline-1",
      "transcripts": ["baseline.jsonl"],
      "evidence": "baseline-evidence.json",
      "controls": {
        "task_fingerprint": "recorded-task",
        "input_fingerprint": "recorded-inputs",
        "requirements_fingerprint": "recorded-requirements",
        "environment_fingerprint": "recorded-environment",
        "pricing_fingerprint": "recorded-price-card"
      },
      "outcome": {
        "task_success": true, "complete": true,
        "checks": {"tests": true}, "approvals": {}
      }
    },
    "candidate": {
      "run_id": "candidate-1",
      "transcripts": ["candidate.jsonl"],
      "evidence": "candidate-evidence.json",
      "controls": {
        "task_fingerprint": "recorded-task",
        "input_fingerprint": "recorded-inputs",
        "requirements_fingerprint": "recorded-requirements",
        "environment_fingerprint": "recorded-environment",
        "pricing_fingerprint": "recorded-price-card"
      },
      "outcome": {
        "task_success": true, "complete": true,
        "checks": {"tests": true}, "approvals": {}
      }
    }
  }]
}
```

Inputs and evidence resolve beside the manifest and must be explicitly named.
Run IDs and pair names must be unique. Recordings, message usage, cumulative
session scopes, or billing receipts cannot be reused across different runs;
copies are not independent observations. Unknown outcomes may be omitted or
represented by null booleans; malformed consumed values fail explicitly.

Both runs must match the recorded task/input/requirements/environment controls.
Estimated comparisons additionally require a matching pricing fingerprint and
consistent supplied rates/weights for each common model. USD estimates use all
reported execution, worker, paid-review and legacy overhead scopes under the
accounting rules above; supplied billing is a separate, never-mixed basis.
Missing costs, uncertain overlap, or unpriced overhead do not become zero.

All runs, including failures, contribute to totals and quality results. A cheaper
failure is not a useful reduction. Required checks and approvals must be explicitly
complete and successful on both sides. Details are capped at 20 pairs, while all
pairs contribute to aggregate results. Unreported failed runs cannot be independently
detected: the caller must supply the complete experiment, not just successes.
Decimal totals preserve supplied receipt precision up to 10,000 digits; unsupported
precision/range fails explicitly instead of overflowing or rounding spend to zero.
Exit 0 means comparable costs and preserved
quality, not guaranteed savings; incomplete or degraded comparisons exit 1, input
errors exit 2. Results remain conditional on caller-supplied evidence, not causal
proof from replaying history.

Measured local wall/CPU timers cover manifest processing and input loading/review;
they exclude serialization, original task time, and later paid report ingestion.
Prune itself makes zero model calls and returns zero paid reviewer tokens. Include
separate paid review traces/receipts if an agent consumed these reports.

## Limits and output

Reports include at most three cuts, source references, replacements, safeguards,
and coverage limits. Implemented checks with missing evidence are explicitly
unassessable, not passed. There are no historical savings estimates or claims
that the whole session was efficient.

By default the report goes to stdout. `--output FILE` creates only the requested
file and refuses to overwrite an existing report or input. Malformed, empty,
unsupported, mixed-format, or unreadable inputs fail with an error and exit 2;
a valid advisory report exits 0 even if it contains findings.

Validate the new behavior with `python -m pytest tests\test_prune_review.py`.
