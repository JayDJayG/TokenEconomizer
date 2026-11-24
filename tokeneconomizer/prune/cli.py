"""Standalone review command; never builds a live plugin context."""
import argparse
import json
import sys

from .review import HOSTS, InputError, load_trace, review
from .evidence import apply_evidence


def add_review_arguments(parser):
    parser.add_argument("action", choices=("review", "calibrate"))
    parser.add_argument("transcripts", nargs="+", metavar="TRANSCRIPT")
    parser.add_argument("--input-host", choices=HOSTS,
                        help="require this input format; otherwise detect each transcript")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", metavar="FILE",
                        help="create a new report file; never overwrite an existing file")
    parser.add_argument("--evidence", metavar="FILE",
                        help="read optional digest-bound metadata for the named transcripts")
    parser.add_argument("--notifications-supported", action="store_true",
                        help="confirm completion notifications for the reviewed agent tasks; "
                             "otherwise only repeated completed-result retrieval is flagged")


def _markdown(report):
    lines = ["# Prune review", "", "Advisory only; no execution controls were changed.", ""]
    if not report["findings"]:
        lines.extend(["No supported high-confidence cut found.", ""])
    for i, finding in enumerate(report["findings"], 1):
        span = (f", lines {finding['first_line']}-{finding['last_line']}"
                if finding["first_line"] is not None else "; supplied metadata")
        lines.extend([
            f"{i}. **{finding['title']}**",
            f"   Evidence: {json.dumps(finding['source'])}{span}; "
            f"{finding['calls']} call(s). {finding['evidence_basis']}",
            f"   Replacement: {finding['replacement']}", "",
        ])
        for source in finding.get("evidence_sources", []):
            lines.append(f"   Supplied evidence: {json.dumps(source)}")
        if finding.get("related_patterns"):
            lines.append(f"   Related: {', '.join(finding['related_patterns'])}.")
            for replacement in finding.get("related_replacements", []):
                lines.append(f"   Also: {replacement}")
    costs = report["costs"]
    lines.extend(["## Usage and cost", "",
                  "Token fields are measured; price calculations are estimates, not billing.",
                  f"Total estimated USD: {costs['total_estimated_usd'] or 'unavailable'}. "
                  f"Supplied billed USD: {costs['reported_billed_usd'] or 'unavailable'}."])
    equivalent = costs["effective_input_token_equivalents"]
    lines.append("Single-model cache-weighted input equivalents: "
                 + (str(equivalent) if equivalent is not None else "unavailable") + ".")
    for note in costs["notes"]:
        lines.append(f"- {note}")
    lines.extend(["", "## Coverage and limits", ""])
    for check in report["checks"]:
        lines.append(f"- {check['pattern']}: {check['status']}. {check['reason']}")
    for source in report["sources"]:
        for limit in source["limitations"]:
            lines.append(f"- {json.dumps(source['path'])}: {limit}")
    if report["additional_findings"]:
        lines.append(f"- {report['additional_findings']} additional finding(s) omitted "
                     "to keep the review to three cuts.")
    lines.extend(["", *report["safeguards"]])
    return "\n".join(lines) + "\n"


def run_review(args):
    try:
        if args.action == "calibrate":
            if len(args.transcripts) != 1 or args.input_host or args.evidence or args.notifications_supported:
                raise InputError("calibrate requires one manifest; specify review options inside each case")
            from .evaluation import calibrate
            report = calibrate(args.transcripts[0])
        else:
            traces = [load_trace(path, args.input_host) for path in args.transcripts]
            if args.evidence:
                traces = apply_evidence(traces, args.evidence)
            report = review(traces, notifications_supported=args.notifications_supported)
        text = (json.dumps(report, indent=2) + "\n" if args.format == "json"
                else _evaluation_markdown(report) if args.action == "calibrate" else _markdown(report))
        if args.output:
            with open(args.output, "x", encoding="utf-8", newline="\n") as output:
                output.write(text)
        else:
            sys.stdout.write(text)
    except (InputError, OSError) as exc:
        print(f"prune: {exc}", file=sys.stderr)
        return 2
    return 1 if args.action == "calibrate" and not report["passed"] else 0


def _evaluation_markdown(report):
    lines = ["# Prune calibration", "", f"Label checks passed: {report['passed']}.",
             json.dumps(report["metrics"], sort_keys=True), ""]
    for case in report["cases"]:
        lines.append(f"- {json.dumps(case['name'])}: "
                     + ", ".join(f"{item['pattern']} {'matched' if item['match'] else 'mismatched'}"
                                 for item in case["outcomes"]))
    return "\n".join(lines + ["", *report["limits"]]) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(prog="te-prune",
                                     description="Review only explicitly supplied transcripts.")
    add_review_arguments(parser)
    return run_review(parser.parse_args(argv))
