"""Explicitly opt-in after-task reporting; never enrolls itself in live hooks."""
from pathlib import Path
import sys
from time import perf_counter, process_time

from ..stage import Stage
from .evaluation import _elapsed, review_case
from .types import InputError, optional_text


class PruneReviewStage(Stage):
    name = "prune_review"

    def __init__(self, enabled=False):
        if type(enabled) is not bool:
            raise InputError("PruneReviewStage enabled must be an explicit boolean")
        self.enabled = enabled

    def after_task(self, task, row, run_dir_path, srow):
        if self.enabled is False:
            return 0
        if self.enabled is not True:
            raise InputError("PruneReviewStage enabled must be an explicit boolean")
        if not isinstance(srow, dict):
            raise InputError("PruneReviewStage requires a caller-owned mutable task record")
        wall, cpu = perf_counter(), process_time()
        try:
            inputs = row.get("prune_inputs") if isinstance(row, dict) else None
            if not isinstance(inputs, dict):
                raise InputError("enabled PruneReviewStage requires explicit row.prune_inputs")
            paths = inputs.get("transcripts")
            if not isinstance(paths, list) or not paths:
                raise InputError("prune_inputs requires named absolute transcript paths")
            for name in [*paths, *([inputs["evidence"]] if inputs.get("evidence") is not None else [])]:
                if optional_text(name, "prune input") is None or not Path(name).is_absolute():
                    raise InputError("PruneReviewStage inputs must be explicit absolute paths")
            report, _traces, _overhead = review_case(inputs, Path())
        except (InputError, OSError) as exc:
            srow[self.name] = {"status": "error", "error": str(exc),
                               "local_review_overhead": _elapsed(wall, cpu)}
            print(f"prune after_task: {exc}", file=sys.stderr)
        else:
            srow[self.name] = {"status": "reviewed", "report": report,
                               "local_review_overhead": _elapsed(wall, cpu)}
        return 0
