from __future__ import annotations

import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pandas as pd


class StepTimer:
    """
    Lightweight step-level timer for profiling pipeline stages.

    Usage:
        timer = StepTimer()
        timer.mark("step name")
        ...
        timer.report()
    """

    def __init__(self):
        self.start = time.perf_counter()
        self.last = self.start
        self.steps = OrderedDict()
        self.extra_lines: list[tuple[str, float, float]] = []

    def mark(self, label: str) -> None:
        now = time.perf_counter()
        self.steps[label] = {
            "step_seconds": now - self.last,
            "total_seconds": now - self.start,
        }
        self.last = now

    def report(self) -> None:
        print("\n=== PIPELINE TIMING REPORT ===")
        for k, v in self.steps.items():
            print(
                f"{k:<45} "
                f"step={v['step_seconds']:>7.3f}s  "
                f"total={v['total_seconds']:>7.3f}s"
            )
        for label, step_seconds, total_seconds in self.extra_lines:
            print(
                f"{label:<45} "
                f"step={step_seconds:>7.3f}s  "
                f"total={total_seconds:>7.3f}s"
            )

    def add_extra(self, label: str, step_seconds: float, total_seconds: float) -> None:
        self.extra_lines.append((label, step_seconds, total_seconds))


def safe_json_dump(obj: Any, path: Path) -> None:
    """Write JSON file safely with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def describe_series(s: pd.Series) -> dict:
    """Compute descriptive statistics for a pandas Series as JSON-serializable dict."""
    d = s.describe()
    return {k: (float(v) if hasattr(v, "item") else v) for k, v in d.to_dict().items()}
