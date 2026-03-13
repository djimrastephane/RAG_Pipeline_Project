from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = ["answer_relevancy", "faithfulness", "context_precision", "context_recall"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build baseline-vs-generated RAGAS comparison chart and CSV.")
    p.add_argument("--baseline-summary", required=True, help="Path to baseline ragas_summary.json")
    p.add_argument("--generated-summary", required=True, help="Path to generated ragas_summary.json")
    p.add_argument("--out-csv", required=True, help="Output CSV path")
    p.add_argument("--out-png", required=True, help="Output PNG path")
    p.add_argument("--title", default="RAGAS 75-query comparison: baseline vs generated answer")
    return p.parse_args()


def _load_means(path: Path) -> dict[str, float]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    means = obj.get("metrics_mean", {}) if isinstance(obj, dict) else {}
    out: dict[str, float] = {}
    for m in METRICS:
        v = means.get(m)
        out[m] = float(v) if v is not None else np.nan
    return out


def main() -> None:
    args = parse_args()
    baseline_path = Path(args.baseline_summary)
    generated_path = Path(args.generated_summary)
    out_csv = Path(args.out_csv)
    out_png = Path(args.out_png)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    baseline = _load_means(baseline_path)
    generated = _load_means(generated_path)

    rows = []
    for m in METRICS:
        b = baseline[m]
        g = generated[m]
        rows.append(
            {
                "metric": m,
                "baseline": b,
                "generated": g,
                "delta_generated_minus_baseline": g - b,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    labels = [m.replace("_", " ").title() for m in METRICS]
    x = np.arange(len(METRICS))
    width = 0.36

    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, df["baseline"].values, width, label="Baseline")
    plt.bar(x + width / 2, df["generated"].values, width, label="Generated answer")
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title(args.title)
    plt.legend(frameon=False)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
