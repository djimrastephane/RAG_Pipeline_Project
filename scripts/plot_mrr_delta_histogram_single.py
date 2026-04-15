from __future__ import annotations

import argparse
from pathlib import Path

import _matplotlib_env
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot single histogram of per-query MRR deltas (Hybrid - Dense)."
    )
    p.add_argument("--input-csv", required=True, help="CSV with delta_mrr_a_minus_b column.")
    p.add_argument("--cohort", required=True, help="Cohort label for title.")
    p.add_argument("--output-path", required=True, help="Output figure path (PNG).")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv).resolve()
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    if "delta_mrr_a_minus_b" not in df.columns:
        raise ValueError("Input CSV missing required column: delta_mrr_a_minus_b")

    x = df["delta_mrr_a_minus_b"].astype(float).to_numpy()
    if x.size == 0:
        raise ValueError("Input CSV has no rows.")

    mean_delta = float(np.mean(x))

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.hist(x, bins=18, color="#c7c7c7", edgecolor="#555555", linewidth=0.8)
    ax.axvline(0.0, color="#4d4d4d", linestyle="--", linewidth=1.2, label="Zero baseline")
    ax.axvline(mean_delta, color="#1f77b4", linestyle="-", linewidth=1.4, label=f"Mean delta = {mean_delta:.3f}")

    ax.set_title("Per-query MRR Delta Distribution\n" + f"(Hybrid - Dense, {args.cohort})")
    ax.set_xlabel("Per-query MRR delta (Hybrid - Dense)")
    ax.set_ylabel("Number of queries")
    ax.grid(axis="y", alpha=0.3, linewidth=0.7)
    ax.legend(frameon=True, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
