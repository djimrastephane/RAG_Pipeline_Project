"""regenerate_bootstrap_ci_panel_postfix.py

Regenerates the paired bootstrap CI panel (Figure C.2) from the post-fix
pipeline artifacts (enable_subsection_boost=False, rrf_k=20, 224/56 chunks).

For each of the five Grampian cohorts:
  - Hybrid metrics: from per_query_results.json (hit_at_1, hit_at_3, reciprocal_rank)
  - Dense metrics: computed from dense_page_hits.jsonl vs gold_pages in per_query_results

Then runs 5000-iteration paired bootstrap and plots a 3-panel CI figure.

Usage:
    python scripts/regenerate_bootstrap_ci_panel_postfix.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO_ROOT  = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPO_ROOT / "results/thesis_ablations/chunk_size_ablation_2026-04-15/pipeline_outputs"
OUT_DIR    = REPO_ROOT / "results/bootstrap_thesis_rag_2026-04-21"

DOCS = [
    ("2020-2021", "Grampian-2020-2021"),
    ("2021-2022", "Grampian-2021-2022"),
    ("2022-2023", "Grampian-2022-2023"),
    ("2023-2024", "Grampian-2023-2024"),
    ("2024-2025", "Grampian-2024-2025"),
]
CFG_DIR = "minilmcap_{doc_id}_chunk_224_56"

N_BOOT = 10000
SEED   = 42
MRR_K  = 10
WEAKEST_COHORT = "2022-2023"


def load_hybrid(exp_dir: Path) -> dict[str, dict]:
    rows = json.loads((exp_dir / "per_query_results.json").read_text())
    return {r["query_id"]: r for r in rows}


def load_dense(exp_dir: Path) -> dict[str, list[int]]:
    """Returns {query_id: [page_number rank1, rank2, ...]} from dense_page_hits.jsonl."""
    per_query: dict[str, list[int]] = defaultdict(list)
    with open(exp_dir / "dense_page_hits.jsonl") as fh:
        for line in fh:
            row = json.loads(line)
            qid = row["query_id"]
            per_query[qid].append(int(row["page_number"]))
    return per_query


def dense_metrics(ranked_pages: list[int], gold_pages: list[int]) -> dict[str, float]:
    gold = set(gold_pages)
    hit1 = float(bool(ranked_pages[:1] and set(ranked_pages[:1]) & gold))
    hit3 = float(bool(ranked_pages[:3] and set(ranked_pages[:3]) & gold))
    rr = 0.0
    for i, p in enumerate(ranked_pages[:MRR_K], 1):
        if p in gold:
            rr = 1.0 / i
            break
    return {"hit_at_1": hit1, "hit_at_3": hit3, "mrr": rr}


def bootstrap_delta(a: np.ndarray, b: np.ndarray, seed: int) -> dict:
    n = len(a)
    rng = np.random.default_rng(seed)
    diffs = np.array([
        float(np.mean(a[rng.integers(0, n, n)]) - np.mean(b[rng.integers(0, n, n)]))
        for _ in range(N_BOOT)
    ])
    obs = float(np.mean(a) - np.mean(b))
    return {
        "observed_delta": obs,
        "ci95_low":  float(np.percentile(diffs, 2.5)),
        "ci95_high": float(np.percentile(diffs, 97.5)),
    }


def run_doc(series: str, doc_id: str) -> dict:
    exp_dir = ARTIFACT_ROOT / CFG_DIR.format(doc_id=doc_id) / doc_id
    hybrid  = load_hybrid(exp_dir)
    dense_ranked = load_dense(exp_dir)

    qids = sorted(hybrid.keys())
    h_h1, h_h3, h_mrr = [], [], []
    d_h1, d_h3, d_mrr = [], [], []

    for qid in qids:
        row = hybrid[qid]
        gold = list(row["gold_pages"])

        h_h1.append(float(row["hit_at_1"]))
        h_h3.append(float(row["hit_at_3"]))
        h_mrr.append(float(row["reciprocal_rank"]))

        dm = dense_metrics(dense_ranked.get(qid, []), gold)
        d_h1.append(dm["hit_at_1"])
        d_h3.append(dm["hit_at_3"])
        d_mrr.append(dm["mrr"])

    h1_a, h1_b = np.array(h_h1), np.array(d_h1)
    h3_a, h3_b = np.array(h_h3), np.array(d_h3)
    mr_a, mr_b = np.array(h_mrr), np.array(d_mrr)

    return {
        "series":  series,
        "doc_id":  doc_id,
        "n":       len(qids),
        "hit_at_1": bootstrap_delta(h1_a, h1_b, SEED),
        "hit_at_3": bootstrap_delta(h3_a, h3_b, SEED + 1),
        "mrr":      bootstrap_delta(mr_a, mr_b, SEED + 2),
    }


def plot_panel(ax, results: list[dict], metric: str, title: str) -> None:
    labels  = [f"{r['series']}\n(N={r['n']})" for r in results]
    x       = np.arange(len(labels))
    markers = []

    for i, r in enumerate(results):
        d   = r[metric]
        obs = d["observed_delta"]
        lo  = d["ci95_low"]
        hi  = d["ci95_high"]
        sig_pos = lo > 0
        sig_neg = hi < 0
        is_weak = r["series"] == WEAKEST_COHORT

        if is_weak:
            ax.axvspan(i - 0.4, i + 0.4, color="#E5E7EB", alpha=0.55, zorder=0)

        color = "#D97706" if sig_pos else ("#1D4ED8" if sig_neg else "#6B7280")
        ax.errorbar(
            x[i], obs,
            yerr=[[obs - lo], [hi - obs]],
            fmt="o", color=color,
            capsize=5, capthick=1.4, elinewidth=1.4,
            markersize=6, markerfacecolor="white", markeredgewidth=1.6,
            zorder=3,
        )
        markers.append((color, sig_pos, sig_neg))

    ax.axhline(0, color="#374151", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=0)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel("Δ metric (Hybrid − Dense)\nwith 95% CI", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Running paired bootstrap from post-fix artifacts...")

    results = []
    for series, doc_id in DOCS:
        print(f"  {series} ({doc_id})...", end=" ", flush=True)
        r = run_doc(series, doc_id)
        results.append(r)
        print(f"Hit@1 Δ={r['hit_at_1']['observed_delta']:+.3f} "
              f"[{r['hit_at_1']['ci95_low']:.3f}, {r['hit_at_1']['ci95_high']:.3f}]")

    # Save summary JSON
    out_json = OUT_DIR / "paired_bootstrap_summary_all.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nSaved summary: {out_json}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Paired Bootstrap Retrieval Comparison — Grampian Cohorts\n"
        r"Bootstrap $n$ = 10,000; $p < 0.05$ → 50 per cohort; orange/red markers indicate cohorts where the 95% CI excludes zero",
        fontsize=9, y=1.02,
    )

    plot_panel(axes[0], results, "hit_at_1", "Hit@1 delta")
    plot_panel(axes[1], results, "hit_at_3", "Hit@3 delta")
    plot_panel(axes[2], results, "mrr",      "MRR@10 delta")

    # Legend
    legend_handles = [
        mpatches.Patch(color="#D97706", label="Sig. positive (CI > 0)"),
        mpatches.Patch(color="#1D4ED8", label="Sig. negative (CI < 0)"),
        mpatches.Patch(color="#6B7280", label="Not significant"),
        mpatches.Patch(color="#E5E7EB", label=f"Weakest cohort ({WEAKEST_COHORT})"),
    ]
    fig.legend(handles=legend_handles, loc="upper center",
               ncol=4, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.98))

    fig.tight_layout()
    out_png = OUT_DIR / "paired_bootstrap_ci_panel_Grampian_2020_2025_hybrid_vs_dense.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved figure: {out_png}")


if __name__ == "__main__":
    main()
