"""
Paired bootstrap for per-query delta between RRF and score-fusion.

Delta definition (per query):
    delta = metric_rrf - metric_score_fusion

Metrics:
- hit1 (top-1 page hit)
- mrr10 (reciprocal rank up to 10 pages)

Decision rule:
- CI includes 0: no reliable difference
- CI < 0: score-fusion wins
- CI > 0: RRF wins
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.ablate_fusion_strategy_temporal import (
    build_query_packs,
    build_temporal_transitions,
    page_hit_at_1,
    page_mrr_at_10,
    rrf_rank,
    weighted_score_rank,
)


def paired_bootstrap_mean_delta(
    deltas: np.ndarray, n_boot: int, seed: int
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(deltas)
    if n == 0:
        return {
            "mean_delta": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "n": 0,
        }
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = float(np.mean(deltas[idx]))
    return {
        "mean_delta": float(np.mean(deltas)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "n": int(n),
    }


def decision_from_ci(ci_low: float, ci_high: float) -> str:
    if ci_low <= 0.0 <= ci_high:
        return "no_reliable_difference"
    if ci_high < 0.0:
        return "score_fusion_wins"
    return "rrf_wins"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paired bootstrap delta: RRF vs score-fusion on temporal test folds.")
    p.add_argument("--run-root", default="data_processed")
    p.add_argument("--model", default="models/all-MiniLM-L6-v2")
    p.add_argument("--dense-weight", type=float, default=0.5)
    p.add_argument("--bm25-weight-rrf", type=float, default=2.0)
    p.add_argument("--bm25-weight-scorefusion", type=float, default=0.5)
    p.add_argument("--rrf-k", type=int, default=20)
    p.add_argument("--max-k-search", type=int, default=200)
    p.add_argument("--bm25-k1", type=float, default=1.5)
    p.add_argument("--bm25-b", type=float, default=0.75)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-dir",
        default="results/ablations/ablation_thesis_5docs_q50/final_selection/fusion_strategy_temporal_compare_2026-03-01/bootstrap_delta_2026-03-01",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    transitions = build_temporal_transitions(run_root)
    if not transitions:
        raise ValueError(f"No consecutive complete transitions found under {run_root}")

    model = SentenceTransformer(str(args.model))

    cache: dict[str, tuple[pd.DataFrame, list[Any]]] = {}
    for _, tr_doc, te_doc in transitions:
        for doc_id in [tr_doc, te_doc]:
            if doc_id not in cache:
                cache[doc_id] = build_query_packs(
                    data_dir=run_root / doc_id,
                    model=model,
                    max_k_search=int(args.max_k_search),
                    bm25_k1=float(args.bm25_k1),
                    bm25_b=float(args.bm25_b),
                )

    per_query_rows: list[dict[str, Any]] = []
    for fold_label, _, te_doc in transitions:
        te_meta, te_packs = cache[te_doc]
        for p in te_packs:
            ranked_rrf = rrf_rank(
                dense_ranked=p.dense_ranked,
                bm25_ranked=p.bm25_ranked,
                dense_weight=float(args.dense_weight),
                bm25_weight=float(args.bm25_weight_rrf),
                rrf_k=int(args.rrf_k),
            )
            ranked_sf = weighted_score_rank(
                dense_score_map=p.dense_score_map,
                bm25_score_map=p.bm25_score_map,
                dense_weight=float(args.dense_weight),
                bm25_weight=float(args.bm25_weight_scorefusion),
            )

            hit1_rrf = page_hit_at_1(p.expected_pages, te_meta, ranked_rrf)
            hit1_sf = page_hit_at_1(p.expected_pages, te_meta, ranked_sf)
            mrr10_rrf = page_mrr_at_10(p.expected_pages, te_meta, ranked_rrf)
            mrr10_sf = page_mrr_at_10(p.expected_pages, te_meta, ranked_sf)

            per_query_rows.append(
                {
                    "fold": fold_label,
                    "test_doc_id": te_doc,
                    "query_id": p.query_id,
                    "hit1_rrf": float(hit1_rrf),
                    "hit1_scorefusion": float(hit1_sf),
                    "delta_hit1_rrf_minus_scorefusion": float(hit1_rrf - hit1_sf),
                    "mrr10_rrf": float(mrr10_rrf),
                    "mrr10_scorefusion": float(mrr10_sf),
                    "delta_mrr10_rrf_minus_scorefusion": float(mrr10_rrf - mrr10_sf),
                }
            )

    qdf = pd.DataFrame(per_query_rows)
    if qdf.empty:
        raise ValueError("No per-query rows computed.")

    hit1_delta = qdf["delta_hit1_rrf_minus_scorefusion"].to_numpy(dtype=np.float64)
    mrr10_delta = qdf["delta_mrr10_rrf_minus_scorefusion"].to_numpy(dtype=np.float64)

    hit1_boot = paired_bootstrap_mean_delta(hit1_delta, n_boot=int(args.n_bootstrap), seed=int(args.seed))
    mrr10_boot = paired_bootstrap_mean_delta(mrr10_delta, n_boot=int(args.n_bootstrap), seed=int(args.seed) + 1)

    summary = {
        "settings": {
            "dense_weight": float(args.dense_weight),
            "bm25_weight_rrf": float(args.bm25_weight_rrf),
            "bm25_weight_scorefusion": float(args.bm25_weight_scorefusion),
            "rrf_k": int(args.rrf_k),
            "max_k_search": int(args.max_k_search),
            "bm25_k1": float(args.bm25_k1),
            "bm25_b": float(args.bm25_b),
            "n_bootstrap": int(args.n_bootstrap),
            "seed": int(args.seed),
            "delta_definition": "metric_rrf - metric_scorefusion",
        },
        "n_queries_total": int(len(qdf)),
        "n_folds": int(qdf["fold"].nunique()),
        "hit1": {
            **hit1_boot,
            "decision": decision_from_ci(hit1_boot["ci95_low"], hit1_boot["ci95_high"]),
        },
        "mrr10": {
            **mrr10_boot,
            "decision": decision_from_ci(mrr10_boot["ci95_low"], mrr10_boot["ci95_high"]),
        },
    }

    per_query_path = out_dir / "paired_bootstrap_per_query_deltas.csv"
    summary_path = out_dir / "paired_bootstrap_summary.json"
    md_path = out_dir / "paired_bootstrap_report.md"
    qdf.to_csv(per_query_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# Paired Bootstrap: RRF vs Score-Fusion",
        "",
        "Delta per query: `metric_rrf - metric_scorefusion`",
        "",
        "## Hit@1",
        f"- mean delta: `{hit1_boot['mean_delta']:.6f}`",
        f"- 95% CI: `[{hit1_boot['ci95_low']:.6f}, {hit1_boot['ci95_high']:.6f}]`",
        f"- decision: `{summary['hit1']['decision']}`",
        "",
        "## MRR@10",
        f"- mean delta: `{mrr10_boot['mean_delta']:.6f}`",
        f"- 95% CI: `[{mrr10_boot['ci95_low']:.6f}, {mrr10_boot['ci95_high']:.6f}]`",
        f"- decision: `{summary['mrr10']['decision']}`",
        "",
        "## Settings",
        f"- dense_weight: `{args.dense_weight}`",
        f"- bm25_weight_rrf: `{args.bm25_weight_rrf}`",
        f"- bm25_weight_scorefusion: `{args.bm25_weight_scorefusion}`",
        f"- rrf_k: `{args.rrf_k}`",
        f"- n_bootstrap: `{args.n_bootstrap}`",
    ]
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    print("Saved:")
    print("-", per_query_path)
    print("-", summary_path)
    print("-", md_path)
    print("Hit@1:", hit1_boot, "decision=", summary["hit1"]["decision"])
    print("MRR@10:", mrr10_boot, "decision=", summary["mrr10"]["decision"])


if __name__ == "__main__":
    main()
