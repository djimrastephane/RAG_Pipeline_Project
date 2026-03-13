from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def parse_int_list(s: str) -> list[int]:
    vals = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not vals or min(vals) <= 0:
        raise ValueError("values must be positive integers")
    return vals


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablate MAX_K_SEARCH sensitivity for hybrid retrieval.")
    p.add_argument("--run-root", required=True)
    p.add_argument("--run-filter", default="chunk_280_90_seg_off_dense_rerank_on")
    p.add_argument("--model", default="models/all-MiniLM-L6-v2")
    p.add_argument("--k-list", default="1,3,5,10")
    p.add_argument("--max-k-search-list", default="25,50,100,150,200")
    p.add_argument("--rrf-k", type=int, default=20)
    p.add_argument("--dense-weight", type=float, default=0.5)
    p.add_argument("--bm25-weight", type=float, default=2.0)
    p.add_argument("--bm25-k1", type=float, default=1.5)
    p.add_argument("--bm25-b", type=float, default=0.75)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def run_eval(data_dir: Path, args: argparse.Namespace, max_k_search: int) -> tuple[dict[str, Any], float]:
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env["MAX_K_SEARCH"] = str(int(max_k_search))

    cmd = [
        ".venv/bin/python",
        "scripts/retrieval_eval_hybrid.py",
        "--data-dir",
        str(data_dir),
        "--model",
        str(args.model),
        "--k-list",
        str(args.k_list),
        "--rrf-k",
        str(int(args.rrf_k)),
        "--dense-weight",
        str(float(args.dense_weight)),
        "--bm25-weight",
        str(float(args.bm25_weight)),
        "--bm25-k1",
        str(float(args.bm25_k1)),
        "--bm25-b",
        str(float(args.bm25_b)),
    ]

    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, env=env)
    elapsed = time.perf_counter() - t0

    metrics_path = data_dir / "retrieval_metrics_hybrid.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return metrics, float(elapsed)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    max_k_list = parse_int_list(args.max_k_search_list)

    run_root = Path(args.run_root).resolve()
    data_dirs = sorted(
        d2
        for d1 in run_root.iterdir()
        if d1.is_dir() and args.run_filter in d1.name
        for d2 in d1.iterdir()
        if d2.is_dir()
    )
    if not data_dirs:
        raise FileNotFoundError(f"No data dirs under {run_root} matching filter {args.run_filter}")

    rows: list[dict[str, Any]] = []
    for mks in max_k_list:
        for d in data_dirs:
            metrics, elapsed = run_eval(d, args, mks)
            run_info = metrics.get("run_info", {})
            n_queries = int(run_info.get("num_queries", 0) or 0)
            latency_ms_per_query = (elapsed * 1000.0 / n_queries) if n_queries > 0 else None
            for k_str, m in (metrics.get("metrics_by_k") or {}).items():
                rows.append(
                    {
                        "doc_id": d.name,
                        "max_k_search": int(mks),
                        "k": int(k_str),
                        "page_hit_rate_at_k": float(m.get("page_hit_rate_at_k", 0.0)),
                        "mean_page_mrr_at_k": float(m.get("mean_page_mrr_at_k", 0.0)),
                        "mean_page_precision_at_k": float(m.get("mean_page_precision_at_k", 0.0)),
                        "chunk_hit_rate_at_k": float(m.get("chunk_hit_rate_at_k", 0.0)),
                        "elapsed_seconds": float(elapsed),
                        "num_queries": int(n_queries),
                        "latency_ms_per_query": float(latency_ms_per_query) if latency_ms_per_query is not None else None,
                    }
                )

    long_df = pd.DataFrame(rows)
    long_csv = out_dir / "max_k_search_sensitivity_long.csv"
    long_df.to_csv(long_csv, index=False)

    agg = (
        long_df.groupby(["max_k_search", "k"], as_index=False)
        .agg(
            page_hit_rate_at_k=("page_hit_rate_at_k", "mean"),
            mean_page_mrr_at_k=("mean_page_mrr_at_k", "mean"),
            mean_page_precision_at_k=("mean_page_precision_at_k", "mean"),
            chunk_hit_rate_at_k=("chunk_hit_rate_at_k", "mean"),
            latency_ms_per_query=("latency_ms_per_query", "mean"),
            elapsed_seconds=("elapsed_seconds", "mean"),
        )
    )
    agg_csv = out_dir / "max_k_search_sensitivity_aggregate.csv"
    agg.to_csv(agg_csv, index=False)

    k1 = agg[agg["k"] == 1].sort_values("max_k_search")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    ax[0].plot(k1["max_k_search"], k1["page_hit_rate_at_k"], marker="o")
    ax[0].set_title("Hit@1 vs MAX_K_SEARCH")
    ax[0].set_xlabel("MAX_K_SEARCH")
    ax[0].set_ylabel("Hit@1")
    ax[0].set_ylim(0, 1)
    ax[0].grid(alpha=0.3)

    ax[1].plot(k1["max_k_search"], k1["latency_ms_per_query"], marker="o")
    ax[1].set_title("Latency/query vs MAX_K_SEARCH")
    ax[1].set_xlabel("MAX_K_SEARCH")
    ax[1].set_ylabel("ms/query")
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    chart = out_dir / "chart_max_k_search_hit1_latency.png"
    fig.savefig(chart)

    # Recommend smallest value within 0.5 percentage points of best Hit@1
    best_hit1 = float(k1["page_hit_rate_at_k"].max()) if not k1.empty else 0.0
    threshold = best_hit1 - 0.005
    candidates = k1[k1["page_hit_rate_at_k"] >= threshold].sort_values("max_k_search")
    recommended = int(candidates.iloc[0]["max_k_search"]) if not candidates.empty else int(max_k_list[0])

    summary = {
        "run_root": str(run_root),
        "run_filter": args.run_filter,
        "docs": [d.name for d in data_dirs],
        "max_k_search_list": max_k_list,
        "rrf_k": int(args.rrf_k),
        "dense_weight": float(args.dense_weight),
        "bm25_weight": float(args.bm25_weight),
        "recommendation_rule": "smallest MAX_K_SEARCH within 0.5 percentage points of best Hit@1",
        "best_hit1": best_hit1,
        "recommended_max_k_search": recommended,
    }
    summary_path = out_dir / "max_k_search_sensitivity_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved: {long_csv}")
    print(f"Saved: {agg_csv}")
    print(f"Saved: {chart}")
    print(f"Saved: {summary_path}")
    print(f"Recommended MAX_K_SEARCH: {recommended}")


if __name__ == "__main__":
    main()
