from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build thesis-ready selection tables/charts from ablation summary.")
    p.add_argument(
        "--summary-csv",
        default="data_processed/ablation_thesis_all_docs/retrieval_ablation_summary.csv",
        help="Input ablation summary CSV.",
    )
    p.add_argument(
        "--out-dir",
        default="data_processed/ablation_thesis_all_docs/final_selection",
        help="Output directory for tables/charts.",
    )
    return p.parse_args()


def _variant_name(exp: str, doc_id: str) -> str:
    prefix = f"thesis_{doc_id}_"
    if exp.startswith(prefix):
        return exp[len(prefix):]
    return exp


def _load_num_queries(data_dir: str, mode: str, k: int) -> int:
    d = Path(data_dir)
    metrics_name = "retrieval_metrics_rewrites.json" if mode == "rewrite" else "retrieval_metrics.json"
    p = d / metrics_name
    if not p.exists():
        return 0
    obj = json.loads(p.read_text(encoding="utf-8"))
    by_k = obj.get("metrics_by_k", {})
    row = by_k.get(str(int(k)), {})
    return int(row.get("num_queries", 0) or 0)


def _weighted_mean(g: pd.DataFrame, col: str) -> float:
    w = g["num_queries"].astype(float)
    x = g[col].astype(float)
    denom = float(w.sum())
    return float((x * w).sum() / denom) if denom > 0 else float("nan")


def _bootstrap_mean_ci(values: list[float], alpha: float = 0.05, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        v = float(arr[0])
        return v, v
    rng = np.random.default_rng(seed)
    n = arr.size
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def _load_query_level(data_dir: str, mode: str) -> dict[int, dict[str, list[float]]]:
    """
    Load per-query hit/mrr/answer arrays per k from retrieval results JSON.
    Returns: {k: {"hit": [...], "mrr": [...], "answer": [...]}}
    """
    d = Path(data_dir)
    name = "retrieval_results_rewrites.json" if mode == "rewrite" else "retrieval_results.json"
    p = d / name
    if not p.exists():
        return {}
    obj = json.loads(p.read_text(encoding="utf-8"))
    results = obj.get("results", [])
    out: dict[int, dict[str, list[float]]] = {}
    for item in results:
        per_k = item.get("per_k", {})
        if not isinstance(per_k, dict):
            continue
        answer_correct = item.get("answer_correct")
        for k_str, k_item in per_k.items():
            try:
                k = int(k_str)
            except Exception:
                continue
            if k not in out:
                out[k] = {"hit": [], "mrr": [], "answer": []}
            if mode == "rewrite":
                rec = float(k_item.get("recall_at_k", 0.0))
                mrr = float(k_item.get("mrr_at_k", 0.0))
            else:
                rec = float(k_item.get("page_recall_at_k", 0.0))
                mrr = float(k_item.get("page_mrr_at_k", 0.0))
            out[k]["hit"].append(1.0 if rec > 0 else 0.0)
            out[k]["mrr"].append(mrr)
            if isinstance(answer_correct, bool):
                out[k]["answer"].append(1.0 if answer_correct else 0.0)
    return out


def _variant_description(variant: str) -> str:
    parts = variant.split("_")
    # Expected pattern examples:
    # chunk_320_90_seg_off_dense_rerank_on
    # chunk_280_90_seg_on_wholemd_dense_rerank_on
    if len(parts) < 8:
        return variant
    chunk_size = parts[1]
    overlap = parts[2]
    seg = parts[4]
    wholemd = "yes" if "wholemd" in parts else "no"
    mode = "rewrite" if "rewrite" in parts else "dense"
    rerank = parts[-1]
    return (
        f"{variant}: chunk={chunk_size}, overlap={overlap}, "
        f"segment_aware={seg}, whole_md={wholemd}, mode={mode}, rerank={rerank}"
    )


def _variant_short_label(variant: str) -> str:
    """Compact human-readable label for legends."""
    parts = variant.split("_")
    if len(parts) < 8:
        return variant
    chunk_size = parts[1]
    overlap = parts[2]
    seg = parts[4]
    rerank = parts[-1]
    return f"{chunk_size}/{overlap}, seg-{seg}, rerank-{rerank}"


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_path)
    if df.empty:
        raise RuntimeError("Summary CSV is empty.")

    df["variant"] = df.apply(lambda r: _variant_name(str(r["experiment"]), str(r["doc_id"])), axis=1)
    df["num_queries"] = df.apply(
        lambda r: _load_num_queries(str(r["data_dir"]), str(r["mode"]), int(r["k"])),
        axis=1,
    )
    qcache: dict[tuple[str, str], dict[int, dict[str, list[float]]]] = {}

    def query_metric_list(r: pd.Series, metric: str) -> list[float]:
        key = (str(r["data_dir"]), str(r["mode"]))
        if key not in qcache:
            qcache[key] = _load_query_level(data_dir=key[0], mode=key[1])
        k = int(r["k"])
        return list(qcache[key].get(k, {}).get(metric, []))

    # Weighted aggregate across docs per variant/k.
    rows: list[dict[str, Any]] = []
    for (variant, k), g in df.groupby(["variant", "k"], sort=True):
        pooled_hit: list[float] = []
        pooled_mrr: list[float] = []
        pooled_ans: list[float] = []
        for _, row in g.iterrows():
            pooled_hit.extend(query_metric_list(row, "hit"))
            pooled_mrr.extend(query_metric_list(row, "mrr"))
            pooled_ans.extend(query_metric_list(row, "answer"))
        hit_lo, hit_hi = _bootstrap_mean_ci(pooled_hit)
        mrr_lo, mrr_hi = _bootstrap_mean_ci(pooled_mrr)
        ans_lo, ans_hi = _bootstrap_mean_ci(pooled_ans) if pooled_ans else (float("nan"), float("nan"))
        rows.append(
            {
                "variant": variant,
                "k": int(k),
                "num_queries_total": int(g["num_queries"].sum()),
                "page_hit_rate_weighted": _weighted_mean(g, "page_hit_rate"),
                "page_hit_ci_low": hit_lo,
                "page_hit_ci_high": hit_hi,
                "page_mrr_weighted": _weighted_mean(g, "page_mrr"),
                "page_mrr_ci_low": mrr_lo,
                "page_mrr_ci_high": mrr_hi,
                "page_precision_weighted": _weighted_mean(g, "page_precision"),
                "answer_accuracy_weighted": _weighted_mean(g.fillna(0.0), "answer_accuracy"),
                "answer_accuracy_ci_low": ans_lo,
                "answer_accuracy_ci_high": ans_hi,
            }
        )
    agg = pd.DataFrame(rows).sort_values(["k", "page_hit_rate_weighted", "page_mrr_weighted"], ascending=[True, False, False])
    agg.to_csv(out_dir / "overall_weighted_by_variant_k.csv", index=False)

    best_overall = agg.groupby("k", as_index=False).head(1).reset_index(drop=True)
    best_overall.to_csv(out_dir / "overall_best_variant_by_k.csv", index=False)

    # Macro aggregate across documents (equal document weight).
    macro_rows: list[dict[str, Any]] = []
    for (variant, k), g in df.groupby(["variant", "k"], sort=True):
        doc_hits = list(g["page_hit_rate"].astype(float))
        doc_mrrs = list(g["page_mrr"].astype(float))
        doc_ans = list(g["answer_accuracy"].fillna(0.0).astype(float))
        h_lo, h_hi = _bootstrap_mean_ci(doc_hits)
        m_lo, m_hi = _bootstrap_mean_ci(doc_mrrs)
        a_lo, a_hi = _bootstrap_mean_ci(doc_ans)
        macro_rows.append(
            {
                "variant": variant,
                "k": int(k),
                "num_docs": int(g["doc_id"].nunique()),
                "page_hit_rate_macro": float(np.mean(doc_hits)) if doc_hits else float("nan"),
                "page_hit_ci_low": h_lo,
                "page_hit_ci_high": h_hi,
                "page_mrr_macro": float(np.mean(doc_mrrs)) if doc_mrrs else float("nan"),
                "page_mrr_ci_low": m_lo,
                "page_mrr_ci_high": m_hi,
                "page_precision_macro": float(np.mean(list(g["page_precision"].astype(float)))) if len(g) else float("nan"),
                "answer_accuracy_macro": float(np.mean(doc_ans)) if doc_ans else float("nan"),
                "answer_accuracy_ci_low": a_lo,
                "answer_accuracy_ci_high": a_hi,
            }
        )
    macro = pd.DataFrame(macro_rows).sort_values(["k", "page_hit_rate_macro", "page_mrr_macro"], ascending=[True, False, False])
    macro.to_csv(out_dir / "overall_macro_by_variant_k.csv", index=False)
    macro_best = macro.groupby("k", as_index=False).head(1).reset_index(drop=True)
    macro_best.to_csv(out_dir / "overall_macro_best_variant_by_k.csv", index=False)

    # Delta Hit@1 relative to best (overall weighted + per doc).
    hit1 = agg[agg["k"] == 1].copy().sort_values(["page_hit_rate_weighted", "page_mrr_weighted"], ascending=False)
    best_hit1 = float(hit1.iloc[0]["page_hit_rate_weighted"]) if len(hit1) else float("nan")
    hit1["delta_hit1_vs_best"] = hit1["page_hit_rate_weighted"] - best_hit1
    hit1.to_csv(out_dir / "delta_hit1_overall_weighted.csv", index=False)

    per_doc_hit1 = df[df["k"] == 1].copy()
    per_doc_hit1["delta_hit1_vs_doc_best"] = 0.0
    for doc, g in per_doc_hit1.groupby("doc_id"):
        best_doc = float(g["page_hit_rate"].max()) if len(g) else 0.0
        per_doc_hit1.loc[g.index, "delta_hit1_vs_doc_best"] = g["page_hit_rate"] - best_doc
    per_doc_hit1.sort_values(["doc_id", "delta_hit1_vs_doc_best", "page_mrr"], ascending=[True, False, False]).to_csv(
        out_dir / "delta_hit1_per_doc.csv",
        index=False,
    )

    # Per-document winners by k.
    per_doc_best = (
        df.sort_values(["doc_id", "k", "page_hit_rate", "page_mrr", "page_precision"], ascending=[True, True, False, False, False])
        .groupby(["doc_id", "k"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    # Attach per-doc CI for selected rows.
    hit_ci_low: list[float] = []
    hit_ci_high: list[float] = []
    mrr_ci_low: list[float] = []
    mrr_ci_high: list[float] = []
    for _, row in per_doc_best.iterrows():
        h = query_metric_list(row, "hit")
        m = query_metric_list(row, "mrr")
        hlo, hhi = _bootstrap_mean_ci(h)
        mlo, mhi = _bootstrap_mean_ci(m)
        hit_ci_low.append(hlo)
        hit_ci_high.append(hhi)
        mrr_ci_low.append(mlo)
        mrr_ci_high.append(mhi)
    per_doc_best["page_hit_ci_low"] = hit_ci_low
    per_doc_best["page_hit_ci_high"] = hit_ci_high
    per_doc_best["page_mrr_ci_low"] = mrr_ci_low
    per_doc_best["page_mrr_ci_high"] = mrr_ci_high
    per_doc_best.to_csv(out_dir / "per_doc_best_variant_by_k.csv", index=False)

    # Markdown summary table.
    md_lines = [
        "# Thesis Final Model Selection",
        "",
        "## Overall Best Variant Per k (Weighted Across Documents)",
        "",
        best_overall.to_markdown(index=False),
        "",
        "## Overall Best Variant Per k (Macro Across Documents)",
        "",
        macro_best.to_markdown(index=False),
        "",
        "## Per-Document Best Variant Per k",
        "",
        per_doc_best[
            [
                "doc_id",
                "k",
                "variant",
                "mode",
                "page_hit_rate",
                "page_hit_ci_low",
                "page_hit_ci_high",
                "page_mrr",
                "page_mrr_ci_low",
                "page_mrr_ci_high",
                "answer_accuracy",
            ]
        ].to_markdown(index=False),
        "",
        "## Delta Hit@1 Relative to Best (Weighted Overall)",
        "",
        hit1[["variant", "page_hit_rate_weighted", "delta_hit1_vs_best"]].to_markdown(index=False),
        "",
    ]
    (out_dir / "thesis_final_selection.md").write_text("\n".join(md_lines), encoding="utf-8")

    # Charts
    plt.style.use("seaborn-v0_8-whitegrid")

    # 1) Weighted page-hit vs k for each variant
    fig, ax = plt.subplots(figsize=(10, 6))
    for variant, g in agg.groupby("variant"):
        gg = g.sort_values("k")
        ax.plot(gg["k"], gg["page_hit_rate_weighted"], marker="o", linewidth=2, label=variant)
    ax.set_title("Weighted Page Hit Rate by k (All Documents)")
    ax.set_xlabel("k")
    ax.set_ylabel("Weighted Page Hit Rate")
    ax.set_xticks(sorted(agg["k"].unique()))
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "chart_weighted_hit_by_k.png", dpi=180)
    plt.close(fig)

    # 1c) Weighted MRR vs k for each variant
    fig, ax = plt.subplots(figsize=(10, 6))
    for variant, g in agg.groupby("variant"):
        gg = g.sort_values("k")
        ax.plot(gg["k"], gg["page_mrr_weighted"], marker="o", linewidth=2, label=variant)
    ax.set_title("Weighted MRR by k (All Documents)")
    ax.set_xlabel("k")
    ax.set_ylabel("Weighted MRR")
    ax.set_xticks(sorted(agg["k"].unique()))
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "chart_weighted_mrr_by_k.png", dpi=180)
    plt.close(fig)

    # 1b) Best-overall with 95% CI error bars (color-coded with legend)
    fig, ax = plt.subplots(figsize=(8, 5))
    bb = best_overall.sort_values("k")
    unique_variants = list(dict.fromkeys(bb["variant"].tolist()))
    variant_codes = {v: f"V{i+1}" for i, v in enumerate(unique_variants)}
    cmap = plt.get_cmap("tab10")
    color_map = {v: cmap(i % 10) for i, v in enumerate(unique_variants)}
    legend_df = pd.DataFrame(
        [
            {
                "variant_code": variant_codes[v],
                "variant": v,
                "short_label": _variant_short_label(v),
                "description": _variant_description(v),
            }
            for v in unique_variants
        ]
    )
    legend_df.to_csv(out_dir / "variant_legend_best_overall.csv", index=False)
    (out_dir / "variant_legend_best_overall.md").write_text(
        legend_df.to_markdown(index=False),
        encoding="utf-8",
    )
    for _, r in bb.iterrows():
        y = float(r["page_hit_rate_weighted"])
        yerr = np.array([[y - float(r["page_hit_ci_low"])], [float(r["page_hit_ci_high"]) - y]])
        ax.errorbar(
            [int(r["k"])],
            [y],
            yerr=yerr,
            fmt="o",
            capsize=5,
            linewidth=2,
            color=color_map[str(r["variant"])],
        )
    ax.plot(bb["k"].values, bb["page_hit_rate_weighted"].values, color="gray", linewidth=1.5, alpha=0.7)
    ax.set_title("Overall Best Variant Per k with 95% CI (Hit Rate)")
    ax.set_xlabel("k")
    ax.set_ylabel("Weighted Page Hit Rate")
    ax.set_xticks(sorted(bb["k"].unique()))
    ax.set_ylim(0.0, 1.05)
    handles = [
        Line2D(
            [0],
            [0],
            color=color_map[v],
            marker="o",
            linestyle="",
            label=f"{variant_codes[v]}: {_variant_short_label(v)}",
        )
        for v in unique_variants
    ]
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=8,
        title="Variant code",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "chart_best_overall_hit_ci.png", dpi=180)
    plt.close(fig)

    # 1d) Best-overall MRR with 95% CI error bars (color-coded with legend)
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, r in bb.iterrows():
        y = float(r["page_mrr_weighted"])
        yerr = np.array([[y - float(r["page_mrr_ci_low"])], [float(r["page_mrr_ci_high"]) - y]])
        ax.errorbar(
            [int(r["k"])],
            [y],
            yerr=yerr,
            fmt="o",
            capsize=5,
            linewidth=2,
            color=color_map[str(r["variant"])],
        )
    ax.plot(bb["k"].values, bb["page_mrr_weighted"].values, color="gray", linewidth=1.5, alpha=0.7)
    ax.set_title("Overall Best Variant Per k with 95% CI (MRR)")
    ax.set_xlabel("k")
    ax.set_ylabel("Weighted MRR")
    ax.set_xticks(sorted(bb["k"].unique()))
    ax.set_ylim(0.0, 1.05)
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=8,
        title="Variant code",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "chart_best_overall_mrr_ci.png", dpi=180)
    plt.close(fig)

    # 2) Heatmap: weighted page hit rate (variant x k)
    pivot = agg.pivot(index="variant", columns="k", values="page_hit_rate_weighted").sort_index()
    fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(pivot))))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_title("Weighted Page Hit Rate Heatmap")
    ax.set_xlabel("k")
    ax.set_ylabel("Variant")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index), fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Weighted Hit Rate")
    fig.tight_layout()
    fig.savefig(out_dir / "chart_weighted_hit_heatmap.png", dpi=180)
    plt.close(fig)

    # 3) Per-doc k=10 comparison for top overall variants
    k10 = agg[agg["k"] == 10].sort_values(["page_hit_rate_weighted", "page_mrr_weighted"], ascending=False)
    top_variants = list(k10["variant"].head(5))
    d10 = df[(df["k"] == 10) & (df["variant"].isin(top_variants))].copy()
    docs = sorted(d10["doc_id"].unique())
    x = np.arange(len(docs))
    width = 0.15 if len(top_variants) > 0 else 0.2
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, v in enumerate(top_variants):
        vals = []
        for doc in docs:
            m = d10[(d10["doc_id"] == doc) & (d10["variant"] == v)]
            vals.append(float(m["page_hit_rate"].iloc[0]) if len(m) else np.nan)
        ax.bar(x + i * width, vals, width=width, label=v)
    ax.set_title("Per-Document Page Hit Rate at k=10 (Top Variants)")
    ax.set_xlabel("Document")
    ax.set_ylabel("Page Hit Rate")
    ax.set_xticks(x + (len(top_variants) - 1) * width / 2 if top_variants else x)
    ax.set_xticklabels(docs, rotation=15, ha="right")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "chart_per_doc_k10_top_variants.png", dpi=180)
    plt.close(fig)

    print("Saved:", out_dir / "overall_weighted_by_variant_k.csv")
    print("Saved:", out_dir / "overall_best_variant_by_k.csv")
    print("Saved:", out_dir / "overall_macro_by_variant_k.csv")
    print("Saved:", out_dir / "overall_macro_best_variant_by_k.csv")
    print("Saved:", out_dir / "delta_hit1_overall_weighted.csv")
    print("Saved:", out_dir / "delta_hit1_per_doc.csv")
    print("Saved:", out_dir / "per_doc_best_variant_by_k.csv")
    print("Saved:", out_dir / "thesis_final_selection.md")
    print("Saved:", out_dir / "chart_weighted_hit_by_k.png")
    print("Saved:", out_dir / "chart_weighted_mrr_by_k.png")
    print("Saved:", out_dir / "chart_best_overall_hit_ci.png")
    print("Saved:", out_dir / "chart_best_overall_mrr_ci.png")
    print("Saved:", out_dir / "variant_legend_best_overall.csv")
    print("Saved:", out_dir / "variant_legend_best_overall.md")
    print("Saved:", out_dir / "chart_weighted_hit_heatmap.png")
    print("Saved:", out_dir / "chart_per_doc_k10_top_variants.png")


if __name__ == "__main__":
    main()
