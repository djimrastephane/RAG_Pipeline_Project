from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts._matplotlib_env import configure_matplotlib_env
except ModuleNotFoundError:
    from _matplotlib_env import configure_matplotlib_env


configure_matplotlib_env()

import matplotlib.pyplot as plt


FAILURE_STAGE_BY_TYPE = {
    "FP1_MISSING_CONTENT": "retrieval",
    "FP2_MISSED_TOP_RANK": "retrieval",
    "FP3_NOT_IN_CONTEXT": "retrieval",
    "FP4_NOT_EXTRACTED": "generation",
    "FP5_WRONG_FORMAT": "generation",
    "FP6_INCORRECT_SPECIFICITY": "generation",
    "FP7_INCOMPLETE": "generation",
    "HIT": "none",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build retrieval metrics report (CSV + Markdown + LaTeX)."
    )
    parser.add_argument(
        "--data-root",
        default="data_processed",
        help="Root directory containing per-document outputs.",
    )
    parser.add_argument(
        "--docs",
        default="Grampian-2022-2023,Grampian-2023-2024,Grampian-2024-2025",
        help="Comma-separated doc IDs to include.",
    )
    parser.add_argument(
        "--out-csv",
        default=None,
        help="Output CSV path. Defaults to <data-root>/retrieval_report.csv.",
    )
    parser.add_argument(
        "--out-md",
        default=None,
        help="Output Markdown path. Defaults to <data-root>/retrieval_report.md.",
    )
    parser.add_argument(
        "--out-tex",
        default=None,
        help="Output LaTeX path. Defaults to <data-root>/retrieval_report.tex.",
    )
    parser.add_argument(
        "--out-queries-csv",
        default=None,
        help="Output per-query CSV path. Defaults to <data-root>/retrieval_queries_report.csv.",
    )
    parser.add_argument(
        "--out-queries-md",
        default=None,
        help="Output per-query Markdown path. Defaults to <data-root>/retrieval_queries_report.md.",
    )
    parser.add_argument(
        "--out-queries-tex",
        default=None,
        help="Output per-query LaTeX path. Defaults to <data-root>/retrieval_queries_report.tex.",
    )
    parser.add_argument(
        "--out-failure-summary",
        default=None,
        help="Output failure-type summary CSV path. Defaults to <data-root>/retrieval_failure_summary.csv.",
    )
    parser.add_argument(
        "--out-table-misses",
        default=None,
        help="Output table-query misses at k=1. Defaults to <data-root>/retrieval_table_misses_k1.csv.",
    )
    parser.add_argument(
        "--out-survival-csv",
        default=None,
        help="Output survival input CSV. Defaults to <data-root>/retrieval_rank_survival.csv.",
    )
    parser.add_argument(
        "--out-survival-km-csv",
        default=None,
        help="Output Kaplan-Meier curve CSV. Defaults to <data-root>/retrieval_rank_km_curve.csv.",
    )
    parser.add_argument(
        "--out-survival-plot",
        default=None,
        help="Output Kaplan-Meier plot path. Defaults to <data-root>/retrieval_rank_km_curve.png.",
    )
    parser.add_argument(
        "--out-survival-compare-csv",
        default=None,
        help="Output dense-vs-hybrid survival input CSV. Defaults to <data-root>/retrieval_rank_survival_compare.csv.",
    )
    parser.add_argument(
        "--out-survival-compare-km-csv",
        default=None,
        help="Output dense-vs-hybrid Kaplan-Meier curve CSV. Defaults to <data-root>/retrieval_rank_km_compare_curve.csv.",
    )
    parser.add_argument(
        "--out-survival-compare-plot",
        default=None,
        help="Output dense-vs-hybrid Kaplan-Meier plot path. Defaults to <data-root>/retrieval_rank_km_compare_curve.png.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=5000,
        help="Number of bootstrap resamples for survival confidence intervals.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
        help="Random seed for bootstrap resampling.",
    )
    return parser.parse_args()


def resolve_output_path(data_root: Path, output_path: str | None, filename: str) -> Path:
    if output_path:
        return Path(output_path)
    return data_root / filename


def _first_correct_rank(
    ranked_pages: list[int], expected_pages: list[int]
) -> int | None:
    expected = {int(page) for page in expected_pages}
    for idx, page in enumerate(ranked_pages, start=1):
        if int(page) in expected:
            return idx
    return None


def build_kaplan_meier_curve(
    survival_df: pd.DataFrame,
) -> pd.DataFrame:
    if survival_df.empty:
        return pd.DataFrame(
            columns=[
                "rank",
                "at_risk",
                "events",
                "censored",
                "survival_probability",
            ]
        )

    max_rank = int(survival_df["time_rank"].max())
    rows: list[dict] = []
    survival_probability = 1.0

    for rank in range(1, max_rank + 1):
        at_risk = int((survival_df["time_rank"] >= rank).sum())
        events = int(
            ((survival_df["time_rank"] == rank) & (survival_df["event"] == 1)).sum()
        )
        censored = int(
            ((survival_df["time_rank"] == rank) & (survival_df["event"] == 0)).sum()
        )
        if at_risk > 0 and events > 0:
            survival_probability *= (at_risk - events) / at_risk
        rows.append(
            {
                "rank": rank,
                "at_risk": at_risk,
                "events": events,
                "censored": censored,
                "survival_probability": survival_probability,
            }
        )

    return pd.DataFrame(rows)


def build_kaplan_meier_curve_by_group(
    survival_df: pd.DataFrame, group_col: str
) -> pd.DataFrame:
    grouped: list[pd.DataFrame] = []
    if survival_df.empty or group_col not in survival_df.columns:
        return pd.DataFrame()
    for group_value, group_df in survival_df.groupby(group_col):
        curve_df = build_kaplan_meier_curve(group_df)
        if curve_df.empty:
            continue
        curve_df.insert(0, group_col, group_value)
        grouped.append(curve_df)
    if not grouped:
        return pd.DataFrame()
    return pd.concat(grouped, ignore_index=True)


def _curve_to_rank_series(curve_df: pd.DataFrame, max_rank: int) -> np.ndarray:
    values = np.ones(max_rank, dtype=float)
    if curve_df.empty or max_rank <= 0:
        return values
    rank_to_survival = {
        int(row["rank"]): float(row["survival_probability"])
        for _, row in curve_df.iterrows()
    }
    last_val = 1.0
    for rank in range(1, max_rank + 1):
        if rank in rank_to_survival:
            last_val = rank_to_survival[rank]
        values[rank - 1] = last_val
    return values


def bootstrap_kaplan_meier_curve(
    survival_df: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    if survival_df.empty:
        return pd.DataFrame()
    base_curve = build_kaplan_meier_curve(survival_df)
    max_rank = int(survival_df["time_rank"].max())
    rng = np.random.default_rng(seed)
    n = len(survival_df)
    bootstrap_curves = np.zeros((n_bootstrap, max_rank), dtype=float)
    for i in range(n_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        sample_df = survival_df.iloc[sample_idx].reset_index(drop=True)
        sample_curve = build_kaplan_meier_curve(sample_df)
        bootstrap_curves[i, :] = _curve_to_rank_series(sample_curve, max_rank)
    out = base_curve.copy()
    out["mean_survival_probability"] = bootstrap_curves.mean(axis=0)
    out["ci_lower"] = np.quantile(bootstrap_curves, 0.025, axis=0)
    out["ci_upper"] = np.quantile(bootstrap_curves, 0.975, axis=0)
    return out


def bootstrap_kaplan_meier_curve_by_group(
    survival_df: pd.DataFrame,
    group_col: str,
    n_bootstrap: int,
    seed: int,
    paired_unit_col: str | None = None,
) -> pd.DataFrame:
    if survival_df.empty or group_col not in survival_df.columns:
        return pd.DataFrame()
    results: list[pd.DataFrame] = []
    if paired_unit_col:
        if paired_unit_col not in survival_df.columns:
            return pd.DataFrame()
        unit_ids = survival_df[paired_unit_col].dropna().astype(str).unique().tolist()
        if not unit_ids:
            return pd.DataFrame()
        unit_to_rows = {
            unit_id: survival_df[survival_df[paired_unit_col].astype(str) == unit_id]
            for unit_id in unit_ids
        }
        rng = np.random.default_rng(seed)
        grouped_base = build_kaplan_meier_curve_by_group(survival_df, group_col)
        max_rank = int(survival_df["time_rank"].max())
        group_values = sorted(survival_df[group_col].dropna().astype(str).unique().tolist())
        boot_map = {
            group_value: np.zeros((n_bootstrap, max_rank), dtype=float)
            for group_value in group_values
        }
        unit_count = len(unit_ids)
        for i in range(n_bootstrap):
            sampled_units = rng.integers(0, unit_count, size=unit_count)
            sampled_frames = [unit_to_rows[unit_ids[idx]] for idx in sampled_units]
            sample_df = pd.concat(sampled_frames, ignore_index=True)
            for group_value in group_values:
                group_sample = sample_df[sample_df[group_col].astype(str) == group_value]
                group_curve = build_kaplan_meier_curve(group_sample)
                boot_map[group_value][i, :] = _curve_to_rank_series(group_curve, max_rank)
        for group_value in group_values:
            group_base = grouped_base[grouped_base[group_col].astype(str) == group_value].copy()
            curves = boot_map[group_value]
            group_base["mean_survival_probability"] = curves.mean(axis=0)
            group_base["ci_lower"] = np.quantile(curves, 0.025, axis=0)
            group_base["ci_upper"] = np.quantile(curves, 0.975, axis=0)
            results.append(group_base)
    else:
        for i, (group_value, group_df) in enumerate(survival_df.groupby(group_col)):
            group_curve = bootstrap_kaplan_meier_curve(
                group_df.reset_index(drop=True),
                n_bootstrap=n_bootstrap,
                seed=seed + i,
            )
            if group_curve.empty:
                continue
            group_curve.insert(0, group_col, group_value)
            results.append(group_curve)
    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


def plot_kaplan_meier_curve(
    curve_df: pd.DataFrame, out_path: Path, bootstrap_samples: int | None = None
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    if curve_df.empty:
        plt.text(0.5, 0.5, "No survival data", ha="center", va="center")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
    else:
        x_vals = [0, *curve_df["rank"].tolist()]
        if {"mean_survival_probability", "ci_lower", "ci_upper"}.issubset(curve_df.columns):
            mean_vals = [1.0, *curve_df["mean_survival_probability"].tolist()]
            lower_vals = [1.0, *curve_df["ci_lower"].tolist()]
            upper_vals = [1.0, *curve_df["ci_upper"].tolist()]
            plt.fill_between(
                x_vals,
                lower_vals,
                upper_vals,
                step="post",
                color="#0b6e4f",
                alpha=0.18,
                label="95% CI",
            )
            plt.step(
                x_vals,
                mean_vals,
                where="post",
                linewidth=2,
                color="#0b6e4f",
                label="Mean",
            )
        else:
            mean_vals = [1.0, *curve_df["survival_probability"].tolist()]
            plt.step(x_vals, mean_vals, where="post", linewidth=2, color="#0b6e4f")
        censor_df = curve_df[curve_df["censored"] > 0]
        if not censor_df.empty:
            censor_y_col = (
                "mean_survival_probability"
                if "mean_survival_probability" in censor_df.columns
                else "survival_probability"
            )
            plt.scatter(
                censor_df["rank"],
                censor_df[censor_y_col],
                marker="+",
                s=60,
                linewidths=1.5,
                color="#c05621",
                label="Censored",
            )
        plt.legend(frameon=False)
        plt.xlim(0, max(x_vals))
        plt.ylim(0, 1.02)
        for k in (1, 3, 5):
            plt.axvline(k, color="#94a3b8", linestyle="--", linewidth=1, alpha=0.8)
        x_max = max(x_vals)
        top3_x = min(2.0, x_max * 0.22 if x_max > 0 else 2.0)
        diminish_x = min(6.2, max(5.2, x_max * 0.62 if x_max > 0 else 6.2))
        plt.text(top3_x, 0.9, "Top-3 region", color="#334155", fontsize=10)
        plt.text(diminish_x, 0.28, "Diminishing returns", color="#334155", fontsize=10)
    plt.xlabel("Rank (k)")
    plt.ylabel("Survival probability")
    plt.title("Rank-as-time Kaplan-Meier Curve")
    if bootstrap_samples is not None and {
        "mean_survival_probability",
        "ci_lower",
        "ci_upper",
    }.issubset(curve_df.columns):
        plt.figtext(
            0.5,
            0.01,
            f"Mean KM curve with 95% bootstrap CI, B={bootstrap_samples}",
            ha="center",
            fontsize=9,
            color="#475569",
        )
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_kaplan_meier_comparison_curve(
    curve_df: pd.DataFrame, out_path: Path, bootstrap_samples: int | None = None
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    if curve_df.empty or "system" not in curve_df.columns:
        plt.text(0.5, 0.5, "No comparison data", ha="center", va="center")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
    else:
        palette = {
            "dense": "#7c3aed",
            "hybrid": "#0b6e4f",
        }
        label_map = {
            "dense": "Dense only",
            "hybrid": "Hybrid",
        }
        max_rank = 1
        for system in ["dense", "hybrid"]:
            system_df = curve_df[curve_df["system"] == system].sort_values("rank")
            if system_df.empty:
                continue
            x_vals = [0, *system_df["rank"].tolist()]
            max_rank = max(max_rank, max(x_vals))
            if {"mean_survival_probability", "ci_lower", "ci_upper"}.issubset(system_df.columns):
                mean_vals = [1.0, *system_df["mean_survival_probability"].tolist()]
                lower_vals = [1.0, *system_df["ci_lower"].tolist()]
                upper_vals = [1.0, *system_df["ci_upper"].tolist()]
                plt.fill_between(
                    x_vals,
                    lower_vals,
                    upper_vals,
                    step="post",
                    color=palette.get(system, None),
                    alpha=0.14,
                )
                plt.step(
                    x_vals,
                    mean_vals,
                    where="post",
                    linewidth=2,
                    color=palette.get(system, None),
                    label=label_map.get(system, system),
                )
            else:
                y_vals = [1.0, *system_df["survival_probability"].tolist()]
                plt.step(
                    x_vals,
                    y_vals,
                    where="post",
                    linewidth=2,
                    color=palette.get(system, None),
                    label=label_map.get(system, system),
                )
            censor_df = system_df[system_df["censored"] > 0]
            if not censor_df.empty:
                censor_y_col = (
                    "mean_survival_probability"
                    if "mean_survival_probability" in censor_df.columns
                    else "survival_probability"
                )
                plt.scatter(
                    censor_df["rank"],
                    censor_df[censor_y_col],
                    marker="+",
                    s=60,
                    linewidths=1.5,
                    color=palette.get(system, None),
                )
        for k in (1, 3, 5):
            plt.axvline(k, color="#94a3b8", linestyle="--", linewidth=1, alpha=0.8)
        top3_x = min(2.0, max_rank * 0.22 if max_rank > 0 else 2.0)
        diminish_x = min(6.2, max(5.2, max_rank * 0.62 if max_rank > 0 else 6.2))
        plt.text(top3_x, 0.9, "Top-3 region", color="#334155", fontsize=10)
        plt.text(diminish_x, 0.28, "Diminishing returns", color="#334155", fontsize=10)
        plt.xlim(0, max_rank)
        plt.ylim(0, 1.02)
        plt.legend(frameon=False)
    plt.xlabel("Rank (k)")
    plt.ylabel("Survival probability")
    plt.title("Rank-as-time Kaplan-Meier: Dense vs Hybrid")
    if bootstrap_samples is not None and {
        "mean_survival_probability",
        "ci_lower",
        "ci_upper",
    }.issubset(curve_df.columns):
        plt.figtext(
            0.5,
            0.01,
            f"Mean KM curve with 95% bootstrap CI, B={bootstrap_samples}",
            ha="center",
            fontsize=9,
            color="#475569",
        )
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def build_survival_rows_for_results(doc_id: str, results: dict, system: str) -> list[dict]:
    rows: list[dict] = []
    for item in results.get("results", []):
        per_k = item.get("per_k", {})
        max_k_key = max(per_k.keys(), key=lambda value: int(value)) if per_k else None
        max_k_data = per_k.get(max_k_key, {}) if max_k_key is not None else {}
        ranked_pages = max_k_data.get("retrieved_pages_ranked") or []
        expected_pages = item.get("expected_pages") or []
        first_correct_rank = _first_correct_rank(ranked_pages, expected_pages)
        top_k_limit = int(max_k_key) if max_k_key is not None else 0
        rows.append(
            {
                "system": system,
                "doc_id": doc_id,
                "query_id": item.get("query_id"),
                "question": item.get("question"),
                "expected_pages": expected_pages,
                "ranked_pages_observed": ranked_pages,
                "top_k_limit": top_k_limit,
                "first_correct_rank": first_correct_rank,
                "event": 1 if first_correct_rank is not None else 0,
                "time_rank": first_correct_rank if first_correct_rank is not None else top_k_limit,
                "censored": 0 if first_correct_rank is not None else 1,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    doc_ids = [d.strip() for d in args.docs.split(",") if d.strip()]

    rows: list[dict] = []
    detail_rows: list[dict] = []
    table_misses: list[dict] = []
    for doc_id in doc_ids:
        metrics_path = data_root / doc_id / "retrieval_metrics_hybrid.json"
        if not metrics_path.exists():
            metrics_path = data_root / doc_id / "retrieval_metrics.json"
        if not metrics_path.exists():
            print(f"Missing: {metrics_path}")
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics_by_k = metrics.get("metrics_by_k", {})
        for k, m in metrics_by_k.items():
            rows.append(
                {
                    "doc_id": doc_id,
                    "k": int(k),
                    "hit_rate": m.get("page_hit_rate_at_k", 0.0),
                    "recall": m.get("mean_page_recall_at_k", 0.0),
                    "mrr": m.get("mean_page_mrr_at_k", 0.0),
                    "precision": m.get("mean_page_precision_at_k", 0.0),
                }
            )
        results_path = data_root / doc_id / "retrieval_results_hybrid.json"
        if not results_path.exists():
            results_path = data_root / doc_id / "retrieval_results.json"
        meta_path = data_root / doc_id / "chunk_meta.parquet"
        if not results_path.exists() or not meta_path.exists():
            print(f"Missing per-doc inputs for table/text breakdown: {doc_id}")
            continue
        results = json.loads(results_path.read_text(encoding="utf-8"))
        meta = pd.read_parquet(meta_path)
        if "is_table" not in meta.columns:
            print(f"Missing is_table in chunk_meta.parquet for {doc_id}")
            continue
        table_pages = set(meta.loc[meta["is_table"] == True, "page_start"].dropna().astype(int).tolist())
        for item in results.get("results", []):
            expected_pages = item.get("expected_pages") or []
            is_table_query = bool(set(expected_pages) & table_pages)
            per_k = item.get("per_k", {})
            for k in per_k.keys():
                kdata = per_k.get(k, {})
                detail_rows.append(
                    {
                        "doc_id": doc_id,
                        "k": int(k),
                        "is_table_query": is_table_query,
                        "page_recall_at_k": kdata.get("page_recall_at_k", 0.0),
                        "page_precision_at_k": kdata.get("page_precision_at_k", 0.0),
                        "page_mrr_at_k": kdata.get("page_mrr_at_k", 0.0),
                    }
                )
            k1 = per_k.get("1", {})
            if is_table_query and k1.get("page_recall_at_k", 0.0) <= 0:
                table_misses.append(
                    {
                        "doc_id": doc_id,
                        "query_id": item.get("query_id"),
                        "question": item.get("question"),
                        "expected_pages": expected_pages,
                        "top_pages": k1.get("retrieved_pages_ranked"),
                        "failure_type": item.get("failure_type") or k1.get("failure_stage"),
                    }
                )

    if not rows:
        print("No metrics found.")
        return

    df = pd.DataFrame(rows).sort_values(["doc_id", "k"])
    if detail_rows:
        ddf = pd.DataFrame(detail_rows)
        table_ddf = ddf[ddf["is_table_query"] == True]
        text_ddf = ddf[ddf["is_table_query"] == False]

        table_stats = (
            table_ddf.groupby(["doc_id", "k"])
            .agg(
                table_query_count=("page_recall_at_k", "size"),
                table_hit_rate=("page_recall_at_k", lambda s: float((s > 0).mean())),
                table_recall=("page_recall_at_k", "mean"),
                table_precision=("page_precision_at_k", "mean"),
                table_mrr=("page_mrr_at_k", "mean"),
            )
            .reset_index()
        )
        text_stats = (
            text_ddf.groupby(["doc_id", "k"])
            .agg(
                text_query_count=("page_recall_at_k", "size"),
                text_hit_rate=("page_recall_at_k", lambda s: float((s > 0).mean())),
                text_recall=("page_recall_at_k", "mean"),
                text_precision=("page_precision_at_k", "mean"),
                text_mrr=("page_mrr_at_k", "mean"),
            )
            .reset_index()
        )
        df = df.merge(table_stats, on=["doc_id", "k"], how="left")
        df = df.merge(text_stats, on=["doc_id", "k"], how="left")
        if "table_hit_rate" in df.columns and "text_hit_rate" in df.columns:
            df["delta_hit_rate"] = df["table_hit_rate"] - df["text_hit_rate"]
        if "table_recall" in df.columns and "text_recall" in df.columns:
            df["delta_recall"] = df["table_recall"] - df["text_recall"]
        if "table_precision" in df.columns and "text_precision" in df.columns:
            df["delta_precision"] = df["table_precision"] - df["text_precision"]
        if "table_mrr" in df.columns and "text_mrr" in df.columns:
            df["delta_mrr"] = df["table_mrr"] - df["text_mrr"]

    out_csv = resolve_output_path(data_root, args.out_csv, "retrieval_report.csv")
    out_md = resolve_output_path(data_root, args.out_md, "retrieval_report.md")
    out_tex = resolve_output_path(data_root, args.out_tex, "retrieval_report.tex")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_csv, index=False)
    out_md.write_text(df.to_markdown(index=False), encoding="utf-8")
    out_tex.write_text(
        df.to_latex(index=False, float_format="%.3f"),
        encoding="utf-8",
    )

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")
    print(f"Wrote: {out_tex}")

    # Per-query report
    query_rows: list[dict] = []
    failure_types: list[str] = []
    survival_rows: list[dict] = []
    survival_compare_rows: list[dict] = []
    for doc_id in doc_ids:
        results_path = data_root / doc_id / "retrieval_results_hybrid.json"
        if not results_path.exists():
            results_path = data_root / doc_id / "retrieval_results.json"
        chunks_path = data_root / doc_id / "chunks.parquet"
        if not results_path.exists() or not chunks_path.exists():
            print(f"Missing per-query inputs for {doc_id}")
            continue
        results = json.loads(results_path.read_text(encoding="utf-8"))
        dense_results_path = data_root / doc_id / "retrieval_results.json"
        hybrid_results_path = data_root / doc_id / "retrieval_results_hybrid.json"
        dense_results = (
            json.loads(dense_results_path.read_text(encoding="utf-8"))
            if dense_results_path.exists()
            else None
        )
        hybrid_results = (
            json.loads(hybrid_results_path.read_text(encoding="utf-8"))
            if hybrid_results_path.exists()
            else None
        )
        if dense_results is not None and hybrid_results is not None:
            dense_survival = build_survival_rows_for_results(doc_id, dense_results, "dense")
            hybrid_survival = build_survival_rows_for_results(doc_id, hybrid_results, "hybrid")
            dense_by_qid = {str(row["query_id"]): row for row in dense_survival}
            hybrid_by_qid = {str(row["query_id"]): row for row in hybrid_survival}
            for query_id in sorted(set(dense_by_qid) & set(hybrid_by_qid)):
                survival_compare_rows.append(dense_by_qid[query_id])
                survival_compare_rows.append(hybrid_by_qid[query_id])
        chunks = pd.read_parquet(chunks_path)
        chunk_text_by_id = {}
        for _, row in chunks.iterrows():
            cid = row.get("chunk_id_global") or row.get("chunk_id")
            if cid:
                chunk_text_by_id[str(cid)] = str(row.get("chunk_text") or "")
        chunk_meta_by_id = {}
        for _, row in chunks.iterrows():
            cid = row.get("chunk_id_global") or row.get("chunk_id")
            if cid:
                chunk_meta_by_id[str(cid)] = {
                    "page_start": row.get("page_start"),
                    "section_title": row.get("section_title"),
                    "subsection_title": row.get("subsection_title"),
                }
        for item in results.get("results", []):
            per_k = item.get("per_k", {})
            k1 = per_k.get("1", {})
            max_k_key = max(per_k.keys(), key=lambda value: int(value)) if per_k else None
            max_k_data = per_k.get(max_k_key, {}) if max_k_key is not None else {}
            chunk_ids = k1.get("retrieved_chunk_ids") or []
            if isinstance(chunk_ids, str):
                chunk_ids = [chunk_ids]
            top_chunk_id = str(chunk_ids[0]) if chunk_ids else ""
            chunk_meta = chunk_meta_by_id.get(top_chunk_id, {})
            page_hit = item.get("page_hit")
            if page_hit is None:
                page_hit = 1 if k1.get("page_recall_at_k", 0.0) > 0 else 0
            failure_type = item.get("failure_type") or k1.get("failure_stage")
            if failure_type:
                failure_types.append(str(failure_type))
            failure_stage = FAILURE_STAGE_BY_TYPE.get(str(failure_type))
            survival_rows.extend(
                build_survival_rows_for_results(
                    doc_id,
                    {"results": [item]},
                    "hybrid",
                )
            )
            query_rows.append(
                {
                    "doc_id": doc_id,
                    "query_id": item.get("query_id"),
                    "question": item.get("question"),
                    "expected_pages": item.get("expected_pages"),
                    "expected_section": item.get("expected_section"),
                    "expected_subsection": item.get("expected_subsection"),
                    "evidence_layout": item.get("evidence_layout"),
                    "acceptable_evidence": item.get("acceptable_evidence"),
                    "filter_hints": item.get("filter_hints"),
                    "page_hit": page_hit,
                    "failure_type": failure_type,
                    "failure_stage": failure_stage,
                    "extracted_answer": item.get("extracted_answer"),
                    "extracted_answer_label": item.get("extracted_answer_label"),
                    "top_chunk_id": top_chunk_id,
                    "top_chunk_text": chunk_text_by_id.get(top_chunk_id, ""),
                    "top_pages": k1.get("retrieved_pages_ranked"),
                    "section_title": chunk_meta.get("section_title"),
                    "subsection_title": chunk_meta.get("subsection_title"),
                    "page_start": chunk_meta.get("page_start"),
                }
            )

    if query_rows:
        qdf = pd.DataFrame(query_rows)
        out_q_csv = resolve_output_path(
            data_root, args.out_queries_csv, "retrieval_queries_report.csv"
        )
        out_q_md = resolve_output_path(
            data_root, args.out_queries_md, "retrieval_queries_report.md"
        )
        out_q_tex = resolve_output_path(
            data_root, args.out_queries_tex, "retrieval_queries_report.tex"
        )
        out_q_csv.parent.mkdir(parents=True, exist_ok=True)
        qdf.to_csv(out_q_csv, index=False)
        out_q_md.write_text(qdf.to_markdown(index=False), encoding="utf-8")
        out_q_tex.write_text(qdf.to_latex(index=False), encoding="utf-8")
        print(f"Wrote: {out_q_csv}")
        print(f"Wrote: {out_q_md}")
        print(f"Wrote: {out_q_tex}")

    if failure_types:
        failure_counts = (
            pd.Series(failure_types, name="failure_type")
            .value_counts(dropna=False)
            .sort_index()
        )
        stage_counts = pd.Series(
            [FAILURE_STAGE_BY_TYPE.get(ft, "unknown") for ft in failure_types],
            name="failure_stage",
        ).value_counts(dropna=False)
        summary = {"total_queries": int(failure_counts.sum())}
        for name, count in stage_counts.items():
            summary[f"count_stage_{name}"] = int(count)
        for name, count in failure_counts.items():
            summary[f"count_{name}"] = int(count)
        out_fail = resolve_output_path(
            data_root, args.out_failure_summary, "retrieval_failure_summary.csv"
        )
        out_fail.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([summary]).to_csv(out_fail, index=False)
        print(f"Wrote: {out_fail}")

    if table_misses:
        out_miss = resolve_output_path(
            data_root, args.out_table_misses, "retrieval_table_misses_k1.csv"
        )
        out_miss.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(table_misses).to_csv(out_miss, index=False)
        print(f"Wrote: {out_miss}")

    if survival_rows:
        sdf = pd.DataFrame(survival_rows).sort_values(["doc_id", "query_id"])
        km_df = bootstrap_kaplan_meier_curve(
            sdf,
            n_bootstrap=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        )
        out_survival_csv = resolve_output_path(
            data_root, args.out_survival_csv, "retrieval_rank_survival.csv"
        )
        out_survival_km_csv = resolve_output_path(
            data_root, args.out_survival_km_csv, "retrieval_rank_km_curve.csv"
        )
        out_survival_plot = resolve_output_path(
            data_root, args.out_survival_plot, "retrieval_rank_km_curve.png"
        )
        out_survival_csv.parent.mkdir(parents=True, exist_ok=True)
        sdf.to_csv(out_survival_csv, index=False)
        km_df.to_csv(out_survival_km_csv, index=False)
        plot_kaplan_meier_curve(
            km_df,
            out_survival_plot,
            bootstrap_samples=args.bootstrap_samples,
        )
        print(f"Wrote: {out_survival_csv}")
        print(f"Wrote: {out_survival_km_csv}")
        print(f"Wrote: {out_survival_plot}")

    if survival_compare_rows:
        compare_df = pd.DataFrame(survival_compare_rows).sort_values(
            ["doc_id", "query_id", "system"]
        )
        compare_df["paired_query_id"] = (
            compare_df["doc_id"].astype(str) + "::" + compare_df["query_id"].astype(str)
        )
        compare_km_df = bootstrap_kaplan_meier_curve_by_group(
            compare_df,
            group_col="system",
            n_bootstrap=args.bootstrap_samples,
            seed=args.bootstrap_seed,
            paired_unit_col="paired_query_id",
        )
        out_compare_csv = resolve_output_path(
            data_root,
            args.out_survival_compare_csv,
            "retrieval_rank_survival_compare.csv",
        )
        out_compare_km_csv = resolve_output_path(
            data_root,
            args.out_survival_compare_km_csv,
            "retrieval_rank_km_compare_curve.csv",
        )
        out_compare_plot = resolve_output_path(
            data_root,
            args.out_survival_compare_plot,
            "retrieval_rank_km_compare_curve.png",
        )
        out_compare_csv.parent.mkdir(parents=True, exist_ok=True)
        compare_df.to_csv(out_compare_csv, index=False)
        compare_km_df.to_csv(out_compare_km_csv, index=False)
        plot_kaplan_meier_comparison_curve(
            compare_km_df,
            out_compare_plot,
            bootstrap_samples=args.bootstrap_samples,
        )
        print(f"Wrote: {out_compare_csv}")
        print(f"Wrote: {out_compare_km_csv}")
        print(f"Wrote: {out_compare_plot}")


if __name__ == "__main__":
    main()
