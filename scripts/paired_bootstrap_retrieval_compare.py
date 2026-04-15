from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Paired bootstrap comparison between two retrieval result JSON files."
    )
    p.add_argument("--system-a", required=True, help="Path to retrieval_results*.json for system A")
    p.add_argument("--system-b", required=True, help="Path to retrieval_results*.json for system B")
    p.add_argument("--mrr-k", type=int, default=10, help="k for MRR metric (default: 10)")
    p.add_argument("--n-bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-dir",
        default="results/paired_bootstrap_retrieval_compare",
        help="Output directory",
    )
    return p.parse_args()


def _load_results(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results", [])
    out: dict[str, dict[str, Any]] = {}
    for row in results:
        qid = str(row.get("query_id", "")).strip()
        if qid:
            out[qid] = row
    return out


def _hit_from_pk(pk: dict[str, Any]) -> float:
    if "page_hit_at_k" in pk and pk["page_hit_at_k"] is not None:
        return float(pk["page_hit_at_k"])
    if "page_hit" in pk and pk["page_hit"] is not None:
        return float(pk["page_hit"])
    return 1.0 if float(pk.get("page_recall_at_k", 0.0)) > 0.0 else 0.0


def _mrr_from_pk(pk: dict[str, Any]) -> float:
    if "page_mrr_at_k" in pk and pk["page_mrr_at_k"] is not None:
        return float(pk["page_mrr_at_k"])
    if "mrr_at_k" in pk and pk["mrr_at_k"] is not None:
        return float(pk["mrr_at_k"])
    return 0.0


def _extract_metrics(row: dict[str, Any], mrr_k: int) -> dict[str, float]:
    per_k = row.get("per_k", {})
    k1 = per_k.get("1", {})
    k3 = per_k.get("3", {})
    km = per_k.get(str(mrr_k), {})
    return {
        "hit_at_1": _hit_from_pk(k1),
        "hit_at_3": _hit_from_pk(k3),
        "mrr": _mrr_from_pk(km),
    }


def _bootstrap_delta(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int) -> dict[str, float]:
    if len(a) != len(b):
        raise ValueError("Arrays must have same length for paired bootstrap.")
    n = len(a)
    if n == 0:
        return {"observed_delta": float("nan"), "mean_delta": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    rng = np.random.default_rng(seed)
    diffs = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        diffs[i] = float(np.mean(a[idx]) - np.mean(b[idx]))
    observed = float(np.mean(a) - np.mean(b))
    return {
        "observed_delta": observed,
        "mean_delta": float(np.mean(diffs)),
        "ci95_low": float(np.percentile(diffs, 2.5)),
        "ci95_high": float(np.percentile(diffs, 97.5)),
    }


def main() -> None:
    args = parse_args()
    path_a = Path(args.system_a).resolve()
    path_b = Path(args.system_b).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_a = _load_results(path_a)
    rows_b = _load_results(path_b)
    common_qids = sorted(set(rows_a).intersection(set(rows_b)))
    if not common_qids:
        raise ValueError("No overlapping query_id values found between system A and B.")

    per_query_rows: list[dict[str, Any]] = []
    for qid in common_qids:
        ma = _extract_metrics(rows_a[qid], mrr_k=int(args.mrr_k))
        mb = _extract_metrics(rows_b[qid], mrr_k=int(args.mrr_k))
        per_query_rows.append(
            {
                "query_id": qid,
                "hit_at_1_a": ma["hit_at_1"],
                "hit_at_1_b": mb["hit_at_1"],
                "delta_hit_at_1_a_minus_b": ma["hit_at_1"] - mb["hit_at_1"],
                "hit_at_3_a": ma["hit_at_3"],
                "hit_at_3_b": mb["hit_at_3"],
                "delta_hit_at_3_a_minus_b": ma["hit_at_3"] - mb["hit_at_3"],
                "mrr_a": ma["mrr"],
                "mrr_b": mb["mrr"],
                "delta_mrr_a_minus_b": ma["mrr"] - mb["mrr"],
            }
        )

    qdf = pd.DataFrame(per_query_rows)
    hit1_a = qdf["hit_at_1_a"].to_numpy(dtype=np.float64)
    hit1_b = qdf["hit_at_1_b"].to_numpy(dtype=np.float64)
    hit3_a = qdf["hit_at_3_a"].to_numpy(dtype=np.float64)
    hit3_b = qdf["hit_at_3_b"].to_numpy(dtype=np.float64)
    mrr_a = qdf["mrr_a"].to_numpy(dtype=np.float64)
    mrr_b = qdf["mrr_b"].to_numpy(dtype=np.float64)

    hit1_boot = _bootstrap_delta(hit1_a, hit1_b, n_boot=int(args.n_bootstrap), seed=int(args.seed))
    hit3_boot = _bootstrap_delta(hit3_a, hit3_b, n_boot=int(args.n_bootstrap), seed=int(args.seed) + 1)
    mrr_boot = _bootstrap_delta(mrr_a, mrr_b, n_boot=int(args.n_bootstrap), seed=int(args.seed) + 2)

    summary = {
        "inputs": {
            "system_a": str(path_a),
            "system_b": str(path_b),
            "n_common_queries": int(len(common_qids)),
            "mrr_k": int(args.mrr_k),
            "n_bootstrap": int(args.n_bootstrap),
            "seed": int(args.seed),
        },
        "metrics": {
            "hit_at_1": hit1_boot,
            "hit_at_3": hit3_boot,
            f"mrr_at_{int(args.mrr_k)}": mrr_boot,
        },
    }

    per_query_path = out_dir / "paired_bootstrap_per_query_deltas.csv"
    summary_path = out_dir / "paired_bootstrap_summary.json"
    report_path = out_dir / "paired_bootstrap_report.md"

    qdf.to_csv(per_query_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Paired Bootstrap Retrieval Comparison",
        "",
        f"- System A: `{path_a}`",
        f"- System B: `{path_b}`",
        f"- Common queries: `{len(common_qids)}`",
        f"- Bootstrap iterations: `{args.n_bootstrap}`",
        "",
        "## Delta = A - B",
        "",
        f"- Hit@1: observed={hit1_boot['observed_delta']:.6f}, 95% CI=[{hit1_boot['ci95_low']:.6f}, {hit1_boot['ci95_high']:.6f}]",
        f"- Hit@3: observed={hit3_boot['observed_delta']:.6f}, 95% CI=[{hit3_boot['ci95_low']:.6f}, {hit3_boot['ci95_high']:.6f}]",
        f"- MRR@{int(args.mrr_k)}: observed={mrr_boot['observed_delta']:.6f}, 95% CI=[{mrr_boot['ci95_low']:.6f}, {mrr_boot['ci95_high']:.6f}]",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("Saved:", per_query_path)
    print("Saved:", summary_path)
    print("Saved:", report_path)
    print("Hit@1:", hit1_boot)
    print("Hit@3:", hit3_boot)
    print(f"MRR@{int(args.mrr_k)}:", mrr_boot)


if __name__ == "__main__":
    main()
