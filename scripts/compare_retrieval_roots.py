from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCS = [
    "Grampian-2020-2021",
    "Grampian-2021-2022",
    "Grampian-2022-2023",
    "Grampian-2023-2024",
    "Grampian-2024-2025",
]


def _load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(obj: dict, k: int, key: str) -> float | None:
    metrics_by_k = obj.get("metrics_by_k", {})
    row = metrics_by_k.get(str(k), {}) if isinstance(metrics_by_k, dict) else {}
    value = row.get(key)
    return None if value is None else float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare retrieval metrics across two processed roots.")
    parser.add_argument("--base-root", default=str(REPO_ROOT / "data_processed"))
    parser.add_argument("--compare-root", default=str(REPO_ROOT / "data_processed_tiktoken_5docs"))
    parser.add_argument("--out-csv", default=str(REPO_ROOT / "results" / "retrieval_compare_fallback_vs_tiktoken_2020_2025.csv"))
    parser.add_argument("--docs", nargs="*", default=DEFAULT_DOCS)
    args = parser.parse_args()

    base_root = Path(args.base_root)
    compare_root = Path(args.compare_root)
    rows: list[dict[str, object]] = []

    for doc_id in args.docs:
        base = _load_metrics(base_root / doc_id / "retrieval_metrics_hybrid.json")
        comp = _load_metrics(compare_root / doc_id / "retrieval_metrics_hybrid.json")
        row: dict[str, object] = {"doc_id": doc_id}
        for k in (1, 3, 5, 10):
            for key in ("page_hit_rate_at_k", "mean_page_mrr_at_k", "chunk_hit_rate_at_k", "mean_chunk_mrr_at_k"):
                base_val = _metric(base, k, key)
                comp_val = _metric(comp, k, key)
                short = key.replace("mean_", "").replace("_at_k", "")
                row[f"fallback_k{k}_{short}"] = base_val
                row[f"tiktoken_k{k}_{short}"] = comp_val
                row[f"delta_k{k}_{short}"] = None if base_val is None or comp_val is None else round(comp_val - base_val, 4)
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(out_df.to_string(index=False))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
