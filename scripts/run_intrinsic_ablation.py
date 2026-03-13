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
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run no-eval intrinsic ablations across multiple PDFs."
    )
    p.add_argument(
        "--config",
        default="configs/intrinsic_ablation_preliminary.yaml",
        help="YAML config path.",
    )
    p.add_argument(
        "--only-variants",
        default="",
        help="Optional comma-separated variant names.",
    )
    p.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="Optional cap on number of PDFs (0 = all).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run even when outputs already exist.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned runs only.",
    )
    return p.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("Config root must be a mapping.")
    return obj


def run_cmd(cmd: list[str], env: dict[str, str] | None = None) -> float:
    start = time.perf_counter()
    subprocess.run(cmd, check=True, env=env)
    return time.perf_counter() - start


def _as_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def _safe_name(s: str) -> str:
    out = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in s.strip())
    return out or "variant"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _size_bytes(path: Path) -> int:
    return int(path.stat().st_size) if path.exists() else 0


def _collect_row(
    *,
    variant_name: str,
    variant_cfg: dict[str, Any],
    doc_id: str,
    pdf_path: Path,
    run_dir: Path,
    preprocess_seconds: float,
    index_seconds: float,
) -> dict[str, Any]:
    doc_dir = run_dir / doc_id
    metrics = _load_json(doc_dir / "metrics.json")
    counts = metrics.get("counts", {}) if isinstance(metrics.get("counts"), dict) else {}
    derived = metrics.get("derived", {}) if isinstance(metrics.get("derived"), dict) else {}
    timing = metrics.get("timing", {}) if isinstance(metrics.get("timing"), dict) else {}
    params = metrics.get("params", {}) if isinstance(metrics.get("params"), dict) else {}
    emb = metrics.get("embedding", {}) if isinstance(metrics.get("embedding"), dict) else {}

    row: dict[str, Any] = {
        "variant": variant_name,
        "doc_id": doc_id,
        "pdf_path": str(pdf_path),
        "run_dir": str(doc_dir),
        "preprocess_wall_seconds": float(preprocess_seconds),
        "index_wall_seconds": float(index_seconds),
        "total_wall_seconds": float(preprocess_seconds + index_seconds),
        "preprocess_time_total_wall": timing.get("time_total_wall"),
        "chunk_size_tokens": params.get("chunk_size_tokens"),
        "chunk_overlap_tokens": params.get("chunk_overlap_tokens"),
        "segment_aware_chunking": params.get("segment_aware_chunking"),
        "whole_doc_markdown_mode": params.get("whole_doc_markdown_mode"),
        "markdown_header_carry_forward": params.get("markdown_header_carry_forward"),
        "markdown_table_injection": params.get("markdown_table_injection"),
        "pages_total": counts.get("pages_total"),
        "pages_text": counts.get("pages_text"),
        "pages_table": counts.get("pages_table"),
        "chunks_total": counts.get("chunks_total"),
        "chunks_text": counts.get("chunks_text"),
        "chunks_table": counts.get("chunks_table"),
        "tables_extracted": counts.get("tables_extracted"),
        "table_facts": counts.get("table_facts"),
        "ocr_raw_pages_detected": counts.get("ocr_raw_pages_detected"),
        "ocr_raw_pages_accepted": counts.get("ocr_raw_pages_accepted"),
        "ocr_short_pages_triggered": counts.get("ocr_short_pages_triggered"),
        "ocr_short_pages_accepted": counts.get("ocr_short_pages_accepted"),
        "ocr_rejected_quality": counts.get("ocr_rejected_quality"),
        "chunks_per_page": derived.get("chunks_per_page"),
        "tables_per_100_pages": derived.get("tables_per_100_pages"),
        "ocr_raw_acceptance_rate": derived.get("ocr_raw_acceptance_rate"),
        "ocr_short_acceptance_rate": derived.get("ocr_short_acceptance_rate"),
        "ocr_quality_reject_rate": derived.get("ocr_quality_reject_rate"),
        "embedding_dim": emb.get("embedding_dim"),
        "chunks_embedded": emb.get("chunks_embedded"),
        "faiss_size_bytes": _size_bytes(doc_dir / "faiss.index"),
        "embeddings_npy_size_bytes": _size_bytes(doc_dir / "embeddings.npy"),
        "chunk_meta_size_bytes": _size_bytes(doc_dir / "chunk_meta.parquet"),
    }
    # Add variant config snapshot for traceability.
    for k, v in variant_cfg.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            row[f"variant_cfg_{k}"] = v
    return row


def _make_charts(df_long: pd.DataFrame, out_dir: Path) -> None:
    if df_long.empty:
        return

    agg = (
        df_long.groupby("variant", as_index=False)
        .agg(
            docs_n=("doc_id", "nunique"),
            chunks_per_page_mean=("chunks_per_page", "mean"),
            tables_per_100_pages_mean=("tables_per_100_pages", "mean"),
            table_facts_mean=("table_facts", "mean"),
            preprocess_wall_seconds_mean=("preprocess_wall_seconds", "mean"),
            index_wall_seconds_mean=("index_wall_seconds", "mean"),
            total_wall_seconds_mean=("total_wall_seconds", "mean"),
        )
        .sort_values("total_wall_seconds_mean", ascending=True)
    )
    code_map = {v: f"V{i+1}" for i, v in enumerate(agg["variant"].tolist())}
    agg["variant_code"] = agg["variant"].map(code_map)
    pd.DataFrame(
        [{"variant_code": code_map[v], "variant": v} for v in agg["variant"].tolist()]
    ).to_csv(out_dir / "intrinsic_ablation_variant_legend.csv", index=False)
    agg.to_csv(out_dir / "intrinsic_ablation_comparison_by_variant.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")

    # Chart 1: runtime by variant
    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(agg))))
    y = range(len(agg))
    ax.barh(y, agg["preprocess_wall_seconds_mean"], label="preprocess")
    ax.barh(y, agg["index_wall_seconds_mean"], left=agg["preprocess_wall_seconds_mean"], label="index")
    ax.set_yticks(list(y))
    ax.set_yticklabels(list(agg["variant_code"]), fontsize=9)
    ax.set_xlabel("Mean Wall Time (seconds)")
    ax.set_title("Intrinsic Ablation: Runtime by Variant")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "chart_runtime_by_variant.png", dpi=180)
    plt.close(fig)

    # Chart 2: chunks/page and tables/100 pages
    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(agg))))
    y = range(len(agg))
    ax.barh(y, agg["chunks_per_page_mean"], label="chunks_per_page")
    ax.barh(y, agg["tables_per_100_pages_mean"], left=agg["chunks_per_page_mean"], label="tables_per_100_pages")
    ax.set_yticks(list(y))
    ax.set_yticklabels(list(agg["variant_code"]), fontsize=9)
    ax.set_xlabel("Mean Intrinsic Density Metrics")
    ax.set_title("Intrinsic Ablation: Content Density by Variant")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "chart_density_by_variant.png", dpi=180)
    plt.close(fig)

    # Chart 3: table facts vs total runtime scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, r in agg.iterrows():
        ax.scatter(float(r["total_wall_seconds_mean"]), float(r["table_facts_mean"]), s=70)
        ax.annotate(str(r["variant_code"]), (float(r["total_wall_seconds_mean"]), float(r["table_facts_mean"])), fontsize=9)
    ax.set_xlabel("Mean Total Wall Time (seconds)")
    ax.set_ylabel("Mean Table Facts")
    ax.set_title("Intrinsic Ablation: Table Facts vs Runtime")
    fig.tight_layout()
    fig.savefig(out_dir / "chart_tablefacts_vs_runtime.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(Path(args.config).resolve())

    python_bin = str(cfg.get("python_bin", ".venv/bin/python"))
    embed_model = str(cfg.get("embed_model", "models/all-MiniLM-L6-v2"))
    run_index = _as_bool(cfg.get("run_index"), default=False)
    pdf_dir = Path(str(cfg.get("pdf_dir", "Data/Annual Accounts NHS Grampian/Preliminary_Test"))).resolve()
    pdf_glob = str(cfg.get("pdf_glob", "*.pdf"))
    out_root = Path(str(cfg.get("out_root", "data_processed_ablation_intrinsic"))).resolve()
    summary_out_dir = Path(str(cfg.get("summary_out_dir", "data_processed/ablation_intrinsic"))).resolve()
    variants = cfg.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("Config must include a non-empty 'variants' list.")

    only = {v.strip() for v in args.only_variants.split(",") if v.strip()}
    selected = [v for v in variants if not only or str(v.get("name", "")) in only]
    if not selected:
        raise ValueError("No variants selected. Check --only-variants.")

    pdfs = sorted(pdf_dir.glob(pdf_glob))
    if args.max_docs and args.max_docs > 0:
        pdfs = pdfs[: args.max_docs]
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir} with glob {pdf_glob}.")

    summary_out_dir.mkdir(parents=True, exist_ok=True)

    offline_env = os.environ.copy()
    offline_env.setdefault("TRANSFORMERS_OFFLINE", "1")
    offline_env.setdefault("HF_HUB_OFFLINE", "1")
    offline_env.setdefault("OMP_NUM_THREADS", "1")
    offline_env.setdefault("MKL_NUM_THREADS", "1")
    offline_env.setdefault("OPENBLAS_NUM_THREADS", "1")
    offline_env.setdefault("NUMEXPR_NUM_THREADS", "1")
    offline_env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    offline_env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    offline_env.setdefault("KMP_INIT_AT_FORK", "FALSE")
    offline_env.setdefault("KMP_AFFINITY", "none")
    offline_env.setdefault("KMP_CREATE_SHM", "0")
    offline_env.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    offline_env.setdefault("TOKENIZERS_PARALLELISM", "false")

    rows: list[dict[str, Any]] = []

    print(f"Variants: {len(selected)} | PDFs: {len(pdfs)}")
    for v in selected:
        vname = _safe_name(str(v.get("name", "")))
        if not vname:
            raise ValueError("Each variant must include a non-empty name.")
        chunk_cfg = v.get("chunking") if isinstance(v.get("chunking"), dict) else {}
        size_tokens = int(chunk_cfg.get("size_tokens", 280))
        overlap_tokens = int(chunk_cfg.get("overlap_tokens", 90))
        segment_aware = _as_bool(chunk_cfg.get("segment_aware"), default=False)
        whole_md = _as_bool(chunk_cfg.get("whole_doc_markdown_mode"), default=False)
        md_header = _as_bool(chunk_cfg.get("markdown_header_carry_forward"), default=True)
        md_table = _as_bool(chunk_cfg.get("markdown_table_injection"), default=True)

        run_root = out_root / vname
        run_root.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Variant: {vname} ===")

        for pdf_path in pdfs:
            doc_id = pdf_path.stem
            doc_dir = run_root / doc_id
            has_metrics = (doc_dir / "metrics.json").exists()
            has_index = (doc_dir / "faiss.index").exists()
            if has_metrics and ((not run_index) or has_index) and not args.force:
                print(f"Skipping existing: {doc_id}")
                row = _collect_row(
                    variant_name=vname,
                    variant_cfg=v,
                    doc_id=doc_id,
                    pdf_path=pdf_path,
                    run_dir=run_root,
                    preprocess_seconds=0.0,
                    index_seconds=0.0,
                )
                row["status"] = "skipped_existing"
                rows.append(row)
                continue

            pre_cmd = [
                python_bin,
                "scripts/preprocess_hybrid.py",
                "--pdf-path",
                str(pdf_path),
                "--out-root",
                str(run_root),
                "--chunk-size-tokens",
                str(size_tokens),
                "--chunk-overlap-tokens",
                str(overlap_tokens),
            ]
            pre_env = os.environ.copy()
            pre_env["SEGMENT_AWARE_CHUNKING"] = "1" if segment_aware else "0"
            pre_env["WHOLE_DOC_MARKDOWN_MODE"] = "1" if whole_md else "0"
            pre_env["MARKDOWN_HEADER_CARRY_FORWARD"] = "1" if md_header else "0"
            pre_env["MARKDOWN_TABLE_INJECTION"] = "1" if md_table else "0"

            idx_cmd = [
                python_bin,
                "scripts/build_index.py",
                "--data-dir",
                str(doc_dir),
                "--model",
                embed_model,
            ]

            print(f"Running {doc_id} ...")
            if args.dry_run:
                print("  PRE:", " ".join(pre_cmd))
                if run_index:
                    print("  IDX:", " ".join(idx_cmd))
                continue

            preprocess_sec = run_cmd(pre_cmd, env=pre_env)
            index_sec = 0.0
            status = "processed_no_index"
            if run_index:
                status = "processed"
                try:
                    index_sec = run_cmd(idx_cmd, env=offline_env)
                except subprocess.CalledProcessError:
                    # Retry once with extra-conservative runtime settings to reduce
                    # OpenMP/shared-memory issues in restricted environments.
                    retry_env = offline_env.copy()
                    retry_env["KMP_SETTINGS"] = "0"
                    retry_env["KMP_HANDLE_SIGNALS"] = "0"
                    retry_env["MKL_SERVICE_FORCE_INTEL"] = "1"
                    retry_env["MALLOC_ARENA_MAX"] = "1"
                    try:
                        index_sec = run_cmd(idx_cmd, env=retry_env)
                    except subprocess.CalledProcessError:
                        status = "index_failed"
            row = _collect_row(
                variant_name=vname,
                variant_cfg=v,
                doc_id=doc_id,
                pdf_path=pdf_path,
                run_dir=run_root,
                preprocess_seconds=preprocess_sec,
                index_seconds=index_sec,
            )
            row["status"] = status
            rows.append(row)
            print(
                f"  done: preprocess={preprocess_sec:.1f}s index={index_sec:.1f}s "
                f"chunks={row.get('chunks_total')} tables={row.get('tables_extracted')} "
                f"status={status}"
            )

    if args.dry_run:
        print("\nDry-run complete.")
        return

    if not rows:
        raise RuntimeError("No rows produced.")

    df_long = pd.DataFrame(rows)
    df_long = df_long.sort_values(["variant", "doc_id"])
    long_csv = summary_out_dir / "intrinsic_ablation_comparison_long.csv"
    df_long.to_csv(long_csv, index=False)
    _make_charts(df_long=df_long, out_dir=summary_out_dir)

    print("\nSaved:", long_csv)
    print("Saved:", summary_out_dir / "intrinsic_ablation_comparison_by_variant.csv")
    print("Saved:", summary_out_dir / "chart_runtime_by_variant.png")
    print("Saved:", summary_out_dir / "chart_density_by_variant.png")
    print("Saved:", summary_out_dir / "chart_tablefacts_vs_runtime.png")


if __name__ == "__main__":
    main()
