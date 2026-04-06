from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml

repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from rag_pdf.retrieval.query_rewrite import generate_query_rewrites


@dataclass
class ExperimentResult:
    """Container for one ablation run summary."""

    name: str
    doc_id: str
    data_dir: str
    mode: str
    k_list: list[int]
    metrics_by_k: dict[str, Any]
    answer_scoring: Optional[dict[str, Any]]
    benchmark: Optional[dict[str, Any]]
    benchmark_path: Optional[str]
    tokenizer_backend: Optional[str]
    tokenizer_exact_counting: Optional[bool]
    chunk_size_tokens: Optional[int]
    chunk_overlap_tokens: Optional[int]
    segment_aware_chunking: Optional[bool]
    whole_doc_markdown_mode: Optional[bool]
    markdown_header_carry_forward: Optional[bool]
    markdown_table_injection: Optional[bool]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for retrieval ablation orchestration."""
    parser = argparse.ArgumentParser(description="Run retrieval A/B ablation experiments from YAML config.")
    parser.add_argument(
        "--config",
        default="configs/retrieval_tuning.yaml",
        help="Path to ablation config YAML file.",
    )
    parser.add_argument(
        "--only",
        default="",
        help="Optional comma-separated experiment names to run.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML config file into a dictionary."""
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise ValueError("Config root must be a mapping.")
    return obj


def run_cmd(cmd: list[str], env: Optional[dict[str, str]] = None) -> None:
    """Run subprocess command and raise on non-zero exit."""
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


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


def _merge_benchmark_cfg(exp: dict[str, Any], global_cfg: dict[str, Any]) -> dict[str, Any]:
    base = dict(global_cfg.get("benchmark") or {})
    override = dict(exp.get("benchmark") or {})
    base.update(override)
    return base


def _safe_name(s: str) -> str:
    out = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in s.strip())
    return out or "experiment"


def _normalize_ablation_path(path_value: Any) -> Path:
    raw = str(path_value or "").strip()
    if not raw:
        return Path(raw)
    p = Path(raw)
    if p.is_absolute():
        return p

    replacements = {
        "data_processed_ablation_intrinsic": "results/ablations/ablation_intrinsic",
        "data_processed_ablation_thesis_all_docs": "results/ablations/ablation_thesis_all_docs",
        "data_processed_ablation_thesis_5docs_q50": "results/ablations/ablation_thesis_5docs_q50",
        "data_processed_ablation": "results/ablations/ablation",
        "data_processed/ablation_intrinsic": "results/ablations/ablation_intrinsic",
        "data_processed/ablation_thesis_all_docs": "results/ablations/ablation_thesis_all_docs",
        "data_processed/ablation_thesis_5docs_q50": "results/ablations/ablation_thesis_5docs_q50",
        "data_processed/ablation_thesis": "results/ablations/ablation_thesis",
        "data_processed/ablation_splade": "results/ablations/ablation_splade",
        "data_processed/ablation_minilm_cap_5docs": "results/ablations/ablation_minilm_cap_5docs",
        "data_processed/ablation_224_56_5docs": "results/ablations/ablation_224_56_5docs",
        "data_processed/ablation": "results/ablations/ablation",
    }
    normalized = raw
    for old, new in replacements.items():
        if normalized == old or normalized.startswith(old + "/"):
            normalized = new + normalized[len(old) :]
            break
    return Path(normalized).resolve()


def run_benchmark(
    exp: dict[str, Any],
    global_cfg: dict[str, Any],
    data_dir: Path,
    doc_id: str,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    bench_cfg = _merge_benchmark_cfg(exp=exp, global_cfg=global_cfg)
    enabled = _as_bool(bench_cfg.get("enabled"), default=False)
    if not enabled:
        return None, None

    python_bin = str(global_cfg.get("python_bin", sys.executable))
    model = str(global_cfg.get("embed_model", "models/all-MiniLM-L6-v2"))
    exp_name = _safe_name(str(exp.get("name", "experiment")))

    output_dir = _normalize_ablation_path(global_cfg.get("output_dir", "results/ablations/ablation"))
    bench_dir = output_dir / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    output_json = bench_dir / f"{exp_name}_benchmark.json"

    cmd = [
        python_bin,
        "scripts/benchmark_search.py",
        "--mode",
        str(bench_cfg.get("mode", "local")),
        "--data-dir",
        str(data_dir),
        "--model",
        model,
        "--doc-id",
        str(bench_cfg.get("doc_id", doc_id)),
        "--api-url",
        str(bench_cfg.get("api_url", "http://127.0.0.1:8000")),
        "--question",
        str(bench_cfg.get("question", "What was the deficit?")),
        "--query-source",
        str(bench_cfg.get("query_source", "eval_set")),
        "--eval-set",
        str(bench_cfg.get("eval_set", "")),
        "--k",
        str(int(bench_cfg.get("k", 5))),
        "--num-queries",
        str(int(bench_cfg.get("num_queries", 100))),
        "--warmup-queries",
        str(int(bench_cfg.get("warmup_queries", 10))),
        "--concurrency",
        str(int(bench_cfg.get("concurrency", 1))),
        "--output-json",
        str(output_json),
    ]

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    run_cmd(cmd, env=env)
    benchmark = json.loads(output_json.read_text(encoding="utf-8"))
    return benchmark, str(output_json)


def ensure_eval_set(data_dir: Path, source_eval_set: Path) -> None:
    """Ensure eval_set.json exists in target data directory."""
    target = data_dir / "eval_set.json"
    if target.exists():
        return
    if not source_eval_set.exists():
        raise FileNotFoundError(f"Missing source eval_set.json: {source_eval_set}")
    shutil.copy2(source_eval_set, target)


def build_eval_set_rewrites(eval_path: Path, out_path: Path) -> None:
    """Generate eval_set_rewrites.json from eval_set.json using deterministic rewrites."""
    raw = json.loads(eval_path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict) and isinstance(raw.get("queries"), list):
        items = raw.get("queries", [])
    else:
        raise ValueError(f"Expected list or {{'queries': [...]}} in {eval_path}")

    rewritten: list[dict[str, Any]] = []
    for item in items:
        q = str(item.get("question", "")).strip()
        if not q:
            continue
        row = dict(item)
        row["rewrites"] = generate_query_rewrites(q)
        rewritten.append(row)

    out_path.write_text(json.dumps(rewritten, indent=2, ensure_ascii=False), encoding="utf-8")


def prepare_data_dir(exp: dict[str, Any], global_cfg: dict[str, Any]) -> tuple[Path, str]:
    """Prepare per-experiment data directory, optionally rebuilding with chunk settings."""
    data_root = Path(global_cfg["data_root"]).resolve()
    ablation_root = _normalize_ablation_path(global_cfg.get("ablation_root", "results/ablations/ablation"))
    python_bin = str(global_cfg.get("python_bin", sys.executable))
    embed_model = str(global_cfg.get("embed_model", "models/all-MiniLM-L6-v2"))
    offline_env = os.environ.copy()
    offline_env.setdefault("TRANSFORMERS_OFFLINE", "1")
    offline_env.setdefault("HF_HUB_OFFLINE", "1")
    offline_env.setdefault("OMP_NUM_THREADS", "1")
    offline_env.setdefault("MKL_NUM_THREADS", "1")
    offline_env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    offline_env.setdefault("TOKENIZERS_PARALLELISM", "false")

    mode = str(exp.get("data_mode", "existing"))
    doc_id = str(exp.get("doc_id") or global_cfg.get("default_doc_id") or "").strip()
    if not doc_id:
        raise ValueError(f"Experiment {exp.get('name')} missing doc_id.")

    if mode == "existing":
        data_dir = data_root / doc_id
        if not data_dir.exists():
            raise FileNotFoundError(f"Missing data dir for existing mode: {data_dir}")
        return data_dir, doc_id

    if mode != "rebuild":
        raise ValueError(f"Unsupported data_mode: {mode}")

    pdf_path = Path(str(exp.get("pdf_path") or global_cfg.get("pdf_path") or "")).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"Missing PDF for rebuild mode: {pdf_path}")

    chunk_cfg = exp.get("chunking") or {}
    chunk_size = int(chunk_cfg.get("size_tokens", 224))
    chunk_overlap = int(chunk_cfg.get("overlap_tokens", 56))
    segment_aware = bool(chunk_cfg.get("segment_aware", False))
    whole_doc_markdown_mode = bool(chunk_cfg.get("whole_doc_markdown_mode", False))
    markdown_header_carry_forward = bool(chunk_cfg.get("markdown_header_carry_forward", True))
    markdown_table_injection = bool(chunk_cfg.get("markdown_table_injection", True))

    run_root = ablation_root / str(exp["name"])
    run_root.mkdir(parents=True, exist_ok=True)

    preprocess_cmd = [
        python_bin,
        "scripts/preprocess_hybrid.py",
        "--pdf-path",
        str(pdf_path),
        "--out-root",
        str(run_root),
        "--chunk-size-tokens",
        str(chunk_size),
        "--chunk-overlap-tokens",
        str(chunk_overlap),
    ]
    # Set preprocessing toggles explicitly to keep ablation runs reproducible.
    pre_env = os.environ.copy()
    pre_env["SEGMENT_AWARE_CHUNKING"] = "1" if segment_aware else "0"
    pre_env["WHOLE_DOC_MARKDOWN_MODE"] = "1" if whole_doc_markdown_mode else "0"
    pre_env["MARKDOWN_HEADER_CARRY_FORWARD"] = "1" if markdown_header_carry_forward else "0"
    pre_env["MARKDOWN_TABLE_INJECTION"] = "1" if markdown_table_injection else "0"
    run_cmd(preprocess_cmd, env=pre_env)
    run_cmd(
        [
            python_bin,
            "scripts/build_index.py",
            "--data-dir",
            str(run_root),
            "--model",
            embed_model,
        ],
        env=offline_env,
    )

    data_dir = run_root / doc_id
    source_eval_set = _normalize_ablation_path(exp.get("source_eval_set") or global_cfg.get("source_eval_set") or "")
    if not source_eval_set.exists():
        fallback = data_root / doc_id / "eval_set.json"
        source_eval_set = fallback
    ensure_eval_set(data_dir, source_eval_set)

    return data_dir, doc_id


def run_experiment(exp: dict[str, Any], global_cfg: dict[str, Any]) -> ExperimentResult:
    """Run one experiment and return parsed metrics summary."""
    data_dir, doc_id = prepare_data_dir(exp=exp, global_cfg=global_cfg)
    mode = str(exp.get("mode", "baseline")).strip().lower()
    if mode not in {"baseline", "rewrite", "bm25", "hybrid", "splade_hybrid"}:
        raise ValueError(f"Unsupported mode in {exp.get('name')}: {mode}")

    model = str(global_cfg.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2"))
    k_list = exp.get("k_list") or global_cfg.get("k_list") or [1, 3, 5, 10]
    k_csv = ",".join(str(int(k)) for k in k_list)

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    rerank = exp.get("rerank") or {}
    rerank_enabled = bool(rerank.get("enabled", True))
    subsection_enabled = bool(rerank.get("enable_subsection_boost", rerank_enabled))

    env["ENABLE_LEXICAL_RERANK"] = "1" if rerank_enabled else "0"
    env["ENABLE_SUBSECTION_BOOST"] = "1" if subsection_enabled else "0"

    env["TABLE_CHUNK_BOOST"] = str(float(rerank.get("table_chunk_boost", 0.08 if rerank_enabled else 0.0)))
    env["MILESTONE_TEXT_BOOST"] = str(float(rerank.get("milestone_text_boost", 0.08 if rerank_enabled else 0.0)))
    env["ENTITY_MATCH_BOOST"] = str(float(rerank.get("entity_match_boost", 0.04 if rerank_enabled else 0.0)))
    env["NUMERIC_DENSITY_BOOST"] = str(float(rerank.get("numeric_density_boost", 0.03 if rerank_enabled else 0.0)))
    env["SEGMENT_SEARCH_HIT_BOOST"] = str(float(rerank.get("segment_search_hit_boost", 0.03 if rerank_enabled else 0.0)))
    env["SUBSECTION_BOOST"] = str(float(rerank.get("subsection_boost", 0.05 if subsection_enabled else 0.0)))

    ce_cfg = exp.get("cross_encoder") or global_cfg.get("cross_encoder") or {}
    ce_enabled = bool(ce_cfg.get("enabled", False))
    ce_model = str(ce_cfg.get("model", "models/bge-reranker-v2-m3"))
    ce_topn = int(ce_cfg.get("topn", 50))
    ce_weight = float(ce_cfg.get("weight", 0.2))

    python_bin = str(global_cfg.get("python_bin", sys.executable))
    if mode == "baseline":
        run_cmd(
            [
                python_bin,
                "scripts/retrieval_eval.py",
                "--data-dir",
                str(data_dir),
                "--model",
                model,
                "--k-list",
                k_csv,
            ],
            env=env,
        )
        metrics_path = data_dir / "retrieval_metrics.json"
    elif mode == "rewrite":
        eval_set_path = data_dir / "eval_set.json"
        rewrites_path = data_dir / "eval_set_rewrites.json"
        build_eval_set_rewrites(eval_set_path, rewrites_path)
        run_cmd(
            [
                python_bin,
                "scripts/retrieval_eval_rewrites.py",
                "--data-dir",
                str(data_dir),
                "--model",
                model,
                "--k-list",
                k_csv,
                "--max-k-per-variant",
                str(int(exp.get("max_k_per_variant", 20))),
            ],
            env=env,
        )
        metrics_path = data_dir / "retrieval_metrics_rewrites.json"
    elif mode == "bm25":
        run_cmd(
            [
                python_bin,
                "scripts/retrieval_eval_bm25.py",
                "--data-dir",
                str(data_dir),
                "--k-list",
                k_csv,
                "--k1",
                str(float(exp.get("bm25_k1", global_cfg.get("bm25_k1", 1.5)))),
                "--b",
                str(float(exp.get("bm25_b", global_cfg.get("bm25_b", 0.75)))),
            ],
            env=env,
        )
        metrics_path = data_dir / "retrieval_metrics_bm25.json"
    elif mode == "hybrid":
        hybrid_cmd = [
            python_bin,
            "scripts/retrieval_eval_hybrid.py",
            "--data-dir",
            str(data_dir),
            "--model",
            model,
            "--k-list",
            k_csv,
            "--rrf-k",
            str(int(exp.get("rrf_k", global_cfg.get("rrf_k", 20)))),
            "--dense-weight",
            str(float(exp.get("dense_weight", global_cfg.get("dense_weight", 0.5)))),
            "--bm25-weight",
            str(float(exp.get("bm25_weight", global_cfg.get("bm25_weight", 2.0)))),
            "--bm25-k1",
            str(float(exp.get("bm25_k1", global_cfg.get("bm25_k1", 1.5)))),
            "--bm25-b",
            str(float(exp.get("bm25_b", global_cfg.get("bm25_b", 0.75)))),
        ]
        if ce_enabled:
            hybrid_cmd.extend(
                [
                    "--enable-cross-encoder-rerank",
                    "--cross-encoder-model",
                    ce_model,
                    "--cross-encoder-topn",
                    str(ce_topn),
                    "--cross-encoder-weight",
                    str(ce_weight),
                ]
            )
        run_cmd(hybrid_cmd, env=env)
        metrics_path = data_dir / "retrieval_metrics_hybrid.json"
    else:
        splade_local_only = bool(exp.get("splade_local_only", global_cfg.get("splade_local_only", False)))
        splade_cmd = [
            python_bin,
            "scripts/retrieval_eval_splade_hybrid.py",
            "--data-dir",
            str(data_dir),
            "--model",
            model,
            "--splade-model",
            str(exp.get("splade_model", global_cfg.get("splade_model", "models/naver-splade-cocondenser-ensembledistil"))),
            "--splade-device",
            str(exp.get("splade_device", global_cfg.get("splade_device", "auto"))),
            "--splade-max-length",
            str(int(exp.get("splade_max_length", global_cfg.get("splade_max_length", 256)))),
            "--splade-doc-batch-size",
            str(int(exp.get("splade_doc_batch_size", global_cfg.get("splade_doc_batch_size", 16)))),
            "--splade-query-batch-size",
            str(int(exp.get("splade_query_batch_size", global_cfg.get("splade_query_batch_size", 16)))),
            "--splade-doc-top-terms",
            str(int(exp.get("splade_doc_top_terms", global_cfg.get("splade_doc_top_terms", 128)))),
            "--splade-query-top-terms",
            str(int(exp.get("splade_query_top_terms", global_cfg.get("splade_query_top_terms", 64)))),
            "--splade-min-weight",
            str(float(exp.get("splade_min_weight", global_cfg.get("splade_min_weight", 0.01)))),
            "--k-list",
            k_csv,
            "--rrf-k",
            str(int(exp.get("rrf_k", global_cfg.get("rrf_k", 20)))),
            "--dense-weight",
            str(float(exp.get("dense_weight", global_cfg.get("dense_weight", 0.5)))),
            "--splade-weight",
            str(float(exp.get("splade_weight", global_cfg.get("splade_weight", 1.0)))),
        ]
        if ce_enabled:
            splade_cmd.extend(
                [
                    "--enable-cross-encoder-rerank",
                    "--cross-encoder-model",
                    ce_model,
                    "--cross-encoder-topn",
                    str(ce_topn),
                    "--cross-encoder-weight",
                    str(ce_weight),
                ]
            )
        if splade_local_only:
            splade_cmd.append("--splade-local-only")
        run_cmd(
            splade_cmd,
            env=env,
        )
        metrics_path = data_dir / "retrieval_metrics_splade_hybrid.json"

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    benchmark, benchmark_path = run_benchmark(
        exp=exp,
        global_cfg=global_cfg,
        data_dir=data_dir,
        doc_id=doc_id,
    )
    preprocess_metrics_path = data_dir / "metrics.json"
    preprocess_metrics: dict[str, Any] = {}
    if preprocess_metrics_path.exists():
        try:
            preprocess_metrics = json.loads(preprocess_metrics_path.read_text(encoding="utf-8"))
        except Exception:
            preprocess_metrics = {}
    preprocess_params = preprocess_metrics.get("params", {}) if isinstance(preprocess_metrics, dict) else {}
    tokenizer_backend = preprocess_params.get("tokenizer_backend")
    tokenizer_exact_counting = preprocess_params.get("tokenizer_exact_counting")
    chunk_size_tokens = preprocess_params.get("chunk_size_tokens")
    chunk_overlap_tokens = preprocess_params.get("chunk_overlap_tokens")
    segment_aware_chunking = preprocess_params.get("segment_aware_chunking")
    whole_doc_markdown_mode = preprocess_params.get("whole_doc_markdown_mode")
    markdown_header_carry_forward = preprocess_params.get("markdown_header_carry_forward")
    markdown_table_injection = preprocess_params.get("markdown_table_injection")

    return ExperimentResult(
        name=str(exp["name"]),
        doc_id=doc_id,
        data_dir=str(data_dir),
        mode=mode,
        k_list=[int(k) for k in k_list],
        metrics_by_k=metrics.get("metrics_by_k", {}),
        answer_scoring=metrics.get("answer_scoring"),
        benchmark=benchmark,
        benchmark_path=benchmark_path,
        tokenizer_backend=(str(tokenizer_backend) if tokenizer_backend is not None else None),
        tokenizer_exact_counting=(bool(tokenizer_exact_counting) if tokenizer_exact_counting is not None else None),
        chunk_size_tokens=(int(chunk_size_tokens) if isinstance(chunk_size_tokens, (int, float)) else None),
        chunk_overlap_tokens=(int(chunk_overlap_tokens) if isinstance(chunk_overlap_tokens, (int, float)) else None),
        segment_aware_chunking=(bool(segment_aware_chunking) if segment_aware_chunking is not None else None),
        whole_doc_markdown_mode=(bool(whole_doc_markdown_mode) if whole_doc_markdown_mode is not None else None),
        markdown_header_carry_forward=(
            bool(markdown_header_carry_forward) if markdown_header_carry_forward is not None else None
        ),
        markdown_table_injection=(bool(markdown_table_injection) if markdown_table_injection is not None else None),
    )


def flatten_results(rows: list[ExperimentResult]) -> pd.DataFrame:
    """Flatten per-experiment metrics into a comparison DataFrame."""
    flat: list[dict[str, Any]] = []
    for row in rows:
        for k_str, m in row.metrics_by_k.items():
            flat.append(
                {
                    "experiment": row.name,
                    "doc_id": row.doc_id,
                    "mode": row.mode,
                    "data_dir": row.data_dir,
                    "k": int(k_str),
                    "page_hit_rate": m.get("page_hit_rate_at_k", m.get("hit_rate_at_k", 0.0)),
                    "page_recall": m.get("mean_page_recall_at_k", m.get("mean_recall_at_k", 0.0)),
                    "page_precision": m.get("mean_page_precision_at_k", m.get("mean_precision_at_k", 0.0)),
                    "page_mrr": m.get("mean_page_mrr_at_k", m.get("mean_mrr_at_k", 0.0)),
                    "chunk_hit_rate": m.get("chunk_hit_rate_at_k"),
                    "chunk_mrr": m.get("mean_chunk_mrr_at_k"),
                    "answer_accuracy": (row.answer_scoring or {}).get("answer_accuracy"),
                    "scored_queries": (row.answer_scoring or {}).get("num_queries_scored"),
                    "bench_mode": (row.benchmark or {}).get("mode"),
                    "bench_concurrency": (row.benchmark or {}).get("concurrency"),
                    "bench_throughput_qps": (row.benchmark or {}).get("throughput_qps"),
                    "bench_latency_p50_ms": ((row.benchmark or {}).get("latency_ms") or {}).get("p50"),
                    "bench_latency_p95_ms": ((row.benchmark or {}).get("latency_ms") or {}).get("p95"),
                    "bench_memory_peak_mb": ((row.benchmark or {}).get("memory") or {}).get("ru_maxrss_mb"),
                    "benchmark_json": row.benchmark_path,
                    "tokenizer_backend": row.tokenizer_backend,
                    "tokenizer_exact_counting": row.tokenizer_exact_counting,
                    "chunk_size_tokens": row.chunk_size_tokens,
                    "chunk_overlap_tokens": row.chunk_overlap_tokens,
                    "segment_aware_chunking": row.segment_aware_chunking,
                    "whole_doc_markdown_mode": row.whole_doc_markdown_mode,
                    "markdown_header_carry_forward": row.markdown_header_carry_forward,
                    "markdown_table_injection": row.markdown_table_injection,
                }
            )
    return pd.DataFrame(flat)


def main() -> None:
    """Entry point for retrieval ablation execution and reporting."""
    args = parse_args()
    cfg = load_config(Path(args.config).resolve())

    experiments = cfg.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("Config must include a non-empty 'experiments' list.")

    only = {x.strip() for x in args.only.split(",") if x.strip()}
    selected = [e for e in experiments if not only or str(e.get("name")) in only]
    if not selected:
        raise ValueError("No experiments selected. Check --only names.")

    results: list[ExperimentResult] = []
    for exp in selected:
        name = str(exp.get("name", "")).strip()
        if not name:
            raise ValueError("Each experiment must include a non-empty 'name'.")
        print(f"\n=== Running experiment: {name} ===")
        results.append(run_experiment(exp=exp, global_cfg=cfg))

    df = flatten_results(results)
    if df.empty:
        raise RuntimeError("No ablation rows produced.")

    out_dir = _normalize_ablation_path(cfg.get("output_dir", "results/ablations/ablation"))
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = out_dir / "retrieval_ablation_summary.csv"
    summary_json = out_dir / "retrieval_ablation_summary.json"
    best_csv = out_dir / "retrieval_ablation_best_by_k.csv"

    df = df.sort_values(["k", "page_hit_rate", "page_mrr", "page_precision"], ascending=[True, False, False, False])
    best = df.groupby("k", as_index=False).head(1).reset_index(drop=True)

    df.to_csv(summary_csv, index=False)
    best.to_csv(best_csv, index=False)

    payload = {
        "config_path": str(Path(args.config).resolve()),
        "experiments_run": [r.name for r in results],
        "benchmark_by_experiment": {
            r.name: {"path": r.benchmark_path, "metrics": r.benchmark}
            for r in results
            if r.benchmark is not None
        },
        "rows": json.loads(df.to_json(orient="records")),
        "best_by_k": json.loads(best.to_json(orient="records")),
    }
    summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Saved:", summary_csv)
    print("Saved:", best_csv)
    print("Saved:", summary_json)


if __name__ == "__main__":
    main()
