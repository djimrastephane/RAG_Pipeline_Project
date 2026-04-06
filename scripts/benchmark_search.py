from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark retrieval latency, throughput, and memory usage."
    )
    parser.add_argument(
        "--mode",
        choices=["local", "api"],
        default="local",
        help="Benchmark local SearchService call path or API endpoint.",
    )
    parser.add_argument(
        "--data-dir",
        default="data_processed/Grampian-2024-2025",
        help="Processed document directory (required for local mode).",
    )
    parser.add_argument(
        "--model",
        default="models/all-MiniLM-L6-v2",
        help="Embedding model path/name (local mode).",
    )
    parser.add_argument(
        "--doc-id",
        default="Grampian-2024-2025",
        help="Doc id for API path /api/v1/docs/{doc_id}/search.",
    )
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000",
        help="Base API URL (api mode).",
    )
    parser.add_argument(
        "--question",
        default="What was the deficit?",
        help="Question used when query source is fixed.",
    )
    parser.add_argument(
        "--query-source",
        choices=["fixed", "eval_set"],
        default="fixed",
        help="Use a single fixed question or cycle questions from eval_set.json.",
    )
    parser.add_argument(
        "--eval-set",
        default="",
        help="Optional eval_set.json path. Defaults to <data-dir>/eval_set.json in eval_set mode.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-k retrieval value.",
    )
    parser.add_argument(
        "--include-generated-answer",
        action="store_true",
        help="Include answer generation in the benchmarked search path.",
    )
    parser.add_argument(
        "--gen-max-context-chunks",
        type=int,
        default=None,
        help="Optional override for generation context chunk count.",
    )
    parser.add_argument(
        "--gen-max-context-chars",
        type=int,
        default=None,
        help="Optional override for generation context char budget.",
    )
    parser.add_argument(
        "--gen-max-chunk-chars",
        type=int,
        default=None,
        help="Optional override for per-chunk char budget in generation.",
    )
    parser.add_argument(
        "--gen-timeout-seconds",
        type=float,
        default=None,
        help="Optional override for generation timeout.",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=100,
        help="Measured query count.",
    )
    parser.add_argument(
        "--warmup-queries",
        type=int,
        default=10,
        help="Warmup query count (excluded from metrics).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent workers.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to save benchmark output JSON.",
    )
    return parser.parse_args()


def _ru_maxrss_bytes() -> int:
    import resource

    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux commonly reports KiB.
    if platform.system() == "Darwin":
        return rss
    return rss * 1024


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if q <= 0:
        return sorted_vals[0]
    if q >= 100:
        return sorted_vals[-1]
    pos = (len(sorted_vals) - 1) * (q / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def _load_questions(args: argparse.Namespace) -> list[str]:
    if args.query_source == "fixed":
        return [str(args.question).strip()]

    eval_path = Path(args.eval_set) if args.eval_set else Path(args.data_dir) / "eval_set.json"
    if not eval_path.exists():
        raise FileNotFoundError(f"eval_set not found: {eval_path}")
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict) and isinstance(payload.get("queries"), list):
        items = payload["queries"]
    else:
        raise ValueError(
            f"Invalid eval_set format in {eval_path}: expected list or dict with 'queries' list"
        )

    qs: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        q = str(item.get("question", "")).strip()
        if q:
            qs.append(q)
    if not qs:
        raise ValueError(f"No valid questions found in {eval_path}")
    return qs


def _make_local_runner(args: argparse.Namespace) -> Callable[[str], None]:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from rag_pdf.services.search_service import SearchService

    service = SearchService(
        repo_root=repo_root,
        model_path=Path(args.model),
    )
    data_dir = Path(args.data_dir)
    k = int(args.k)
    generation_overrides = {
        "max_context_chunks": args.gen_max_context_chunks,
        "max_context_chars": args.gen_max_context_chars,
        "max_chunk_chars": args.gen_max_chunk_chars,
        "timeout_seconds": args.gen_timeout_seconds,
    }
    generation_overrides = {k: v for k, v in generation_overrides.items() if v is not None}

    def run(question: str) -> None:
        service.search(
            data_dir=data_dir,
            question=question,
            k=k,
            query_id=None,
            include_generated_answer=bool(args.include_generated_answer),
            generation_overrides=generation_overrides,
        )

    return run


def _make_api_runner(args: argparse.Namespace) -> Callable[[str], None]:
    base = str(args.api_url).rstrip("/")
    url = f"{base}/api/v1/docs/{args.doc_id}/search"
    headers = {"Content-Type": "application/json"}
    k = int(args.k)

    def run(question: str) -> None:
        body_payload: dict[str, Any] = {
            "question": question,
            "k": k,
            "include_generated_answer": bool(args.include_generated_answer),
        }
        if args.gen_max_context_chunks is not None:
            body_payload["gen_max_context_chunks"] = int(args.gen_max_context_chunks)
        if args.gen_max_context_chars is not None:
            body_payload["gen_max_context_chars"] = int(args.gen_max_context_chars)
        if args.gen_max_chunk_chars is not None:
            body_payload["gen_max_chunk_chars"] = int(args.gen_max_chunk_chars)
        if args.gen_timeout_seconds is not None:
            body_payload["gen_timeout_seconds"] = float(args.gen_timeout_seconds)
        body = json.dumps(body_payload).encode("utf-8")
        req = urllib.request.Request(url=url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                if resp.status < 200 or resp.status >= 300:
                    raise RuntimeError(f"HTTP {resp.status} from {url}")
                _ = resp.read()
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP error {e.code}: {msg}") from e

    return run


def _run_queries(
    run_one: Callable[[str], None],
    questions: list[str],
    total_queries: int,
    concurrency: int,
) -> tuple[list[float], float]:
    latencies_ms: list[float] = []
    t0 = time.perf_counter()

    if concurrency <= 1:
        for i in range(total_queries):
            q = questions[i % len(questions)]
            s = time.perf_counter()
            run_one(q)
            latencies_ms.append((time.perf_counter() - s) * 1000.0)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futs = []
            for i in range(total_queries):
                q = questions[i % len(questions)]
                futs.append(pool.submit(_timed_run, run_one, q))
            for fut in as_completed(futs):
                latencies_ms.append(fut.result())

    elapsed = time.perf_counter() - t0
    return latencies_ms, elapsed


def _timed_run(run_one: Callable[[str], None], question: str) -> float:
    s = time.perf_counter()
    run_one(question)
    return (time.perf_counter() - s) * 1000.0


def _summarize(
    latencies_ms: list[float],
    elapsed_s: float,
    measured_queries: int,
    warmup_queries: int,
    mode: str,
    concurrency: int,
    include_generated_answer: bool,
) -> dict[str, Any]:
    vals = sorted(latencies_ms)
    mean_ms = statistics.fmean(vals) if vals else float("nan")
    stdev_ms = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    qps = (measured_queries / elapsed_s) if elapsed_s > 0 else 0.0
    rss_bytes = _ru_maxrss_bytes()

    return {
        "mode": mode,
        "concurrency": int(concurrency),
        "include_generated_answer": bool(include_generated_answer),
        "queries_measured": int(measured_queries),
        "queries_warmup": int(warmup_queries),
        "throughput_qps": float(qps),
        "total_elapsed_s": float(elapsed_s),
        "latency_ms": {
            "min": float(vals[0]) if vals else None,
            "mean": float(mean_ms) if vals else None,
            "p50": float(_percentile(vals, 50)) if vals else None,
            "p90": float(_percentile(vals, 90)) if vals else None,
            "p95": float(_percentile(vals, 95)) if vals else None,
            "p99": float(_percentile(vals, 99)) if vals else None,
            "max": float(vals[-1]) if vals else None,
            "stdev": float(stdev_ms) if vals else None,
        },
        "memory": {
            "ru_maxrss_bytes": int(rss_bytes),
            "ru_maxrss_mb": float(rss_bytes / (1024.0 * 1024.0)),
        },
    }


def main() -> None:
    args = parse_args()
    if args.num_queries <= 0:
        raise ValueError("--num-queries must be > 0")
    if args.warmup_queries < 0:
        raise ValueError("--warmup-queries must be >= 0")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be > 0")

    questions = _load_questions(args)
    run_one = _make_local_runner(args) if args.mode == "local" else _make_api_runner(args)

    if args.warmup_queries:
        _run_queries(
            run_one=run_one,
            questions=questions,
            total_queries=int(args.warmup_queries),
            concurrency=int(args.concurrency),
        )

    latencies_ms, elapsed_s = _run_queries(
        run_one=run_one,
        questions=questions,
        total_queries=int(args.num_queries),
        concurrency=int(args.concurrency),
    )

    result = _summarize(
        latencies_ms=latencies_ms,
        elapsed_s=elapsed_s,
        measured_queries=int(args.num_queries),
        warmup_queries=int(args.warmup_queries),
        mode=str(args.mode),
        concurrency=int(args.concurrency),
        include_generated_answer=bool(args.include_generated_answer),
    )

    print(json.dumps(result, indent=2))
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
