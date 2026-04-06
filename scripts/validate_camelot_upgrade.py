from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

try:
    import pymupdf as fitz  # type: ignore
except Exception as e:
    raise RuntimeError("PyMuPDF is required for page counting. Install pymupdf.") from e


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rag_pdf.table_extract import TableResult, extract_tables_for_page, table_to_markdown


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate Camelot 3-pass extraction upgrade with structural checks, distribution stats, and retrieval impact."
    )
    p.add_argument(
        "--pdfs",
        nargs="+",
        required=True,
        help="PDF paths or glob patterns (e.g. 'Data/*.pdf').",
    )
    p.add_argument(
        "--sample-pages",
        default="",
        help="Comma-separated sample pages for CHECK 1 (applies to each PDF). Example: 21,22,24,26,27",
    )
    p.add_argument(
        "--baseline-metrics",
        default="",
        help="Optional baseline metrics path (json file or directory). If omitted, uses data_processed/<doc_id>/retrieval_*_hybrid files.",
    )
    p.add_argument(
        "--eval-set",
        default="",
        help="Optional eval_set.json path to copy into rebuilt output. If omitted, uses source data_processed/<doc_id>/eval_set.json.",
    )
    p.add_argument(
        "--output-dir",
        default="data_processed_camelot_validate",
        help="Output root for rebuilt artifacts and reports.",
    )
    p.add_argument(
        "--model",
        default="models/all-MiniLM-L6-v2",
        help="Embedding model path for build/eval scripts.",
    )
    p.add_argument(
        "--k-list",
        default="1,3,5,10",
        help="k-list for retrieval_eval_hybrid.py",
    )
    p.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable to run pipeline scripts.",
    )
    p.add_argument(
        "--checks",
        default="1,2,3",
        help="Comma-separated checks to run (any of: 1,2,3). Example: 1 or 1,2",
    )
    return p.parse_args()


def resolve_pdfs(pdf_inputs: list[str]) -> list[Path]:
    out: list[Path] = []
    for s in pdf_inputs:
        p = Path(s)
        if any(ch in s for ch in ["*", "?", "["]):
            out.extend(sorted(REPO_ROOT.glob(s)))
        elif p.exists():
            out.append(p.resolve())
        elif (REPO_ROOT / s).exists():
            out.append((REPO_ROOT / s).resolve())
    uniq = []
    seen = set()
    for p in out:
        key = str(p)
        if key not in seen and p.suffix.lower() == ".pdf":
            seen.add(key)
            uniq.append(p)
    return uniq


def parse_pages_arg(s: str) -> list[int]:
    s = str(s or "").strip()
    if not s:
        return []
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok.isdigit():
            out.append(int(tok))
    return sorted(set(out))


def parse_checks_arg(s: str) -> set[str]:
    toks = [x.strip() for x in str(s or "").split(",") if x.strip()]
    checks = {x for x in toks if x in {"1", "2", "3"}}
    return checks or {"1", "2", "3"}


def _preview_table_text(df: pd.DataFrame, max_lines: int = 30, max_chars: int = 800) -> str:
    md = table_to_markdown(df)
    lines = md.splitlines()[:max_lines]
    txt = "\n".join(lines)
    if len(txt) > max_chars:
        txt = txt[:max_chars].rstrip() + " ..."
    return txt


def run_extract_with_logs(pdf_path: Path, page_no: int) -> tuple[list[TableResult], list[str], Optional[str]]:
    buf = io.StringIO()
    err: Optional[str] = None
    with contextlib.redirect_stdout(buf):
        try:
            results = extract_tables_for_page(pdf_path=pdf_path, page_no=page_no, config=None)
        except Exception as e:
            results = []
            err = f"{type(e).__name__}: {e}"
    logs = [ln.strip() for ln in buf.getvalue().splitlines() if ln.strip()]
    if err:
        logs.append(f"exception: {err}")
    return results, logs, err


def classify_failure_reasons(logs: list[str]) -> dict[str, int]:
    reasons = {
        "no_tables_returned": 0,
        "accuracy_below_threshold": 0,
        "whitespace_above_max": 0,
        "empty_invalid_table": 0,
        "exception": 0,
        "all_passes_failed": 0,
    }
    for ln in logs:
        l = ln.lower()
        if "failed (no tables)" in l:
            reasons["no_tables_returned"] += 1
        if "failed (accuracy" in l:
            reasons["accuracy_below_threshold"] += 1
        if "failed (whitespace" in l:
            reasons["whitespace_above_max"] += 1
        if "failed (empty/invalid parsed tables)" in l:
            reasons["empty_invalid_table"] += 1
        if "exception:" in l:
            reasons["exception"] += 1
        if "all camelot passes failed" in l:
            reasons["all_passes_failed"] += 1
    return reasons


def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def check1_structural_sanity(pdf_path: Path, sample_pages: list[int]) -> None:
    print_section("CHECK 1: Structural Sanity Check on Sampled Pages")
    if not sample_pages:
        print("No --sample-pages provided. Skipping CHECK 1.")
        return
    print(f"PDF: {pdf_path}")
    for pg in sample_pages:
        results, logs, err = run_extract_with_logs(pdf_path=pdf_path, page_no=pg)
        if results:
            tr = results[0]
            preview = _preview_table_text(tr.dataframe)
            print(f"\n[page {pg}] flavor={tr.flavor}")
            print(
                f"  parsing_report: accuracy={tr.parsing_report.get('accuracy')}, "
                f"whitespace={tr.parsing_report.get('whitespace')}, "
                f"order={tr.parsing_report.get('order')}, page={tr.parsing_report.get('page')}"
            )
            print("  preview:")
            for ln in preview.splitlines():
                print(f"    {ln}")
        else:
            print(f"\n[page {pg}] flavor=none")
            if err:
                print(f"  exception={err}")
            if logs:
                print("  logs:")
                for ln in logs[-8:]:
                    print(f"    {ln}")


def check2_acceptance_distribution(pdf_paths: list[Path], output_dir: Path) -> pd.DataFrame:
    print_section("CHECK 2: Acceptance Distribution Across Whole Document Set")
    rows = []
    for pdf in pdf_paths:
        doc = fitz.open(pdf)
        n_pages = doc.page_count
        doc.close()
        doc_id = pdf.stem

        counts = {
            "accepted_by_lattice": 0,
            "accepted_by_hybrid": 0,
            "accepted_by_stream": 0,
            "no_table": 0,
        }
        reason_totals = {
            "no_tables_returned": 0,
            "accuracy_below_threshold": 0,
            "whitespace_above_max": 0,
            "empty_invalid_table": 0,
            "exception": 0,
            "all_passes_failed": 0,
        }

        for page_no in range(1, n_pages + 1):
            results, logs, _ = run_extract_with_logs(pdf_path=pdf, page_no=page_no)
            if results:
                flavor = str(results[0].flavor)
                if flavor == "lattice":
                    counts["accepted_by_lattice"] += 1
                elif flavor == "hybrid":
                    counts["accepted_by_hybrid"] += 1
                elif flavor == "stream":
                    counts["accepted_by_stream"] += 1
                else:
                    counts["no_table"] += 1
            else:
                counts["no_table"] += 1
            rs = classify_failure_reasons(logs)
            for k, v in rs.items():
                reason_totals[k] += v

        total_pages = max(1, n_pages)
        print(f"\nPDF: {doc_id} ({n_pages} pages)")
        print(
            f"  lattice={counts['accepted_by_lattice']} ({counts['accepted_by_lattice']/total_pages:.1%}) | "
            f"hybrid={counts['accepted_by_hybrid']} ({counts['accepted_by_hybrid']/total_pages:.1%}) | "
            f"stream={counts['accepted_by_stream']} ({counts['accepted_by_stream']/total_pages:.1%}) | "
            f"no_table={counts['no_table']} ({counts['no_table']/total_pages:.1%})"
        )
        print("  failures_by_reason:")
        for rk, rv in reason_totals.items():
            print(f"    - {rk}: {rv} ({rv/total_pages:.1%})")
        rows.append({"doc_id": doc_id, "pdf_path": str(pdf), "pages": n_pages, **counts, **reason_totals})

    df = pd.DataFrame(rows)
    if not df.empty:
        totals = df.sum(numeric_only=True).to_dict()
        pages_total = max(1, int(totals.get("pages", 0)))
        print("\nOVERALL TOTALS")
        print(
            f"  lattice={int(totals.get('accepted_by_lattice',0))} ({totals.get('accepted_by_lattice',0)/pages_total:.1%}) | "
            f"hybrid={int(totals.get('accepted_by_hybrid',0))} ({totals.get('accepted_by_hybrid',0)/pages_total:.1%}) | "
            f"stream={int(totals.get('accepted_by_stream',0))} ({totals.get('accepted_by_stream',0)/pages_total:.1%}) | "
            f"no_table={int(totals.get('no_table',0))} ({totals.get('no_table',0)/pages_total:.1%})"
        )
        print("  failures_by_reason:")
        for rk in (
            "no_tables_returned",
            "accuracy_below_threshold",
            "whitespace_above_max",
            "empty_invalid_table",
            "exception",
            "all_passes_failed",
        ):
            rv = int(totals.get(rk, 0))
            print(f"    - {rk}: {rv} ({rv/pages_total:.1%})")
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "check2_distribution.csv", index=False)
    return df


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def load_eval_items(eval_path: Path) -> list[dict[str, Any]]:
    obj = json.loads(eval_path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict) and isinstance(obj.get("queries"), list):
        return [x for x in obj["queries"] if isinstance(x, dict)]
    return []


def difficulty_hit_at_1(summary_csv: Path, eval_set: Path) -> dict[str, float]:
    df = pd.read_csv(summary_csv)
    df = df[df["k"] == 1].copy()
    items = pd.DataFrame(load_eval_items(eval_set))
    if items.empty or "query_id" not in items.columns:
        return {}
    items["query_id"] = items["query_id"].astype(str)
    if "difficulty" not in items.columns:
        items["difficulty"] = "MISSING"
    m = df.merge(items[["query_id", "difficulty"]], on="query_id", how="left")
    m["difficulty"] = m["difficulty"].fillna("MISSING").astype(str)
    m["hit"] = (m["page_recall_at_k"] > 0).astype(int)
    out = m.groupby("difficulty")["hit"].mean().to_dict()
    return {str(k): float(v) for k, v in out.items()}


def summary_metrics(summary_csv: Path, eval_set: Path) -> dict[str, Any]:
    df = pd.read_csv(summary_csv)
    if df.empty:
        return {"query_count": 0, "hit1": None, "mrr": None, "mrr_k": None, "diff": {}}

    k1 = df[df["k"] == 1].copy()
    hit1 = float((k1["page_recall_at_k"] > 0).mean()) if not k1.empty else None
    query_count = int(k1["query_id"].nunique()) if "query_id" in k1.columns else 0

    k_values = sorted(int(x) for x in df["k"].dropna().unique())
    mrr_k = int(max(k_values)) if k_values else 1
    km = df[df["k"] == mrr_k].copy()
    mrr = float(km["page_mrr_at_k"].mean()) if not km.empty else None

    diff_rates = difficulty_hit_at_1(summary_csv, eval_set)
    diff_counts: dict[str, int] = {}
    items = pd.DataFrame(load_eval_items(eval_set))
    if not items.empty and "difficulty" in items.columns:
        for d in ["LEX", "MOD", "STR"]:
            diff_counts[d] = int((items["difficulty"].astype(str) == d).sum())

    return {
        "query_count": query_count,
        "hit1": hit1,
        "mrr": mrr,
        "mrr_k": mrr_k,
        "diff": diff_rates,
        "diff_counts": diff_counts,
    }


def _pick_mrr_k(metrics_obj: dict[str, Any]) -> str:
    by_k = metrics_obj.get("metrics_by_k", {})
    if not isinstance(by_k, dict) or not by_k:
        return "1"
    ks = sorted(int(k) for k in by_k.keys() if str(k).isdigit())
    if not ks:
        return "1"
    return str(max(ks))


def load_baseline_doc_metrics(doc_id: str, baseline_metrics: Optional[Path]) -> tuple[Optional[dict[str, Any]], Optional[Path]]:
    # 1) explicit baseline path
    if baseline_metrics:
        if baseline_metrics.is_file():
            obj = json.loads(baseline_metrics.read_text(encoding="utf-8"))
            if isinstance(obj, dict) and "metrics_by_k" in obj:
                return obj, None
            if isinstance(obj, dict) and "by_doc" in obj and isinstance(obj["by_doc"], dict):
                d = obj["by_doc"].get(doc_id)
                if isinstance(d, dict):
                    return d, None
        elif baseline_metrics.is_dir():
            p = baseline_metrics / doc_id / "retrieval_metrics_hybrid.json"
            if p.exists():
                return json.loads(p.read_text(encoding="utf-8")), baseline_metrics / doc_id / "retrieval_summary_hybrid.csv"

    # 2) fallback current data_processed baseline
    p = REPO_ROOT / "data_processed" / doc_id / "retrieval_metrics_hybrid.json"
    s = REPO_ROOT / "data_processed" / doc_id / "retrieval_summary_hybrid.csv"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8")), (s if s.exists() else None)
    return None, None


def check3_retrieval_impact(
    pdf_paths: list[Path],
    args: argparse.Namespace,
    output_dir: Path,
) -> pd.DataFrame:
    print_section("CHECK 3: Retrieval Impact Evaluation")
    rows = []
    baseline_path = Path(args.baseline_metrics).resolve() if args.baseline_metrics else None
    eval_set_override = Path(args.eval_set).resolve() if args.eval_set else None
    agg = {
        "base_hit1_num": 0.0,
        "base_hit1_den": 0,
        "new_hit1_num": 0.0,
        "new_hit1_den": 0,
        "base_mrr_num": 0.0,
        "base_mrr_den": 0,
        "new_mrr_num": 0.0,
        "new_mrr_den": 0,
        "base_diff_num": {"LEX": 0.0, "MOD": 0.0, "STR": 0.0},
        "base_diff_den": {"LEX": 0, "MOD": 0, "STR": 0},
        "new_diff_num": {"LEX": 0.0, "MOD": 0.0, "STR": 0.0},
        "new_diff_den": {"LEX": 0, "MOD": 0, "STR": 0},
    }

    for pdf in pdf_paths:
        doc_id = pdf.stem
        run_root = output_dir / "rebuild"
        doc_out = run_root / doc_id
        run_root.mkdir(parents=True, exist_ok=True)

        # Baseline before rebuild.
        base_metrics, base_summary_csv = load_baseline_doc_metrics(doc_id, baseline_path)

        # Rebuild + index + eval using upgraded extraction logic.
        run_cmd(
            [
                args.python_bin,
                "scripts/preprocess_hybrid.py",
                "--pdf-path",
                str(pdf),
                "--out-root",
                str(run_root),
            ],
            cwd=REPO_ROOT,
        )
        run_cmd(
            [
                args.python_bin,
                "scripts/build_index.py",
                "--data-dir",
                str(run_root),
                "--model",
                str(args.model),
            ],
            cwd=REPO_ROOT,
        )

        # Ensure eval set exists.
        target_eval = doc_out / "eval_set.json"
        if eval_set_override and eval_set_override.exists():
            shutil.copy2(eval_set_override, target_eval)
        elif not target_eval.exists():
            source_eval = REPO_ROOT / "data_processed" / doc_id / "eval_set.json"
            if source_eval.exists():
                shutil.copy2(source_eval, target_eval)

        run_cmd(
            [
                args.python_bin,
                "scripts/retrieval_eval_hybrid.py",
                "--data-dir",
                str(doc_out),
                "--model",
                str(args.model),
                "--k-list",
                str(args.k_list),
            ],
            cwd=REPO_ROOT,
        )

        new_metrics_path = doc_out / "retrieval_metrics_hybrid.json"
        new_summary_csv = doc_out / "retrieval_summary_hybrid.csv"
        new_metrics = json.loads(new_metrics_path.read_text(encoding="utf-8"))

        # New metrics
        new_hit1 = float(new_metrics["metrics_by_k"]["1"]["page_hit_rate_at_k"])
        mrr_k = _pick_mrr_k(new_metrics)
        new_mrr = float(new_metrics["metrics_by_k"][mrr_k]["mean_page_mrr_at_k"])
        new_diff = difficulty_hit_at_1(new_summary_csv, target_eval) if target_eval.exists() else {}
        new_sm = summary_metrics(new_summary_csv, target_eval) if target_eval.exists() else None

        # Baseline metrics
        base_hit1 = None
        base_mrr = None
        base_diff: dict[str, float] = {}
        base_sm: Optional[dict[str, Any]] = None
        if isinstance(base_metrics, dict) and "metrics_by_k" in base_metrics:
            base_hit1 = float(base_metrics["metrics_by_k"]["1"]["page_hit_rate_at_k"])
            base_mrr_k = _pick_mrr_k(base_metrics)
            base_mrr = float(base_metrics["metrics_by_k"][base_mrr_k]["mean_page_mrr_at_k"])
            if base_summary_csv and base_summary_csv.exists():
                eval_for_base = REPO_ROOT / "data_processed" / doc_id / "eval_set.json"
                if eval_for_base.exists():
                    base_diff = difficulty_hit_at_1(base_summary_csv, eval_for_base)
                    base_sm = summary_metrics(base_summary_csv, eval_for_base)

        # Print side-by-side
        print(f"\nDOC: {doc_id}")
        print(f"  Hit@1: baseline={base_hit1} | new={new_hit1} | delta={(new_hit1 - base_hit1) if base_hit1 is not None else 'N/A'}")
        print(f"  MRR@{mrr_k}: baseline={base_mrr} | new={new_mrr} | delta={(new_mrr - base_mrr) if base_mrr is not None else 'N/A'}")
        for d in ["LEX", "MOD", "STR"]:
            b = base_diff.get(d)
            n = new_diff.get(d)
            delta = (n - b) if (b is not None and n is not None) else None
            print(f"  Hit@1 {d}: baseline={b} | new={n} | delta={delta}")

        if new_sm and new_sm.get("query_count"):
            qn = int(new_sm["query_count"])
            if new_sm.get("hit1") is not None:
                agg["new_hit1_num"] += float(new_sm["hit1"]) * qn
                agg["new_hit1_den"] += qn
            if new_sm.get("mrr") is not None:
                agg["new_mrr_num"] += float(new_sm["mrr"]) * qn
                agg["new_mrr_den"] += qn
            for d in ["LEX", "MOD", "STR"]:
                cnt = int((new_sm.get("diff_counts") or {}).get(d, 0))
                val = (new_sm.get("diff") or {}).get(d)
                if cnt > 0 and val is not None:
                    agg["new_diff_num"][d] += float(val) * cnt
                    agg["new_diff_den"][d] += cnt

        if base_sm and base_sm.get("query_count"):
            qn = int(base_sm["query_count"])
            if base_sm.get("hit1") is not None:
                agg["base_hit1_num"] += float(base_sm["hit1"]) * qn
                agg["base_hit1_den"] += qn
            if base_sm.get("mrr") is not None:
                agg["base_mrr_num"] += float(base_sm["mrr"]) * qn
                agg["base_mrr_den"] += qn
            for d in ["LEX", "MOD", "STR"]:
                cnt = int((base_sm.get("diff_counts") or {}).get(d, 0))
                val = (base_sm.get("diff") or {}).get(d)
                if cnt > 0 and val is not None:
                    agg["base_diff_num"][d] += float(val) * cnt
                    agg["base_diff_den"][d] += cnt

        rows.append(
            {
                "doc_id": doc_id,
                "baseline_hit1": base_hit1,
                "new_hit1": new_hit1,
                "delta_hit1": (new_hit1 - base_hit1) if base_hit1 is not None else None,
                f"baseline_mrr@{mrr_k}": base_mrr,
                f"new_mrr@{mrr_k}": new_mrr,
                f"delta_mrr@{mrr_k}": (new_mrr - base_mrr) if base_mrr is not None else None,
                "baseline_hit1_lex": base_diff.get("LEX"),
                "new_hit1_lex": new_diff.get("LEX"),
                "delta_hit1_lex": (new_diff.get("LEX") - base_diff.get("LEX")) if ("LEX" in new_diff and "LEX" in base_diff) else None,
                "baseline_hit1_mod": base_diff.get("MOD"),
                "new_hit1_mod": new_diff.get("MOD"),
                "delta_hit1_mod": (new_diff.get("MOD") - base_diff.get("MOD")) if ("MOD" in new_diff and "MOD" in base_diff) else None,
                "baseline_hit1_str": base_diff.get("STR"),
                "new_hit1_str": new_diff.get("STR"),
                "delta_hit1_str": (new_diff.get("STR") - base_diff.get("STR")) if ("STR" in new_diff and "STR" in base_diff) else None,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "check3_retrieval_impact.csv", index=False)

    base_hit1_all = (agg["base_hit1_num"] / agg["base_hit1_den"]) if agg["base_hit1_den"] else None
    new_hit1_all = (agg["new_hit1_num"] / agg["new_hit1_den"]) if agg["new_hit1_den"] else None
    base_mrr_all = (agg["base_mrr_num"] / agg["base_mrr_den"]) if agg["base_mrr_den"] else None
    new_mrr_all = (agg["new_mrr_num"] / agg["new_mrr_den"]) if agg["new_mrr_den"] else None
    print("\nOVERALL (query-weighted across processed docs)")
    print(
        f"  Hit@1: baseline={base_hit1_all} | new={new_hit1_all} | "
        f"delta={(new_hit1_all - base_hit1_all) if (base_hit1_all is not None and new_hit1_all is not None) else 'N/A'}"
    )
    print(
        f"  MRR: baseline={base_mrr_all} | new={new_mrr_all} | "
        f"delta={(new_mrr_all - base_mrr_all) if (base_mrr_all is not None and new_mrr_all is not None) else 'N/A'}"
    )
    for d in ["LEX", "MOD", "STR"]:
        b = (agg["base_diff_num"][d] / agg["base_diff_den"][d]) if agg["base_diff_den"][d] else None
        n = (agg["new_diff_num"][d] / agg["new_diff_den"][d]) if agg["new_diff_den"][d] else None
        delta = (n - b) if (b is not None and n is not None) else None
        print(f"  Hit@1 {d}: baseline={b} | new={n} | delta={delta}")
    return df


def main() -> None:
    args = parse_args()
    pdf_paths = resolve_pdfs(args.pdfs)
    if not pdf_paths:
        raise FileNotFoundError("No PDFs resolved from --pdfs inputs.")

    out_dir = (REPO_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_pages = parse_pages_arg(args.sample_pages)
    checks = parse_checks_arg(args.checks)
    if "1" in checks:
        check1_structural_sanity(pdf_paths[0], sample_pages)
    if "2" in checks:
        check2_acceptance_distribution(pdf_paths, out_dir)
    if "3" in checks:
        check3_retrieval_impact(pdf_paths, args, out_dir)

    print("\nDone. Reports written to:", out_dir)


if __name__ == "__main__":
    main()
