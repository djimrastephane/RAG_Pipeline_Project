from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
scripts_path = repo_root / "scripts"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
if scripts_path.exists() and str(scripts_path) not in sys.path:
    sys.path.insert(0, str(scripts_path))

from ablate_generation_prompts import _parse_json_object, _run_prompt_arm_detailed
from rag_pdf.services.local_llm_service import LocalLLMService
from rag_pdf.services.search_service import SearchService


ARM_DISPLAY_ORDER = [
    "baseline",
    "grounded_reasoning",
    "quote_then_answer",
    "constrained_extraction",
]

DEFAULT_LABELS = [
    "quote_supports_gold_but_answer_misread",
    "quote_supports_gold_but_answer_truncated",
    "quote_supports_gold_but_answer_overgeneralized",
    "quote_partial_only",
    "quote_irrelevant",
    "answer_formatting_only",
    "retrieval_context_insufficient",
    "judge_uncertain",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Use a judge LLM to diagnose quoted-evidence answer failures.")
    p.add_argument("--ablation-dir", required=True, help="Directory produced by ablate_generation_prompts.py")
    p.add_argument("--data-root", default="data_processed")
    p.add_argument("--model-path", default="models/all-MiniLM-L6-v2")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--max-context-chunks", type=int, default=5)
    p.add_argument("--max-context-chars", type=int, default=9000)
    p.add_argument("--max-chunk-chars", type=int, default=2200)
    p.add_argument("--gen-timeout-seconds", type=float, default=20.0)
    p.add_argument("--limit", type=int, default=0, help="Optional cap on judged failures (0 = all).")
    p.add_argument(
        "--arms",
        default="quote_then_answer",
        help="Comma-separated ablation arms to inspect. Defaults to quote_then_answer.",
    )
    p.add_argument("--judge-model", default="", help="Optional override model name for the judge LLM.")
    p.add_argument("--out-dir", default="", help="Optional output dir; defaults to <ablation-dir>/judge_quoted_failures")
    return p.parse_args()


def _judge_prompt(
    *,
    arm: str,
    question: str,
    expected_answer: str,
    model_answer: str,
    evidence_quote: str,
    context_text: str,
    allowed_labels: list[str],
) -> str:
    labels_text = "\n".join(f"- {label}" for label in allowed_labels)
    return (
        "You are a diagnostic judge for a retrieval-grounded QA pipeline.\n"
        "Your task is to explain why the quoted evidence did or did not lead to a correct answer.\n"
        "Use only the inputs below.\n"
        "Return JSON only, with no markdown and no extra text.\n\n"
        "Allowed judge_label values:\n"
        f"{labels_text}\n\n"
        "Return exactly this JSON schema:\n"
        "{"
        "\"judge_label\":\"...\","
        "\"judge_rationale_short\":\"...\","
        "\"quote_contains_gold\":true,"
        "\"answer_mismatch_type\":\"...\""
        "}\n\n"
        "Definitions:\n"
        "- quote_contains_gold = true only if the quoted text itself clearly contains the gold answer or enough exact support for it.\n"
        "- answer_mismatch_type should be a short phrase such as misread_number, omitted_qualifier, wrong_span, incomplete_list, formatting_only, unsupported, or uncertain.\n\n"
        f"ARM:\n{arm}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"EXPECTED ANSWER:\n{expected_answer}\n\n"
        f"MODEL ANSWER:\n{model_answer}\n\n"
        f"EVIDENCE QUOTE:\n{evidence_quote}\n\n"
        f"RETRIEVED CONTEXT:\n{context_text}\n\n"
        "OUTPUT JSON:"
    )


def _write_charts(detail_df: pd.DataFrame, summary_df: pd.DataFrame, out_dir: Path) -> list[str]:
    chart_dir = out_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[str] = []
    n_queries = int(detail_df["query_id"].nunique()) if "query_id" in detail_df.columns and len(detail_df) else 0
    caption = f"Evaluation performed on n = {n_queries} queries."

    if len(summary_df):
        pivot = summary_df.pivot(index="judge_label", columns="arm", values="count").fillna(0.0)
        ordered_cols = [arm for arm in ARM_DISPLAY_ORDER if arm in pivot.columns] + [
            arm for arm in pivot.columns if arm not in ARM_DISPLAY_ORDER
        ]
        pivot = pivot.reindex(columns=ordered_cols)
        label_totals = pivot.sum(axis=1).sort_values(ascending=False)
        pivot = pivot.reindex(label_totals.index)
        arm_totals = pivot.sum(axis=0).replace(0.0, 1.0)
        fig, ax = plt.subplots(figsize=(10.0, max(4.5, 0.45 * len(pivot))), constrained_layout=True)
        arms = list(pivot.columns)
        y = np.arange(len(pivot.index))
        bar_h = 0.72 / max(1, len(arms))
        colors = {
            "baseline": "#355070",
            "grounded_reasoning": "#457b9d",
            "quote_then_answer": "#2a9d8f",
            "constrained_extraction": "#b56576",
        }
        for i, arm in enumerate(arms):
            vals = pivot[arm].to_numpy(dtype=float)
            ax.barh(
                y + (i - (len(arms) - 1) / 2.0) * bar_h,
                vals,
                height=bar_h,
                label=str(arm),
                color=colors.get(str(arm), "#8d99ae"),
                edgecolor="#ffffff",
                linewidth=0.6,
            )
            for yy, vv in zip(y + (i - (len(arms) - 1) / 2.0) * bar_h, vals):
                if vv <= 0:
                    continue
                pct = 100.0 * float(vv) / float(arm_totals.get(arm, 1.0))
                ax.text(vv + 0.1, yy, f"{pct:.0f}%", va="center", ha="left", fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels([str(v) for v in pivot.index])
        ax.set_xlabel("Count")
        ax.set_title(f"Judge Labels for Quoted-Evidence Failures\n{caption}")
        ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc")
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        p1 = chart_dir / "judge_label_distribution.png"
        fig.savefig(p1, dpi=180)
        plt.close(fig)
        out_paths.append(str(p1))

    valid_detail = detail_df[detail_df["judge_label"].astype(str) != ""].copy()
    if len(valid_detail):
        ctab = (
            valid_detail.groupby(["quote_contains_gold", "answer_mismatch_type"], dropna=False)
            .size()
            .rename("count")
            .reset_index()
        )
        pivot2 = ctab.pivot(index="answer_mismatch_type", columns="quote_contains_gold", values="count").fillna(0.0)
        pivot2 = pivot2.sort_values(list(pivot2.columns), ascending=False)
        fig, ax = plt.subplots(figsize=(9.0, max(4.0, 0.45 * len(pivot2))), constrained_layout=True)
        y = np.arange(len(pivot2.index))
        false_vals = pivot2[False].to_numpy(dtype=float) if False in pivot2.columns else np.zeros(len(pivot2))
        true_vals = pivot2[True].to_numpy(dtype=float) if True in pivot2.columns else np.zeros(len(pivot2))
        ax.barh(y, false_vals, color="#f4a261", label="quote_contains_gold = False")
        ax.barh(y, true_vals, left=false_vals, color="#2a9d8f", label="quote_contains_gold = True")
        totals = false_vals + true_vals
        for idx, (fval, tval, total) in enumerate(zip(false_vals, true_vals, totals)):
            if total <= 0:
                continue
            if fval > 0:
                ax.text(fval / 2.0, idx, f"{(100.0 * fval / total):.0f}%", va="center", ha="center", fontsize=8)
            if tval > 0:
                ax.text(fval + (tval / 2.0), idx, f"{(100.0 * tval / total):.0f}%", va="center", ha="center", fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels([str(v) for v in pivot2.index])
        ax.set_xlabel("Count")
        ax.set_title(f"Judge Mismatch Types by Quote Support\n{caption}")
        ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc")
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        p2 = chart_dir / "judge_mismatch_by_quote_support.png"
        fig.savefig(p2, dpi=180)
        plt.close(fig)
        out_paths.append(str(p2))

    return out_paths


def main() -> None:
    args = parse_args()
    ablation_dir = Path(args.ablation_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (ablation_dir / "judge_quoted_failures")
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_path = ablation_dir / "generation_prompt_ablation_detail.csv"
    sample_path = ablation_dir / "sampled_queries.csv"
    if not detail_path.exists():
        raise FileNotFoundError(f"Missing detail CSV: {detail_path}")
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing sampled queries CSV: {sample_path}")

    detail_df = pd.read_csv(detail_path)
    sample_df = pd.read_csv(sample_path)
    arms = [a.strip() for a in str(args.arms).split(",") if a.strip()]

    merged = detail_df.merge(
        sample_df,
        on=["doc_id", "query_id", "difficulty", "answer_type"],
        how="left",
        suffixes=("", "_sample"),
    )
    filt = merged[
        merged["arm"].astype(str).isin(arms)
        & merged["answer_correct"].fillna(False).eq(False)
        & merged["generation_status"].astype(str).eq("ok")
    ].copy()
    if int(args.limit) > 0:
        filt = filt.head(int(args.limit)).copy()

    svc = SearchService(repo_root=Path(".").resolve(), model_path=Path(args.model_path))
    svc.gen_timeout_seconds = float(args.gen_timeout_seconds)
    judge_llm = LocalLLMService()
    if str(args.judge_model).strip():
        judge_llm.model = str(args.judge_model).strip()

    rows: list[dict[str, Any]] = []
    for i, rec in filt.iterrows():
        data_dir = Path(args.data_root) / str(rec["doc_id"])
        question = str(rec.get("question") or "")
        retrieval_out = svc.search(
            data_dir=data_dir,
            question=question,
            k=int(args.k),
            query_id=str(rec.get("query_id") or "") or None,
            include_generated_answer=False,
        )
        rerun = _run_prompt_arm_detailed(
            svc=svc,
            arm=str(rec["arm"]),
            question=question,
            results=list(retrieval_out.get("results") or []),
            max_context_chunks=int(args.max_context_chunks),
            max_context_chars=int(args.max_context_chars),
            max_chunk_chars=int(args.max_chunk_chars),
            timeout_seconds=float(args.gen_timeout_seconds),
        )
        evidence_quote = str(rerun.get("evidence_quote") or "").strip()
        model_answer = str(rerun.get("answer") or "").strip()
        context_text = str(rerun.get("context_text") or "").strip()

        if not evidence_quote:
            rows.append(
                {
                    "arm": rec["arm"],
                    "doc_id": rec["doc_id"],
                    "query_id": rec["query_id"],
                    "difficulty": rec["difficulty"],
                    "judge_status": "skipped_no_quote",
                    "judge_label": "",
                    "judge_rationale_short": "",
                    "quote_contains_gold": None,
                    "answer_mismatch_type": "",
                    "expected_answer": rec.get("expected_answer"),
                    "model_answer": model_answer,
                    "evidence_quote": evidence_quote,
                }
            )
            continue

        prompt = _judge_prompt(
            arm=str(rec["arm"]),
            question=question,
            expected_answer=str(rec.get("expected_answer") or ""),
            model_answer=model_answer,
            evidence_quote=evidence_quote,
            context_text=context_text,
            allowed_labels=DEFAULT_LABELS,
        )
        judge_out = judge_llm.generate(prompt, timeout_seconds=float(args.gen_timeout_seconds))
        parsed = _parse_json_object(judge_out.answer or "")
        rows.append(
            {
                "arm": rec["arm"],
                "doc_id": rec["doc_id"],
                "query_id": rec["query_id"],
                "difficulty": rec["difficulty"],
                "judge_status": str(judge_out.status),
                "judge_label": str((parsed or {}).get("judge_label") or ""),
                "judge_rationale_short": str((parsed or {}).get("judge_rationale_short") or ""),
                "quote_contains_gold": (parsed or {}).get("quote_contains_gold"),
                "answer_mismatch_type": str((parsed or {}).get("answer_mismatch_type") or ""),
                "expected_answer": rec.get("expected_answer"),
                "model_answer": model_answer,
                "evidence_quote": evidence_quote,
                "judge_raw_output": str(judge_out.answer or ""),
            }
        )
        if (len(rows) % 10) == 0:
            print(f"Judged {len(rows)}/{len(filt)} failures")

    detail_out = pd.DataFrame(rows)
    detail_out.to_csv(out_dir / "judge_quoted_failures_detail.csv", index=False)

    summary = (
        detail_out[detail_out["judge_label"].astype(str) != ""]
        .groupby(["arm", "judge_label"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    if len(summary):
        ordered_arms = [arm for arm in ARM_DISPLAY_ORDER if arm in set(summary["arm"].astype(str))]
        summary["arm"] = pd.Categorical(summary["arm"], categories=ordered_arms, ordered=True)
        summary["pct_within_arm"] = summary.groupby("arm")["count"].transform(lambda s: s / max(1, s.sum()))
        summary = summary.sort_values(["arm", "count"], ascending=[True, False]).reset_index(drop=True)
    summary.to_csv(out_dir / "judge_quoted_failures_summary.csv", index=False)
    chart_paths = _write_charts(detail_df=detail_out, summary_df=summary, out_dir=out_dir)

    report = {
        "ablation_dir": str(ablation_dir),
        "n_candidate_failures": int(len(filt)),
        "n_judged_rows": int(len(detail_out)),
        "arms": arms,
        "judge_model": str(judge_llm.model),
        "labels": DEFAULT_LABELS,
        "chart_paths": chart_paths,
    }
    (out_dir / "judge_quoted_failures_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = [
        "# Judge Analysis: Quoted-Evidence Failures",
        "",
        f"- Source ablation dir: `{ablation_dir}`",
        f"- Arms inspected: `{', '.join(arms)}`",
        f"- Candidate failures: `{len(filt)}`",
        f"- Judge model: `{judge_llm.model}`",
        f"- Evaluation performed on `n = {int(detail_out['query_id'].nunique()) if len(detail_out) else 0}` queries.",
        "",
        "## Label Summary",
        "",
        (summary.to_markdown(index=False) if len(summary) else "_No judged rows with parsed labels._"),
        "",
        "## Charts",
        "",
        f"- Evaluation performed on n = {int(detail_out['query_id'].nunique()) if len(detail_out) else 0} queries.",
        "",
        *([f"- `{p}`" for p in chart_paths] if chart_paths else ["- No charts generated."]),
    ]
    (out_dir / "judge_quoted_failures_summary.md").write_text("\n".join(md), encoding="utf-8")

    print("Wrote:", out_dir / "judge_quoted_failures_detail.csv")
    print("Wrote:", out_dir / "judge_quoted_failures_summary.csv")
    print("Wrote:", out_dir / "judge_quoted_failures_summary.md")
    print("Wrote:", out_dir / "judge_quoted_failures_report.json")
    for chart_path in chart_paths:
        print("Wrote:", chart_path)


if __name__ == "__main__":
    main()
