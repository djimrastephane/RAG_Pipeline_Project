from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from datasets import Dataset
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from ragas.run_config import RunConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run RAGAS on exported JSONL dataset.")
    p.add_argument("--input-jsonl", default="results/ragas/ragas_input_250q.jsonl")
    p.add_argument("--out-dir", default="results/ragas")
    p.add_argument("--llm-model", default="qwen2.5:7b-instruct")
    p.add_argument("--embedding-model", default="models/all-MiniLM-L6-v2")
    p.add_argument("--sample-n", type=int, default=0, help="Optional row cap for pilot runs (0=all).")
    p.add_argument("--timeout", type=float, default=90.0, help="Per-call timeout seconds for evaluator LLM.")
    p.add_argument("--max-retries", type=int, default=1, help="Retries for failed evaluator calls.")
    p.add_argument("--batch-size", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_json(in_path, lines=True)
    if args.sample_n and int(args.sample_n) > 0:
        df = df.head(int(args.sample_n)).copy()

    ds = Dataset.from_pandas(df, preserve_index=False)
    llm = LangchainLLMWrapper(ChatOllama(model=str(args.llm_model), temperature=0.0))
    embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=str(args.embedding_model)))
    run_config = RunConfig(timeout=float(args.timeout), max_retries=int(args.max_retries))

    result = evaluate(
        dataset=ds,
        metrics=[answer_relevancy, faithfulness, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
        column_map={
            "user_input": "question",
            "response": "answer",
            "retrieved_contexts": "contexts",
            "reference": "ground_truth",
        },
        run_config=run_config,
        batch_size=int(args.batch_size),
        show_progress=True,
        raise_exceptions=False,
    )

    res_df = result.to_pandas()
    per_query_csv = out_dir / "ragas_per_query.csv"
    res_df.to_csv(per_query_csv, index=False)

    metric_cols = [c for c in ("answer_relevancy", "faithfulness", "context_precision", "context_recall") if c in res_df.columns]
    summary = {
        "input_rows": int(len(df)),
        "scored_rows": int(len(res_df)),
        "llm_model": str(args.llm_model),
        "embedding_model": str(args.embedding_model),
        "timeout": float(args.timeout),
        "max_retries": int(args.max_retries),
        "batch_size": int(args.batch_size),
        "metrics_mean": {
            c: (None if res_df[c].dropna().empty else float(res_df[c].dropna().mean()))
            for c in metric_cols
        },
        "metrics_non_null": {
            c: int(res_df[c].notna().sum())
            for c in metric_cols
        },
    }
    summary_json = out_dir / "ragas_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Wrote {per_query_csv}")
    print(f"Wrote {summary_json}")


if __name__ == "__main__":
    main()
