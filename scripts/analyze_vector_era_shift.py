#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import _matplotlib_env
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


@dataclass
class EraData:
    name: str
    doc_ids: list[str]
    vectors: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare embedding distributions across two eras and generate charts."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data_processed"),
        help="Root directory containing per-doc embeddings.npy files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results"),
        help="Directory where CSV and chart outputs will be saved.",
    )
    parser.add_argument(
        "--old-docs",
        nargs="+",
        default=[
            "Grampian-2004-2005",
            "Grampian-2005-2006",
            "Grampian-2010-2011",
            "Grampian-2014-2015",
            "Grampian-2016-2017",
        ],
        help="Document IDs for the older era.",
    )
    parser.add_argument(
        "--new-docs",
        nargs="+",
        default=[
            "Grampian-2022-2023",
            "Grampian-2023-2024",
            "Grampian-2024-2025",
        ],
        help="Document IDs for the newer era.",
    )
    parser.add_argument(
        "--old-label",
        default="old_2004_2017",
        help="Label to use in output filenames/CSV for old era.",
    )
    parser.add_argument(
        "--new-label",
        default="new_2022_2025",
        help="Label to use in output filenames/CSV for new era.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20000,
        help="Random sample size for cosine pair similarity estimates.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def load_era(name: str, doc_ids: list[str], data_root: Path) -> EraData:
    vectors = []
    for doc_id in doc_ids:
        emb_path = data_root / doc_id / "embeddings.npy"
        if not emb_path.exists():
            raise FileNotFoundError(f"Missing embeddings file: {emb_path}")
        arr = np.load(emb_path)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D embeddings array in {emb_path}, got {arr.shape}")
        vectors.append(arr.astype(np.float32, copy=False))
    return EraData(name=name, doc_ids=doc_ids, vectors=np.vstack(vectors))


def sample_cosine_pairs(vectors: np.ndarray, sample_size: int, rng: np.random.Generator) -> np.ndarray:
    n = vectors.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 vectors to sample cosine pairs.")
    i = rng.integers(0, n, size=sample_size)
    j = rng.integers(0, n, size=sample_size)
    same = i == j
    while np.any(same):
        j[same] = rng.integers(0, n, size=np.sum(same))
        same = i == j
    return np.sum(vectors[i] * vectors[j], axis=1)


def compute_metrics(era: EraData, cos_pairs: np.ndarray) -> dict[str, float | int | str]:
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(era.vectors)
    radius = np.sqrt(np.sum(coords**2, axis=1))

    return {
        "era": era.name,
        "num_vectors": int(era.vectors.shape[0]),
        "embedding_dim": int(era.vectors.shape[1]),
        "l2_mean": float(np.linalg.norm(era.vectors, axis=1).mean()),
        "l2_std": float(np.linalg.norm(era.vectors, axis=1).std()),
        "cos_pair_mean": float(cos_pairs.mean()),
        "cos_pair_std": float(cos_pairs.std()),
        "cos_pair_p5": float(np.percentile(cos_pairs, 5)),
        "cos_pair_p50": float(np.percentile(cos_pairs, 50)),
        "cos_pair_p95": float(np.percentile(cos_pairs, 95)),
        "pca_var_ratio_pc1": float(pca.explained_variance_ratio_[0]),
        "pca_var_ratio_pc2": float(pca.explained_variance_ratio_[1]),
        "pca_compactness_mean_radius": float(radius.mean()),
        "pca_compactness_p95_radius": float(np.percentile(radius, 95)),
    }


def plot_pca_overlay(old_era: EraData, new_era: EraData, out_path: Path, seed: int) -> None:
    stacked = np.vstack([old_era.vectors, new_era.vectors])
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(stacked)
    old_n = old_era.vectors.shape[0]
    old_xy = coords[:old_n]
    new_xy = coords[old_n:]

    rng = np.random.default_rng(seed)
    old_take = rng.choice(old_n, size=min(5000, old_n), replace=False)
    new_take = rng.choice(new_era.vectors.shape[0], size=min(5000, new_era.vectors.shape[0]), replace=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(old_xy[old_take, 0], old_xy[old_take, 1], s=6, alpha=0.20, label=old_era.name)
    ax.scatter(new_xy[new_take, 0], new_xy[new_take, 1], s=6, alpha=0.20, label=new_era.name)
    ax.set_title("PCA of Embeddings: Old vs New Era")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_similarity_hist_overlay(
    old_cos: np.ndarray, new_cos: np.ndarray, old_name: str, new_name: str, out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    bins = np.linspace(-0.1, 1.0, 60)
    ax.hist(old_cos, bins=bins, alpha=0.55, density=True, label=old_name)
    ax.hist(new_cos, bins=bins, alpha=0.55, density=True, label=new_name)
    ax.set_title("Cosine Similarity Distribution: Old vs New Era")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_panel(
    old_era: EraData,
    new_era: EraData,
    old_cos: np.ndarray,
    new_cos: np.ndarray,
    out_path: Path,
    seed: int,
) -> None:
    stacked = np.vstack([old_era.vectors, new_era.vectors])
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(stacked)
    old_n = old_era.vectors.shape[0]
    old_xy = coords[:old_n]
    new_xy = coords[old_n:]

    rng = np.random.default_rng(seed)
    old_take = rng.choice(old_n, size=min(5000, old_n), replace=False)
    new_take = rng.choice(new_era.vectors.shape[0], size=min(5000, new_era.vectors.shape[0]), replace=False)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
    axs[0].scatter(old_xy[old_take, 0], old_xy[old_take, 1], s=6, alpha=0.20, label=old_era.name)
    axs[0].scatter(new_xy[new_take, 0], new_xy[new_take, 1], s=6, alpha=0.20, label=new_era.name)
    axs[0].set_title("PCA Overlay")
    axs[0].set_xlabel("PC1")
    axs[0].set_ylabel("PC2")
    axs[0].legend()

    bins = np.linspace(-0.1, 1.0, 60)
    axs[1].hist(old_cos, bins=bins, alpha=0.55, density=True, label=old_era.name)
    axs[1].hist(new_cos, bins=bins, alpha=0.55, density=True, label=new_era.name)
    axs[1].set_title("Cosine Similarity Histogram")
    axs[1].set_xlabel("Cosine similarity")
    axs[1].set_ylabel("Density")
    axs[1].legend()

    fig.suptitle("Embedding Distribution Comparison by Era", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    old_era = load_era(args.old_label, args.old_docs, args.data_root)
    new_era = load_era(args.new_label, args.new_docs, args.data_root)

    old_cos = sample_cosine_pairs(old_era.vectors, args.sample_size, rng)
    new_cos = sample_cosine_pairs(new_era.vectors, args.sample_size, rng)

    old_metrics = compute_metrics(old_era, old_cos)
    new_metrics = compute_metrics(new_era, new_cos)
    summary_df = pd.DataFrame([old_metrics, new_metrics])

    compare_row = {
        "old_era": args.old_label,
        "new_era": args.new_label,
        "num_vectors_old": old_metrics["num_vectors"],
        "num_vectors_new": new_metrics["num_vectors"],
        "num_vectors_delta": int(new_metrics["num_vectors"]) - int(old_metrics["num_vectors"]),
        "cos_pair_mean_old": old_metrics["cos_pair_mean"],
        "cos_pair_mean_new": new_metrics["cos_pair_mean"],
        "cos_pair_mean_delta_new_minus_old": float(new_metrics["cos_pair_mean"]) - float(old_metrics["cos_pair_mean"]),
        "cos_pair_std_old": old_metrics["cos_pair_std"],
        "cos_pair_std_new": new_metrics["cos_pair_std"],
        "pca_var_ratio_pc1_old": old_metrics["pca_var_ratio_pc1"],
        "pca_var_ratio_pc1_new": new_metrics["pca_var_ratio_pc1"],
        "pca_var_ratio_pc1_delta_new_minus_old": float(new_metrics["pca_var_ratio_pc1"]) - float(old_metrics["pca_var_ratio_pc1"]),
        "pca_compactness_mean_radius_old": old_metrics["pca_compactness_mean_radius"],
        "pca_compactness_mean_radius_new": new_metrics["pca_compactness_mean_radius"],
        "pca_compactness_mean_radius_delta_new_minus_old": float(new_metrics["pca_compactness_mean_radius"]) - float(old_metrics["pca_compactness_mean_radius"]),
        "old_docs": "|".join(args.old_docs),
        "new_docs": "|".join(args.new_docs),
    }
    compare_df = pd.DataFrame([compare_row])

    base = f"{args.old_label}_vs_{args.new_label}"
    summary_csv = args.out_dir / f"vector_distribution_summary_{base}.csv"
    compare_csv = args.out_dir / f"vector_distribution_comparison_{base}.csv"
    pca_png = args.out_dir / f"vector_pca_{base}.png"
    hist_png = args.out_dir / f"vector_similarity_hist_{base}.png"
    panel_png = args.out_dir / f"vector_old_vs_new_panel_{base}.png"

    summary_df.to_csv(summary_csv, index=False)
    compare_df.to_csv(compare_csv, index=False)
    plot_pca_overlay(old_era, new_era, pca_png, seed=args.seed)
    plot_similarity_hist_overlay(old_cos, new_cos, old_era.name, new_era.name, hist_png)
    plot_panel(old_era, new_era, old_cos, new_cos, panel_png, seed=args.seed)

    print(f"Wrote {summary_csv}")
    print(f"Wrote {compare_csv}")
    print(f"Wrote {pca_png}")
    print(f"Wrote {hist_png}")
    print(f"Wrote {panel_png}")


if __name__ == "__main__":
    main()
