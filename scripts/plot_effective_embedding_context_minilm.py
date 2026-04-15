#!/usr/bin/env python3
"""Plot the effective embedding context under MiniLM token cap."""

from pathlib import Path

import _matplotlib_env
import matplotlib.pyplot as plt


def effective_tokens(chunk_size: int, cap: int = 256) -> int:
    """Return effective tokens seen by the embedding model."""
    return min(chunk_size, cap)


def main() -> None:
    # Study settings
    tested_chunk_sizes = [280, 320, 500, 700]
    minilm_cap = 256

    # Smooth curve across the plotting range
    x_values = list(range(0, 751))
    y_values = [effective_tokens(x, minilm_cap) for x in x_values]

    # Values at tested chunk sizes
    tested_effective = [effective_tokens(x, minilm_cap) for x in tested_chunk_sizes]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    # Main curve: y = min(x, 256)
    ax.plot(x_values, y_values, linewidth=2, label=r"$\mathrm{effective\_tokens}=\min(\mathrm{chunk\_size},256)$")

    # Vertical cap reference line
    ax.axvline(minilm_cap, linestyle="--", linewidth=1.5, label="MiniLM cap = 256 tokens")

    # Mark tested chunk sizes on the curve
    ax.scatter(tested_chunk_sizes, tested_effective, s=36, zorder=3)

    # Annotate tested points with staggered offsets to avoid overlap
    label_offsets = {
        280: (-28, 18),
        320: (10, 32),
        500: (10, 18),
        700: (-56, 32),
    }
    for x, y in zip(tested_chunk_sizes, tested_effective):
        dx, dy = label_offsets.get(x, (6, 8))
        ax.annotate(
            f"{x} \u2192 {y}",
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=10,
            arrowprops={"arrowstyle": "-", "linewidth": 0.8},
        )

    # Short explanatory note near the plateau
    ax.text(
        430,
        210,
        "Beyond 256 tokens, additional chunk text is\ntruncated before embedding",
        fontsize=10,
    )

    # Thesis-friendly labels and styling
    ax.set_title("Effective Embedding Context Under the MiniLM 256-Token Cap", fontsize=13)
    ax.set_xlabel("Chunk size (tokens)", fontsize=11)
    ax.set_ylabel("Effective tokens seen by MiniLM", fontsize=11)
    ax.set_xlim(0, 750)
    ax.set_ylim(0, 300)
    ax.tick_params(labelsize=10)
    ax.grid(True, linewidth=0.5, alpha=0.4)
    ax.legend(frameon=False, loc="lower right")

    # Save high-resolution output
    repo_root = Path(__file__).resolve().parents[1]
    output_path = repo_root / "docs" / "figures" / "effective_embedding_context_minilm.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)

    # Console summary
    print("Tested chunk sizes:", tested_chunk_sizes)
    print("Effective tokens per chunk size:")
    for chunk_size, eff in zip(tested_chunk_sizes, tested_effective):
        print(f"  {chunk_size} -> {eff}")


if __name__ == "__main__":
    main()
