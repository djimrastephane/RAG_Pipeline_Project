# make_titles_before_after.py
# Generates BEFORE/AFTER section-title frequency charts for Figure 3.3.
# Output files:
#   figures/top_section_titles_before.png
#   figures/top_section_titles_after.png
#
# Run inside your rag_nhs environment:
#   python make_titles_before_after.py

from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path("/Users/djimra/Offline Projects/RAG_NHS")
DOC_ID = "nss_annual_report_and_accounts_2023-24"
DATA_DIR = PROJECT_ROOT / "data_processed" / DOC_ID

SECTIONS_PATH = DATA_DIR / "sections.parquet"
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_BEFORE = FIG_DIR / "top_section_titles_before.png"
OUT_AFTER = FIG_DIR / "top_section_titles_after.png"


# -----------------------------------------------------------------------------
# Config (adjust if needed)
# -----------------------------------------------------------------------------
TOP_N = 10

# Titles you consider structural boilerplate rather than meaningful headings
STOP_TITLES = {
    "nhs national services scotland",
    "annual report and accounts",
    "annual report and accounts 2023/24",
    "part a",
    "part b",
    "contents",
    "at 31 march 2024",
    "at 1 april 2022",
    "current year",
}

# Optional: treat obvious date-only titles as boilerplate
DATE_TITLE_RE = re.compile(r"^(at\s+\d{1,2}\s+[a-z]+\s+\d{4}|at\s+\d{1,2}\s+[a-z]+\s+\d{2,4})$", re.IGNORECASE)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def is_boilerplate_title(title: str) -> bool:
    t = (title or "").strip().lower()
    if t in STOP_TITLES:
        return True
    if DATE_TITLE_RE.match(t):
        return True
    return False


def apply_title_normalisation(sections_df: pd.DataFrame) -> pd.DataFrame:
    """
    Carry-forward rule:
    - If title is boilerplate or Unknown, replace with last good title.
    """
    df = sections_df.copy()

    cleaned = []
    last_good = "Unknown"

    for t in df["section_title"].fillna("Unknown").tolist():
        if t == "Unknown" or is_boilerplate_title(t):
            cleaned.append(last_good)
        else:
            cleaned.append(t)
            last_good = t

    df["section_title_norm"] = cleaned
    return df


def plot_top_titles(series: pd.Series, title: str, out_path: Path, top_n: int = 10):
    vc = series.value_counts().head(top_n)
    vc = vc.sort_values(ascending=True)

    plt.figure(figsize=(10, 5.5))
    plt.barh(vc.index.tolist(), vc.values.tolist())
    plt.xlabel("Frequency")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    if not SECTIONS_PATH.exists():
        raise FileNotFoundError(f"Missing sections.parquet: {SECTIONS_PATH}")

    sections = pd.read_parquet(SECTIONS_PATH)

    # BEFORE: raw titles
    plot_top_titles(
        sections["section_title"].fillna("Unknown"),
        title="Top section titles (before normalisation)",
        out_path=OUT_BEFORE,
        top_n=TOP_N,
    )

    # AFTER: normalised titles
    sections2 = apply_title_normalisation(sections)

    plot_top_titles(
        sections2["section_title_norm"].fillna("Unknown"),
        title="Top section titles (after normalisation)",
        out_path=OUT_AFTER,
        top_n=TOP_N,
    )

    print("Saved:")
    print(" -", OUT_BEFORE)
    print(" -", OUT_AFTER)


if __name__ == "__main__":
    main()