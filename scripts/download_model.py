"""
Populate models/all-MiniLM-L6-v2 from the HuggingFace Hub (or local cache).

Run once after a fresh clone or environment rebuild before starting the API:

    python scripts/download_model.py

The API (app/api/main.py) loads the embedding model from models/all-MiniLM-L6-v2
at startup and will raise FileNotFoundError if the directory is missing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HF_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DEST = REPO_ROOT / "models" / "all-MiniLM-L6-v2"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download embedding model weights.")
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Destination directory (default: {DEFAULT_DEST})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the destination already exists.",
    )
    args = parser.parse_args()
    dest: Path = args.dest

    if dest.exists() and not args.force:
        print(f"Model already present at {dest}  (use --force to re-download)")
        sys.exit(0)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence-transformers is not installed. Run: pip install sentence-transformers")
        sys.exit(1)

    print(f"Downloading {HF_MODEL_ID} → {dest} ...")
    model = SentenceTransformer(HF_MODEL_ID)
    dest.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(dest))
    print(f"Saved to {dest}")


if __name__ == "__main__":
    main()
