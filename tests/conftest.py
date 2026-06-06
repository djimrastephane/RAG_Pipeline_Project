from __future__ import annotations

import sys
import types
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Stub heavy ML packages so tests collect without requiring GPU/large installs.
for _name in ("torch", "faiss", "sentence_transformers"):
    if _name not in sys.modules:
        sys.modules[_name] = types.ModuleType(_name)

_st = sys.modules["sentence_transformers"]
if not hasattr(_st, "SentenceTransformer"):
    _st.SentenceTransformer = object  # type: ignore[attr-defined]
if not hasattr(_st, "CrossEncoder"):
    _st.CrossEncoder = object  # type: ignore[attr-defined]
