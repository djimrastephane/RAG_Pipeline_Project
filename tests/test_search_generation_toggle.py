from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import unittest
import sys
import types
import threading

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Lightweight stubs so unit tests do not require heavy optional runtime deps.
if "faiss" not in sys.modules:
    sys.modules["faiss"] = types.ModuleType("faiss")
if "sentence_transformers" not in sys.modules:
    st_mod = types.ModuleType("sentence_transformers")
    st_mod.SentenceTransformer = object
    st_mod.CrossEncoder = object
    sys.modules["sentence_transformers"] = st_mod

from rag_pdf.services.search_service import SearchService


class _FakeModel:
    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=False):
        return np.array([[1.0, 0.0]], dtype="float32")


class _FakeIndex:
    def search(self, emb, k):
        return np.array([[0.9]], dtype="float32"), np.array([[0]], dtype="int64")


@dataclass
class _FakeLoadedDoc:
    index: _FakeIndex
    meta: pd.DataFrame
    eval_items: list[dict]
    bm25: object | None
    chunk_text_by_id: dict[str, str]
    chunk_section_by_id: dict[str, str]
    chunk_subsection_by_id: dict[str, str]


class SearchGenerationToggleTests(unittest.TestCase):
    def _build_service(self) -> SearchService:
        svc = SearchService.__new__(SearchService)
        svc.model = _FakeModel()
        svc.local_llm = type("LLM", (), {"model": "unit-test-llm"})()
        svc.cross_encoder = None
        svc.cross_encoder_topn = 50
        svc.cross_encoder_weight = 0.2
        svc._cache = {}
        svc._global_cache = {}
        svc.gen_max_context_chunks = 5
        svc.gen_max_context_chars = 9000
        svc.gen_max_chunk_chars = 2200
        svc.gen_timeout_seconds = 20.0
        svc._obs_lock = threading.Lock()
        svc._obs = {
            "generation_total": 0.0,
            "generation_ok": 0.0,
            "generation_skipped": 0.0,
            "generation_insufficient_evidence": 0.0,
            "generation_error": 0.0,
            "citations_parsed_total": 0.0,
            "citations_valid_total": 0.0,
            "citations_rejected_total": 0.0,
            "generation_latency_ms_sum": 0.0,
        }
        return svc

    @staticmethod
    def _build_loaded_doc() -> _FakeLoadedDoc:
        meta = pd.DataFrame(
            [
                {
                    "chunk_id": "c1",
                    "chunk_id_global": "doc:c1",
                    "pages": [1],
                    "page_start": 1,
                    "page_end": 1,
                    "section_title": "Financial Statements",
                    "subsection_title": "Core",
                    "is_table": False,
                    "doc_id": "DocA-2024-2025",
                }
            ]
        )
        return _FakeLoadedDoc(
            index=_FakeIndex(),
            meta=meta,
            eval_items=[],
            bm25=None,
            chunk_text_by_id={"doc:c1": "Example chunk text"},
            chunk_section_by_id={"doc:c1": "Financial Statements"},
            chunk_subsection_by_id={"doc:c1": "Core"},
        )

    def test_generation_is_skipped_when_flag_false(self) -> None:
        svc = self._build_service()
        loaded = self._build_loaded_doc()

        svc._load_doc = lambda data_dir: loaded  # type: ignore[attr-defined]
        svc._predict_answer = lambda **kwargs: (None, None, {"status": "ok"})  # type: ignore[attr-defined]

        calls = {"n": 0}

        def _gen(**kwargs):
            calls["n"] += 1
            return "should-not-be-called", {"status": "ok"}

        svc._generate_local_answer = _gen  # type: ignore[attr-defined]

        out = svc.search(
            data_dir=Path("data_processed/DocA-2024-2025"),
            question="What is the deficit?",
            k=1,
            include_generated_answer=False,
        )

        self.assertEqual(calls["n"], 0)
        self.assertIsNone(out.get("generated_answer"))
        self.assertEqual(out.get("generation_debug", {}).get("status"), "skipped")
        self.assertEqual(out.get("include_generated_answer"), False)

    def test_generation_is_called_when_flag_true(self) -> None:
        svc = self._build_service()
        loaded = self._build_loaded_doc()

        svc._load_doc = lambda data_dir: loaded  # type: ignore[attr-defined]
        svc._predict_answer = lambda **kwargs: (None, None, {"status": "ok"})  # type: ignore[attr-defined]

        calls = {"n": 0}

        def _gen(**kwargs):
            calls["n"] += 1
            return (
                "Grounded answer [chunk_id=doc:c1, page=1]",
                {"status": "ok", "provider": "local_ollama"},
            )

        svc._generate_local_answer = _gen  # type: ignore[attr-defined]

        out = svc.search(
            data_dir=Path("data_processed/DocA-2024-2025"),
            question="What is the deficit?",
            k=1,
            include_generated_answer=True,
        )

        self.assertEqual(calls["n"], 1)
        self.assertEqual(out.get("generated_answer"), "Grounded answer [chunk_id=doc:c1, page=1]")
        self.assertEqual(out.get("generation_status"), "ok")
        self.assertEqual(out.get("generation_debug", {}).get("status"), "ok")
        self.assertEqual(out.get("generated_citations"), [{"chunk_id": "doc:c1", "page": 1}])
        self.assertEqual(out.get("include_generated_answer"), True)

    def test_generation_is_gated_when_citations_missing(self) -> None:
        svc = self._build_service()
        loaded = self._build_loaded_doc()

        svc._load_doc = lambda data_dir: loaded  # type: ignore[attr-defined]
        svc._predict_answer = lambda **kwargs: (None, None, {"status": "ok"})  # type: ignore[attr-defined]
        svc._generate_local_answer = lambda **kwargs: ("Grounded answer", {"status": "ok"})  # type: ignore[attr-defined]

        out = svc.search(
            data_dir=Path("data_processed/DocA-2024-2025"),
            question="What is the deficit?",
            k=1,
            include_generated_answer=True,
        )

        self.assertIsNone(out.get("generated_answer"))
        self.assertEqual(out.get("generation_status"), "insufficient_evidence")
        self.assertEqual(out.get("generated_citations"), [])


if __name__ == "__main__":
    unittest.main()
