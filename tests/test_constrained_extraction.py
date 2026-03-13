from __future__ import annotations

from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from generation.constrained_extraction import (
    build_constrained_extraction_messages,
    parse_constrained_extraction_response,
    validate_span_in_context,
)


class ConstrainedExtractionTests(unittest.TestCase):
    def test_parse_valid_json(self) -> None:
        out = parse_constrained_extraction_response('{"answer":"42","evidence_span":"42"}')
        self.assertEqual(out["answer"], "42")
        self.assertEqual(out["evidence_span"], "42")

    def test_parse_recovers_json_with_prefix_suffix(self) -> None:
        text = 'ignored prefix\n{"answer":"£1,200","evidence_span":"£1,200"}\nignored suffix'
        out = parse_constrained_extraction_response(text)
        self.assertEqual(out["answer"], "£1,200")
        self.assertEqual(out["evidence_span"], "£1,200")

    def test_schema_enforcement_rejects_missing_or_extra_keys(self) -> None:
        with self.assertRaises(ValueError):
            parse_constrained_extraction_response('{"answer":"x"}')
        with self.assertRaises(ValueError):
            parse_constrained_extraction_response('{"answer":"x","evidence_span":"x","extra":1}')

    def test_validate_span_in_context_passes_when_present(self) -> None:
        ctx = "Total deficit was £12.5 million in 2024/25."
        out = validate_span_in_context("£12.5 million", "£12.5 million", ctx)
        self.assertEqual(out["answer"], "£12.5 million")
        self.assertEqual(out["evidence_span"], "£12.5 million")
        self.assertEqual(out["violations"], [])

    def test_validate_span_in_context_nulls_when_missing(self) -> None:
        ctx = "Total deficit was £12.5 million in 2024/25."
        out = validate_span_in_context("£9.0 million", "£9.0 million", ctx)
        self.assertIsNone(out["answer"])
        self.assertIsNone(out["evidence_span"])
        self.assertIn("answer_not_in_context", out["violations"])

    def test_build_messages_has_two_messages_and_required_sections(self) -> None:
        q = "What was total deficit?"
        c = "The total deficit was £12.5 million."
        msgs = build_constrained_extraction_messages(q, c)

        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[1]["role"], "user")

        system = msgs[0]["content"].lower()
        user = msgs[1]["content"].lower()

        self.assertIn("exact extraction engine", system)
        self.assertIn("exact json only", system)
        self.assertIn("question:", user)
        self.assertIn("context:", user)
        self.assertIn("you must follow these rules", user)
        self.assertIn('"answer"', user)
        self.assertIn('"evidence_span"', user)
        self.assertIn("return nothing other than this json object", user)


if __name__ == "__main__":
    unittest.main()
