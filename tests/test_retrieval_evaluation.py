import json
import tempfile
import unittest
from pathlib import Path

import _paths  # noqa: F401
from evaluation.metrics import aggregate_summary, ndcg_at_k, score_case
from evaluation.report import load_golden_cases, render_html_report


class RetrievalMetricTests(unittest.TestCase):
    def test_score_case_covers_ranked_metrics_and_forbidden_scope(self):
        score = score_case(
            retrieved_pages=["noise", "weekly-3", "weekly-4"],
            relevant_pages=["weekly-3", "weekly-4"],
            forbidden_pages=["noise"],
            k=3,
        )

        self.assertEqual(score["hit_at_k"], 1.0)
        self.assertEqual(score["recall_at_k"], 1.0)
        self.assertEqual(score["precision_at_k"], 2 / 3)
        self.assertEqual(score["reciprocal_rank"], 0.5)
        self.assertTrue(score["has_forbidden_scope"])

    def test_graded_ndcg_and_aggregate_summary(self):
        self.assertGreater(ndcg_at_k(["useful", "noise", "most-relevant"], {"most-relevant": 3, "useful": 2}, 3), 0.0)

        summary = aggregate_summary(
            [
                {"hit_at_k": 1.0, "recall_at_k": 1.0, "precision_at_k": 0.5, "reciprocal_rank": 1.0, "ndcg_at_k": 1.0, "has_forbidden_scope": False, "latency_ms": 10.0},
                {"hit_at_k": 0.0, "recall_at_k": 0.0, "precision_at_k": 0.0, "reciprocal_rank": 0.0, "ndcg_at_k": 0.0, "has_forbidden_scope": True, "latency_ms": 30.0},
            ]
        )

        self.assertEqual(summary["case_count"], 2)
        self.assertEqual(summary["hit_at_k"], 0.5)
        self.assertEqual(summary["forbidden_scope_rate"], 0.5)


class GoldenCaseLoaderTests(unittest.TestCase):
    def test_loads_jsonl_relevance_and_rejects_missing_relevance(self):
        rows = [
            {"id": "weekly", "category": "weekly", "query": "4주차", "history": [], "relevant_pages": ["weekly-4"]},
            {"id": "graded", "category": "meeting", "query": "5월 회의", "history": [], "relevance_grades": {"meeting-0526": 3}},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cases.jsonl"
            path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
            cases = load_golden_cases(path)

            missing = Path(temp_dir) / "missing.jsonl"
            missing.write_text('{"id":"missing","category":"negative","query":"none"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "relevant_pages or relevance_grades"):
                load_golden_cases(missing)

        self.assertEqual(cases[0]["relevant_pages"], ["weekly-4"])
        self.assertEqual(cases[1]["relevance_grades"]["meeting-0526"], 3)


class HtmlReportTests(unittest.TestCase):
    def test_renders_escaped_html_with_svg_summary(self):
        html = render_html_report(
            summary={"case_count": 1, "hit_at_k": 1.0, "mrr": 1.0},
            category_summaries={"weekly": {"case_count": 1, "hit_at_k": 1.0}},
            case_results=[
                {
                    "id": "weekly-4",
                    "category": "weekly",
                    "query": "<script>alert(1)</script>",
                    "resolved_query": "홍길동 5월 4주차 주간보고",
                    "scope_changed": True,
                    "route": "exact_lookup",
                    "long_document_stats": {"long_document_mode": "lookup"},
                    "latency_ms": 12.5,
                    "retrieved_pages": ["weekly-4"],
                    "hit_at_k": 1.0,
                    "recall_at_k": 1.0,
                    "precision_at_k": 0.2,
                    "reciprocal_rank": 1.0,
                    "ndcg_at_k": 1.0,
                    "forbidden_pages": [],
                }
            ],
            title="RAG <Quality>",
            targets={"hit_at_k": 0.9},
            baseline_summary={"hit_at_k": 0.25},
            baseline_case_results=[{"id": "weekly-4", "hit_at_k": 0.0}],
        )

        self.assertIn("<svg", html)
        self.assertIn("RAG &lt;Quality&gt;", html)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", html)
        self.assertIn("Baseline vs Current", html)
        self.assertNotIn("<script>", html)


if __name__ == "__main__":
    unittest.main()
