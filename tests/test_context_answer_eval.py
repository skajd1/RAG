import tempfile
import unittest
from pathlib import Path

import _paths  # noqa: F401

from evaluation.context_answer_eval import answer_fact_score, render_report, source_hit_score, summarize


class ContextAnswerEvalTests(unittest.TestCase):
    def test_fact_and_source_scores_cover_answer_quality_contract(self):
        fact_score, missing = answer_fact_score("AM-174 샘플보험 2026-04-21 완료", ["AM-174", "샘플보험", "2026-04-21"])
        source_score = source_hit_score([{"page_id": "p1", "title": "문서"}, {"title": "title-only"}], ["p1", "title-only"])

        self.assertEqual(fact_score, 1.0)
        self.assertEqual(missing, [])
        self.assertEqual(source_score, 1.0)

    def test_summarize_aggregates_accuracy_latency_and_prompt_size(self):
        summaries = summarize(
            [
                {"axis_value": 4000, "axis_label": "Context chars", "context_size": 4000, "fact_score": 1.0, "source_score": 0.5, "passed": True, "latency_seconds": 2.0, "prompt_context_chars": 1200, "prompt_context_est_tokens": 400, "answer_chars": 120, "answer_est_tokens": 40, "prompt_chunk_count": 3},
                {"axis_value": 4000, "axis_label": "Context chars", "context_size": 4000, "fact_score": 0.0, "source_score": 0.5, "passed": False, "latency_seconds": 4.0, "prompt_context_chars": 1800, "prompt_context_est_tokens": 600, "answer_chars": 180, "answer_est_tokens": 60, "prompt_chunk_count": 5},
            ]
        )

        self.assertEqual(summaries[0]["answer_accuracy"], 0.5)
        self.assertEqual(summaries[0]["average_latency_seconds"], 3.0)
        self.assertEqual(summaries[0]["average_prompt_chunks"], 4)

    def test_render_report_writes_smoke_html(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "report.html"
            render_report(
                [{"axis_value": 4000, "axis_label": "Context chars", "context_size": 4000, "answer_accuracy": 1.0, "source_accuracy": 1.0, "pass_rate": 1.0, "average_latency_seconds": 2.0, "average_prompt_chars": 1200, "average_prompt_est_tokens": 400, "average_answer_chars": 150}],
                [{"axis_value": 4000, "context_size": 4000, "id": "case", "category": "cat", "fact_score": 1.0, "source_score": 1.0, "passed": True, "missing_facts": [], "latency_seconds": 2.0, "prompt_chunk_count": 3, "prompt_context_chars": 1200, "prompt_context_est_tokens": 400, "answer_chars": 150}],
                output,
            )

            html = output.read_text(encoding="utf-8")

        self.assertIn("<svg", html)
        self.assertIn("case", html)
        self.assertIn("Correlation Summary", html)


if __name__ == "__main__":
    unittest.main()
