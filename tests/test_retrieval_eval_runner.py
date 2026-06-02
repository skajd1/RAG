import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import _paths  # noqa: F401
from evaluation.query_log_analyzer import load_records, render
from evaluation.run_retrieval_eval import FixtureRetrievalAdapter, LiveRetrievalAdapter, run_evaluation


class RetrievalEvalRunnerTests(unittest.TestCase):
    def test_fixture_adapter_requires_explicit_rankings(self):
        adapter = FixtureRetrievalAdapter()

        result = adapter.retrieve({"fixture_ranked_pages": [{"page_id": "page-2"}, {"page_id": "page-1"}]}, "query")
        self.assertEqual([page["page_id"] for page in result["ranked_pages"]], ["page-2", "page-1"])

        with self.assertRaisesRegex(ValueError, "fixture_ranked_pages"):
            adapter.retrieve({"id": "missing-fixture-ranking", "relevant_pages": ["page-1"]}, "query")

    def test_live_adapter_deduplicates_docs_and_preserves_route_metadata(self):
        docs = [
            SimpleNamespace(metadata={"page_id": "page-1", "title": "First"}),
            SimpleNamespace(metadata={"page_id": "page-1", "title": "First duplicate"}),
            SimpleNamespace(metadata={"title": "Title only"}),
            SimpleNamespace(metadata={"title": "Title only"}),
        ]
        context = SimpleNamespace(
            docs=docs,
            route=SimpleNamespace(name="exact_lookup", reason="exact"),
            long_document_stats={"long_document_mode": "lookup"},
        )
        chain = SimpleNamespace(retrieve_context=lambda *args, **kwargs: context)

        result = LiveRetrievalAdapter(chain_factory=lambda: chain).retrieve({"query": "q"}, "resolved query")

        self.assertEqual([page["page_id"] for page in result["ranked_pages"]], ["page-1", ""])
        self.assertEqual(result["route"], "exact_lookup")
        self.assertEqual(result["long_document_stats"]["long_document_mode"], "lookup")

    def test_runs_fixture_evaluation_writes_html(self):
        cases = [
            {
                "id": "followup",
                "category": "weekly",
                "query": "3주차도 보여줘",
                "history": [{"role": "user", "content": "홍길동의 5월 4주차 주간보고 요약해줘"}],
                "relevant_pages": ["weekly-3"],
                "forbidden_pages": ["weekly-4"],
                "expected_query": "홍길동 5월 3주차 주간보고",
                "fixture_ranked_pages": [{"page_id": "weekly-3"}, {"page_id": "weekly-4"}],
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cases_path = root / "cases.jsonl"
            output_path = root / "report.html"
            cases_path.write_text("\n".join(json.dumps(case, ensure_ascii=False) for case in cases), encoding="utf-8")

            result = run_evaluation(cases_path, output_path, mode="fixture", k=2)

            self.assertTrue(output_path.exists())
            html = output_path.read_text(encoding="utf-8")

        self.assertIn("<h1>RAG Retrieval Quality (fixture)</h1>", html)
        self.assertEqual(result["case_results"][0]["resolved_query"], "홍길동 5월 3주차 주간보고")
        self.assertEqual(result["summary"]["case_count"], 1)

    def test_default_live_chain_loads_env_and_query_log_analyzer_renders(self):
        with patch("evaluation.run_retrieval_eval.load_dotenv") as load, patch("rag_chain.RAGChain", return_value="chain"):
            self.assertEqual(LiveRetrievalAdapter()._create_chain(), "chain")
        self.assertEqual(load.call_count, 2)
        self.assertTrue(load.call_args_list[1].kwargs["override"])

        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "query"
            log_dir.mkdir()
            (log_dir / "sample.jsonl").write_text(
                '{"question":"사내 전결 규정 알려줘","route":"exact_lookup","total_seconds":4.2,"prompt_chunks":5,"finish_reason":"end"}\n',
                encoding="utf-8-sig",
            )
            records = load_records(log_dir)
            html = render(records)

        self.assertEqual(len(records), 1)
        self.assertIn("exact_lookup", html)
        self.assertIn("4.20s", html)


if __name__ == "__main__":
    unittest.main()
