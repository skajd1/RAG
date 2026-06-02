import os
import unittest

import _paths  # noqa: F401
from query_routing import RetrievalRoutePlanner


class RetrievalRoutePlannerTests(unittest.TestCase):
    def setUp(self):
        self.env_keys = [
            "RAG_ROUTE_EXACT_LOOKUP_TOP_K",
            "RAG_ROUTE_EXACT_LOOKUP_KEYWORD_TOP_K",
            "RAG_ROUTE_EXACT_LOOKUP_EXPAND_CHUNKS",
            "RAG_ROUTE_EXACT_LOOKUP_MAX_CONTEXT_CHARS",
            "RAG_ROUTE_LONG_DOC_LOOKUP_LONG_DOC_NEIGHBOR_WINDOW",
        ]
        for key in self.env_keys:
            os.environ.pop(key, None)

    def tearDown(self):
        for key in self.env_keys:
            os.environ.pop(key, None)

    def test_core_query_types_select_expected_profiles(self):
        planner = RetrievalRoutePlanner()

        self.assertEqual(planner.plan("사내 전결 규정 알려줘").name, "exact_lookup")
        self.assertEqual(planner.plan("5월 1주차~5주차 주간보고 표로 정리해줘").name, "multi_doc_summary")
        self.assertEqual(planner.plan("@SAMPLE_MAINTENANCE 5월 진행중 이슈 알려줘").name, "jira_status_summary")
        self.assertEqual(planner.plan("SAMPLE_APP V2 성능지표 문서에서 TPS 값 알려줘").name, "long_doc_lookup")
        self.assertEqual(planner.plan("관련 문서 알려줘").name, "broad_exploration")

    def test_exact_lookup_profile_defaults_and_env_override(self):
        route = RetrievalRoutePlanner().plan("사내 전결 규정 알려줘")

        self.assertEqual(route.profile.vector_top_k, 4)
        self.assertEqual(route.profile.keyword_top_k, 6)
        self.assertEqual(route.profile.expand_chunks, 12)
        self.assertEqual(route.profile.max_context_chars, 12000)

        os.environ["RAG_ROUTE_EXACT_LOOKUP_TOP_K"] = "3"
        self.assertEqual(RetrievalRoutePlanner().plan("사내 전결 규정 알려줘").profile.vector_top_k, 3)

    def test_invalid_env_values_are_ignored(self):
        os.environ["RAG_ROUTE_EXACT_LOOKUP_TOP_K"] = "abc"
        os.environ["RAG_ROUTE_EXACT_LOOKUP_KEYWORD_TOP_K"] = "0"
        os.environ["RAG_ROUTE_EXACT_LOOKUP_EXPAND_CHUNKS"] = "-1"

        route = RetrievalRoutePlanner().plan("사내 전결 규정 알려줘")

        self.assertEqual(route.profile.vector_top_k, 4)
        self.assertEqual(route.profile.keyword_top_k, 6)
        self.assertEqual(route.profile.expand_chunks, 12)


if __name__ == "__main__":
    unittest.main()
