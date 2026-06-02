import unittest

import _paths  # noqa: F401

from conversation_query import StructuredFollowupQueryResolver


class StructuredFollowupQueryResolverTests(unittest.TestCase):
    def setUp(self):
        self.resolver = StructuredFollowupQueryResolver()
        self.previous_report = [{"role": "user", "content": "홍길동의 5월 4주차 주간보고 요약해줘"}]

    def test_week_followup_reuses_previous_subject_month_and_document_type(self):
        result = self.resolver.resolve("3주차도 보여줘", self.previous_report)

        self.assertEqual(result.retrieval_query, "홍길동 5월 3주차 주간보고")
        self.assertTrue(result.scope_changed)

    def test_subject_followup_reuses_previous_month_week_and_document_type(self):
        result = self.resolver.resolve("이영희 것도 보여줘", self.previous_report)

        self.assertEqual(result.retrieval_query, "이영희 5월 4주차 주간보고")
        self.assertTrue(result.scope_changed)

    def test_explicit_or_formatting_queries_keep_original_query(self):
        self.assertEqual(self.resolver.resolve("표로 정리해줘", self.previous_report).retrieval_query, "표로 정리해줘")
        self.assertEqual(
            self.resolver.resolve("이영희 5월 3주차 주간보고 알려줘", self.previous_report).retrieval_query,
            "이영희 5월 3주차 주간보고 알려줘",
        )
        self.assertEqual(self.resolver.resolve("@TEAM 3주차 주간보고 알려줘", self.previous_report).retrieval_query, "@TEAM 3주차 주간보고 알려줘")

    def test_assistant_history_is_not_used_for_anchor_extraction(self):
        history = [
            {"role": "user", "content": "홍길동의 5월 4주차 주간보고 요약해줘"},
            {"role": "assistant", "content": "이영희의 6월 2주차 회의록입니다."},
        ]

        result = self.resolver.resolve("3주차도 보여줘", history)

        self.assertEqual(result.retrieval_query, "홍길동 5월 3주차 주간보고")

    def test_week_followup_can_use_prior_source_title_when_history_is_missing(self):
        source_texts = ["26-05-04주차 홍길동 주간보고 SAMPLE TEAM SPACE > 업무보고 > 주간보고 > 홍길동 주간보고"]

        result = self.resolver.resolve("3주차도 보여줘", [], source_texts=source_texts)

        self.assertEqual(result.retrieval_query, "홍길동 5월 3주차 주간보고")
        self.assertTrue(result.scope_changed)


if __name__ == "__main__":
    unittest.main()
