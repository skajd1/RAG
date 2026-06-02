import unittest

import _paths  # noqa: F401

from temporal_query import TemporalQueryPlanner


class TemporalQueryPlannerTests(unittest.TestCase):
    def setUp(self):
        self.planner = TemporalQueryPlanner()

    def test_quarter_query_expands_to_month_buckets(self):
        plan = self.planner.plan("홍길동 3분기 보고서 요약해줘")

        self.assertTrue(plan.is_multi_bucket)
        self.assertEqual([bucket.label for bucket in plan.buckets], ["7월", "8월", "9월"])
        self.assertEqual(
            [bucket.query for bucket in plan.buckets],
            [
                "홍길동 7월 보고서 요약해줘",
                "홍길동 8월 보고서 요약해줘",
                "홍길동 9월 보고서 요약해줘",
            ],
        )

    def test_week_range_query_expands_to_week_buckets(self):
        plan = self.planner.plan("홍길동 5월 1주차~5주차 주간보고 표로 정리해줘")

        self.assertTrue(plan.is_multi_bucket)
        self.assertEqual([bucket.label for bucket in plan.buckets], ["1주차", "2주차", "3주차", "4주차", "5주차"])
        self.assertEqual(plan.buckets[0].query, "홍길동 5월 1주차 주간보고 표로 정리해줘")
        self.assertEqual(plan.buckets[-1].query, "홍길동 5월 5주차 주간보고 표로 정리해줘")

    def test_non_range_query_keeps_single_bucket(self):
        plan = self.planner.plan("홍길동 5월 4주차 주간보고 요약해줘")

        self.assertFalse(plan.is_multi_bucket)
        self.assertEqual(len(plan.buckets), 1)
        self.assertEqual(plan.buckets[0].query, "홍길동 5월 4주차 주간보고 요약해줘")


if __name__ == "__main__":
    unittest.main()
