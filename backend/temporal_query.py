import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TemporalQueryBucket:
    label: str
    query: str


@dataclass(frozen=True)
class TemporalQueryPlan:
    original_query: str
    buckets: list[TemporalQueryBucket]

    @property
    def is_multi_bucket(self) -> bool:
        return len(self.buckets) > 1


class TemporalQueryPlanner:
    QUARTER_MONTHS = {
        1: (1, 2, 3),
        2: (4, 5, 6),
        3: (7, 8, 9),
        4: (10, 11, 12),
    }

    def plan(self, query: str) -> TemporalQueryPlan:
        quarter = self._quarter(query)
        if quarter:
            months = self.QUARTER_MONTHS[quarter]
            return TemporalQueryPlan(
                original_query=query,
                buckets=[
                    TemporalQueryBucket(label=f"{month}월", query=self._replace_quarter(query, quarter, month))
                    for month in months
                ],
            )

        half_months = self._half_year(query)
        if half_months:
            return TemporalQueryPlan(
                original_query=query,
                buckets=[
                    TemporalQueryBucket(label=f"{month}월", query=self._replace_half_year(query, month))
                    for month in half_months
                ],
            )

        week_range = self._week_range(query)
        if week_range:
            start, end = week_range
            return TemporalQueryPlan(
                original_query=query,
                buckets=[
                    TemporalQueryBucket(label=f"{week}주차", query=self._replace_week_range(query, week))
                    for week in range(start, end + 1)
                ],
            )

        return TemporalQueryPlan(
            original_query=query,
            buckets=[TemporalQueryBucket(label="single", query=query)],
        )

    def _quarter(self, query: str) -> int | None:
        match = re.search(r"([1-4])\s*분기", query)
        return int(match.group(1)) if match else None

    def _replace_quarter(self, query: str, quarter: int, month: int) -> str:
        return re.sub(rf"{quarter}\s*분기", f"{month}월", query, count=1)

    def _half_year(self, query: str) -> tuple[int, ...] | None:
        if "상반기" in query:
            return (1, 2, 3, 4, 5, 6)
        if "하반기" in query:
            return (7, 8, 9, 10, 11, 12)
        return None

    def _replace_half_year(self, query: str, month: int) -> str:
        return re.sub(r"상반기|하반기", f"{month}월", query, count=1)

    def _week_range(self, query: str) -> tuple[int, int] | None:
        patterns = (
            r"([1-5])\s*주차\s*[~\-]\s*([1-5])\s*주차",
            r"([1-5])\s*주차\s*(?:부터|에서)\s*([1-5])\s*주차\s*까지",
            r"([1-5])\s*[~\-]\s*([1-5])\s*주차",
        )
        for pattern in patterns:
            match = re.search(pattern, query)
            if not match:
                continue
            start = int(match.group(1))
            end = int(match.group(2))
            if start <= end:
                return start, end
        return None

    def _replace_week_range(self, query: str, week: int) -> str:
        replacements = (
            r"[1-5]\s*주차\s*[~\-]\s*[1-5]\s*주차",
            r"[1-5]\s*주차\s*(?:부터|에서)\s*[1-5]\s*주차\s*까지",
            r"[1-5]\s*[~\-]\s*[1-5]\s*주차",
        )
        for pattern in replacements:
            if re.search(pattern, query):
                return re.sub(pattern, f"{week}주차", query, count=1)
        return query
