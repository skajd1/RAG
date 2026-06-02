import re
from dataclasses import dataclass, replace


DOCUMENT_TYPES = ("주간보고", "회의록", "기술회의", "점검", "프로젝트", "이슈")
ANCHOR_FIELDS = ("subject", "year", "month", "week", "document_type")
SUBJECT_STOP_WORDS = {
    "보고서",
    "문서",
    "회의록",
    "기술회의",
    "주간보고",
    "주차",
    "프로젝트",
    "이슈",
    "점검",
    "내용",
    "이번주",
    "지난주",
}


@dataclass(frozen=True)
class QueryAnchors:
    subject: str | None = None
    year: str | None = None
    month: str | None = None
    week: str | None = None
    document_type: str | None = None


@dataclass(frozen=True)
class ResolvedFollowupQuery:
    retrieval_query: str
    scope_changed: bool
    anchors: QueryAnchors


class StructuredFollowupQueryResolver:
    def resolve(
        self,
        query: str,
        history: list[dict],
        source_texts: list[str] | None = None,
    ) -> ResolvedFollowupQuery:
        current = self._extract_anchors(query)
        previous = self._previous_user_anchors(history) or self._source_text_anchors(source_texts)
        if "@" in query or self._is_standalone(current) or not previous:
            return ResolvedFollowupQuery(query, False, current)

        changed_fields = {
            field
            for field in ANCHOR_FIELDS
            if getattr(current, field) and getattr(current, field) != getattr(previous, field)
        }
        if not changed_fields:
            return ResolvedFollowupQuery(query, False, current)

        merged = previous
        for field in ANCHOR_FIELDS:
            value = getattr(current, field)
            if value:
                merged = replace(merged, **{field: value})
        return ResolvedFollowupQuery(self._format_anchors(merged), True, merged)

    def _previous_user_anchors(self, history: list[dict]) -> QueryAnchors | None:
        merged = QueryAnchors()
        for item in reversed(history or []):
            if item.get("role") != "user":
                continue
            anchors = self._extract_anchors(item.get("content", ""))
            merged = self._fill_missing_anchors(merged, anchors)
        return merged if any(vars(merged).values()) else None

    def _fill_missing_anchors(self, primary: QueryAnchors, fallback: QueryAnchors) -> QueryAnchors:
        values = {
            field: getattr(primary, field) or getattr(fallback, field)
            for field in ANCHOR_FIELDS
        }
        return QueryAnchors(**values)

    def _source_text_anchors(self, source_texts: list[str] | None) -> QueryAnchors | None:
        merged = QueryAnchors()
        for text in source_texts or []:
            anchors = self._extract_anchors(text)
            merged = self._fill_missing_anchors(merged, anchors)
        return merged if any(vars(merged).values()) else None

    def _extract_anchors(self, query: str) -> QueryAnchors:
        return QueryAnchors(
            subject=self._first_subject_match(
                query,
                r"([가-힣]{2,4})\s*의",
                r"([가-힣]{2,4})\s+것(?:도|으로)(?:\s|$)",
                r"([가-힣]{2,4})\s*주간보고",
                r"^([가-힣]{2,4})\s+(?=(?:20\d{2}\s*년|\d{1,2}\s*월|\d\s*주차|주간보고|회의록|기술회의|점검|프로젝트|이슈))",
            ),
            year=self._first_match(query, r"(20\d{2})\s*년"),
            month=self._first_number_match(
                query,
                r"(?<!\d)(1[0-2]|0?[1-9])\s*월",
                r"(?:20)?\d{2}[-_.년\s]+(1[0-2]|0?[1-9])[-_.월\s]+\d{1,2}\s*주차",
            ),
            week=self._first_number_match(
                query,
                r"(?<!\d)([1-5])\s*주차",
                r"(?:20)?\d{2}[-_.년\s]+(?:1[0-2]|0?[1-9])[-_.월\s]+(0?[1-5])\s*주차",
            ),
            document_type=next((value for value in DOCUMENT_TYPES if value in query), None),
        )

    def _first_subject_match(self, query: str, *patterns: str) -> str | None:
        for pattern in patterns:
            for match in re.finditer(pattern, query):
                candidate = match.group(1)
                normalized_candidate = candidate.removesuffix("의")
                if normalized_candidate not in SUBJECT_STOP_WORDS:
                    return candidate
        return None

    def _first_match(self, query: str, *patterns: str) -> str | None:
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                return match.group(1)
        return None

    def _first_number_match(self, query: str, *patterns: str) -> str | None:
        value = self._first_match(query, *patterns)
        if value is None:
            return None
        return str(int(value))

    def _is_standalone(self, anchors: QueryAnchors) -> bool:
        return bool(
            anchors.document_type
            and anchors.subject
            and (anchors.year or anchors.month or anchors.week)
        )

    def _format_anchors(self, anchors: QueryAnchors) -> str:
        values = [
            anchors.subject,
            f"{anchors.year}년" if anchors.year else None,
            f"{anchors.month}월" if anchors.month else None,
            f"{anchors.week}주차" if anchors.week else None,
            anchors.document_type,
        ]
        return " ".join(value for value in values if value)
