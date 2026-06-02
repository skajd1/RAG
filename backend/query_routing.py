import os
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalProfile:
    vector_top_k: int
    keyword_top_k: int
    jira_top_k: int
    expand_chunks: int
    max_context_chars: int
    long_doc_neighbor_window: int = 4
    long_doc_detail_chunks: int = 48
    long_doc_summary_window_chunks: int = 8
    use_long_document_summary: bool = False


@dataclass(frozen=True)
class RetrievalRoute:
    name: str
    reason: str
    profile: RetrievalProfile


class RetrievalRoutePlanner:
    DEFAULTS = {
        "exact_lookup": RetrievalProfile(4, 6, 5, 12, 12000),
        "single_doc_detail": RetrievalProfile(4, 6, 5, 32, 24000),
        "multi_doc_summary": RetrievalProfile(8, 6, 5, 12, 36000),
        "jira_status_summary": RetrievalProfile(4, 4, 8, 8, 18000),
        "project_overview": RetrievalProfile(8, 6, 8, 16, 24000),
        "broad_exploration": RetrievalProfile(12, 8, 5, 8, 18000),
        "long_doc_lookup": RetrievalProfile(4, 8, 5, 8, 18000, long_doc_neighbor_window=4),
        "long_doc_detail": RetrievalProfile(4, 8, 5, 48, 36000, long_doc_detail_chunks=48),
        "long_doc_summary": RetrievalProfile(
            4,
            6,
            5,
            8,
            24000,
            long_doc_summary_window_chunks=8,
            use_long_document_summary=True,
        ),
        "fallback": RetrievalProfile(5, 3, 5, 20, 18000),
    }

    ENV_FIELDS = {
        "TOP_K": "vector_top_k",
        "KEYWORD_TOP_K": "keyword_top_k",
        "JIRA_TOP_K": "jira_top_k",
        "EXPAND_CHUNKS": "expand_chunks",
        "MAX_CONTEXT_CHARS": "max_context_chars",
        "LONG_DOC_NEIGHBOR_WINDOW": "long_doc_neighbor_window",
        "LONG_DOC_DETAIL_CHUNKS": "long_doc_detail_chunks",
        "LONG_DOC_SUMMARY_WINDOW_CHUNKS": "long_doc_summary_window_chunks",
    }

    def __init__(self):
        self.profiles = {
            name: self.apply_env_overrides(name, profile)
            for name, profile in self.DEFAULTS.items()
        }

    def apply_env_overrides(self, name: str, profile: RetrievalProfile) -> RetrievalProfile:
        values = profile.__dict__.copy()
        prefix = f"RAG_ROUTE_{name.upper()}_"
        for suffix, field in self.ENV_FIELDS.items():
            raw = os.getenv(prefix + suffix)
            if raw is None:
                continue
            try:
                value = int(raw)
            except ValueError:
                print(f"[RAG_ROUTE] ignored invalid integer env {prefix + suffix}={raw!r}")
                continue
            if value <= 0:
                print(f"[RAG_ROUTE] ignored non-positive env {prefix + suffix}={raw!r}")
                continue
            values[field] = value
        return RetrievalProfile(**values)

    def plan(self, query: str, mentions: list[dict] | None = None) -> RetrievalRoute:
        text = query or ""
        normalized = self.normalize(text)
        active_mentions = mentions or []

        if self.is_jira_status_query(normalized, active_mentions):
            return self.route("jira_status_summary", "jira/status terms")
        if self.is_long_doc_summary_query(normalized):
            return self.route("long_doc_summary", "whole-document summary terms")
        if self.is_long_doc_detail_query(normalized):
            return self.route("long_doc_detail", "single-document detail terms")
        if self.is_long_doc_lookup_query(normalized):
            return self.route("long_doc_lookup", "single-document lookup terms")
        if self.is_temporal_range_query(text):
            return self.route("multi_doc_summary", "temporal range terms")
        if self.is_project_overview_query(normalized):
            return self.route("project_overview", "project overview terms")
        if self.is_broad_exploration_query(normalized):
            return self.route("broad_exploration", "broad exploration terms")
        if self.is_exact_lookup_query(normalized):
            return self.route("exact_lookup", "exact lookup terms")
        return self.route("fallback", "no route rule matched")

    def route(self, name: str, reason: str) -> RetrievalRoute:
        return RetrievalRoute(name=name, reason=reason, profile=self.profiles[name])

    def normalize(self, value: str) -> str:
        return re.sub(r"\s+", "", value.lower())

    def is_jira_status_query(self, normalized: str, mentions: list[dict]) -> bool:
        has_jira_mention = any(mention.get("source_type") == "jira" for mention in mentions)
        has_jira_scope = has_jira_mention or any(
            term in normalized
            for term in ["jira", "issue", "이슈", "작업", "maintenance", "SAMPLE_MAINTENANCE"]
        )
        has_status_term = any(
            term in normalized
            for term in ["진행", "예정", "완료", "해야", "todo", "done", "status", "상태"]
        )
        return has_jira_scope and has_status_term

    def is_long_doc_summary_query(self, normalized: str) -> bool:
        has_document_scope = any(term in normalized for term in ["문서전체", "전체문서"])
        has_summary_intent = any(term in normalized for term in ["요약", "정리", "summary"])
        return has_document_scope and has_summary_intent

    def is_long_doc_detail_query(self, normalized: str) -> bool:
        has_document_scope = "문서" in normalized or "document" in normalized
        has_detail_intent = any(term in normalized for term in ["자세히", "상세", "전체내용", "detail"])
        return has_document_scope and has_detail_intent

    def is_long_doc_lookup_query(self, normalized: str) -> bool:
        has_document_scope = "문서" in normalized or "document" in normalized
        return has_document_scope and self.has_lookup_term(normalized)

    def is_temporal_range_query(self, query: str) -> bool:
        return bool(
            re.search(r"\d+\s*주차\s*(?:[~-]|부터)\s*\d+\s*주차", query)
            or re.search(r"[1234]\s*분기", query)
            or re.search(r"(상반기|하반기)", query)
        )

    def is_project_overview_query(self, normalized: str) -> bool:
        has_project_scope = "프로젝트" in normalized or "project" in normalized
        has_overview_intent = any(term in normalized for term in ["현황", "개요", "진행", "overview"])
        return has_project_scope and has_overview_intent

    def is_exact_lookup_query(self, normalized: str) -> bool:
        return self.has_lookup_term(normalized)

    def has_lookup_term(self, normalized: str) -> bool:
        return any(
            term in normalized
            for term in [
                "규정",
                "접속정보",
                "주소",
                "계정",
                "일정",
                "날짜",
                "값",
                "번호",
                "ip",
                "url",
                "서버",
                "포트",
            ]
        )

    def is_broad_exploration_query(self, normalized: str) -> bool:
        return any(term in normalized for term in ["관련문서", "찾아", "검색", "explore"])
