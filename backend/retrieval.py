import copy
import re
from dataclasses import dataclass
from datetime import datetime

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models


@dataclass
class HybridRetrievalResult:
    docs: list[Document]
    vector_docs: list[Document]
    keyword_docs: list[Document]
    jira_docs: list[Document]


class QueryKeywordAnalyzer:
    KOREAN_STOP_WORDS = {
        "관련",
        "문서",
        "요약",
        "내용",
        "정리",
        "알려줘",
        "찾아줘",
        "요약해",
        "요약해줘",
        "해줘",
        "주간보고",
    }
    PARTICLES = ("에게서", "에게", "으로", "에서", "부터", "까지", "은", "는", "이", "가", "을", "를", "의")
    SYNONYMS = {
        "기술회의": {"기술회의", "주간회의", "회의록", "회의"},
        "주간회의": {"주간회의", "기술회의", "회의록", "회의"},
        "회의록": {"회의록", "회의", "기술회의", "주간회의"},
        "점검": {"점검", "정기점검", "유지보수"},
        "정기점검": {"정기점검", "점검", "유지보수"},
        "해야할일": {"해야할일", "todo", "pending", "open"},
        "진행중": {"진행중", "inprogress", "inprogress", "현재"},
        "완료": {"완료", "done", "closed", "resolved"},
        "사내": {"사내", "전사", "회사", "내부"},
        "전사": {"전사", "사내", "회사", "내부"},
        "규정": {"규정", "정책"},
    }

    def normalize(self, value: str) -> str:
        return re.sub(r"[^0-9a-zA-Z가-힣]+", "", value or "").lower()

    def strip_particle(self, term: str) -> str:
        for suffix in self.PARTICLES:
            if term.endswith(suffix) and len(term) > len(suffix) + 1:
                return term[: -len(suffix)]
        return term

    def extract(self, query: str) -> dict[str, set[str]]:
        exact_terms = set()
        broad_terms = set()

        exact_terms.update(match.lower() for match in re.findall(r"\b[A-Z][A-Z0-9]+-\d+\b", query, re.IGNORECASE))
        exact_terms.update(self.temporal_terms(query))

        for raw_term in re.findall(r"[가-힣]{2,12}", query):
            term = self.strip_particle(raw_term)
            if term in self.KOREAN_STOP_WORDS or term.endswith("주차"):
                continue
            normalized = self.normalize(term)
            if normalized:
                broad_terms.add(normalized)
                broad_terms.update(self.expand_synonyms(normalized))
                if normalized.endswith("회의록"):
                    broad_terms.add(normalized[:-1])

        return {
            "exact": {self.normalize(term) for term in exact_terms if self.normalize(term)},
            "broad": broad_terms,
        }

    def expand_synonyms(self, term: str) -> set[str]:
        return {self.normalize(value) for value in self.SYNONYMS.get(term, set()) if self.normalize(value)}

    def temporal_terms(self, query: str) -> set[str]:
        return self.calendar_date_terms(query) | self.week_terms(query)

    def requested_months(self, query: str) -> set[int]:
        return {int(month) for month in re.findall(r"(?<!\d)(1[0-2]|0?[1-9])\s*월", query)}

    def locator_months(self, value: str) -> set[int]:
        months = {
            int(month)
            for month in re.findall(r"(?<!\d)(?:\d{2})?(0[1-9]|1[0-2])(?:[0-3]\d)(?!\d)", value)
        }
        months.update(
            int(month)
            for month in re.findall(r"\b(?:20)?\d{2}[-_.](1[0-2]|0[1-9])(?:[-_.]\d{1,2})?\b", value)
        )
        months.update(int(month) for month in re.findall(r"(?<!\d)(1[0-2]|0?[1-9])\s*월", value))
        return months

    def calendar_date_terms(self, query: str) -> set[str]:
        terms = set()

        for year, month, day in re.findall(r"(20\d{2}|\d{2})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일(?:자)?", query):
            terms.update(self.date_variants(year, int(month), int(day)))

        for year, month, day in re.findall(r"\b(20\d{2}|\d{2})[-_.](\d{1,2})[-_.](\d{1,2})\b", query):
            terms.update(self.date_variants(year, int(month), int(day)))

        for month, day in re.findall(r"(?<!\d)(\d{1,2})\s*월\s*(\d{1,2})\s*일(?:자)?", query):
            terms.update(self.date_variants(str(datetime.now().year), int(month), int(day)))

        for compact in re.findall(r"(?<!\d)(0[1-9]|1[0-2])([0-3]\d)(?!\d)", query):
            month, day = compact
            if 1 <= int(day) <= 31:
                terms.add(f"{month}{day}")

        return terms

    def week_terms(self, query: str) -> set[str]:
        terms = set()

        for year, month, week in re.findall(r"(20\d{2}|\d{2})[-_.\s년월]+(\d{1,2})[-_.\s월]+(\d{1,2})\s*(?:주차|w)", query, re.IGNORECASE):
            short_year = year[-2:]
            terms.update(self.week_variants(short_year, int(month), int(week)))

        for month, week in re.findall(r"(\d{1,2})\s*월\s*(\d{1,2})\s*주차", query):
            month_int = int(month)
            week_int = int(week)
            short_year = str(datetime.now().year)[-2:]
            terms.update(
                {
                    f"{month_int}월{week_int}주차",
                    f"{month_int:02d}월{week_int}주차",
                    f"{month_int}월{week_int}w",
                    f"{month_int:02d}월{week_int}w",
                }
            )
            terms.update(self.week_variants(short_year, month_int, week_int))

        return terms

    def date_variants(self, year: str, month: int, day: int) -> set[str]:
        if not 1 <= month <= 12 or not 1 <= day <= 31:
            return set()
        short_year = year[-2:]
        return {
            f"{month:02d}{day:02d}",
            f"{short_year}{month:02d}{day:02d}",
            f"20{short_year}{month:02d}{day:02d}",
        }

    def week_variants(self, short_year: str, month: int, week: int) -> set[str]:
        return {
            f"{short_year}{month:02d}{week}주차",
            f"{short_year}{month:02d}{week:02d}주차",
            f"{short_year}{month:02d}{week}w",
            f"{short_year}-{month:02d}-{week}주차",
            f"{short_year}-{month:02d}-{week:02d}주차",
            f"{short_year}-{month:02d}-{week}w",
        }


class KeywordRetriever:
    def __init__(self, client: QdrantClient, collection_name: str, limit: int = 3):
        self.client = client
        self.collection_name = collection_name
        self.limit = limit
        self.analyzer = QueryKeywordAnalyzer()

    def retrieve(self, query: str, limit: int | None = None) -> list[Document]:
        active_limit = self.positive_limit(limit, self.limit)
        tokens = self.analyzer.extract(query)
        exact_terms = tokens["exact"]
        broad_terms = tokens["broad"]
        requested_months = self.analyzer.requested_months(query)
        month_only_query = bool(requested_months) and not exact_terms
        if not exact_terms and not requested_months and not broad_terms:
            return []

        scanned_docs = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in points:
                payload = point.payload or {}
                metadata = payload.get("metadata") or {}
                page_content = payload.get("page_content")
                if not page_content:
                    continue

                title_raw = metadata.get("title", "")
                breadcrumb_raw = metadata.get("breadcrumb", "")
                title_text = self.analyzer.normalize(title_raw)
                breadcrumb_text = self.analyzer.normalize(breadcrumb_raw)
                content_text = self.analyzer.normalize(page_content)
                title_text += self.analyzer.normalize(" ".join(self.analyzer.temporal_terms(title_raw)))
                breadcrumb_text += self.analyzer.normalize(" ".join(self.analyzer.temporal_terms(breadcrumb_raw)))
                content_text += self.analyzer.normalize(" ".join(self.analyzer.temporal_terms(page_content)))
                searchable_text = f"{title_text}{breadcrumb_text}{content_text}"
                locator = f"{metadata.get('title', '')} {metadata.get('breadcrumb', '')}"
                month_match = bool(requested_months & self.analyzer.locator_months(locator))
                if exact_terms and not any(term in searchable_text for term in exact_terms):
                    continue
                if exact_terms and broad_terms and not any(term in searchable_text for term in broad_terms):
                    continue
                if month_only_query and not month_match:
                    continue

                scanned_docs.append(
                    {
                        "id": str(point.id),
                        "metadata": metadata,
                        "page_content": page_content,
                        "title": title_text,
                        "keyword": f"{breadcrumb_text}{self.analyzer.normalize(metadata.get('space_name', ''))}{self.analyzer.normalize(metadata.get('space', ''))}",
                        "content": content_text,
                        "month_match": month_match,
                    }
                )

            if offset is None:
                break

        query_terms = sorted(exact_terms | broad_terms)
        candidates = []
        for item in scanned_docs:
            bm25_score = (
                self.bm25_score(query_terms, item["title"], [doc["title"] for doc in scanned_docs]) * 3.0
                + self.bm25_score(query_terms, item["keyword"], [doc["keyword"] for doc in scanned_docs]) * 2.0
                + self.bm25_score(query_terms, item["content"], [doc["content"] for doc in scanned_docs])
            )
            title_match_score = self.title_match_score(query_terms, item["title"])
            if exact_terms and any(term in f"{item['title']}{item['keyword']}" for term in exact_terms):
                title_match_score = 1.0
            if item["month_match"]:
                bm25_score += 2.0
            if bm25_score <= 0 and title_match_score <= 0:
                continue
            metadata = {
                **item["metadata"],
                "_id": item["id"],
                "_collection_name": self.collection_name,
                "_bm25_score": bm25_score,
                "_title_match_score": title_match_score,
            }
            candidates.append((bm25_score + title_match_score, Document(page_content=item["page_content"], metadata=metadata)))

        ranked_candidates = candidates
        ranked_candidates.sort(key=lambda item: item[0], reverse=True)
        return self.dedupe_by_page([doc for _, doc in ranked_candidates])[:active_limit]

    def positive_limit(self, value, default: int) -> int:
        try:
            active = int(default if value is None else value)
        except (TypeError, ValueError):
            return max(1, int(default))
        return max(1, active)

    def bm25_score(self, query_terms: list[str], text: str, corpus_texts: list[str]) -> float:
        if not query_terms or not corpus_texts:
            return 0.0
        avg_len = sum(max(1, len(text)) for text in corpus_texts) / len(corpus_texts)
        doc_len = max(1, len(text))
        score = 0.0
        k1 = 1.2
        b = 0.75
        for term in query_terms:
            tf = text.count(term)
            if tf <= 0:
                continue
            docs_with_term = sum(1 for corpus_text in corpus_texts if term in corpus_text)
            idf = max(0.1, ((len(corpus_texts) - docs_with_term + 0.5) / (docs_with_term + 0.5)))
            score += idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_len))))
        return score

    def title_match_score(self, query_terms: list[str], title: str) -> float:
        if not query_terms:
            return 0.0
        matches = sum(1 for term in query_terms if term in title)
        return matches / len(query_terms)

    def dedupe_by_page(self, docs: list[Document]) -> list[Document]:
        results = []
        seen = set()
        for doc in docs:
            key = doc.metadata.get("page_id") or doc.metadata.get("_id")
            if key in seen:
                continue
            seen.add(key)
            results.append(doc)
        return results


class JiraStatusRetriever:
    STATUS_TERMS = {
        "todo": ("todo", "to do", "pending", "open", "해야", "할일", "예정", "대기"),
        "in_progress": ("inprogress", "in progress", "진행", "진행중", "작업중", "현재"),
        "done": ("done", "closed", "resolved", "완료", "끝난", "종료"),
    }
    INTENT_TERMS = ("jira", "이슈", "issue", "프로젝트", "진행", "진행중", "해야", "완료", "todo", "done", "현재")
    STATUS_QUERY_TERMS = ("jira", "이슈", "issue", "진행", "진행중", "해야", "완료", "todo", "done", "현재")

    def __init__(self, client: QdrantClient, collection_name: str, limit: int = 5):
        self.client = client
        self.collection_name = collection_name
        self.limit = limit
        self.analyzer = QueryKeywordAnalyzer()

    def should_retrieve(self, query: str) -> bool:
        normalized = self.analyzer.normalize(query)
        return any(self.analyzer.normalize(term) in normalized for term in self.INTENT_TERMS)

    def is_status_query(self, query: str) -> bool:
        normalized = self.analyzer.normalize(query)
        return any(self.analyzer.normalize(term) in normalized for term in self.STATUS_QUERY_TERMS)

    def requested_status_categories(self, query: str) -> set[str]:
        normalized = self.analyzer.normalize(query)
        categories = set()
        for category, terms in self.STATUS_TERMS.items():
            if any(self.analyzer.normalize(term) in normalized for term in terms):
                categories.add(category)
        return categories

    def retrieve(self, query: str, limit: int | None = None) -> list[Document]:
        if not self.should_retrieve(query):
            return []
        active_limit = self.positive_limit(limit, self.limit)

        tokens = self.analyzer.extract(query)
        query_terms = tokens["exact"] | tokens["broad"]
        status_categories = self.requested_status_categories(query)
        candidates = []
        offset = None

        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.source_type",
                            match=models.MatchValue(value="jira"),
                        )
                    ]
                ),
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in points:
                payload = point.payload or {}
                metadata = payload.get("metadata") or {}
                page_content = payload.get("page_content")
                if not page_content:
                    continue

                content_type = metadata.get("content_type")
                if content_type not in {"jira_issue", "jira_project_status_summary"}:
                    continue

                issue_status_category = metadata.get("issue_status_category")
                if status_categories and issue_status_category not in status_categories and content_type != "jira_project_status_summary":
                    continue

                title_text = self.analyzer.normalize(metadata.get("title", ""))
                breadcrumb_text = self.analyzer.normalize(metadata.get("breadcrumb", ""))
                space_text = self.analyzer.normalize(metadata.get("space", ""))
                space_name_text = self.analyzer.normalize(metadata.get("space_name", ""))
                content_text = self.analyzer.normalize(page_content)
                haystack = f"{title_text}{breadcrumb_text}{space_text}{space_name_text}{content_text}"

                term_matches = sum(1 for term in query_terms if term in haystack)
                if query_terms and term_matches == 0:
                    continue

                score = term_matches * 5
                if content_type == "jira_project_status_summary":
                    score += 10
                if status_categories and issue_status_category in status_categories:
                    score += 6

                candidates.append(
                    (
                        score,
                        Document(
                            page_content=page_content,
                            metadata={**metadata, "_id": str(point.id), "_collection_name": self.collection_name},
                        ),
                    )
                )

            if offset is None:
                break

        candidates.sort(key=lambda item: item[0], reverse=True)
        return self.dedupe_by_page([doc for _, doc in candidates])[:active_limit]

    def positive_limit(self, value, default: int) -> int:
        try:
            active = int(default if value is None else value)
        except (TypeError, ValueError):
            return max(1, int(default))
        return max(1, active)

    def dedupe_by_page(self, docs: list[Document]) -> list[Document]:
        results = []
        seen = set()
        for doc in docs:
            key = doc.metadata.get("page_id") or doc.metadata.get("_id")
            if key in seen:
                continue
            seen.add(key)
            results.append(doc)
        return results


class HybridRetriever:
    def __init__(self, vector_retriever, keyword_retriever: KeywordRetriever, jira_status_retriever: JiraStatusRetriever | None = None):
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.jira_status_retriever = jira_status_retriever

    def retrieve(self, query: str, profile=None) -> HybridRetrievalResult:
        vector_docs = self.retrieve_vector_docs(query, profile)

        keyword_limit = profile.keyword_top_k if profile is not None else None
        jira_limit = profile.jira_top_k if profile is not None else None
        keyword_docs = self.keyword_retriever.retrieve(query, limit=keyword_limit)
        jira_docs = self.jira_status_retriever.retrieve(query, limit=jira_limit) if self.jira_status_retriever else []
        if self.jira_status_retriever and self.jira_status_retriever.is_status_query(query) and not jira_docs:
            return HybridRetrievalResult(
                docs=[],
                vector_docs=vector_docs,
                keyword_docs=keyword_docs,
                jira_docs=jira_docs,
            )
        if jira_docs:
            return HybridRetrievalResult(
                docs=self.rerank_docs(query, jira_docs, keyword_docs, vector_docs),
                vector_docs=vector_docs,
                keyword_docs=keyword_docs,
                jira_docs=jira_docs,
            )
        if keyword_docs:
            return HybridRetrievalResult(
                docs=self.rerank_docs(query, [], keyword_docs, vector_docs),
                vector_docs=vector_docs,
                keyword_docs=keyword_docs,
                jira_docs=jira_docs,
            )

        return HybridRetrievalResult(
            docs=self.rerank_docs(query, [], [], vector_docs),
            vector_docs=vector_docs,
            keyword_docs=keyword_docs,
            jira_docs=jira_docs,
        )

    def retrieve_vector_docs(self, query: str, profile=None) -> list[Document]:
        if profile is None or not hasattr(self.vector_retriever, "search_kwargs"):
            return self.vector_retriever.invoke(query)

        search_kwargs = dict(getattr(self.vector_retriever, "search_kwargs", {}) or {})
        search_kwargs["k"] = self.positive_limit(getattr(profile, "vector_top_k", None), search_kwargs.get("k", 4))
        try:
            scoped_retriever = copy.copy(self.vector_retriever)
            scoped_retriever.search_kwargs = search_kwargs
            return scoped_retriever.invoke(query)
        except Exception as exc:
            print(f"[RAG] request-scoped vector retrieval fallback: {exc}")
            return self.vector_retriever.invoke(query)

    def positive_limit(self, value, default: int) -> int:
        try:
            active = int(default if value is None else value)
        except (TypeError, ValueError):
            return max(1, int(default))
        return max(1, active)

    def rerank_docs(
        self,
        query: str,
        jira_docs: list[Document],
        keyword_docs: list[Document],
        vector_docs: list[Document],
    ) -> list[Document]:
        all_docs = self.merge_docs(jira_docs + keyword_docs + vector_docs)
        vector_scores = self.rank_scores(vector_docs)
        keyword_scores = self.normalized_metadata_scores(keyword_docs, "_bm25_score")
        title_scores = self.normalized_title_scores(query, all_docs)
        rrf_scores = self.rrf_scores([jira_docs, keyword_docs, vector_docs])

        scored_docs = []
        for doc in all_docs:
            key = self.doc_key(doc)
            vector_score = vector_scores.get(key, 0.0)
            bm25_score = keyword_scores.get(key, doc.metadata.get("_bm25_score", 0.0))
            title_score = max(title_scores.get(key, 0.0), float(doc.metadata.get("_title_match_score", 0.0) or 0.0))
            final_score = 0.4 * vector_score + 0.4 * bm25_score + 0.2 * title_score
            if doc in jira_docs:
                final_score += 0.1
            metadata = {
                **doc.metadata,
                "_vector_score": vector_score,
                "_bm25_score": bm25_score,
                "_title_match_score": title_score,
                "_rrf_score": rrf_scores.get(key, 0.0),
                "_final_score": final_score,
            }
            scored_docs.append((final_score, rrf_scores.get(key, 0.0), Document(page_content=doc.page_content, metadata=metadata)))

        scored_docs.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [doc for _, _, doc in scored_docs]

    def rank_scores(self, docs: list[Document]) -> dict:
        scores = {}
        for rank, doc in enumerate(docs, start=1):
            scores[self.doc_key(doc)] = 1 / rank
        return scores

    def normalized_metadata_scores(self, docs: list[Document], field: str) -> dict:
        raw = {self.doc_key(doc): float(doc.metadata.get(field, 0.0) or 0.0) for doc in docs}
        max_score = max(raw.values(), default=0.0)
        if max_score <= 0:
            return raw
        return {key: value / max_score for key, value in raw.items()}

    def normalized_title_scores(self, query: str, docs: list[Document]) -> dict:
        analyzer = QueryKeywordAnalyzer()
        terms = analyzer.extract(query)["exact"] | analyzer.extract(query)["broad"]
        if not terms:
            return {}
        return {
            self.doc_key(doc): sum(1 for term in terms if term in analyzer.normalize(doc.metadata.get("title", ""))) / len(terms)
            for doc in docs
        }

    def rrf_scores(self, ranked_lists: list[list[Document]]) -> dict:
        scores = {}
        rrf_k = 60
        for docs in ranked_lists:
            for rank, doc in enumerate(docs, start=1):
                key = self.doc_key(doc)
                scores[key] = scores.get(key, 0.0) + 1 / (rrf_k + rank)
        return scores

    def merge_docs(self, docs: list[Document]) -> list[Document]:
        merged = []
        seen = set()
        for doc in docs:
            doc_key = self.doc_key(doc)
            if doc_key in seen:
                continue
            seen.add(doc_key)
            merged.append(doc)
        return merged

    def doc_key(self, doc: Document):
        return doc.metadata.get("_id") or (doc.metadata.get("page_id"), doc.metadata.get("chunk_index"), doc.metadata.get("url"))
