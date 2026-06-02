import asyncio
import json
import os
import re
import time
from datetime import datetime
from types import SimpleNamespace
from typing import AsyncGenerator

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models

from conversation_query import StructuredFollowupQueryResolver
from long_document import LongDocumentPolicy
from query_logging import QueryLogWriter
from query_routing import RetrievalRoutePlanner
from retrieval import HybridRetriever, JiraStatusRetriever, KeywordRetriever, QueryKeywordAnalyzer
from temporal_query import TemporalQueryPlanner


class RAGChain:
    def __init__(self):
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.qdrant_host = os.getenv("QDRANT_HOST", "http://localhost:6333")
        self.model_name = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:8b")
        self.top_k = int(os.getenv("RAG_TOP_K", "5"))
        self.expand_chunks = int(os.getenv("RAG_CONTEXT_EXPAND_CHUNKS", "20"))
        self.max_context_chars = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "18000"))
        self.num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
        self.aggregation_num_ctx = int(os.getenv("OLLAMA_AGGREGATION_NUM_CTX", "8192"))
        self.num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "8192"))
        self.reasoning = os.getenv("OLLAMA_REASONING", "false").lower() in {"1", "true", "yes", "on"}
        self.timeout_seconds = float(os.getenv("RAG_TIMEOUT_SECONDS", "45"))
        self.chat_history_turns = int(os.getenv("RAG_CHAT_HISTORY_TURNS", "8"))
        self.chat_history_max_chars = int(os.getenv("RAG_CHAT_HISTORY_MAX_CHARS", "4000"))
        self.conversation_hint_max_chars = int(os.getenv("RAG_CONVERSATION_HINT_MAX_CHARS", "1000"))
        self.prior_source_limit = int(os.getenv("RAG_PRIOR_SOURCE_LIMIT", "4"))
        self.prior_source_chunks = int(os.getenv("RAG_PRIOR_SOURCE_CHUNKS", "12"))
        self.route_planner = RetrievalRoutePlanner()
        self.query_log_writer = QueryLogWriter()

        self.llm = ChatOllama(
            base_url=self.ollama_host,
            model=self.model_name,
            temperature=0,
            num_ctx=self.num_ctx,
            num_predict=self.num_predict,
            reasoning=self.reasoning,
            keep_alive="24h",
        )

        self.embeddings = OllamaEmbeddings(base_url=self.ollama_host, model=self.embedding_model)
        self.client = QdrantClient(url=self.qdrant_host)
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name="confluence_docs",
            embedding=self.embeddings,
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": self.top_k})
        self.keyword_retriever = KeywordRetriever(
            client=self.client,
            collection_name="confluence_docs",
            limit=int(os.getenv("RAG_KEYWORD_TOP_K", "3")),
        )
        self.jira_status_retriever = JiraStatusRetriever(
            client=self.client,
            collection_name="confluence_docs",
            limit=int(os.getenv("RAG_JIRA_STATUS_TOP_K", "5")),
        )
        self.hybrid_retriever = HybridRetriever(self.retriever, self.keyword_retriever, self.jira_status_retriever)

        self.prompt = ChatPromptTemplate.from_template(
            """You are MetsaBrain, an internal knowledge assistant.
Answer in Korean using only the reference context.
If the context does not contain enough information, say that you could not find the information.
Use enough detail to be useful, but do not invent facts outside the context.
Include specific names, dates, paths, or values when the context contains them.
If a relevant document is present but only partial text is available, answer from the available text and clearly say which details are not in the indexed text.

[Conversation Context]
{conversation_context}

Use conversation context only to understand follow-up references, pronouns, and requested formatting.
Do not treat conversation context as factual source material when reference context does not support it.

[Reference Context]
{context}

[User Question]
{question}

[Answer]
"""
        )

    def format_docs(self, docs):
        parts = []
        used_chars = 0
        for doc in docs:
            content = doc.page_content
            remaining = self.max_context_chars - used_chars
            if remaining <= 0:
                break
            if len(content) > remaining:
                content = content[:remaining]
            parts.append(content)
            used_chars += len(content)
        return "\n\n---\n\n".join(parts)

    def compact_history_text(self, value: str, max_chars: int) -> str:
        text = re.sub(r"\s+", " ", value or "").strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "..."

    def normalize_chat_history(self, history: list[dict] | None) -> list[dict]:
        normalized = []
        history_turns = getattr(self, "chat_history_turns", 8)
        history_max_chars = getattr(self, "chat_history_max_chars", 4000)
        for item in (history or [])[-history_turns:]:
            role = item.get("role")
            if role not in {"user", "assistant"}:
                continue
            content_limit = min(900 if role == "assistant" else 700, max(80, history_max_chars // 2))
            content = self.compact_history_text(item.get("content", ""), content_limit)
            if content:
                normalized.append({"role": role, "content": content})

        used_chars = 0
        bounded = []
        for item in reversed(normalized):
            remaining = history_max_chars - used_chars
            if remaining <= 0:
                break
            content = self.compact_history_text(item["content"], remaining)
            bounded.append({"role": item["role"], "content": content})
            used_chars += len(content)
        return list(reversed(bounded))

    def format_conversation_hint(self, history: list[dict]) -> str:
        if not history:
            return "No prior conversation context."
        hint_max_chars = getattr(self, "conversation_hint_max_chars", 1000)
        labels = {"user": "User", "assistant": "Assistant summary"}
        lines = [f"{labels[item['role']]}: {item['content']}" for item in history]
        return self.compact_history_text("\n".join(lines), hint_max_chars)

    def build_retrieval_query(self, query: str, history: list[dict]) -> str:
        return StructuredFollowupQueryResolver().resolve(query, history).retrieval_query

    def prior_source_anchor_texts(self, sources: list[dict] | None) -> list[str]:
        texts = []
        for source in sources or []:
            parts = [
                source.get("title"),
                source.get("breadcrumb"),
                source.get("space_name"),
                source.get("space"),
            ]
            text = " ".join(str(part) for part in parts if part)
            if text:
                texts.append(text)
        return texts

    def retrieve_context(
        self,
        query: str,
        mentions: list[dict] | None = None,
        history: list[dict] | None = None,
        prior_sources: list[dict] | None = None,
    ):
        retrieval_start = time.perf_counter()
        chat_history = self.normalize_chat_history(history)
        conversation_context = self.format_conversation_hint(chat_history)
        query_resolution = StructuredFollowupQueryResolver().resolve(
            query,
            chat_history,
            source_texts=self.prior_source_anchor_texts(prior_sources),
        )
        retrieval_query = query_resolution.retrieval_query
        inline_mentions = self.resolve_query_mentions(retrieval_query)
        inferred_mentions = self.infer_jira_summary_mentions(retrieval_query)
        resolved_mentions = self.merge_mentions(mentions, inline_mentions + inferred_mentions)
        route_planner = getattr(self, "route_planner", None)
        if route_planner is None:
            route_planner = RetrievalRoutePlanner()
            self.route_planner = route_planner
        route = route_planner.plan(retrieval_query, resolved_mentions)
        profile = route.profile
        reusable_sources = [] if query_resolution.scope_changed else prior_sources
        prior_docs = self.prior_source_docs(reusable_sources, query, resolved_mentions)
        temporal_plan = TemporalQueryPlanner().plan(retrieval_query)
        retrieval_result = self.retrieve_temporal_plan(temporal_plan, profile=profile)
        scoped_retrieval_docs = self.filter_docs_to_mention_scope(
            retrieval_result.docs,
            resolved_mentions,
        )
        jira_scope_summary_docs = self.build_jira_scope_summary_docs(
            resolved_mentions,
            retrieval_query,
        )
        mention_docs = self.mention_context_docs(resolved_mentions, retrieval_query)
        has_month_scoped_confluence = bool(
            mention_docs
            and self.requested_months(retrieval_query)
            and any(mention.get("source_type") == "confluence" for mention in resolved_mentions)
        )
        retrieval_docs = mention_docs
        if not has_month_scoped_confluence:
            retrieval_docs += scoped_retrieval_docs
        expanded_retrieval_docs = self.expand_retrieved_docs(retrieval_docs, profile=profile)
        long_policy = LongDocumentPolicy(
            neighbor_window=profile.long_doc_neighbor_window,
            detail_chunks=profile.long_doc_detail_chunks,
            summary_window_chunks=profile.long_doc_summary_window_chunks,
        )
        selected_retrieval_docs, long_document_stats = long_policy.apply(
            route.name,
            expanded_retrieval_docs,
            retrieval_query,
        )
        docs = prior_docs + jira_scope_summary_docs + selected_retrieval_docs
        return SimpleNamespace(
            docs=docs,
            route=route,
            long_document_stats=long_document_stats,
            conversation_context=conversation_context,
            query_resolution=query_resolution,
            retrieval_query=retrieval_query,
            resolved_mentions=resolved_mentions,
            retrieval_result=retrieval_result,
            temporal_plan=temporal_plan,
            prior_docs=prior_docs,
            jira_scope_summary_docs=jira_scope_summary_docs,
            mention_docs=mention_docs,
            has_month_scoped_confluence=has_month_scoped_confluence,
            retrieval_elapsed=time.perf_counter() - retrieval_start,
        )

    def retrieve_temporal_plan(self, temporal_plan, profile=None):
        if not temporal_plan.is_multi_bucket:
            return self.hybrid_retriever.retrieve(temporal_plan.buckets[0].query, profile=profile)

        docs = []
        vector_docs = []
        keyword_docs = []
        jira_docs = []
        seen = set()
        per_bucket_limit = 2
        for bucket in temporal_plan.buckets:
            result = self.hybrid_retriever.retrieve(bucket.query, profile=profile)
            vector_docs.extend(result.vector_docs)
            keyword_docs.extend(result.keyword_docs)
            jira_docs.extend(result.jira_docs)
            for doc in result.docs[:per_bucket_limit]:
                key = doc.metadata.get("_id") or (
                    doc.metadata.get("page_id"),
                    doc.metadata.get("chunk_index"),
                    doc.metadata.get("url"),
                )
                if key in seen:
                    continue
                seen.add(key)
                docs.append(
                    Document(
                        page_content=f"[Target: {bucket.label}]\n{doc.page_content}",
                        metadata={**doc.metadata, "_temporal_bucket": bucket.label},
                    )
                )

        return SimpleNamespace(
            docs=docs,
            vector_docs=vector_docs,
            keyword_docs=keyword_docs,
            jira_docs=jira_docs,
        )

    def has_standalone_context_anchor(self, query: str) -> bool:
        normalized = re.sub(r"\s+", "", query.lower())
        has_period = bool(
            re.search(r"\d{2,4}\s*년|\d{1,2}\s*월|\d{1,2}\s*주차|\d{2}-\d{2}", query)
        )
        domain_terms = [
            "주간보고",
            "회의록",
            "프로젝트",
            "이슈",
            "문서",
            "점검",
            "서버",
            "목록",
            "정보",
        ]
        has_domain_term = any(term in normalized for term in domain_terms)
        has_korean_name_like_token = bool(
            re.search(r"(?<![가-힣])[가-힣]{2,4}(?:님|의|이야|은|는|팀장|팀원)?", query)
        )
        return has_period or (has_domain_term and has_korean_name_like_token)

    def should_reuse_prior_sources(self, query: str, mentions: list[dict] | None) -> bool:
        if mentions:
            return False
        if re.search(r"@[^\s@]+", query):
            return False
        normalized = re.sub(r"\s+", "", query.lower())
        standalone_anchor = self.has_standalone_context_anchor(query)
        reference_markers = [
            "그거",
            "그것",
            "위내용",
            "방금",
            "아까",
            "이내용",
            "해당내용",
            "앞서",
            "이전",
        ]
        expansion_markers = [
            "다른",
            "나머지",
            "그외",
            "추가로",
            "또",
        ]
        formatting_markers = [
            "표로",
            "다시",
            "자세히",
            "간단히",
            "요약",
            "정리",
        ]
        if any(marker in normalized for marker in reference_markers + expansion_markers):
            return True
        if any(marker in normalized for marker in formatting_markers):
            return not standalone_anchor and len(query.strip()) <= 20
        return len(query.strip()) <= 20 and not standalone_anchor

    def prior_source_filter(self, source: dict):
        must = []
        should = []

        if source.get("page_id"):
            should.append(
                models.FieldCondition(
                    key="metadata.page_id",
                    match=models.MatchValue(value=source["page_id"]),
                )
            )
        if source.get("url"):
            should.append(
                models.FieldCondition(
                    key="metadata.url",
                    match=models.MatchValue(value=source["url"]),
                )
            )
        if source.get("source_type"):
            must.append(
                models.FieldCondition(
                    key="metadata.source_type",
                    match=models.MatchValue(value=source["source_type"]),
                )
            )
        if source.get("space"):
            must.append(
                models.FieldCondition(
                    key="metadata.space",
                    match=models.MatchValue(value=source["space"]),
                )
            )
        if source.get("title") and not should:
            must.append(
                models.FieldCondition(
                    key="metadata.title",
                    match=models.MatchValue(value=source["title"]),
                )
            )
        if should:
            return models.Filter(must=must, should=should)
        if must:
            return models.Filter(must=must)
        return None

    def prior_source_docs(self, sources: list[dict] | None, query: str, mentions: list[dict] | None):
        if not sources or not self.should_reuse_prior_sources(query, mentions):
            return []

        docs = []
        seen = set()
        source_limit = getattr(self, "prior_source_limit", 4)
        source_chunks = getattr(self, "prior_source_chunks", 12)
        for source in list(sources)[-source_limit:]:
            source_filter = self.prior_source_filter(source)
            if source_filter is None:
                continue
            try:
                points, _ = self.client.scroll(
                    collection_name="confluence_docs",
                    scroll_filter=source_filter,
                    limit=max(1, source_chunks),
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as exc:
                print(f"[RAG] prior source reuse failed: {exc}")
                continue

            for point in points:
                payload = point.payload or {}
                metadata = payload.get("metadata") or {}
                content = payload.get("page_content") or ""
                doc_key = metadata.get("page_id") or metadata.get("url") or point.id
                chunk_key = (doc_key, metadata.get("chunk_index"))
                if not content or chunk_key in seen:
                    continue
                seen.add(chunk_key)
                docs.append(Document(page_content=content, metadata=metadata))

        docs.sort(key=lambda doc: (doc.metadata.get("page_id") or "", doc.metadata.get("chunk_index", 0)))
        return docs

    def expand_retrieved_docs(self, docs, profile=None):
        expand_limit = getattr(profile, "expand_chunks", self.expand_chunks)
        expanded = []
        seen = set()

        for doc in docs:
            page_id = doc.metadata.get("page_id")
            if not page_id:
                doc_key = doc.metadata.get("_id") or (doc.metadata.get("url"), doc.metadata.get("chunk_index"))
                if doc_key not in seen:
                    expanded.append(doc)
                    seen.add(doc_key)
                continue

            source_type = doc.metadata.get("source_type")
            filters = [
                models.FieldCondition(
                    key="metadata.page_id",
                    match=models.MatchValue(value=page_id),
                )
            ]
            if source_type:
                filters.append(
                    models.FieldCondition(
                        key="metadata.source_type",
                        match=models.MatchValue(value=source_type),
                    )
                )

            try:
                points = []
                offset = None
                while len(points) < max(expand_limit, 1):
                    batch, offset = self.client.scroll(
                        collection_name="confluence_docs",
                        scroll_filter=models.Filter(must=filters),
                        limit=min(max(expand_limit - len(points), 1), 64),
                        offset=offset,
                        with_payload=True,
                        with_vectors=False,
                    )
                    points.extend(batch)
                    if offset is None:
                        break
            except Exception as e:
                print(f"[RAG] sibling chunk expansion failed for {page_id}: {e}")
                points = []

            sibling_docs = []
            for point in points:
                payload = point.payload or {}
                metadata = payload.get("metadata") or {}
                page_content = payload.get("page_content")
                if not page_content:
                    continue

                metadata = {**metadata, "_id": str(point.id), "_collection_name": "confluence_docs"}
                sibling_docs.append(Document(page_content=page_content, metadata=metadata))

            if not sibling_docs:
                sibling_docs = [doc]

            sibling_docs.sort(key=lambda item: item.metadata.get("chunk_index", 0))
            for sibling in sibling_docs:
                doc_key = sibling.metadata.get("_id") or (sibling.metadata.get("page_id"), sibling.metadata.get("chunk_index"))
                if doc_key in seen:
                    continue
                expanded.append(sibling)
                seen.add(doc_key)

        return expanded

    def mention_keyword_score(self, query: str, doc: Document):
        tokens = QueryKeywordAnalyzer().extract(query)
        query_terms = sorted(tokens["exact"] | tokens["broad"])
        if not query_terms:
            return 0

        metadata = doc.metadata
        title = metadata.get("title") or ""
        breadcrumb = metadata.get("breadcrumb") or ""
        title_text = self.normalize_mention_text(title)
        breadcrumb_text = self.normalize_mention_text(breadcrumb)
        content_text = self.normalize_mention_text(doc.page_content[:4000])

        score = 0
        for term in query_terms:
            if not term:
                continue
            if term in title_text:
                score += 3
            if term in breadcrumb_text:
                score += 2
            if term in content_text:
                score += 1
        if self.document_matches_requested_month(doc, self.requested_months(query)):
            score += 10
        if metadata.get("content_type") == "jira_project_status_summary":
            score += 2
        return score

    def normalize_mention_text(self, value: str | None):
        return re.sub(r"[^0-9a-zA-Z가-힣]+", "", value or "").lower()

    def mention_scope_key(self, mention: dict):
        source_type = mention.get("source_type")
        space = mention.get("space")
        if source_type and space:
            return f"{source_type}:{space}"
        return None

    def extract_query_mentions(self, query: str):
        return [token.strip() for token in re.findall(r"@([^\s@]+)", query) if token.strip()]

    def resolve_query_mentions(self, query: str):
        tokens = {self.normalize_mention_text(token) for token in self.extract_query_mentions(query)}
        tokens = {token for token in tokens if token}
        if not tokens:
            return []

        resolved = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name="confluence_docs",
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in points:
                payload = point.payload or {}
                metadata = payload.get("metadata") or {}
                source_type = metadata.get("source_type", "confluence")
                space = metadata.get("space")
                if not space:
                    continue

                space_name = metadata.get("space_name") or space
                normalized_values = {
                    self.normalize_mention_text(space),
                    self.normalize_mention_text(space_name),
                }
                if tokens.isdisjoint(normalized_values):
                    continue

                key = f"{source_type}:{space}"
                resolved[key] = {
                    "mention_type": "space",
                    "source_type": source_type,
                    "space": space,
                    "space_name": space_name,
                    "title": space_name,
                    "content_type": "space",
                }

            if offset is None:
                break

        if resolved:
            print(f"[RAG] resolved inline mentions: {list(resolved.keys())}")
        return list(resolved.values())

    def merge_mentions(self, explicit_mentions: list[dict] | None, inline_mentions: list[dict]):
        merged = {}
        for mention in (explicit_mentions or []) + inline_mentions:
            key = self.mention_scope_key(mention)
            if key:
                merged[key] = mention
        return list(merged.values())

    def filter_docs_to_mention_scope(self, docs: list[Document], mentions: list[dict]):
        scope_keys = {self.mention_scope_key(mention) for mention in mentions}
        scope_keys.discard(None)
        if not scope_keys:
            return docs
        return [
            doc
            for doc in docs
            if f"{doc.metadata.get('source_type', 'confluence')}:{doc.metadata.get('space')}" in scope_keys
        ]

    def requested_months(self, query: str):
        months = {int(month) for month in re.findall(r"(?<!\d)(1[0-2]|0?[1-9])\s*월", query)}
        months.update(int(month) for month in re.findall(r"\b(?:20)?\d{2}[-_.](1[0-2]|0[1-9])\b", query))
        return months

    def document_matches_requested_month(self, doc: Document, months: set[int]):
        if not months:
            return False

        metadata = doc.metadata
        locator = f"{metadata.get('title') or ''} {metadata.get('breadcrumb') or ''}"
        document_months = {
            int(month)
            for month in re.findall(r"(?<!\d)(?:\d{2})?(0[1-9]|1[0-2])(?:[0-3]\d)(?!\d)", locator)
        }
        document_months.update(
            int(month) for month in re.findall(r"\b(?:20)?\d{2}[-_.](1[0-2]|0[1-9])(?:[-_.]\d{1,2})?\b", locator)
        )
        document_months.update(int(month) for month in re.findall(r"(?<!\d)(1[0-2]|0?[1-9])\s*월", locator))
        return bool(months & document_months)

    def filter_month_scoped_mention_docs(self, query: str, docs: list[Document]):
        months = self.requested_months(query)
        if not months:
            return docs

        monthly_docs = [doc for doc in docs if self.document_matches_requested_month(doc, months)]
        if not monthly_docs:
            return docs

        normalized_query = self.normalize_mention_text(query)
        if "기술회의" in normalized_query:
            technical_meeting_docs = [
                doc
                for doc in monthly_docs
                if "기술회의" in self.normalize_mention_text(
                    f"{doc.metadata.get('title') or ''} {doc.metadata.get('breadcrumb') or ''}"
                )
            ]
            if technical_meeting_docs:
                monthly_docs = technical_meeting_docs

        structural_terms = QueryKeywordAnalyzer().extract(query)["broad"]
        if not structural_terms:
            return monthly_docs

        def structural_score(doc):
            metadata = doc.metadata
            locator = self.normalize_mention_text(
                f"{metadata.get('title') or ''} {metadata.get('breadcrumb') or ''}"
            )
            return sum(1 for term in structural_terms if term in locator)

        best_score = max(structural_score(doc) for doc in monthly_docs)
        if best_score == 0:
            return monthly_docs
        return monthly_docs

    def mention_month_sort_key(self, doc: Document):
        locator = f"{doc.metadata.get('title') or ''} {doc.metadata.get('breadcrumb') or ''}"
        candidates = []
        for month, day in re.findall(r"(?<!\d)(?:\d{2})?(0[1-9]|1[0-2])([0-3]\d)(?!\d)", locator):
            candidates.append((int(month), int(day)))
        for month, day in re.findall(r"\b(?:20)?\d{2}[-_.](1[0-2]|0[1-9])[-_.](\d{1,2})\b", locator):
            candidates.append((int(month), int(day)))
        return min(candidates) if candidates else (99, 99)

    def requested_jira_status_categories(self, query: str):
        normalized = self.normalize_mention_text(query)
        categories = set()
        past_activity = any(term in normalized for term in ("진행했던", "진행한", "했던", "수행한"))
        if any(term in normalized for term in ("진행예정", "예정", "해야", "할일", "todo", "pending", "open")):
            categories.add("todo")
        if any(term in normalized for term in ("진행중", "작업중", "inprogress")):
            categories.add("in_progress")
        if any(term in normalized for term in ("완료", "종료", "done", "closed", "resolved")):
            categories.add("done")
        if not categories and not past_activity and "진행" in normalized:
            categories.add("in_progress")
        return categories

    def requested_jira_status_categories_legacy(self, query: str):
        normalized = self.normalize_mention_text(query)
        categories = set()
        if any(term in normalized for term in ("해야", "할일", "예정", "todo", "pending", "open")):
            categories.add("todo")
        if any(term in normalized for term in ("진행", "진행중", "작업중", "inprogress")):
            categories.add("in_progress")
        if any(term in normalized for term in ("완료", "종료", "done", "closed", "resolved")):
            categories.add("done")
        return categories

    def jira_issue_dates(self, metadata: dict):
        values = []
        for key in ("due_date", "updated", "created"):
            value = metadata.get(key)
            if value:
                values.append(str(value))
        for value in (metadata.get("jira_date_fields") or {}).values():
            if value:
                values.append(str(value))
        return values

    def content_field(self, page_content: str, field_name: str):
        match = re.search(rf"^{re.escape(field_name)}:\s*(.+)$", page_content or "", re.MULTILINE)
        return match.group(1).strip() if match else ""

    def infer_issue_status_category(self, metadata: dict, page_content: str):
        category = metadata.get("issue_status_category")
        if category:
            return category
        status_text = self.normalize_mention_text(
            f"{metadata.get('issue_status') or ''} {self.content_field(page_content, 'Status')}"
        )
        if any(term in status_text for term in ("완료", "done", "closed", "resolved")):
            return "done"
        if any(term in status_text for term in ("해야", "할일", "todo", "pending", "open")):
            return "todo"
        if any(term in status_text for term in ("진행", "inprogress")):
            return "in_progress"
        return category

    def issue_date_values(self, metadata: dict, page_content: str):
        values = self.jira_issue_dates(metadata)
        for field_name in ("Updated", "Created", "Due date", "Date"):
            value = self.content_field(page_content, field_name)
            if value:
                values.append(value)
        values.append(metadata.get("title") or "")
        return values

    def date_matches_month(self, values: list[str], months: set[int]):
        if not months:
            return True
        for value in values:
            match = re.search(r"\b(?:20)?\d{2}[-_.](1[0-2]|0[1-9])[-_.]\d{1,2}\b", value)
            if match and int(match.group(1)) in months:
                return True
            title_month_match = re.search(r"\b\d{2}[-_.](1[0-2]|0[1-9])\b", value)
            if title_month_match and int(title_month_match.group(1)) in months:
                return True
        return False

    def issue_matches_jira_summary_query(self, metadata: dict, page_content: str, query: str, months: set[int], status_categories: set[str], require_maintenance: bool):
        title = metadata.get("title") or ""
        if require_maintenance and "점검" not in f"{title} {page_content}":
            return False
        issue_status_category = self.infer_issue_status_category(metadata, page_content)
        if status_categories and issue_status_category not in status_categories:
            return False
        return self.date_matches_month(self.issue_date_values(metadata, page_content), months)

    def infer_jira_summary_mentions(self, query: str):
        months = self.requested_months(query)
        status_categories = self.requested_jira_status_categories(query)
        normalized_query = self.normalize_mention_text(query)
        require_maintenance = "점검" in normalized_query
        mentions_maintenance = "유지보수" in normalized_query or "maintenance" in normalized_query
        mentions_jira = "jira" in normalized_query or "이슈" in normalized_query or "issue" in normalized_query
        if not require_maintenance and not mentions_maintenance and not status_categories and not mentions_jira:
            return []
        if not months and not status_categories and not require_maintenance and not mentions_maintenance:
            return []

        candidates = {}
        matched_issues = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name="confluence_docs",
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.source_type",
                            match=models.MatchValue(value="jira"),
                        ),
                        models.FieldCondition(
                            key="metadata.content_type",
                            match=models.MatchValue(value="jira_issue"),
                        ),
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
                space = metadata.get("space")
                if not space:
                    continue
                page_content = payload.get("page_content") or ""
                if not self.issue_matches_jira_summary_query(metadata, page_content, query, months, status_categories, require_maintenance):
                    continue

                space_name = metadata.get("space_name") or space
                normalized_space_name = self.normalize_mention_text(space_name)
                normalized_space = self.normalize_mention_text(space)
                explicit_space_match = normalized_space in normalized_query or normalized_space_name in normalized_query
                if mentions_maintenance and "maintenance" not in normalized_space_name and "maint" not in normalized_space_name:
                    continue

                key = f"jira:{space}"
                issue_key = metadata.get("issue_key") or metadata.get("page_id") or str(point.id)
                match_key = (key, issue_key)
                if match_key in matched_issues:
                    continue
                matched_issues.add(match_key)
                current = candidates.setdefault(
                    key,
                    {
                        "mention_type": "space",
                        "source_type": "jira",
                        "space": space,
                        "space_name": space_name,
                        "title": space_name,
                        "content_type": "space",
                        "_score": 0,
                    },
                )
                current["_score"] += 1
                if explicit_space_match:
                    current["_score"] += 100
                    current["_explicit"] = True
                if mentions_maintenance and ("maintenance" in normalized_space_name or "maint" in normalized_space_name):
                    current["_score"] += 5

            if offset is None:
                break

        ranked_candidates = sorted(candidates.values(), key=lambda item: item["_score"], reverse=True)
        explicit_candidates = [item for item in ranked_candidates if item.get("_explicit")]
        inferred = (explicit_candidates or ranked_candidates)[:3]
        for mention in inferred:
            mention.pop("_score", None)
            mention.pop("_explicit", None)
        if inferred:
            print(f"[RAG] inferred Jira summary scopes: {[self.mention_scope_key(mention) for mention in inferred]}")
        return inferred

    def build_jira_scope_summary_docs(self, mentions: list[dict], query: str):
        jira_mentions = [mention for mention in mentions if mention.get("source_type") == "jira" and mention.get("space")]
        if not jira_mentions:
            return []

        months = self.requested_months(query)
        status_categories = self.requested_jira_status_categories(query)
        normalized_query = self.normalize_mention_text(query)
        require_maintenance = "점검" in normalized_query
        if not months and not status_categories and not require_maintenance:
            return []

        summary_docs = []
        for mention in jira_mentions:
            space = mention.get("space")
            space_name = mention.get("space_name") or space
            issues_by_key = {}
            offset = None

            while True:
                points, offset = self.client.scroll(
                    collection_name="confluence_docs",
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="metadata.source_type",
                                match=models.MatchValue(value="jira"),
                            ),
                            models.FieldCondition(
                                key="metadata.space",
                                match=models.MatchValue(value=space),
                            ),
                            models.FieldCondition(
                                key="metadata.content_type",
                                match=models.MatchValue(value="jira_issue"),
                            ),
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
                    page_content = payload.get("page_content") or ""
                    title = metadata.get("title") or ""
                    if not self.issue_matches_jira_summary_query(metadata, page_content, query, months, status_categories, require_maintenance):
                        continue

                    issue_status_category = self.infer_issue_status_category(metadata, page_content)
                    issue_date = (
                        metadata.get("due_date")
                        or next(iter((metadata.get("jira_date_fields") or {}).values()), "")
                        or self.content_field(page_content, "Updated")
                        or self.content_field(page_content, "Created")
                    )
                    issue_key = metadata.get("issue_key") or metadata.get("page_id") or title or str(point.id)
                    issues_by_key.setdefault(
                        issue_key,
                        {
                            "key": issue_key,
                            "title": title,
                            "status": metadata.get("issue_status") or self.content_field(page_content, "Status") or issue_status_category or "",
                            "status_category": issue_status_category or "",
                            "assignee": metadata.get("assignee") or self.content_field(page_content, "Assignee") or "unassigned",
                            "date": issue_date,
                            "updated": metadata.get("updated") or self.content_field(page_content, "Updated") or "",
                            "url": metadata.get("url") or "",
                        }
                    )

                if offset is None:
                    break

            issues = list(issues_by_key.values())
            if not issues:
                continue

            issues.sort(key=lambda item: (item["date"] or "9999-99-99", item["key"] or ""))
            lines = [
                f"Jira project scoped issue summary for {space_name} ({space}).",
                f"Query: {query}",
                f"Matched issue count: {len(issues)}",
                "",
                "| Issue | Title | Status | Assignee | Date | Updated |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
            for issue in issues:
                lines.append(
                    f"| {issue['key']} | {issue['title']} | {issue['status']} ({issue['status_category']}) | {issue['assignee']} | {issue['date']} | {issue['updated']} |"
                )

            summary_docs.append(
                Document(
                    page_content="\n".join(lines),
                    metadata={
                        "title": f"{space_name} Jira 조건 검색 요약",
                        "source_type": "jira",
                        "space": space,
                        "space_name": space_name,
                        "content_type": "jira_mention_scope_summary",
                        "breadcrumb": f"{space_name} > Jira 조건 검색 요약",
                        "generated_at": datetime.now().isoformat(timespec="seconds"),
                    },
                )
            )

        return summary_docs

    def mention_context_docs(self, mentions: list[dict] | None, query: str = ""):
        if not mentions:
            return []

        docs = []
        seen = set()
        has_space_mention = False
        for mention in mentions:
            source_type = mention.get("source_type")
            mention_type = mention.get("mention_type")
            page_id = mention.get("page_id")
            space = mention.get("space")
            filters = []

            if source_type:
                filters.append(
                    models.FieldCondition(
                        key="metadata.source_type",
                        match=models.MatchValue(value=source_type),
                    )
                )
            is_space_mention = mention_type == "space" and space
            has_space_mention = has_space_mention or bool(is_space_mention)
            if mention_type == "document" and page_id:
                filters.append(
                    models.FieldCondition(
                        key="metadata.page_id",
                        match=models.MatchValue(value=page_id),
                    )
                )
            elif is_space_mention:
                filters.append(
                    models.FieldCondition(
                        key="metadata.space",
                        match=models.MatchValue(value=space),
                    )
                )
            else:
                continue

            try:
                points = []
                offset = None
                while True:
                    batch, offset = self.client.scroll(
                        collection_name="confluence_docs",
                        scroll_filter=models.Filter(must=filters),
                        limit=256 if is_space_mention else max(self.expand_chunks, 20),
                        offset=offset,
                        with_payload=True,
                        with_vectors=False,
                    )
                    points.extend(batch)
                    if not is_space_mention or offset is None:
                        break
            except Exception as e:
                print(f"[RAG] mention context lookup failed for {mention}: {e}")
                continue

            for point in points:
                payload = point.payload or {}
                metadata = payload.get("metadata") or {}
                page_content = payload.get("page_content")
                if not page_content:
                    continue
                doc_key = str(point.id)
                if doc_key in seen:
                    continue
                seen.add(doc_key)
                docs.append(
                    Document(
                        page_content=page_content,
                        metadata={**metadata, "_id": doc_key, "_collection_name": "confluence_docs"},
                    )
                )

        if query and has_space_mention:
            months = self.requested_months(query)
            docs = self.filter_month_scoped_mention_docs(query, docs)
            if months:
                docs.sort(
                    key=lambda doc: (
                        self.mention_month_sort_key(doc),
                        -self.mention_keyword_score(query, doc),
                    )
                )
            else:
                docs.sort(
                    key=lambda doc: (
                        self.mention_keyword_score(query, doc),
                        1 if doc.metadata.get("content_type") == "jira_project_status_summary" else 0,
                    ),
                    reverse=True,
                )
            docs = docs[: max(self.expand_chunks, 20)]
        else:
            docs.sort(
                key=lambda doc: (
                    0 if doc.metadata.get("content_type") == "jira_project_status_summary" else 1,
                    doc.metadata.get("page_id") or "",
                    doc.metadata.get("chunk_index", 0),
                )
            )
        return docs

    def write_chat_query_log(
        self,
        query: str,
        start_time: float,
        context_result=None,
        docs: list[Document] | None = None,
        unique_sources: list[dict] | None = None,
        context_text: str = "",
        finish_reason: str | None = None,
        generation_seconds: float = 0.0,
        answer_chars: int = 0,
        error: str | None = None,
    ):
        query_log_writer = getattr(self, "query_log_writer", None)
        if query_log_writer is None:
            query_log_writer = QueryLogWriter()
            self.query_log_writer = query_log_writer

        docs = docs or []
        unique_sources = unique_sources or []
        record = {
            "question": query,
            "question_chars": len(query or ""),
            "retrieved_docs": len(docs),
            "prompt_chunks": len(docs),
            "prompt_chars": len(context_text or ""),
            "source_count": len(unique_sources),
            "finish_reason": finish_reason or "end",
            "generation_seconds": round(generation_seconds, 3),
            "total_seconds": round(time.perf_counter() - start_time, 3),
            "answer_chars": answer_chars,
        }
        if error:
            record["error"] = error
        if context_result is not None:
            route = context_result.route
            record.update(
                {
                    "mentions": context_result.resolved_mentions,
                    "route": route.name,
                    "route_reason": route.reason,
                    "vector_top_k": route.profile.vector_top_k,
                    "keyword_top_k": route.profile.keyword_top_k,
                    "jira_top_k": route.profile.jira_top_k,
                    "expand_chunks": route.profile.expand_chunks,
                    "max_context_chars": route.profile.max_context_chars,
                    "retrieval_seconds": round(context_result.retrieval_elapsed, 3),
                    **context_result.long_document_stats,
                }
            )
        query_log_writer.write(record)

    async def stream_ask(
        self,
        query: str,
        mentions: list[dict] | None = None,
        history: list[dict] | None = None,
        prior_sources: list[dict] | None = None,
    ) -> AsyncGenerator[str, None]:
        print(f"\n[STREAM] Request Start: chars={len(query or '')}")
        start_time = time.perf_counter()
        context_result = None
        docs = []
        unique_sources = []
        context_text = ""
        generation_elapsed = 0.0
        answer_chars = 0

        try:
            yield json.dumps({"type": "status", "content": "관련 문서를 찾는 중입니다"}) + "\n\n"

            context_result = await asyncio.to_thread(
                self.retrieve_context,
                query,
                mentions,
                history,
                prior_sources,
            )
            docs = context_result.docs
            conversation_context = context_result.conversation_context
            retrieval_query = context_result.retrieval_query
            resolved_mentions = context_result.resolved_mentions
            retrieval_result = context_result.retrieval_result
            has_month_scoped_confluence = context_result.has_month_scoped_confluence
            route = context_result.route
            print(f"[TIMING] retrieval={context_result.retrieval_elapsed:.2f}s docs={len(docs)} mention_scopes={len(resolved_mentions)} jira_scope_summary_docs={len(context_result.jira_scope_summary_docs)} mention_docs={len(context_result.mention_docs)} prior_docs={len(context_result.prior_docs)} vector_docs={len(retrieval_result.vector_docs)} keyword_docs={len(retrieval_result.keyword_docs)} jira_docs={len(retrieval_result.jira_docs)} route={route.name} top_k={self.top_k} expand_chunks={self.expand_chunks}")

            unique_sources = []
            seen_urls = set()
            for doc in docs:
                url = doc.metadata.get("url")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    unique_sources.append(
                        {
                            "title": doc.metadata.get("title"),
                            "url": url,
                            "page_id": doc.metadata.get("page_id"),
                            "breadcrumb": doc.metadata.get("breadcrumb"),
                            "content_type": doc.metadata.get("content_type"),
                            "source_type": doc.metadata.get("source_type", "confluence"),
                            "space": doc.metadata.get("space"),
                            "space_name": doc.metadata.get("space_name"),
                        }
                    )

            yield json.dumps({"type": "sources", "content": unique_sources}) + "\n\n"

            if not docs:
                yield json.dumps({"type": "answer", "content": "관련 정보를 찾을 수 없습니다."}) + "\n\n"
                self.write_chat_query_log(
                    query,
                    start_time,
                    context_result=context_result,
                    docs=docs,
                    unique_sources=unique_sources,
                    finish_reason="no_context",
                )
                yield json.dumps({"type": "done", "content": "end"}) + "\n\n"
                return

            original_max_context_chars = self.max_context_chars
            self.max_context_chars = route.profile.max_context_chars
            try:
                self._last_retrieval_context = context_result
                context_text = self.format_docs(docs)
            finally:
                self.max_context_chars = original_max_context_chars
            formatted_prompt = self.prompt.format(conversation_context=conversation_context, context=context_text, question=query)
            yield json.dumps({"type": "status", "content": "답변을 생성하는 중입니다"}) + "\n\n"

            generation_start = time.perf_counter()
            first_token_at = None
            has_content = False
            finish_reason = None
            distinct_pages = {
                doc.metadata.get("page_id") or doc.metadata.get("title")
                for doc in docs
                if doc.metadata.get("page_id") or doc.metadata.get("title")
            }
            use_aggregation_context = has_month_scoped_confluence and len(distinct_pages) > 1
            generation_llm = self.llm
            if use_aggregation_context and self.aggregation_num_ctx > self.num_ctx:
                generation_llm = ChatOllama(
                    base_url=self.ollama_host,
                    model=self.model_name,
                    temperature=0,
                    num_ctx=self.aggregation_num_ctx,
                    num_predict=self.num_predict,
                    reasoning=self.reasoning,
                    keep_alive="24h",
                )
                print(f"[RAG] expanded aggregation context num_ctx={self.aggregation_num_ctx} pages={len(distinct_pages)}")

            try:
                stream = generation_llm.astream(formatted_prompt).__aiter__()
                loop = asyncio.get_running_loop()
                answer_deadline = loop.time() + self.timeout_seconds
                while True:
                    remaining_seconds = answer_deadline - loop.time()
                    if remaining_seconds <= 0:
                        raise asyncio.TimeoutError
                    try:
                        chunk = await asyncio.wait_for(anext(stream), timeout=remaining_seconds)
                    except StopAsyncIteration:
                        break
                    chunk_done_reason = (
                        getattr(chunk, "response_metadata", {}).get("done_reason")
                        or getattr(chunk, "response_metadata", {}).get("finish_reason")
                        or getattr(chunk, "generation_info", {}).get("finish_reason")
                    )
                    if chunk_done_reason in {"length", "max_tokens"}:
                        finish_reason = "length"
                    if chunk.content:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                            print(f"[TIMING] first_token={first_token_at - generation_start:.2f}s")
                        has_content = True
                        answer_chars += len(chunk.content)
                        answer_deadline = loop.time() + self.timeout_seconds
                        yield json.dumps({"type": "answer", "content": chunk.content}) + "\n\n"
            except (asyncio.TimeoutError, TimeoutError):
                print("[TIMING] generation inactivity timeout")
                finish_reason = "timeout"
                yield json.dumps({"type": "answer", "content": "\n\n응답 시간이 길어져 여기까지 생성했습니다."}) + "\n\n"

            generation_elapsed = time.perf_counter() - generation_start
            print(f"[TIMING] generation={generation_elapsed:.2f}s")

            if not has_content and finish_reason is None:
                finish_reason = "error"
                yield json.dumps({"type": "answer", "content": "응답 생성에 실패했습니다."}) + "\n\n"

            self.write_chat_query_log(
                query,
                start_time,
                context_result=context_result,
                docs=docs,
                unique_sources=unique_sources,
                context_text=context_text,
                finish_reason=finish_reason or "end",
                generation_seconds=generation_elapsed,
                answer_chars=answer_chars,
            )

            yield json.dumps({"type": "done", "content": "end", "finish_reason": finish_reason}) + "\n\n"

        except Exception as e:
            print(f"\n[STREAM] Error: {str(e)}")
            self.write_chat_query_log(
                query,
                start_time,
                context_result=context_result,
                docs=docs,
                unique_sources=unique_sources,
                context_text=context_text,
                finish_reason="error",
                generation_seconds=generation_elapsed,
                answer_chars=answer_chars,
                error=str(e),
            )
            yield json.dumps({"type": "answer", "content": f"\n\n오류 발생: {str(e)}"}) + "\n\n"
            yield json.dumps({"type": "done", "content": "end", "finish_reason": "error"}) + "\n\n"

        finally:
            total_elapsed = time.perf_counter() - start_time
            print(f"[TIMING] total={total_elapsed:.2f}s\n", flush=True)
