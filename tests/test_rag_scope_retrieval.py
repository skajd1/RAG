import asyncio
import json
import unittest
from types import SimpleNamespace

from langchain_core.documents import Document

import _paths  # noqa: F401
from long_document import LongDocumentPolicy
from query_logging import QueryLogWriter
from rag_chain import RAGChain
from retrieval import HybridRetriever, JiraStatusRetriever, KeywordRetriever, QueryKeywordAnalyzer


def point(point_id: str, metadata: dict, page_content: str):
    return SimpleNamespace(id=point_id, payload={"metadata": metadata, "page_content": page_content})


class PagedClient:
    def __init__(self, batches):
        self.batches = batches
        self.calls = 0
        self.kwargs = []

    def scroll(self, **kwargs):
        self.kwargs.append(kwargs)
        batch = self.batches[self.calls]
        self.calls += 1
        return batch


class RagScopeRetrievalTests(unittest.TestCase):
    def test_jira_summary_deduplicates_issue_chunks_and_prefers_explicit_project(self):
        metadata = {
            "source_type": "jira",
            "space": "AM",
            "space_name": "SAMPLE_MAINTENANCE",
            "content_type": "jira_issue",
            "issue_key": "AM-180",
            "title": "AM-180 26-04 샘플보험 정기점검",
            "issue_status": "DONE",
            "issue_status_category": "done",
            "assignee": "홍길동",
            "due_date": "2026-04-21",
        }
        chain = RAGChain.__new__(RAGChain)
        chain.client = PagedClient(
            [
                (
                    [
                        point("chunk-1", metadata, "Title: 26-04 샘플보험 정기점검\nStatus: DONE"),
                        point("chunk-2", metadata, "Description: 점검 결과 상세 내용"),
                    ],
                    None,
                ),
                (
                    [
                        point(
                            "DEMOAPP",
                            {
                                "source_type": "jira",
                                "content_type": "jira_issue",
                                "space": "DEMOAPP",
                                "space_name": "DEMOAPP",
                                "issue_key": "DEMOAPP-131",
                                "issue_status_category": "in_progress",
                            },
                            "DEMOAPP 진행중 이슈",
                        ),
                        point("am", metadata, "AM 진행중 이슈"),
                    ],
                    None,
                ),
            ]
        )

        docs = chain.build_jira_scope_summary_docs(
            [{"source_type": "jira", "space": "AM", "space_name": "SAMPLE_MAINTENANCE"}],
            "4월 점검 완료 목록 알려줘",
        )
        mentions = chain.infer_jira_summary_mentions("DEMOAPP 프로젝트에서 현재 진행중인 이슈가 뭐야?")

        self.assertEqual(docs[0].page_content.count("| AM-180 |"), 1)
        self.assertEqual([mention["space"] for mention in mentions], ["DEMOAPP"])

    def test_space_mention_scans_pages_and_prefers_title_match(self):
        chain = RAGChain.__new__(RAGChain)
        chain.expand_chunks = 20
        chain.client = PagedClient(
            [
                (
                    [point("first", {"source_type": "confluence", "space": "TEAM", "title": "AT Team 온보딩 문서"}, "사내 내부 회사 안내")],
                    "next-page",
                ),
                (
                    [point("target", {"source_type": "confluence", "space": "TEAM", "title": "2026 전사 전결 규정"}, "근태관리 팀장 결재")],
                    None,
                ),
            ]
        )

        docs = chain.mention_context_docs(
            [{"mention_type": "space", "source_type": "confluence", "space": "TEAM"}],
            "@AT_TEAM_SPACE 사내 전결 규정 알려줘",
        )

        self.assertEqual(chain.client.calls, 2)
        self.assertEqual(docs[0].metadata["title"], "2026 전사 전결 규정")

    def test_hybrid_retriever_reranks_shared_vector_and_keyword_hits(self):
        class VectorRetriever:
            def invoke(self, query):
                return [
                    Document(page_content="vector", metadata={"_id": "v", "title": "내용 문서"}),
                    Document(page_content="shared-vector", metadata={"_id": "s", "title": "샘플보험 점검"}),
                ]

        class KeywordRetrieverStub:
            def retrieve(self, query, limit=None):
                return [
                    Document(
                        page_content="shared-keyword",
                        metadata={"_id": "s", "title": "샘플보험 점검", "_bm25_score": 10, "_title_match_score": 1},
                    ),
                    Document(
                        page_content="keyword",
                        metadata={"_id": "k", "title": "점검 문서", "_bm25_score": 5, "_title_match_score": 0.5},
                    ),
                ]

        result = HybridRetriever(VectorRetriever(), KeywordRetrieverStub()).retrieve("샘플보험 점검")

        self.assertEqual(result.docs[0].metadata["_id"], "s")
        self.assertGreater(result.docs[0].metadata["_final_score"], result.docs[1].metadata["_final_score"])


class RoutedRetrievalIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def test_hybrid_retriever_uses_request_scoped_profile_without_mutating_defaults(self):
        class VectorRetriever:
            def __init__(self):
                self.search_kwargs = {"k": 5}
                self.seen_k = []

            def invoke(self, query):
                self.seen_k.append(self.search_kwargs.get("k"))
                return [Document(page_content="vector", metadata={"_id": "v", "title": "url guide"})]

        class RecordingRetriever:
            def __init__(self):
                self.seen_limits = []

            def retrieve(self, query, limit=None):
                self.seen_limits.append(limit)
                return []

            def is_status_query(self, query):
                return False

        vector = VectorRetriever()
        keyword = RecordingRetriever()
        jira = RecordingRetriever()
        profile = SimpleNamespace(vector_top_k=2, keyword_top_k=7, jira_top_k=9)

        HybridRetriever(vector, keyword, jira).retrieve("url", profile=profile)

        self.assertEqual(vector.seen_k, [2])
        self.assertEqual(vector.search_kwargs["k"], 5)
        self.assertEqual(keyword.seen_limits, [7])
        self.assertEqual(jira.seen_limits, [9])

    def test_retrieve_context_applies_route_profile_and_long_doc_policy(self):
        chain = RAGChain.__new__(RAGChain)
        chain.top_k = 5
        chain.expand_chunks = 20
        chain.max_context_chars = 18000
        chain.normalize_chat_history = lambda history: []
        chain.format_conversation_hint = lambda history: "No prior conversation context."
        chain.prior_source_anchor_texts = lambda sources: []
        chain.resolve_query_mentions = lambda query: []
        chain.infer_jira_summary_mentions = lambda query: []
        chain.merge_mentions = lambda explicit, inline: []
        chain.prior_source_docs = lambda sources, query, mentions: []
        chain.filter_docs_to_mention_scope = lambda docs, mentions: docs
        chain.build_jira_scope_summary_docs = lambda mentions, query: []
        chain.mention_context_docs = lambda mentions, query: []
        chain.requested_months = lambda query: set()
        chain.expand_retrieved_docs = lambda docs, profile=None: docs

        seen_profiles = []
        chain.hybrid_retriever = SimpleNamespace(
            retrieve=lambda query, profile=None: seen_profiles.append(profile)
            or SimpleNamespace(
                docs=[
                    Document(
                        page_content=f"chunk {index} " + ("TPS 130" if index == 12 else "noise"),
                        metadata={"title": "long", "page_id": "long", "chunk_index": index},
                    )
                    for index in range(25)
                ],
                vector_docs=[],
                keyword_docs=[],
                jira_docs=[],
            )
        )

        context = chain.retrieve_context("SAMPLE_APP V2 성능지표 문서에서 TPS 값 알려줘")

        self.assertEqual(context.route.name, "long_doc_lookup")
        self.assertEqual(seen_profiles[0].long_doc_neighbor_window, 4)
        self.assertEqual([doc.metadata["chunk_index"] for doc in context.docs], [8, 9, 10, 11, 12, 13, 14, 15, 16])
        self.assertEqual(chain.max_context_chars, 18000)

    async def test_stream_writes_route_log_and_keeps_generation_realtime(self):
        class ProgressiveLlm:
            async def astream(self, prompt):
                for content in ["첫째", "둘째"]:
                    await asyncio.sleep(0.01)
                    yield SimpleNamespace(content=content)

        chain = build_stream_chain(ProgressiveLlm(), 0.05)
        log_records = []
        chain.query_log_writer.write = lambda record: log_records.append(record)

        events = [json.loads(event.strip()) async for event in chain.stream_ask("show url")]
        answer = "".join(event["content"] for event in events if event["type"] == "answer")

        self.assertEqual(answer, "첫째둘째")
        self.assertEqual(log_records[0]["route"], "exact_lookup")
        self.assertIsNone(events[-1]["finish_reason"])


class LongDocumentPolicyTests(unittest.TestCase):
    def test_targeted_window_selects_late_matching_chunk_and_neighbors(self):
        docs = [
            Document(
                page_content=f"chunk {index} " + ("TPS 130" if index == 45 else "noise"),
                metadata={"page_id": "long", "chunk_index": index, "title": "SAMPLE_APP V2 성능지표"},
            )
            for index in range(53)
        ]

        selected, stats = LongDocumentPolicy(neighbor_window=2).select_targeted_window(docs, query="TPS 값 알려줘")

        self.assertEqual([doc.metadata["chunk_index"] for doc in selected], [43, 44, 45, 46, 47])
        self.assertEqual(stats["long_document_mode"], "lookup")

    def test_summary_windows_cover_extra_large_document(self):
        docs = [Document(page_content=f"chunk {index}", metadata={"page_id": "long", "chunk_index": index}) for index in range(18)]

        windows = LongDocumentPolicy(summary_window_chunks=5).summary_windows(docs)

        self.assertEqual([len(window) for window in windows], [5, 5, 5, 3])
        self.assertEqual([doc.metadata["chunk_index"] for window in windows for doc in window], list(range(18)))


class ChatHistoryMemoryTests(unittest.TestCase):
    def build_chain(self):
        chain = RAGChain.__new__(RAGChain)
        chain.chat_history_turns = 4
        chain.chat_history_max_chars = 220
        chain.conversation_hint_max_chars = 80
        return chain

    def test_retrieval_query_excludes_raw_history_and_short_followups_reuse_sources(self):
        chain = self.build_chain()
        history = [
            {"role": "user", "content": "홍길동 주간보고 알려줘"},
            {"role": "assistant", "content": "assistant answer should not pollute retrieval"},
        ]

        self.assertEqual(chain.build_retrieval_query("다른사람은?", history), "다른사람은?")
        self.assertTrue(chain.should_reuse_prior_sources("다른 사람은?", []))
        self.assertFalse(chain.should_reuse_prior_sources("홍길동 주간보고 요약해줘", []))

    def test_prior_source_docs_fetches_by_page_id(self):
        chain = self.build_chain()
        chain.prior_source_limit = 4
        chain.prior_source_chunks = 12
        chain.client = PagedClient(
            [
                (
                    [point("chunk-1", {"page_id": "p1", "title": "문서", "chunk_index": 0, "source_type": "confluence"}, "본문")],
                    None,
                )
            ]
        )

        docs = chain.prior_source_docs([{"page_id": "p1", "source_type": "confluence"}], "표로 정리해줘", [])

        self.assertEqual(docs[0].metadata["page_id"], "p1")
        self.assertEqual(chain.client.kwargs[0]["scroll_filter"].should[0].match.value, "p1")


class KeywordRetrievalTests(unittest.TestCase):
    def test_keyword_analyzer_handles_calendar_dates_weeks_and_synonyms(self):
        date_terms = QueryKeywordAnalyzer().extract("26년 5월 26일자로 기술회의록 알려줘")
        week_terms = QueryKeywordAnalyzer().extract("26년 5월 4주차 홍길동 주간보고 요약해줘")

        self.assertIn("0526", date_terms["exact"])
        self.assertNotIn("260526주차", date_terms["exact"])
        self.assertIn("26054주차", week_terms["exact"])
        self.assertIn("기술회의", date_terms["broad"])

    def test_keyword_retrieval_finds_compact_date_and_synonym_title_pages(self):
        batch = (
            [
                point(
                    "meeting",
                    {"title": "0526 주간회의", "breadcrumb": "기술연구소 > 26년 주간 기술회의 > 0526 주간회의", "page_id": "meeting"},
                    "기술백서 리뷰",
                ),
                point(
                    "policy",
                    {"title": "2026 전사 전결 규정", "breadcrumb": "SAMPLE TEAM SPACE > 전사 / 허브 / 팀 공지사항", "page_id": "policy"},
                    "근태관리 팀장 결재",
                ),
            ],
            None,
        )

        meeting_docs = KeywordRetriever(PagedClient([batch]), "confluence_docs", limit=3).retrieve("26년 5월 26일자로 기술회의록 알려줘")
        policy_docs = KeywordRetriever(PagedClient([batch]), "confluence_docs", limit=3).retrieve("사내 전결 규정 알려줘")

        self.assertEqual(meeting_docs[0].metadata["page_id"], "meeting")
        self.assertEqual(policy_docs[0].metadata["page_id"], "policy")


class RagStreamFinishReasonTests(unittest.IsolatedAsyncioTestCase):
    async def test_scope_changed_followup_rewrites_query_without_reusing_prior_sources(self):
        seen_retrieval_queries = []
        seen_prior_sources = []

        class FastLlm:
            async def astream(self, prompt):
                yield SimpleNamespace(content="답변")

        chain = build_stream_chain(FastLlm(), 1)
        chain.prior_source_docs = lambda sources, query, mentions: seen_prior_sources.extend(sources or []) or []
        chain.hybrid_retriever = SimpleNamespace(
            retrieve=lambda query, profile=None: seen_retrieval_queries.append(query)
            or SimpleNamespace(
                docs=[Document(page_content="본문", metadata={"title": "문서", "page_id": "p1", "url": "u1"})],
                vector_docs=[],
                keyword_docs=[],
                jira_docs=[],
            )
        )

        async for _ in chain.stream_ask(
            "3주차도 보여줘",
            [],
            [{"role": "user", "content": "홍길동의 5월 4주차 주간보고 요약해줘"}],
            [{"page_id": "weekly-4"}],
        ):
            pass

        self.assertEqual(seen_retrieval_queries, ["홍길동 5월 3주차 주간보고"])
        self.assertEqual(seen_prior_sources, [])

    async def test_stream_fans_out_week_range_queries_by_bucket(self):
        seen_retrieval_queries = []

        class FastLlm:
            async def astream(self, prompt):
                yield SimpleNamespace(content="답변")

        def retrieve(query, profile=None):
            seen_retrieval_queries.append(query)
            week = query.split("주차")[0].split()[-1]
            return SimpleNamespace(
                docs=[Document(page_content=f"{week}주차 내용", metadata={"title": f"홍길동 5월 {week}주차 주간보고", "page_id": f"w{week}"})],
                vector_docs=[],
                keyword_docs=[],
                jira_docs=[],
            )

        chain = build_stream_chain(FastLlm(), 1)
        chain.prior_source_docs = lambda sources, query, mentions: []
        chain.hybrid_retriever = SimpleNamespace(retrieve=retrieve)

        async for _ in chain.stream_ask("홍길동 5월 1주차~3주차 주간보고 표로 정리해줘"):
            pass

        self.assertEqual(
            seen_retrieval_queries,
            [
                "홍길동 5월 1주차 주간보고 표로 정리해줘",
                "홍길동 5월 2주차 주간보고 표로 정리해줘",
                "홍길동 5월 3주차 주간보고 표로 정리해줘",
            ],
        )

    async def test_timeout_and_length_finish_reasons_are_reported(self):
        class SlowLlm:
            async def astream(self, prompt):
                await asyncio.sleep(0.02)
                yield SimpleNamespace(content="too late")

        class LengthLimitedLlm:
            async def astream(self, prompt):
                yield SimpleNamespace(content="partial", response_metadata={"done_reason": "length"})

        timeout_events = [json.loads(event.strip()) async for event in build_stream_chain(SlowLlm(), 0.001).stream_ask("질문")]
        length_events = [json.loads(event.strip()) async for event in build_stream_chain(LengthLimitedLlm(), 1).stream_ask("질문")]

        self.assertEqual(timeout_events[-1]["finish_reason"], "timeout")
        self.assertEqual(length_events[-1]["finish_reason"], "length")


def build_stream_chain(llm, timeout_seconds):
    chain = RAGChain.__new__(RAGChain)
    chain.timeout_seconds = timeout_seconds
    chain.top_k = 5
    chain.expand_chunks = 20
    chain.max_context_chars = 18000
    chain.num_ctx = 4096
    chain.aggregation_num_ctx = 4096
    chain.ollama_host = "http://localhost:11434"
    chain.model_name = "test"
    chain.num_predict = 100
    chain.reasoning = False
    chain.chat_history_turns = 4
    chain.chat_history_max_chars = 220
    chain.conversation_hint_max_chars = 80
    chain.prior_source_limit = 4
    chain.prior_source_chunks = 12
    chain.prior_source_anchor_texts = lambda sources: []
    chain.resolve_query_mentions = lambda query: []
    chain.infer_jira_summary_mentions = lambda query: []
    chain.merge_mentions = lambda explicit, inline: []
    chain.filter_docs_to_mention_scope = lambda docs, mentions: docs
    chain.build_jira_scope_summary_docs = lambda mentions, query: []
    chain.mention_context_docs = lambda mentions, query: []
    chain.expand_retrieved_docs = lambda docs, profile=None: docs
    chain.format_docs = lambda docs: docs[0].page_content
    chain.prompt = SimpleNamespace(format=lambda **kwargs: "prompt")
    chain.query_log_writer = QueryLogWriter(enabled=False)
    chain.hybrid_retriever = SimpleNamespace(
        retrieve=lambda query, profile=None: SimpleNamespace(
            docs=[Document(page_content="본문", metadata={"title": "문서", "page_id": "p1", "url": "u1"})],
            vector_docs=[],
            keyword_docs=[],
            jira_docs=[],
        )
    )
    chain.llm = llm
    return chain


if __name__ == "__main__":
    unittest.main()
